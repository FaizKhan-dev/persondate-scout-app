"""PersonDate Scout core pipeline.

This module implements sentence-level multi-label classification for person and
date presence. It is not token-level named entity recognition.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import spacy
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB

RANDOM_STATE = 42
TARGET_COLUMNS = ("person", "date")


class DatasetValidationError(ValueError):
    """Raised when input data cannot support training or prediction."""


def load_dataset(csv_path: str | os.PathLike[str] | Iterable[str]) -> pd.DataFrame:
    """Load and validate a CSV, normalizing the required headers."""
    if isinstance(csv_path, (str, os.PathLike)):
        path = Path(csv_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        if not path.is_file():
            raise DatasetValidationError(f"CSV path is not a file: {path}")
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError as exc:
            raise DatasetValidationError("The CSV file is empty.") from exc
        except pd.errors.ParserError as exc:
            raise DatasetValidationError(f"The CSV could not be parsed: {exc}") from exc
    else:
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError as exc:
            raise DatasetValidationError("The CSV content is empty.") from exc
        except pd.errors.ParserError as exc:
            raise DatasetValidationError(f"The CSV could not be parsed: {exc}") from exc

    if df.empty:
        raise DatasetValidationError("The dataset has no data rows.")

    df.columns = [str(column).strip().lower() for column in df.columns]
    if "name" in df.columns and "person" not in df.columns:
        df = df.rename(columns={"name": "person"})

    required = {"sentence", "person", "date"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise DatasetValidationError("Dataset columns are missing: " + ", ".join(missing))

    df = df[["sentence", "person", "date"]].copy()
    df["sentence"] = df["sentence"].fillna("").astype(str).str.strip()
    df["person"] = df["person"].fillna("").astype(str).str.strip()
    df["date"] = df["date"].fillna("").astype(str).str.strip()

    empty_count = int(df["sentence"].eq("").sum())
    if empty_count:
        raise DatasetValidationError(
            f"Found {empty_count} row(s) with an empty sentence. Remove or fill them before training."
        )
    return df


def build_spacy_preprocessor() -> spacy.language.Language:
    """Build the lightweight English preprocessing pipeline."""
    return spacy.blank("en")


def preprocess_text(text: str, nlp: spacy.language.Language) -> str:
    """Lowercase, tokenize, remove stop words/punctuation, and retain words."""
    doc = nlp(str(text).lower())
    return " ".join(token.text for token in doc if token.is_alpha and not token.is_stop)


def prepare_features(df: pd.DataFrame, nlp: spacy.language.Language) -> Tuple[pd.Series, pd.DataFrame]:
    """Create processed text and binary labels from a validated dataset."""
    x = df["sentence"].apply(lambda sentence: preprocess_text(sentence, nlp))
    y = pd.DataFrame(
        {
            "person": df["person"].ne("").astype(int),
            "date": df["date"].ne("").astype(int),
        }
    )
    return x, y


def extract_handcrafted_features(sentences: Iterable[str]) -> pd.DataFrame:
    """Create three small explainable features alongside TF-IDF features."""
    year_pattern = re.compile(r"\b(?:19|20)\d{2}\b")
    rows = []
    for sentence in sentences:
        text = str(sentence)
        words = text.split()
        rows.append(
            {
                "has_capitalized_word": int(any(word[:1].isupper() for word in words if word)),
                "has_year": int(bool(year_pattern.search(text))),
                "sentence_length": len(words),
            }
        )
    return pd.DataFrame(rows)


def replace_entity(sentence: str, entity: str, replacement: str) -> str:
    """Replace the first exact entity mention, case-insensitively."""
    if not entity:
        return sentence
    updated, count = re.subn(re.escape(entity), replacement, sentence, count=1, flags=re.IGNORECASE)
    return updated if count else sentence


def build_augmented_dataset(
    df: pd.DataFrame,
    nlp: spacy.language.Language,
) -> Tuple[pd.Series, pd.DataFrame, list[int]]:
    """Build synthetic negatives while retaining source-row IDs for grouping.

    Synthetic augmentation creates controlled negatives for the small sample
    dataset. It does not replace the need for a larger real-world dataset.
    """
    augmented_sentences: list[str] = []
    augmented_labels: list[tuple[int, int]] = []
    source_ids: list[int] = []

    for source_id, row in df.reset_index(drop=True).iterrows():
        sentence = str(row["sentence"])
        person = str(row["person"])
        date = str(row["date"])

        person_present = bool(person) and re.search(re.escape(person), sentence, re.IGNORECASE) is not None
        date_present = bool(date) and re.search(re.escape(date), sentence, re.IGNORECASE) is not None

        original_person_label = int(person_present)
        original_date_label = int(date_present)

        augmented_sentences.append(sentence)
        augmented_labels.append((original_person_label, original_date_label))
        source_ids.append(source_id)

        if person_present:
            person_negative = replace_entity(sentence, person, "[person omitted]")
            if person_negative != sentence:
                augmented_sentences.append(person_negative)
                augmented_labels.append((0, original_date_label))
                source_ids.append(source_id)

        if date_present:
            date_negative = replace_entity(sentence, date, "[date omitted]")
            if date_negative != sentence:
                augmented_sentences.append(date_negative)
                augmented_labels.append((original_person_label, 0))
                source_ids.append(source_id)

    x_processed = pd.Series(augmented_sentences, name="sentence")
    y = pd.DataFrame(augmented_labels, columns=list(TARGET_COLUMNS))
    x_processed = x_processed.apply(lambda sentence: preprocess_text(sentence, nlp))

    if x_processed.empty:
        raise DatasetValidationError("No usable training rows could be created.")
    return x_processed, y, source_ids


def split_data(
    x: pd.Series,
    y: pd.DataFrame,
    source_ids: list[int] | None = None,
    test_size: float = 0.20,
):
    """Create a reproducible held-out split, keeping source variants together."""
    if len(x) == 0:
        raise DatasetValidationError("The dataset is empty after preprocessing.")
    if len(x) != len(y):
        raise DatasetValidationError("Text and label counts do not match.")

    if source_ids is not None:
        unique_ids = sorted(set(source_ids))
        if len(unique_ids) < 2:
            raise DatasetValidationError("At least two source rows are required for a train/test split.")

        source_labels: dict[int, str] = {}
        for index, source_id in enumerate(source_ids):
            source_labels.setdefault(
                source_id,
                f"{int(y.iloc[index]['person'])}_{int(y.iloc[index]['date'])}",
            )

        counts = Counter(source_labels.values())
        stratify_ids = [source_labels[source_id] for source_id in unique_ids]
        use_stratify = all(counts[label] >= 2 for label in set(stratify_ids))

        test_count = max(1, int(len(unique_ids) * test_size))
        test_count = min(test_count, len(unique_ids) - 1)
        train_ids, test_ids = train_test_split(
            unique_ids,
            test_size=test_count,
            random_state=RANDOM_STATE,
            stratify=stratify_ids if use_stratify else None,
        )
        train_set, test_set = set(train_ids), set(test_ids)

        train_mask = [source_id in train_set for source_id in source_ids]
        test_mask = [source_id in test_set for source_id in source_ids]
        train_indices = [i for i, keep in enumerate(train_mask) if keep]
        test_indices = [i for i, keep in enumerate(test_mask) if keep]
        if not train_indices or not test_indices:
            raise DatasetValidationError("Could not create a non-empty train/test split.")

        return (
            x.iloc[train_indices],
            x.iloc[test_indices],
            y.iloc[train_indices],
            y.iloc[test_indices],
        )

    n = len(x)
    test_count = max(1, int(n * test_size))
    test_count = min(test_count, n - 1)
    stratify_labels = (y["person"].astype(str) + "_" + y["date"].astype(str)).values
    counts = Counter(stratify_labels)
    stratify = stratify_labels if all(count >= 2 for count in counts.values()) else None
    return train_test_split(
        x,
        y,
        test_size=test_count,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )


def train_models(x_train_vec, y_train) -> Dict[str, MultiOutputClassifier]:
    """Train reproducible baseline classifiers."""
    models = {
        "Logistic Regression": MultiOutputClassifier(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "Naive Bayes": MultiOutputClassifier(MultinomialNB()),
    }
    for model in models.values():
        model.fit(x_train_vec, y_train)
    return models


def evaluate_model(name: str, model, x_test_vec, y_test) -> Dict[str, object]:
    """Evaluate exact-match accuracy and micro-averaged label metrics."""
    y_pred = model.predict(x_test_vec)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="micro",
        zero_division=0,
    )
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predictions": y_pred,
    }


def build_feature_matrix(vectorizer: TfidfVectorizer, sentences: pd.Series, fit: bool):
    """Fit or transform TF-IDF and combine it with handcrafted features."""
    tfidf = vectorizer.fit_transform(sentences) if fit else vectorizer.transform(sentences)
    handcrafted = extract_handcrafted_features(sentences).values
    return hstack([tfidf, handcrafted]).tocsr()


def train_pipeline(csv_path: str):
    """Load, split, fit, train, and evaluate a PersonDate Scout pipeline."""
    dataset = load_dataset(csv_path)
    nlp = build_spacy_preprocessor()
    x_processed, y, source_ids = build_augmented_dataset(dataset, nlp)
    x_train, x_test, y_train, y_test = split_data(x_processed, y, source_ids)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    x_train_vec = build_feature_matrix(vectorizer, x_train, fit=True)
    x_test_vec = build_feature_matrix(vectorizer, x_test, fit=False)

    models = train_models(x_train_vec, y_train)
    lr_results = evaluate_model("Logistic Regression", models["Logistic Regression"], x_test_vec, y_test)
    nb_results = evaluate_model("Naive Bayes", models["Naive Bayes"], x_test_vec, y_test)

    best_name = "Logistic Regression" if lr_results["f1"] >= nb_results["f1"] else "Naive Bayes"
    return {
        "dataset": dataset,
        "x_processed": x_processed,
        "augmented_rows": len(x_processed),
        "source_rows": len(dataset),
        "nlp": nlp,
        "vectorizer": vectorizer,
        "models": models,
        "x_test": x_test,
        "y_test": y_test,
        "lr_results": lr_results,
        "nb_results": nb_results,
        "best_model_name": best_name,
        "best_model": models[best_name],
    }


def predict_sentence(sentence: str, nlp, vectorizer, model) -> Tuple[list[int], list[list[float]]]:
    """Predict both labels and the model-estimator probability for each class."""
    original = str(sentence).strip()
    if not original:
        raise DatasetValidationError("Enter a sentence before predicting.")

    processed = preprocess_text(original, nlp)
    if not processed:
        raise DatasetValidationError("The sentence does not contain usable alphabetic text.")

    tfidf = vectorizer.transform([processed])
    handcrafted = extract_handcrafted_features([processed]).values
    features = hstack([tfidf, handcrafted]).tocsr()
    prediction = model.predict(features)[0].tolist()

    probabilities = []
    for estimator in model.estimators_:
        if hasattr(estimator, "predict_proba"):
            probabilities.append(estimator.predict_proba(features)[0].tolist())
        else:
            probabilities.append([0.0, 1.0])
    return prediction, probabilities


def plot_confusion_matrices(y_test: pd.DataFrame, lr_pred, nb_pred):
    """Create four confusion-matrix plots for CLI use."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    figure.suptitle("PersonDate Scout: Confusion Matrices", fontsize=16)
    plots = (
        (axes[0, 0], y_test["person"], lr_pred[:, 0], "Logistic Regression - Person"),
        (axes[0, 1], y_test["date"], lr_pred[:, 1], "Logistic Regression - Date"),
        (axes[1, 0], y_test["person"], nb_pred[:, 0], "Naive Bayes - Person"),
        (axes[1, 1], y_test["date"], nb_pred[:, 1], "Naive Bayes - Date"),
    )
    for axis, actual, predicted, title in plots:
        matrix = confusion_matrix(actual, predicted, labels=[0, 1])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(title)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
    figure.tight_layout()
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description="PersonDate Scout CLI")
    parser.add_argument("--csv-path", default="ner_data.csv", help="Path to a sentence/person/date CSV")
    args = parser.parse_args()

    try:
        result = train_pipeline(args.csv_path)
    except (FileNotFoundError, DatasetValidationError) as exc:
        raise SystemExit(f"Unable to train: {exc}")

    print("\nPersonDate Scout Model Comparison")
    for key in ("lr_results", "nb_results"):
        metrics = result[key]
        print(
            f"{metrics['name']}: accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}"
        )
    print(f"Selected model: {result['best_model_name']}")

    try:
        import matplotlib.pyplot as plt
        figure = plot_confusion_matrices(
            result["y_test"],
            result["lr_results"]["predictions"],
            result["nb_results"]["predictions"],
        )
        plt.show()
    except Exception as exc:
        print(f"Could not display confusion matrices: {exc}")

    sentence = input("\nEnter a sentence to classify (or press Enter to exit): ").strip()
    if sentence:
        try:
            labels, probabilities = predict_sentence(
                sentence,
                result["nlp"],
                result["vectorizer"],
                result["best_model"],
            )
        except DatasetValidationError as exc:
            print(f"Prediction error: {exc}")
            return
        print("Person-related: " + ("Yes" if labels[0] else "No"))
        print("Date-related: " + ("Yes" if labels[1] else "No"))
        print("Person confidence: {:.1%}".format(probabilities[0][labels[0]]))
        print("Date confidence: {:.1%}".format(probabilities[1][labels[1]]))


if __name__ == "__main__":
    main()
