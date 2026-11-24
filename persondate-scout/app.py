"""PersonDate Scout Streamlit application."""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from ner_project import (
    DatasetValidationError,
    load_dataset,
    predict_sentence,
    train_pipeline,
)


st.set_page_config(page_title="PersonDate Scout", page_icon="", layout="wide")


@st.cache_data(show_spinner=False)
def load_uploaded_csv(uploaded_bytes: bytes) -> pd.DataFrame:
    return load_dataset(io.BytesIO(uploaded_bytes))


@st.cache_resource(show_spinner=False)
def get_training_result(csv_path: str, uploaded_bytes: bytes | None = None):
    if uploaded_bytes is not None:
        dataset = load_dataset(io.BytesIO(uploaded_bytes))
        # Use bytes through the core pipeline without rewriting a temporary file.
        return train_dataset(dataset)
    return train_pipeline(csv_path)


def train_dataset(dataset: pd.DataFrame):
    """Train from an already validated DataFrame, used by CSV uploads."""
    from ner_project import (
        build_augmented_dataset,
        build_spacy_preprocessor,
        build_feature_matrix,
        evaluate_model,
        split_data,
        train_models,
    )
    from sklearn.feature_extraction.text import TfidfVectorizer

    nlp = build_spacy_preprocessor()
    x_processed, y, source_ids = build_augmented_dataset(dataset, nlp)
    x_train, x_test, y_train, y_test = split_data(x_processed, y, source_ids)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    x_train_vec = build_feature_matrix(vectorizer, x_train, fit=True)
    x_test_vec = build_feature_matrix(vectorizer, x_test, fit=False)
    models = train_models(x_train_vec, y_train)
    lr = evaluate_model("Logistic Regression", models["Logistic Regression"], x_test_vec, y_test)
    nb = evaluate_model("Naive Bayes", models["Naive Bayes"], x_test_vec, y_test)
    best_name = "Logistic Regression" if lr["f1"] >= nb["f1"] else "Naive Bayes"
    return {
        "dataset": dataset,
        "augmented_rows": len(x_processed),
        "source_rows": len(dataset),
        "nlp": nlp,
        "vectorizer": vectorizer,
        "models": models,
        "x_test": x_test,
        "y_test": y_test,
        "lr_results": lr,
        "nb_results": nb,
        "best_model_name": best_name,
        "best_model": models[best_name],
    }


def build_metrics_table(result: dict) -> pd.DataFrame:
    rows = []
    for key in ("lr_results", "nb_results"):
        metrics = result[key]
        rows.append(
            {
                "Model": metrics["name"],
                "Exact-match accuracy": metrics["accuracy"],
                "Micro precision": metrics["precision"],
                "Micro recall": metrics["recall"],
                "Micro F1": metrics["f1"],
            }
        )
    return pd.DataFrame(rows)


def create_confusion_figure(y_test: pd.DataFrame, lr_pred, nb_pred):
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    figure.suptitle("PersonDate Scout: Confusion Matrices", fontsize=16)
    plots = (
        (axes[0, 0], y_test["person"], lr_pred[:, 0], "Logistic Regression - Person"),
        (axes[0, 1], y_test["date"], lr_pred[:, 1], "Logistic Regression - Date"),
        (axes[1, 0], y_test["person"], nb_pred[:, 0], "Naive Bayes - Person"),
        (axes[1, 1], y_test["date"], nb_pred[:, 1], "Naive Bayes - Date"),
    )
    for axis, actual, predicted, title in plots:
        matrix = pd.crosstab(pd.Series(actual, name="Actual"), pd.Series(predicted, name="Predicted"), dropna=False)
        matrix = matrix.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(title)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
    figure.tight_layout()
    return figure


def render_confidence(probabilities: list[list[float]], label: int):
    return probabilities[label]


def main() -> None:
    st.title("PersonDate Scout")
    st.caption("Sentence-level classification for person-related and date-related information")

    st.info(
        "This tool predicts whether a sentence contains person-related and/or date-related information. "
        "It is a sentence classifier, not a token-level entity extractor."
    )

    with st.expander("Expected CSV format", expanded=False):
        st.code("Sentence,Name,Date\n\"Elon Musk founded Tesla on March 1, 2003\",\"Elon Musk\",\"March 1, 2003\"")
        st.caption("Required columns: Sentence, Person/Name, and Date.")

    tab_upload, tab_path = st.tabs(["Upload CSV", "Use file path"])
    uploaded = tab_upload.file_uploader("Training dataset", type=["csv"])

    csv_path = tab_path.text_input("CSV file path", value="ner_data.csv")

    if "training_result" not in st.session_state:
        st.session_state.training_result = None

    train_requested = st.button("Load data and train models", type="primary")
    if train_requested:
        try:
            with st.spinner("Validating data and training the baseline models..."):
                if uploaded is not None:
                    data = load_uploaded_csv(uploaded.getvalue())
                    st.session_state.training_result = train_dataset(data)
                else:
                    st.session_state.training_result = train_pipeline(csv_path)
        except (FileNotFoundError, DatasetValidationError) as exc:
            st.session_state.training_result = None
            st.error(f"Could not train: {exc}")
        except Exception as exc:
            st.session_state.training_result = None
            st.error(f"Training failed unexpectedly: {exc}")

    result = st.session_state.training_result
    if result is None:
        st.stop()

    st.divider()
    st.subheader("Model comparison")
    metrics = build_metrics_table(result)
    st.dataframe(metrics.style.format({
        "Exact-match accuracy": "{:.4f}",
        "Micro precision": "{:.4f}",
        "Micro recall": "{:.1%}",
        "Micro F1": "{:.4f}",
    }), use_container_width=True, hide_index=True)

    st.success(f"Selected model: **{result['best_model_name']}**, based on micro F1 score.")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Held-out confusion matrices")
        figure = create_confusion_figure(
            result["y_test"],
            result["lr_results"]["predictions"],
            result["nb_results"]["predictions"],
        )
        st.pyplot(figure)
        plt.close(figure)

    with col2:
        st.subheader("Dataset")
        st.metric("Annotated source rows", result["source_rows"])
        st.metric("Augmented dataset rows", result["augmented_rows"])
        st.caption("The app reports held-out source groups; augmentation is not a substitute for real negative examples.")

        csv_export = result["dataset"].to_csv(index=False).encode("utf-8")
        st.download_button("Download validated dataset", csv_export, "persondate_scout_validated.csv", "text/csv")

    st.divider()
    st.subheader("Classify a sentence")
    user_sentence = st.text_area(
        "Sentence",
        value="Elon Musk founded Tesla on March 1, 2003",
        height=100,
        help="Enter one complete sentence in English.",
    )

    if st.button("Analyze sentence"):
        if not user_sentence.strip():
            st.warning("Enter a sentence before analysis.")
        else:
            try:
                labels, probabilities = predict_sentence(
                    user_sentence,
                    result["nlp"],
                    result["vectorizer"],
                    result["best_model"],
                )

                person_col, date_col = st.columns(2)
                person_col.metric(
                    "Person-related",
                    "Detected" if labels[0] else "Not detected",
                    f"{render_confidence(probabilities[0], labels[0]):.1%} confidence",
                )
                date_col.metric(
                    "Date-related",
                    "Detected" if labels[1] else "Not detected",
                    f"{render_confidence(probabilities[1], labels[1]):.1%} confidence",
                )

                output = pd.DataFrame(
                    {
                        "sentence": [user_sentence],
                        "person_related": [bool(labels[0])],
                        "date_related": [bool(labels[1])],
                        "person_confidence": [render_confidence(probabilities[0], labels[0])],
                        "date_confidence": [render_confidence(probabilities[1], labels[1])],
                    }
                )
                st.download_button("Download result", output.to_csv(index=False), "persondate_scout_result.csv", "text/csv")
            except DatasetValidationError as exc:
                st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
