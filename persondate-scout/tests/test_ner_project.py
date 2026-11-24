import io

import pandas as pd
import pytest

from ner_project import (
    DatasetValidationError,
    build_augmented_dataset,
    build_feature_matrix,
    build_spacy_preprocessor,
    extract_handcrafted_features,
    load_dataset,
    predict_sentence,
    preprocess_text,
    split_data,
    train_pipeline,
)


def test_load_dataset_normalizes_name_header():
    csv = io.BytesIO(b"Sentence,Name,Date\nA person met on January 1,2024,Alex,January 1\n")
    df = load_dataset(csv)
    assert df.columns.tolist() == ["sentence", "person", "date"]
    assert df.iloc[0]["person"] == "Alex"


def test_load_dataset_rejects_missing_columns():
    csv = io.BytesIO(b"Sentence,Date\nHello,2024\n")
    with pytest.raises(DatasetValidationError):
        load_dataset(csv)


def test_load_dataset_rejects_empty_sentence():
    csv = io.BytesIO(b"Sentence,Name,Date\n,Alex,2024\n")
    with pytest.raises(DatasetValidationError):
        load_dataset(csv)


def test_preprocessing_removes_punctuation_stopwords_and_lowercases():
    nlp = build_spacy_preprocessor()
    output = preprocess_text("The Elon Musk!", nlp)
    assert output == "elon musk"


def test_handcrafted_features():
    frame = extract_handcrafted_features(["Elon Musk founded Tesla in 2003"])
    assert frame.iloc[0].to_dict() == {
        "has_capitalized_word": 1,
        "has_year": 1,
        "sentence_length": 6,
    }


def test_augmentation_and_grouped_split_keep_source_rows_together():
    df = pd.DataFrame(
        {
            "sentence": ["Alice met Bob on January 1", "Carol met Dan on February 2", "Eve met Frank on March 3"],
            "person": ["Alice", "Carol", "Eve"],
            "date": ["January 1", "February 2", "March 3"],
        }
    )
    nlp = build_spacy_preprocessor()
    x, y, source_ids = build_augmented_dataset(df, nlp)
    x_train, x_test, y_train, y_test = split_data(x, y, source_ids)
    assert len(x_train) > 0
    assert len(x_test) > 0
    train_ids = set(source_ids[i] for i in x_train.index)
    test_ids = set(source_ids[i] for i in x_test.index)
    assert train_ids.isdisjoint(test_ids)


def test_training_and_prediction_smoke():
    result = train_pipeline("ner_data.csv")
    assert set(result["lr_results"]) >= {"accuracy", "precision", "recall", "f1", "predictions"}
    assert set(result["nb_results"]) >= {"accuracy", "precision", "recall", "f1", "predictions"}

    labels, probabilities = predict_sentence(
        "Elon Musk founded Tesla in 2003",
        result["nlp"],
        result["vectorizer"],
        result["best_model"],
    )
    assert len(labels) == 2
    assert all(label in (0, 1) for label in labels)
    assert len(probabilities) == 2


def test_prediction_rejects_empty_input():
    result = train_pipeline("ner_data.csv")
    with pytest.raises(DatasetValidationError):
        predict_sentence("", result["nlp"], result["vectorizer"], result["best_model"])
