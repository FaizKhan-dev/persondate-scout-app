# PersonDate Scout Methodology

## Task definition

PersonDate Scout performs binary classification for two labels:

- `person`: whether the sentence contains person-related information.
- `date`: whether the sentence contains date-related information.

This is not token-level extraction. The system does not identify character spans or produce BIO tags.

## Text preprocessing

The application uses a blank spaCy English pipeline. It lowercases text, tokenizes it, removes stop words and punctuation, and retains alphabetic tokens.

## Features

### TF-IDF

The TF-IDF vectorizer uses unigrams and bigrams, up to 5,000 features, and sublinear term-frequency scaling. It is fitted only on training text.

### Handcrafted features

- Whether any word begins with a capital letter.
- Whether the text contains a four-digit year beginning with 19 or 20.
- Number of processed words.

## Models

The baseline models are:

- Logistic Regression with 1,000 maximum iterations.
- Multinomial Naive Bayes.

Both labels are trained using `MultiOutputClassifier` wrappers.

## Evaluation

The implementation reports exact-match multi-label accuracy and micro-averaged precision, recall, and F1. The Streamlit and CLI model-selection rules use micro F1.

## Leakage controls

Synthetic variants retain source-row IDs. The split keeps variants from the same source row together, preventing a person-negative and its original sentence from appearing in both partitions.

## Data limitations

The bundled dataset is small and narrow. Synthetic negatives replace real negative examples and can make the task easier than natural text. Metrics are local demonstration evidence and are not suitable for production claims.

## Confidence interpretation

Class probabilities are returned by each estimator's `predict_proba` method. They indicate the model's relative score for the selected class. They are not calibrated estimates of real-world correctness.
