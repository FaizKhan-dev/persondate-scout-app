# Project Audit: Named Entity Recognition Demo

## Final Implementation Update — 2026-11-24

PersonDate Scout has been improved from a demo-style script to a reusable local pipeline with a Streamlit interface, CSV upload support, grouped source-row splitting, input validation, tests, dependency metadata, and technical documentation.

Verified local results for the bundled 9-row dataset were:

- Logistic Regression: exact-match accuracy `1.0000`, micro precision `1.0000`, micro recall `1.0000`, micro F1 `1.0000`.
- Naive Bayes: exact-match accuracy `0.3333`, micro precision `0.6667`, micro recall `1.0000`, micro F1 `0.8000`.

These metrics use synthetic augmented examples and a small dataset. They are demonstration results, not generalizable production benchmarks.

## Executive Summary

This repository is a small, educational Python project that trains text classifiers to decide whether a sentence contains a **person** entity and/or a **date** entity. It is not a conventional token-level Named Entity Recognition system: for each complete sentence it predicts two binary labels, while spaCy's pretrained English model is used separately to extract person/date text for display in the Streamlit interface.

The portfolio-ready boundary is currently a prototype/demonstration. The code shows a complete local training-to-prediction loop, but it has limited data, synthetic labels, no model persistence/API/database, inconsistent project packaging, no automated test suite, and several correctness and maintainability risks.

## 1. What the Application Actually Does

- Loads a CSV containing sentence text and person/date annotations.
- Normalizes the CSV headers: trims whitespace and converts them to lowercase.
- Accepts `Name` as an alternative header for `person`.
- Converts non-empty person/date fields into binary labels (`1`) and empty fields into `0`.
- Cleans text with a blank spaCy English pipeline: lowercase, tokenize, remove stop words and punctuation, and retain alphabetic tokens.
- Creates synthetic augmented examples by substituting person or date values in each source row.
- Splits processed text and labels into training/testing sets.
- Combines 1–2 gram TF-IDF features with three handcrafted features:
  - capitalization signal,
  - four-digit year pattern,
  - sentence word count.
- Trains separate `MultiOutputClassifier` wrappers containing Logistic Regression and Multinomial Naive Bayes estimators.
- Evaluates both models and selects the model with the higher F1 score.
- Prints metrics/confusion matrices in the CLI.
- Displays metrics/confusion matrices and accepts interactive predictions in Streamlit.
- For a predicted person or date, uses a pretrained spaCy `en_core_web_sm` NER pipeline to extract entity text, with rule-based fallbacks.

## 2. Real-World Problem

The demonstrated problem is **sentence-level screening for the presence of people and dates**. It could be used as a lightweight demo for triaging text before a more complete extraction process—for example, identifying whether a passage is likely to describe a person and/or an event date.

The evidence does not support a claim that the system performs production-grade entity extraction or general information retrieval. In particular:

- It does not return character offsets or BIO/entity-span labels.
- It does not recognize arbitrary entity types such as organizations, locations, or products.
- Its person/date extraction depends heavily on spaCy rules and fallbacks.
- The binary model is a presence classifier, not a grounded extraction model.

## 3. Target Users and Business Use Cases

### Likely target users

- Students learning NLP and machine learning.
- Developers demonstrating a basic text-classification pipeline.
- Technical recruiters reviewing a portfolio project.
- Small teams needing a simple prototype for a person/date document screen.

### Potential business use cases

- Rapidly flagging whether incoming text mentions people or dates.
- Demonstrating NLP and model-comparison concepts to non-specialist stakeholders.
- A baseline classifier for a manual review workflow.
- An educational starting point for a future document-processing service.

These uses are inferred only from implemented behavior; no domain, customer, SLA, compliance, or deployment requirement is documented in the repository.

## 4. End-to-End Workflow

### CLI workflow

1. `ner_project.py` receives `--csv-path`, defaulting to `ner_data.csv`.
2. `load_dataset()` reads the CSV and normalizes headers.
3. `build_augmented_dataset()` creates original and synthetic rows.
4. `preprocess_text()` normalizes text with a blank spaCy pipeline.
5. `split_data()` creates train/test partitions.
6. Text is converted to TF-IDF vectors and combined with handcrafted features.
7. Logistic Regression and Naive Bayes models are trained and evaluated.
8. The F1 winner is selected; confusion matrices are shown.
9. The user may enter a sentence for a binary person/date prediction.

### Streamlit workflow

1. The app asks for a CSV path and waits for a user action.
2. `Load and Train Models` loads/trains both models in a cached resource.
3. The UI compares accuracy, precision, recall, and F1, then plots four confusion matrices.
4. It chooses the same F1-winning model in ordinary ties (whereas the CLI has explicit tie handling).
5. The user enters a sentence and presses `Predict`.
6. The app preprocesses and vectorizes the text, runs the best binary classifier, and loads spaCy NER once.
7. spaCy returns PERSON/DATE spans; fallbacks search the first lowercase word, known dataset names, or four-digit years.

## 5. Architecture and Technology Stack

### Architecture

- Single-machine, in-memory Python prototype.
- No separate frontend/backend or API tier.
- Streamlit serves the local UI directly.
- pandas handles CSV and label data.
- spaCy performs preprocessing and pretrained named-entity extraction.
- scikit-learn performs TF-IDF vectorization, training, and evaluation.
- Matplotlib/Seaborn render confusion matrices.
- No database, queue, cache, container, or external service integration is implemented.

### Main files

- `IDS-Project/app.py`: Streamlit interface and application orchestration.
- `IDS-Project/ner_project.py`: data loading, augmentation, preprocessing, training, evaluation, CLI, and plotting.
- `IDS-Project/test_detect.py`: manual smoke/demo script.
- `IDS-Project/debug_check.py`: prints basic environment and library checks.
- `IDS-Project/debug_csv_test.py`: inspects raw CSV parsing behavior.
- `IDS-Project/ner_data.csv`: active 9-row project dataset.
- `IDS-Project/ner_data.bak`: 19-byte backup file; contents are not a usable dataset backup.
- `IDS-Project/README.md`: setup and usage documentation.
- `IDS-Project/.venv/`: committed or present local virtual environment.
- `ner_data.csv` at workspace root: separate 147-row CSV with the same `Sentence`, `Name`, `Date` schema; no documented authority or synchronization rule identifies which copy should be used.

### Verified environment

The selected environment is Python 3.13.3 with pandas 3.0.3, scikit-learn 1.8.0, spaCy 3.8.14, Streamlit 1.57.0, matplotlib 3.10.9, Seaborn 0.13.2, SciPy 1.17.1, and the `en_core_web_sm` 3.8.0 package.

The inner project dataset has exactly 9 rows and no missing values. After augmentation, it produces 27 rows: 18 positive and 9 negative labels for each target.

## 6. AI and Automation

- **Traditional ML:** TF-IDF plus Logistic Regression and Multinomial Naive Bayes classify sentence-level person/date presence.
- **NLP preprocessing:** spaCy tokenization and stop-word removal.
- **Pretrained NER:** spaCy `en_core_web_sm` is used for span extraction, not model training.
- **Automated augmentation:** simple entity substitutions generate synthetic negative examples.
- **Rule automation:** capitalization, year regex, dataset-name matching, and first-word fallback post-process predictions.
- **Model selection:** F1 score determines the preferred model.
- **Caching:** Streamlit caches the trained pipeline and loaded spaCy NER model.

There is no transformer model, fine-tuning, LLM, external AI API, agent, online learning, or model-monitoring system.

## 7. Implemented, Incomplete, and Missing

### Implemented

- CSV loading and header normalization.
- Sentence preprocessing.
- Synthetic data augmentation.
- Train/test splitting.
- TF-IDF and handcrafted features.
- Two model baselines.
- Micro-averaged evaluation metrics.
- Per-label and overall confusion matrices.
- CLI and Streamlit interfaces.
- Basic spaCy entity extraction and fallbacks.
- Debug and manual test scripts.

### Incomplete or fragile

- Augmentation is not a semantic data-generation strategy and can create invalid or duplicate labels.
- The app augments original data without first proving that source person/date values actually occur in the source sentence.
- Pretrained spaCy output can be incongruent with the binary model output.
- The fallback logic can label an arbitrary first word as a person.
- The fallback known-name matching can return a name part rather than a verified full entity.
- Tie selection differs between CLI and Streamlit implementations.
- The README describes 27? The implementation, but no measured evidence is provided and README metrics may not be reproducible.
- There is no explicit class or API boundary; application, training, and presentation logic are loosely combined.

### Missing

- `requirements.txt`, `pyproject.toml`, or another reproducible dependency manifest.
- Locked dependency versions and Python version declaration.
- Automated unit/integration tests and test fixtures.
- Model/vectorizer serialization and reload support.
- Prediction API or batch processing endpoint.
- Database, authentication, authorization, audit log, or user management.
- Real token-level entity spans, offsets, confidence scores, and entity-type coverage.
- Cross-validation, held-out benchmark, per-label metrics, calibration, and leakage analysis.
- Data validation and schema tooling.
- CI/CD, linting, formatting, and build/deploy configuration.
- Docker or other deployment packaging.
- Logging and observability.
- Dataset provenance, licensing, and update process.
- Privacy, security, and retention controls.

## 8. Engineering, Security, and Maintainability Issues

### Validated functional issue

Using the inner 9-row dataset, the end-to-end normalized load and training pipeline produced:

- Logistic Regression: accuracy `0.4000`, precision `0.7000`, recall `1.0000`, F1 `0.8235`.
- Naive Bayes: accuracy `0.4000`, precision `0.7000`, recall `1.0000`, F1 `0.8235`.

These are a one-time local run on a tiny synthetic-augmented split, not evidence of production quality or generalization. In this run all test predictions were `(1,1)`, showing recall but poor exact-match accuracy and a possible negative-label prediction failure.

### Data and model correctness

- `build_augmented_dataset()` encodes every original row as `(1,1)` even if a source person's/date field is empty or absent; it assumes all source rows contain both entities.
- Synthetic negative rows are likely duplicates or near-duplicates of positives and do not establish meaningful real negatives.
- `replace_entity()` does nothing if the annotation is not actually present in the sentence, but the label still changes to negative or may remain unchanged without detection.
- Random split behavior can leak highly similar original/synthetic variants across train and test sets, inflating or biasing results.
- No train-only feature fitting check, stratified cross-validation, or data-quality baseline is documented.
- Evaluation uses micro-averaging across labels; exact-match accuracy may be more informative for this UI but is already included as subset accuracy.
- The handcrafted feature extraction runs on already preprocessed text in one location and original text in another; behavior differs in capitalization and word lengths.
- Multinomial Naive Bayes receives sparse TF-IDF plus dense handcrafted features, but positive counts/feature scaling are not formally analyzed for this setup.

### Security and operational risk

- Streamlit accepts arbitrary CSV paths; in a remote deployment this is a local-file path exposure unless the service boundary is tightly restricted.
- CSV loading is unbounded and may consume excessive memory or trigger parser/resource exceptions.
- Uploaded/trusted external text is not a clear threat boundary in the current local app, but remote deployment would need path allow-listing, limits, and authentication.
- There is no model provenance, content/version hash, or dataset lineage.
- Models and data are held in memory only.
- Dependency execution and the pretrained spaCy model require an environment setup; no reproducible lockfile is present.
- The virtual environment directory is included in the workspace, creating repository-size and platform-reproducibility risks.
- Public deployment would expose no rate limiting, logging, input validation, output filtering, or abuse protection.

### Maintainability

- `ner_project.py` combines loading, augmentation, training, evaluation, plotting, and CLI concerns.
- `app.py` repeats training and feature-combination logic already present in `ner_project.py` instead of exposing a reusable pipeline.
- Model selection is duplicated between CLI and Streamlit.
- Debug scripts are ad hoc and not pytest/unittest-compatible in every case.
- No package structure or import-time separation exists.
- README setup references `requirements.txt`, but no such file is present.
- README's dataset and preprocessing examples claim more than the inner 9-row file demonstrates.
- The workspace has no Git repository metadata (`.git`), so version-history and contributor review workflows cannot be inspected.
- There are two CSV locations with no stated authoritative dataset or synchronization process.
- Backup and cache/test artifacts make the project structure less professional and potentially confusing.

## 9. Documentation and Portfolio Assessment

### Strengths

- Clear beginner-oriented explanation of code concepts.
- A runnable Streamlit demo and CLI alternative are present.
- It demonstrates preprocessing, feature engineering, model comparison, evaluation, and a live prediction path.
- It provides a concrete educational artifact rather than only a notebook or source snippet.

### Gaps for professional portfolio use

- Add a precise project statement: it is a sentence-level binary classifier, not full NER.
- State scope, data limitations, and measured validation honestly.
- Remove or regularize `.venv`, caches, `.bak`, and duplicate CSV state.
- Add dependency and run instructions that match the actual files and environment.
- Add tests and screenshots/demo evidence.
- Separate reusable pipeline code from UI and CLI entry points.
- Move toward model persistence, robust preprocessing, and a real held-out dataset.
- If retention of the educational scope is intended, label it as a prototype and list the production roadmap.

## 10. Recommended Product Names

1. **EntityLens** — Clear and concise; communicates the app's inspection-oriented approach to detecting person/date entities.
2. **TextFacet** — Positions the system as an easy way to classify a text passage into useful facets, but is broader than current person/date behavior.
3. **SignalSpan** — Evokes span detection and information signal extraction; slightly less intuitive for beginners.
4. **EntityPulse** — Product-like name suggesting quick entity screening and real-time insight.
5. **PersonDate Scout** — Explicitly describes the only two supported categories, making it honest and searchable.

### Recommendation

**PersonDate Scout** is the strongest fit for the current implementation because it does not overstate product capability and accurately communicates the two-category scope. If the project is expected to expand beyond person/date detection, **EntityLens** is the more scalable and portfolio-friendly option.

## Priority Next Steps

1. Establish one authoritative dataset and clearly document real versus synthetic examples.
2. Replace synthetic augmentation with valid held-out negative examples or a real multi-label benchmark.
3. Add `pyproject.toml`/`requirements*.txt`, tests, and a `.gitignore` that excludes `.venv`, caches, and backup files.
4. Refactor shared pipeline logic out of duplicate CLI/UI implementations.
5. Persist vectorizer and model artifacts with a model card/metadata file.
6. Report per-label precision/recall/F1 and run cross-validation with leakage-safe grouping.
7. Rename the README description to “sentence-level person/date presence classification” until true token/entity-span extraction is implemented.
8. Add a small API or batch interface only after the core pipeline and validation are stabilized.
