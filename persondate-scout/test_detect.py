from ner_project import load_dataset, build_spacy_preprocessor, extract_handcrafted_features, train_models
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import spacy

# Load dataset
csv = 'ner_data.csv'
df = load_dataset(csv)

# Preprocess using blank spaCy
nlp_blank = build_spacy_preprocessor()

# Prepare simple features: use original sentence column from load_dataset
x = df['sentence'].fillna('').astype(str)
y = df[['person','date']]

# Fit TF-IDF
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(x)

# handcrafted
hf = extract_handcrafted_features(x).values
X_full = hstack([X_tfidf, hf]).tocsr()

# Train models
models = train_models(X_full, y)
best_model = models['Logistic Regression']

# Load spaCy NER
try:
    nlp_spacy = spacy.load('en_core_web_sm')
except Exception as e:
    print('Could not load en_core_web_sm:', e)
    nlp_spacy = spacy.blank('en')

# Test sentence
sentence = 'Elon work at tesla'

# Predict using model
proc = ' '.join([tok.text for tok in nlp_blank(sentence.lower()) if tok.is_alpha and not tok.is_stop])
vec = hstack([vectorizer.transform([proc]), extract_handcrafted_features([sentence]).values]).tocsr()
pred = best_model.predict(vec)[0]
print('Model predicts person/date:', pred)

# Try spaCy NER
doc = nlp_spacy(sentence)
persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
print('spaCy persons:', persons)

# Fallback: search dataset names
if pred[0] == 1 and not persons:
    known = df['person'].dropna().astype(str).unique().tolist()
    for name in known:
        if name and name.lower() in sentence.lower().split():
            persons = [name]
            break

print('Final detected person:', persons if persons else 'Not detected')
print('Done')
