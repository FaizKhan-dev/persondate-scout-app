from ner_project import load_dataset, build_spacy_preprocessor, build_augmented_dataset

if __name__ == '__main__':
    import os
    csv = os.path.join(os.path.dirname(__file__), 'ner_data.csv')
    df = load_dataset(csv)
    print('columns:', df.columns.tolist())
    print('rows:', len(df))
    nlp = build_spacy_preprocessor()
    x, y = build_augmented_dataset(df, nlp)
    print('x length:', len(x))
    print('y shape:', y.shape)
    print('x sample (first 5):')
    for i, xi in enumerate(x[:5]):
        print(i, xi)
    print('y head:')
    print(y.head())
