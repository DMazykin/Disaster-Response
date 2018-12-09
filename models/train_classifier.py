import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    
    df = pd.read_sql_table('message_with_cat', engine)
    labels = df.columns[4:].tolist()

    X = df['message']
    y = df[labels].values

    return X, y, labels


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    scorer = make_scorer(f1_score, average='micro')
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, stop_words='english')),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(LinearSVC()))
            ])

    parameters = {
        'clf__estimator__C': [0.1, 1, 10],
        'clf__estimator__multi_class': ['ovr', 'crammer_singer']
    }

    model = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer)
    
    return model
    


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print (classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()