import sys
import os

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd

import sqlite3
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer

import pickle


def load_data(database_filepath):
    """
    Load Data from the Database Function
    
    Arguments:
        database_filepath -> Path to SQLite destination database (e.g. disaster_response_db.db)
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> List of categories name
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
      
    #the 'related' column has a few rows with a value of 2 instead of 1 or 0, this would result in an error in the model
    #the values will be replaced by a  value of 1 in this case.
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    """
    The function separates text into smaller units called tokens. it also applies lamentization on the text 
    and returns clean tokens that are lower case.
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the provided text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build Pipeline function
    
    Output:
        A Scikit ML Pipeline that process text messages and applies a classifier.
        
    """
    VECT = CountVectorizer(tokenizer=tokenize)
    TFIDF = TfidfTransformer()
    CLF = MultiOutputClassifier(AdaBoostClassifier())
    
    pipeline = Pipeline([
        ('vect', VECT),
        ('tfidf', TFIDF),
        ('clf', CLF)
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [10, 20, 40],
        'clf__estimator__learning_rate': [0.01, 0.02, 0.05],
    }
    
    pipeline = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', n_jobs=-1)
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function applies a ML pipeline to a test set and prints out the model performance using the clasification report
    
    Args:
        pipeline -> A valid scikit ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names (multi-output) 
    """
    Y_pred = model.predict(X_test)
    test_report = classification_report(Y_test, Y_pred, target_names=category_names)
    print(test_report)
   

def save_model(model, model_filepath):
    """
    Save Pipeline function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        pipeline -> GridSearchCV or Scikit Pipelin object
        pickle_filepath -> destination path to save .pkl file
    
    """
    final_output=pickle.dump(model, open(model_filepath, 'wb'))
    return final_output


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