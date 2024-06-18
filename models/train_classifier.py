import sys
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('averaged_perceptron_tagger')
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Global varibales
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    '''
    This function is used to load data.
    
    Input:
    database_filepath: database file path for SQLite
    
    Output:
    X: Feature colums
    Y: Category columns
    category_names: Category column name
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('select * from message_processed',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X,Y,category_names


def tokenize(text):
    '''
    This function is used to tokenize text data.
    
    Input:
    text: text information that need to be tokenized
    
    Output:
    clean_tokens: cleaned tokens
    '''
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


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    '''
    This function is used to build AI model.
    
    Input:
    None
    
    Output:
    cv: Grid searched model
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': [(1, 1)],
        'clf__estimator__n_estimators': [3, 5],
        #'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function is used to evaluate ai model.
    
    Input:
    model: AI model
    X_test: Feature column on test sample
    Y_test: Category column on test sample
    category_names: Category column name
    
    Output:
    None: print out classification report and average overall accuracy.
    '''    
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns = category_names)
    for column in Y_test.columns:
        print(classification_report(Y_test[column],y_pred_df[column]))

    overall_accuracy_improve = (y_pred == Y_test).mean().mean()
    print("Average overall accuracy is {0:.2f}%".format(overall_accuracy_improve * 100))

def save_model(model, model_filepath):
    '''
    This function is used to save ai model.
    
    Input:
    model: AI model
    model_filepath: File path for saving AI model
    
    Output:
    None: print out AI model save successfully.
    '''  
    with open(model_filepath,'wb') as f:
        pickle.dump(model,f)
    print('Save ml_pipeline_model successfully!!!')


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