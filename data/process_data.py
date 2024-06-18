import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function is used to load data.
    
    Input:
    messages_filepath: messages filepath
    categories_filepath: categories filepath
    
    Output:
    return merged dataframe
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id')
    #print(df.head(5))
    return df


def clean_data(df):
    '''
    This function is used to clean data.
    
    Input:
    df: dataframe for clearning
    
    Output:
    return cleaned dataframe
    '''
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[[1]]
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    return df
    
    
    
def save_data(df, database_filename):
    '''
    This function is used to save data.
    
    Input:
    df: dataframe for clearning
    database_filename: database filname
    
    Output:
    None
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('message_processed', engine, index=False)
    print('Save data successfully.')

def main():
    # Check system argument
    print(sys.argv)
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()