import sys
import os
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Args:
        messages_filepath   --> str: path to message CSV file
        categories_filepath --> str: path to categories CSV file
    Return:
        combined Dataframe from the mesages and categories files
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on='id')

def clean_data(df):
    """
    expands the categories columns to 36 new columns
    
    Arguments:
        df -> Combined data containing messages and categories
    Outputs:
        df -> Combined data containing messages and categories with new categories cleaned up
    """
    # split the categories column into new columns
    categories = df['categories'].str.split(pat=';', expand = True)
    
    # Get new column names for the new expanded category columns
    columns = [x.split("-")[0] for x in categories.iloc[0].values]
    
    # determining column names for the expanded new category columns
    # use first row to compute column name
    row = categories.iloc[[0]]
    # Use a split to slice the column to retain everything on the column name before the ('-') character
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    
    #convert category values to binary
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    
    df = pd.concat([df,categories],axis=1)
    df = df.drop('categories',axis=1)
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Save Data to SQLite Database Function
    
    Arguments:
        df -> Combined data containing messages and categories with categories cleaned up
        database_filename -> Path to SQLite destination database
    """
    engine = create_engine("sqlite:///" + database_filename)
    db_name = os.path.basename(database_filename).split(".")[0]
    try:
        df.to_sql(db_name, engine, index=False)
    except ValueError as error:
        print(error)
    

def main():
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