import os
import pandas as pd


# Load data from multiple CSVs in multiple folders ##########

def merge_csv_mult_dir(path_to_dir):
    # Get all non-zip folders/files in dir
    names = os.listdir(path_to_dir)
    folders = [name for name in names if not name.endswith('.zip')] # Remove names found for zips
    print('Non-zip folders/files found in {}: {}'.format(path_to_dir, folders))
    # From each folder, read all CSV files into Pandas df
    dfs = []
    for folder in folders:
        try: 
            data_dir = path_to_dir + '/' + folder + '/'
            csv_filenames = os.listdir(data_dir)    
            for name in csv_filenames:
                df = pd.read_csv(data_dir + name)
                df['csv'] = name
                df['folder'] = folder
                dfs.append(df)
        except NotADirectoryError:
            print('\tSkipping (not a directory):', data_dir)
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all
    
############################################################


# Functions for data cleaning & prep ####################

def add_city_col(df):
    df['city'] = df['url'].str.split('-', expand=True).iloc[:, -2]
    return df

def clean_usernames(df):

    df['user_name_clean'] = df['user_name'].str.split('<', expand=True).iloc[:, 0]
    return df

#######################################################


# Fully clean & prepare data #########################

def clean_and_prep(df):
    # # Change 'review_date' to datetime type
    # df['review_date'] = pd.to_datetime(df['review_date'])

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)
    # Fill nulls for 'user_location' with 'n/a'
    df.fillna({'user_location': 'n/a'}, inplace=True)

    # # Add col 'review_length' from 'review_body'
    # df['review_length'] = df['review_body'].str.len()

    # Get 'city' from 'url'
    df = add_city_col(df)
    # Clean 'user_name'
    df = clean_usernames(df)

    # Add 'sentiment' column mapped by 'rating'
    df['sentiment'] = df['rating'].map({1: 'negative', 2: 'negative', 3: 'neutral', 4:'positive', 5:'positive'})
    # Add 'sentiment' column mapped by 'sentiment'
    df['polarity'] = df['sentiment'].map({'negative': 0, 'neutral': 0.5, 'positive': 1})
    # Add 'sentiment_int' column mapped by 'sentiment'
    df['sentiment_int'] = (df['polarity'] * 2).astype(int)
    # Move 'sentiment' col to be last
    last_col = df.pop('sentiment')
    df.insert(df.shape[1], 'sentiment', last_col)
    return df

#######################################################


