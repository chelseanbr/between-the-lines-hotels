import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


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


### NOTE: Train-test-val split and train undersampling below use random state/seed=42 
### for reproducibility.

# Train-test-val split ###################################

def train_test_val_split(df, target, features):
    indices = df.index

    X = df[features]
    y = df[target].to_numpy()

    X_train_init, X_test, y_train_init, y_test, indices_train_init, indices_test = \
        train_test_split(X, y, indices, test_size=0.2, shuffle=True, stratify=y, random_state=42)

    # Get train_init df with train indices
    train_init_df = df.iloc[indices_train_init,:]

    X_train_init2 = train_init_df['review_body']
    y_train_init2 = train_init_df[target].to_numpy()
    train_init_indices2 = train_init_df.index

    X_train, X_val, y_train, y_val, indices_train, indices_val = \
        train_test_split(X_train_init, y_train_init, train_init_indices2, test_size=0.2, 
                         shuffle=True, stratify=y_train_init, random_state=42)
    
    print('train: {}, val: {}, test: {}'.format(X_train.shape[0], X_val.shape[0], X_test.shape[0]))
    return X_train, X_val, X_test, y_train, y_val, y_test, indices_train, indices_val, indices_test

#######################################################

# Undersample train due to class imbalance ###################################

def undersample_train(df, target, indices_train, y_train):
    # Get train df with train indices
    train_df = df.iloc[indices_train,:]
    train_df.shape

    # Get classes and counts
    unique, counts = np.unique(y_train, return_counts=True)

    # Determine majority, middle, and minority classes
    majority_class = unique[np.argmax(counts)]
    minority_class = unique[np.argmin(counts)]
    mid_class = np.unique(y_train[(y_train!=majority_class) & (y_train!=minority_class)])[0]
    print('Majority Class: {}, Middle Class: {}, Minority Class: {}'.format(majority_class, mid_class, minority_class))

    # Get indices per class
    class_indices = dict.fromkeys([majority_class, mid_class, minority_class])
    for key in class_indices:
        class_indices[key] = train_df[train_df[target]==key].index
        print('{} indices length: {}'.format(key, class_indices[key].shape[0]))

    # Randomly under-sample majority and middle class indices to get new under-sampled train df
    np.random.seed(42)
    rand_maj_indices = np.random.choice(class_indices[majority_class], class_indices[minority_class].shape[0], replace=False)
    rand_mid_indices = np.random.choice(class_indices[mid_class], class_indices[minority_class].shape[0], replace=False)
    undersample_indices = np.concatenate([class_indices[minority_class], rand_mid_indices, rand_maj_indices])

    train_df_us = df.iloc[undersample_indices,:]
    print('undersampled train shape:', train_df_us.shape)
    return train_df_us

#######################################################

