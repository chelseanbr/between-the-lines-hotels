import os
import sys
import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

import nltk
from nltk.tokenize import RegexpTokenizer
nltk.download('wordnet', quiet=True, raise_on_error=True)
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb


 # Load data from single CSV or from multiple CSVs in multiple folders #################

def load_data(path_to_dir):
    """Load data from single CSV or from multiple CSVs in multiple folders"""
    if '.csv' in path_to_dir:
        print('Reading {}...'.format(path_to_dir))
        df = pd.read_csv(path_to_dir)
        return df
    else:
        print('Processing files in {}...'.format(path_to_dir))
        # Get all non-zip folders/files in dir
        names = os.listdir(path_to_dir)
        folders = [name for name in names if not name.endswith('.zip')] # Remove names found for zips
        print('\tNon-zip folders/files found in {}: {}'.format(path_to_dir, folders))
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
    """Add 'city' column from 'url'"""
    df['city'] = df['url'].str.split('-', expand=True).iloc[:, -2]
    return df

def add_loc_col(df):
    """Add clean location column 'loc' from 'city'"""
    df['loc'] = df['city']
    locs_dict = {'New_York':'New_York', 'Tokyo':'Tokyo', 'Phuket':'Thailand', 'Bali':'Bali', \
        'Cuba':'Cuba', 'Domi':'Dominican_Republic', 'Dubai':'Dubai', 'Cayo_Guillermo':'Cuba', \
            'Pattaya':'Thailand', 'Uvero_Alto_Punta_Cana_La_Altagracia_Province_Do':'Dominican_Republic',\
                'Krabi':'Thailand', 'Chiang_Mai':'Thailand', 'Khao_Lak_Phang_Nga_Province':'Thailand',\
                    'Bangkok':'Thailand'}
    for loc in locs_dict:
        df.loc[df[df['city'].str.contains(loc)].index, 'loc'] = locs_dict[loc]
    return df

def clean_usernames(df):
    """Add 'user_name_clean' column from 'user_name'"""
    df['user_name_clean'] = df['user_name'].str.split('<', expand=True).iloc[:, 0]
    return df

#######################################################


# Fully clean & prepare data #########################

def clean_and_prep(df):
    """Fully clean & prepare data"""
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
    # Get clean location 'loc' from city col
    df = add_loc_col(df)
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

def train_test_val_split(df, target):
    """Train-test-val split - shuffled, stratified, 80:20 ratios --> 64/20/16 train/test/val"""
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, \
        stratify=df[target], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=True, \
        stratify=train_df[target], random_state=42)

    print('\tTrain: {}, Test: {}, Val: {}'.format(train_df.shape[0], test_df.shape[0], val_df.shape[0]))
    
    return train_df, test_df, val_df

#######################################################

# Undersample train due to class imbalance ###################################

def undersample_train(train_df, target):
    """Undersample train due to class imbalance"""
    y_train = train_df[target]
    # Get classes and counts
    unique, counts = np.unique(y_train, return_counts=True)

    # Determine majority, middle, and minority classes
    majority_class = unique[np.argmax(counts)]
    minority_class = unique[np.argmin(counts)]
    mid_class = (set(unique) - set([majority_class, minority_class])).pop()
    print('\tMajority Class: {}, Middle Class: {}, Minority Class: {}'.format(majority_class, mid_class, minority_class))

    # Get indices per class
    class_indices = dict.fromkeys([majority_class, mid_class, minority_class])
    for key in class_indices:
        class_indices[key] = train_df[train_df[target]==key].index
        print('\t\tNumber {} in train: {}'.format(key, class_indices[key].shape[0]))

    # Randomly under-sample majority and middle class indices to get new under-sampled train df
    np.random.seed(42)
    rand_maj_indices = np.random.choice(class_indices[majority_class], class_indices[minority_class].shape[0], replace=False)
    rand_mid_indices = np.random.choice(class_indices[mid_class], class_indices[minority_class].shape[0], replace=False)
    undersample_indices = np.concatenate([class_indices[minority_class], rand_mid_indices, rand_maj_indices])

    train_df_us = train_df.loc[undersample_indices,:]
    print('\tFinal undersampled train size:', train_df_us.shape[0])
    return train_df_us

#######################################################


# Complete preprocessing, splitting, undersampling #################################################

def preprocess_split_undersample(path):
    """Complete preprocessing, splitting, undersampling"""
    train_df, test_df, val_df = preprocess_split(path)

    train_df_us = undersample_train(train_df, target)
    
    return train_df_us, test_df, val_df

############################################################


# Complete preprocessing, train-test-val split #################################################

def preprocess_split(path):
    """Complete preprocessing, train-test-val split"""
    # Data preprocessing
    df = load_data(path)
    df = clean_and_prep(df)

    # Train/test/val split
    print('\nSplitting data into train/test/val...')
    target = 'sentiment'
    features = ['review_body']
    feature = 'review_body'
    train_df, test_df, val_df = train_test_val_split(df, target)

    return train_df, test_df, val_df

############################################################


# NLP ##################################################

# Use NLTK's WordNetLemmatizer
def tokenizer(str_input):
    lem = WordNetLemmatizer()
    nltk.download('stopwords', quiet=True, raise_on_error=True)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokenized_stop_words = [lem.lemmatize(word) for word in stop_words]

    tokenizer = RegexpTokenizer(r"[a-zA-Z]+")
    tokens = tokenizer.tokenize(str_input)

    words = [lem.lemmatize(word) for word in tokens if word not in tokenized_stop_words]
    return words

def set_stopwords():
    lem = WordNetLemmatizer()
    nltk.download('stopwords', quiet=True, raise_on_error=True)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokenized_stop_words = [lem.lemmatize(word) for word in stop_words]
    return tokenized_stop_words

def count_vectorize(texts):
    tokenized_stop_words = set_stopwords()
    count_vect = CountVectorizer(tokenizer=tokenizer, stop_words=tokenized_stop_words, max_features=5000)
    matrix = count_vect.fit_transform(texts)
    results = pd.DataFrame(matrix.toarray(), columns=count_vect.get_feature_names())
    return results

def tfidf_cv(X, y, model, cv=5, scoring=['accuracy']):
    clf = Pipeline([('tfidf', TfidfTransformer()), \
                    ('model', model)])
    scores = cross_validate(clf, X, y, scoring=scoring, cv=cv, return_train_score=True)
    print('\t\tScores: {}'.format(scores))
    return scores

############################################################


if __name__ == "__main__":
    try: 
        path = sys.argv[1]
        action = sys.argv[2]
    except IndexError:
        print('Please specify path to data files and action ("model").')
        sys.exit()

    # Data preprocessing
    train_df_us, test_df, val_df = preprocess_split_undersample(path)
    
    target = 'sentiment'
    features = ['review_body']
    feature = 'review_body'

    X_train_us = train_df_us[feature].to_list()
    y_train_us = train_df_us[target].to_numpy()

    print('\nGetting bag of words for train data...')
    X_train_us_vect = count_vectorize(X_train_us)

    if action == 'model': 
        # Modeling
        print('\nStarting modeling...')

        lr = LogisticRegression(multi_class='multinomial', solver='newton-cg')
        mnb = MultinomialNB()
        rf = RandomForestClassifier()
        gb = GradientBoostingClassifier()
        xc = xgb.XGBClassifier()

        # models = dict.fromkeys([lr]) # single model test
        models = dict.fromkeys([lr, mnb, rf, gb, xc])

        for key in models:
            print('\n\tFitting {}...'.format(key.__class__.__name__))
            scores = tfidf_cv(X_train_us_vect, y_train_us, key, cv=5)
            models[key] = scores
            print('\t\tAverage train accuracy:', np.mean(models[key]['train_accuracy']))
            print('\t\tAverage test accuracy:', np.mean(models[key]['test_accuracy']))
            print('\n')

    else:
        print('Unknown action:', action)