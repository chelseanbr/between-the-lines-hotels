from pprint import pprint
from time import time
import logging
import sys


# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import src.preprocess as prep

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# #############################################################################
# 



# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    # ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', model),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
lr_params = {
    'model__penalty': ['l1', 'l2'],
    'model__C': np.logspace(-4, 4, 20),  # unigrams or bigrams
}


if __name__ == "__main__":

    try: 
        path = sys.argv[1]
        action = sys.argv[2]
    except IndexError:
        print('Please specify path to data files and model (lr/mnb/rf/gb/xc).')
        sys.exit()

    # Data preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, \
        indices_train, indices_val, indices_test, \
            train_df_us, df = preprocess_split_undersample(path)
    
    target = 'sentiment'
    features = ['review_body']
    feature = 'review_body'

    X_train_us = train_df_us[feature].to_list()
    y_train_us = train_df_us[target].to_numpy()

    print('\nGetting bag of words for train data...')
    X_train_us_vect = prep.count_vectorize(X_train_us)

    param_dict = {'lr': lr_params, 'mnb': mnb_params, 'rf': rf_params, 'gb': gb_params, 'xc': xc_params}
    model_dict = {'lr': LogisticRegression(multi_class='multinomial', solver='newton-cg'),\
        'mnb': MultinomialNB(), 'rf': RandomForestClassifier(), 'gb': GradientBoostingClassifier(),\
            'xc': xgb.XGBClassifier()}
    if action in param_dict: 
        parameters = param_dict[action]
        model = model_dict[action]
    else:
        print('Unkwnown model:' action)
        sys.exit()

    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X_train_us_vect, y_train_us)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    else:
        print('Unkwnown model:' action)