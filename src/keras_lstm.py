import numpy as np
# Setting the seed for numpy-generated random numbers
np.random.seed(42)
import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
# Just disables annoying TF warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Setting the seed for python random numbers
rn.seed(42)
# Setting the graph-level random seed.
tf.compat.v1.set_random_seed(42)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from nltk.stem import SnowballStemmer
import pandas as pd
import matplotlib.pyplot as plt
import sys
from datetime import datetime
import timeit
import preprocess as prep
import re

# Constants
TARGET = 'sentiment'
FEATURE = 'review_body'
FEATURES = [FEATURE]

# Set plot sizes
SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 26
BIGGEST_SIZE = 28
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGEST_SIZE)  # fontsize of the figure title

# Plotting
def plot_graphs(history, string, model_name):
    """Plot history graphs for loss/accuracy"""
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(history.history[string])
    ax.plot(history.history['val_'+string])
    ax.set_xlabel("Epochs")
    ax.set_ylabel(string)
    ax.set_title('string')
    ax.legend([string, 'val_'+ string])
    fig.tight_layout()
    fig.savefig('imgs/' + model_name + '_' + string)

# Get epochs, time, save path
def get_epochs_save_path():
    """Get # epochs from sys args and time to define save model path"""
    try: 
        num_epochs = int(sys.argv[-1])
    except IndexError:
        print('Using default num_epochs = 5')
        num_epochs = 5

    saved_model_path = \
        "saved_models/lstm_{}epochs_{}".format(num_epochs,
                                                             datetime.now()
                                                             .strftime("%Y%m%d-%H:%M:%S")) 
    print('\nWill save model to: {}\n'.format(saved_model_path))
    model_name = saved_model_path.split('/')[-1]
    model_name = model_name.split('.')[0]
    return num_epochs, saved_model_path, model_name
 
# Define custom metrics ###############
def recall(y_true, y_pred):
    """Calculate multiclass recall"""
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Calculate multiclass precision"""
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

#############################

if __name__ == "__main__":
    try: 
        path = sys.argv[1]
        action = sys.argv[2]
    except IndexError:
        print('Please specify path to data and action ("new_model"/"load"/"train_more").')
        sys.exit()
    
    if action == 'load' or action == 'train_more':
        try: 
            model_path = sys.argv[3]
        except IndexError:
            print('Please specify path to saved model.')
            sys.exit()
        # Load saved model
        prev_model_path = model_path
        print('\nLoading model: {}\n'.format(prev_model_path))
        model = load_model(prev_model_path, custom_objects={'precision':precision, 'recall':recall})

    # Data preprocessing #############
    train_df, test_df, val_df = prep.preprocess_split(path)
    

    # Optional, to save time ##############
    # Get smaller samples of data
    train_df, _ = train_test_split(train_df, train_size=0.5, shuffle=True, \
        stratify=train_df[TARGET], random_state=42)
    val_df, _ = train_test_split(val_df, train_size=0.5, shuffle=True, \
            stratify=val_df[TARGET],random_state=42)
    test_df, _ = train_test_split(test_df, train_size=0.5, shuffle=True, \
            stratify=test_df[TARGET],random_state=42)
    print('Taking 50pct of data - Train: {}, Val: {}, Test: {}'.format(train_df.shape[0], val_df.shape[0], test_df.shape[0]))

    ########################
    
    # Further prepare data #############
    # Change Train and Val labels into ints
    y_train = train_df[TARGET]
    y_val = val_df[TARGET]
    y_test = test_df[TARGET]

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)

    training_label_seq = label_encoder.transform(y_train)
    validation_label_seq = label_encoder.transform(y_val)
    test_label_seq = label_encoder.transform(y_test)

    # Get class weights
    class_weights = class_weight.compute_class_weight('balanced',
                                                 label_encoder.classes_,
                                                 y_train)
    class_weights_dict = {k:v for k,v in zip(np.arange(label_encoder.classes_.shape[0]), class_weights)}
    
    # Lower case, remove punctuation and stop words from X data
    X_train_vals = train_df[FEATURE].str.lower()
    X_val_vals = val_df[FEATURE].str.lower()
    X_test_vals = test_df[FEATURE].str.lower()

    # Get stopwords
    stopwords = prep.set_stopwords()
    stop_pat = ' | '.join(stopwords)
    
    # Remove punctuation digits, and stop words
    regex = '[^a-zA-Z\s]'
    print('\nRemoving punctuation, digits, and stop words from X_train/val/test data...')
    X_train_vals = X_train_vals.str.replace(regex, ' ')
    X_val_vals = X_val_vals.str.replace(regex, ' ')
    X_test_vals = X_test_vals.str.replace(regex, ' ')

    X_train_vals = X_train_vals.str.replace(stop_pat, ' ')
    X_val_vals = X_val_vals.str.replace(stop_pat, ' ')
    X_test_vals = X_test_vals.str.replace(stop_pat, ' ')
    
    # Snowball stemming
    stemmer = SnowballStemmer('english')
    X_train_vals = X_train_vals.apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    X_val_vals = X_val_vals.apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    X_test_vals = X_test_vals.apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    
    # Tokenize data ######################
    
    print('\nTokenizing X_train/val/test data...')
    X_train_vals = X_train_vals.values
    X_val_vals = X_val_vals.values
    X_test_vals = X_test_vals.values

    # Set tokenization params
    maxlen = 300
    oov_tok = '<OOV>'
    num_words = 5000
    trunc_type = 'post'
    padding_type = 'post'
    
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_tok)
    tonkenize = tokenizer.fit_on_texts(X_train_vals)
    xtrain_tkns = tokenizer.texts_to_sequences(X_train_vals)
    xval_tkns = tokenizer.texts_to_sequences(X_val_vals)
    xtest_tkns = tokenizer.texts_to_sequences(X_test_vals)

    vocab_size=len(tokenizer.word_index)+1
        
    # Pad with zeros
    xtrain_tkns = pad_sequences(xtrain_tkns,padding=padding_type, truncating=trunc_type, maxlen=maxlen)
    xval_tkns = pad_sequences(xval_tkns,padding=padding_type, truncating=trunc_type, maxlen=maxlen)
    xtest_tkns = pad_sequences(xtest_tkns,padding=padding_type, truncating=trunc_type, maxlen=maxlen)
        
    # Start modeling ##################
    print('\nStarting modeling...')

    if action == 'load':
        loss, acc, precision, recall = model.evaluate(xtrain_tkns, training_label_seq)
        print("Training Accuracy: ", acc)
        loss, acc, precision, recall = model.evaluate(xval_tkns, validation_label_seq)
        print("Val Accuracy: ", acc)
        loss, acc, precision, recall = model.evaluate(xtest_tkns, test_label_seq)
        print("Test Accuracy: ", acc)
    
    elif action == 'train_more': 
        num_epochs, saved_model_path, model_name = get_epochs_save_path()
        
        # Train
        start_time = timeit.default_timer()

        checkpoint = ModelCheckpoint("BEST_saved_models/" + model_name, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=False)
        callbacks_list = [checkpoint]
        
        history = model.fit(xtrain_tkns, training_label_seq, epochs=num_epochs, batch_size=batch_size,
                  callbacks=callbacks_list, validation_data=(xval_tkns, validation_label_seq), class_weight=class_weights_dict, shuffle=False)
        
        elapsed = timeit.default_timer() - start_time
        print('\nTook {:.2f}s to train'.format(elapsed))
        
        plot_graphs(history, "accuracy", model_name)
        plot_graphs(history, "loss", model_name)
        plot_graphs(history, "precision", model_name)
        plot_graphs(history, "recall", model_name)

        loss, acc, precision, recall = model.evaluate(xtrain_tkns, training_label_seq)
        print("Training Accuracy: ", acc)
        loss, acc, precision, recall = model.evaluate(xval_tkns, validation_label_seq)
        print("Val Accuracy: ", acc)
        loss, acc, precision, recall = model.evaluate(xtest_tkns, test_label_seq)
        print("Test Accuracy: ", acc)

        # Save entire model
        model.save(saved_model_path, include_optimizer=True)
        
    elif action == 'new_model': 
        num_epochs, saved_model_path, model_name = get_epochs_save_path()
        
        # Set params
        embedding_dim = 128 #PARAMS
        lstm_cells = 100 #PARAMS
        batch_size = 32 #PARAMS
        
        # MODEL
        model = Sequential([
        # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
        layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
        # layers.Embedding(vocab_size, embedding_dim),

        layers.Dropout(0.25),

        layers.Conv1D(embedding_dim/2, 5, padding='valid', activation='relu', strides=1),
        layers.MaxPooling1D(pool_size=4),

        layers.Bidirectional(layers.LSTM(lstm_cells, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(lstm_cells, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(lstm_cells)),

        # layers.LSTM(lstm_cells,return_sequences=True),
        # layers.LSTM(lstm_cells),

        layers.Dropout(0.5),
        # use ReLU in place of tanh function since they are very good alternatives of each other.
        layers.Dense(embedding_dim, activation='relu'),
        layers.Dropout(0.2),
        # layers.Dense(8),
        # layers.Dropout(0.2),
        # Add a Dense layer with 4 units and softmax activation.
        # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
        layers.Dense(3, activation='softmax')
        ])
        
        # Compile model, show summary
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", 
            weighted_metrics=['accuracy', precision, recall])
        print('\n')
        model.summary()
        
        # Train
        start_time = timeit.default_timer()

        checkpoint = ModelCheckpoint("BEST_saved_models/" + model_name, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=False)
        callbacks_list = [checkpoint]
        
        history = model.fit(xtrain_tkns, training_label_seq, epochs=num_epochs, batch_size=batch_size,
                  callbacks=callbacks_list, validation_data=(xval_tkns, validation_label_seq), class_weight=class_weights_dict, shuffle=False)
        
        elapsed = timeit.default_timer() - start_time
        print('\nTook {:.2f}s to train'.format(elapsed))
        
        plot_graphs(history, "accuracy", model_name)
        plot_graphs(history, "loss", model_name)
        plot_graphs(history, "precision", model_name)
        plot_graphs(history, "recall", model_name)

        loss, acc, precision, recall = model.evaluate(xtrain_tkns, training_label_seq)
        print("Training Accuracy: ", acc)
        loss, acc, precision, recall = model.evaluate(xval_tkns, validation_label_seq)
        print("Val Accuracy: ", acc)
        loss, acc, precision, recall = model.evaluate(xtest_tkns, test_label_seq)
        print("Test Accuracy: ", acc)

        # Save entire model
        model.save(saved_model_path, include_optimizer=True)

    else:
        print('Unknown action:', action)