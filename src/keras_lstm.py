from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime
import timeit
import preprocess as prep
import re
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# Just disables annoying TF warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

def plot_graphs(history, string, model_name):
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
    
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+ string])
#     plt.show()
    plt.tight_layout()
    plt.savefig('imgs/' + model_name + '_' + string)

def get_epochs_save_path():
    try: 
        num_epochs = int(sys.argv[-1])
    except IndexError:
        print('Using default num_epochs = 5')
        num_epochs = 5

    saved_model_path = \
        "saved_models/lstm_tokens5000_{}epochs_{}.h5".format(num_epochs,
                                                             datetime.now()
                                                             .strftime("%Y%m%d-%H:%M:%S")) 
    print('\nWill save model to: {}\n'.format(saved_model_path))
    model_name = saved_model_path.split('/')[-1]
    model_name = model_name.split('.')[0]
    return num_epochs, saved_model_path, model_name
        

if __name__ == "__main__":
    try: 
        path = sys.argv[1]
        action = sys.argv[2]
    except IndexError:
        print('Please specify path to data files and action ("new_model"/"load"/"train_more").')
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
        model = load_model(prev_model_path)

    # Data preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, \
        indices_train, indices_val, indices_test, \
            train_df_us, df = prep.preprocess_split_undersample(path)

    target = 'sentiment'
    features = ['review_body']
    feature = 'review_body'

    y_train_us = train_df_us[target]
    
    # # encode class values as integers
    # encoder = LabelEncoder()
    # encoded_y_train_us = encoder.fit_transform(y_train_us)
    # # convert integers to dummy variables (i.e. one hot encoded)
    # dummy_y_train_us = utils.to_categorical(encoded_y_train_us)
    
    # encoded_y_val = encoder.transform(y_val)
    # dummy_y_val = utils.to_categorical(encoded_y_val)

    # Change Train and Val labels into ints
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(set(y_train_us))
    training_label_seq = np.array(label_tokenizer.texts_to_sequences(y_train_us))
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(y_val))

    # Removing punctuation and stop words from X data
    X_train_us_vals = train_df_us[feature].str.lower()
    X_val_vals = df.loc[indices_val, feature].str.lower()

    stop = [re.sub('[^\w\s]', '', stopword) for stopword in STOPWORDS]
    stop_pat = ' | '.join(stop)
    
    print('\nRemoving punctuation and stop words from X_train/val data...')
    X_train_us_vals = X_train_us_vals.str.replace('[^\w\s]', '')
    X_val_vals = X_val_vals.str.replace('[^\w\s]', '')

    X_train_us_vals = X_train_us_vals.str.replace(stop_pat, ' ')
    X_val_vals = X_val_vals.str.replace(stop_pat, ' ')

    # Tokenize X data
    print('\nTokenizing X_train/val data...')
    X_train_us_vals = X_train_us_vals.values
    X_val_vals = X_val_vals.values

    maxlen = 400 #PARAMS
    oov_tok = '<OOV>'
    num_words = 5000
    trunc_type = 'post'
    padding_type = 'post'
    
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_tok)
    tonkenize = tokenizer.fit_on_texts(X_train_us_vals)
    xtrain_tkns = tokenizer.texts_to_sequences(X_train_us_vals)
    xval_tkns = tokenizer.texts_to_sequences(X_val_vals)

    vocab_size=len(tokenizer.word_index)+1
        
    xtrain_tkns = pad_sequences(xtrain_tkns,padding=padding_type, truncating=trunc_type, maxlen=maxlen)
    xval_tkns = pad_sequences(xval_tkns,padding=padding_type, truncating=trunc_type, maxlen=maxlen)
        
    print('\nStarting modeling...')

    if action == 'load':
        
        loss, acc = model.evaluate(xtrain_tkns, training_label_seq)
        print("Training Accuracy: ", acc)
    
        loss, acc = model.evaluate(xval_tkns, validation_label_seq)
        print("Test Accuracy: ", acc)
    
    elif action == 'train_more': 
        num_epochs, saved_model_path, model_name = get_epochs_save_path()
        
        # Train
        start_time = timeit.default_timer()
        
        history = model.fit(xtrain_tkns, training_label_seq, epochs=num_epochs, 
                  validation_data=(xval_tkns, validation_label_seq))

        # Save entire model to a HDF5 file
        model.save(saved_model_path)
        
        elapsed = timeit.default_timer() - start_time
        print('\nTook {:.2f}s to train'.format(elapsed))
        
        plot_graphs(history, "accuracy", model_name)
        plot_graphs(history, "loss", model_name)

        loss, acc = model.evaluate(xtrain_tkns, training_label_seq)
        print("Training Accuracy: ", acc)

        loss, acc = model.evaluate(xval_tkns, validation_label_seq)
        print("Test Accuracy: ", acc)
        
    elif action == 'new_model': 
        num_epochs, saved_model_path, model_name = get_epochs_save_path()
        
        embedding_dim=64 #PARAMS
        lstm_cells = 96 #PARAMS
        batch_size=32 #PARAMS
        
#         model=Sequential()
#         model.add(layers.Embedding(input_dim=vocab_size,
#             output_dim=embedding_dim,
#             input_length=maxlen))
#         model.add(layers.LSTM(units=embedding_dim,return_sequences=True))
#         model.add(layers.LSTM(units=maxlen))
#         model.add(layers.Dropout(0.5))
#         model.add(layers.Dense(8))
#         model.add(layers.Dense(3, activation="softmax"))
        
        model = Sequential([
        # Add an Embedding layer expecting input vocab of size 10000, and output embedding dimension of size 100 we set at the top
        layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
        # layers.Embedding(vocab_size, embedding_dim),
        
        # layers.Dropout(0.2),

        # layers.Bidirectional(layers.LSTM(lstm_cells, return_sequences=True)),
        # layers.Bidirectional(layers.LSTM(lstm_cells)),

        # use ReLU in place of tanh function since they are very good alternatives of each other.

        layers.LSTM(lstm_cells,return_sequences=True),
        layers.LSTM(lstm_cells),

        layers.Dropout(0.5),
        layers.Dense(embedding_dim, activation='relu'),
        # layers.Dropout(0.5),
        # Add a Dense layer with 6 units and softmax activation.
        # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
        layers.Dense(4, activation='softmax')
        ])
        
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", 
            metrics=['accuracy'])
        print('\n')
        model.summary()
        
        # Train
        start_time = timeit.default_timer()
        
        history = model.fit(xtrain_tkns, training_label_seq, epochs=num_epochs, batch_size=batch_size,
                  validation_data=(xval_tkns, validation_label_seq))

        # Save entire model to a HDF5 file
        model.save(saved_model_path)
        
        elapsed = timeit.default_timer() - start_time
        print('\nTook {:.2f}s to train'.format(elapsed))
        
        plot_graphs(history, "accuracy", model_name)
        plot_graphs(history, "loss", model_name)

        loss, acc = model.evaluate(xtrain_tkns, training_label_seq)
        print("Training Accuracy: ", acc)

        loss, acc = model.evaluate(xval_tkns, validation_label_seq)
        print("Test Accuracy: ", acc)
        
    else:
        print('Unknown action:', action)
        
