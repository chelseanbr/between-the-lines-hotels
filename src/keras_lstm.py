from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import utils
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import sys
from datetime import datetime
import timeit
import preprocess as prep

if __name__ == "__main__":
    print('updated')

    # Data preprocessing
    try: 
        path = sys.argv[1]
    except IndexError:
        print('Please specify path to data files.')
        sys.exit()
    print('Processing files in {}...'.format(path))
    df = prep.merge_csv_mult_dir(sys.argv[1])
    df = prep.clean_and_prep(df)

    # Train/val/test split
    print('\nSplitting data into train/val/test...')
    target = 'sentiment'
    features = ['review_body']
    feature = 'review_body'
    X_train, X_val, X_test, y_train, y_val, y_test, indices_train, indices_val, indices_test = \
        prep.train_test_val_split(df, target, features)
    train_df_us = prep.undersample_train(df, target, indices_train, y_train)

    y_train_us = train_df_us[target]
    
    # encode class values as integers
    encoder = LabelEncoder()
    encoded_y_train_us = encoder.fit_transform(y_train_us)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y_train_us = utils.to_categorical(encoded_y_train_us)
    
    encoded_y_val = encoder.transform(y_val)
    dummy_y_val = utils.to_categorical(encoded_y_val)
        
    # Tokenize X data
    print('\nTokenizing X_train/val data...')
    X_train_us_vals = train_df_us[feature].values
    X_val_vals = df.loc[indices_val, feature].values
    
    tokenizer = Tokenizer(num_words=5000)
    tonkenize = tokenizer.fit_on_texts(df[feature].values)
    xtrain_tkns = tokenizer.texts_to_sequences(X_train_us_vals)
    xval_tkns = tokenizer.texts_to_sequences(X_val_vals)

    vocab_size=len(tokenizer.word_index)+1
    
    maxlen = 250
    xtrain_tkns = pad_sequences(xtrain_tkns,padding='post', maxlen=maxlen)
    xval_tkns = pad_sequences(xval_tkns,padding='post', maxlen=maxlen)
        
    saved_model_path = \
        "saved_models/lstm_tokens5000_10epochs_{}.h5".format(datetime.now().strftime("%Y%m%d-%H:%M:%S"))
    print('\nStarting modeling...')
    
    print('\nWill save model to ', saved_model_path) 
    
    embedding_dim=50
    model=Sequential()
    model.add(layers.Embedding(input_dim=vocab_size,
          output_dim=embedding_dim,
          input_length=maxlen))
    model.add(layers.LSTM(units=50,return_sequences=True))
    model.add(layers.LSTM(units=10))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(8))
    model.add(layers.Dense(3, activation="sigmoid"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", 
         metrics=['accuracy'])
    model.summary()
    
    start_time = timeit.default_timer()
    
    model.fit(xtrain_tkns, dummy_y_train_us, epochs=10, batch_size=64)
    
    # Save entire model to a HDF5 file
    model.save(saved_model_path)
    
    model = load_model('saved_models/lstm_tokens5000_10epochs_20200602.h5')
    
    loss, acc = model.evaluate(xtrain_tkns, dummy_y_train_us)
    print("Training Accuracy: ", acc)
 
    loss, acc = model.evaluate(xval_tkns, dummy_y_val)
    model.fit(xtrain_tkns, dummy_y_train_us, epochs=20, batch_size=16, verbose=2)
    
    loss, acc = model.evaluate(xtrain_tkns, dummy_y_train_us, verbose=2)
    print("Training Accuracy: ", acc)
 
    loss, acc = model.evaluate(xval_tkns, dummy_y_val, verbose=2)
    print("Test Accuracy: ", acc)
    
    elapsed = timeit.default_timer() - start_time
    print('\nTook {:.2f}s to finish'.format(elapsed))