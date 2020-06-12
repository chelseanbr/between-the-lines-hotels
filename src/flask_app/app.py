# Docker: docker exec -it quizzical_taussig /bin/bash

from flask import Flask, render_template, request
app = Flask(__name__)

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import numpy as np
import pickle

# Set params
embedding_dim = 128 #PARAMS
lstm_cells = 100 #PARAMS
batch_size = 64 #PARAMS

# Tokenization params
maxlen = 550
num_words = 5000

# Path to model
model_path = 'BEST_saved_models/lstm_6epochs_20200608-07:34:50/'

# Home page
@app.route('/')
def index():
    return render_template('jumbotron.html', title='Predict an Airbnb Rating')

# About page
@app.route('/about')
def about():
    return render_template('about.html', title='About')

# My sentiment predictor app
@app.route('/prediction', methods=['POST'])
def show_pred(maxlen=maxlen, num_words=num_words, model_path=model_path):
    text = str(request.form['user_input'])
    
    tokenized_text = process_input(text, maxlen, num_words)
    # PREDICT
    model = build_model()
    model.load_weights(model_path)

    y_pred_proba = model.predict(tokenized_text)
    neg_prob, neut_prob, pos_prob = tuple(100*y_pred_proba[0])
    y_pred = np.argmax(y_pred_proba)
    labels = ['Negative', 'Neutral', 'Positive']
    sentiment_pred = labels[y_pred]
    rating_dict = {'Negative':'(1-2 Stars)', 'Neutral':'(3 Stars)', 'Positive':'(4-5 Stars)'}
    reaction_gifs_dict = {'Negative': 'negative.gif', 'Neutral': 'neutral.gif', 'Positive': 'positive.gif'}
    return render_template('predict.html', sentiment_pred=sentiment_pred, y_pred=y_pred,
                            neg_prob="{:.2f}".format(neg_prob), neut_prob="{:.2f}".format(neut_prob), pos_prob="{:.2f}".format(pos_prob),
                            rating=rating_dict[sentiment_pred], reaction_gif=reaction_gifs_dict[sentiment_pred], review=text, 
                            tokenized_review=str(tokenized_text))

# Build model
def build_model(num_words=num_words, embedding_dim=embedding_dim, lstm_cells=lstm_cells, batch_size=batch_size, maxlen=maxlen):
    """Build model"""
    # MODEL
    model = Sequential([
    # Add an Embedding layer expecting input vocab size, output embedding dimension set at the top
    layers.Embedding(num_words, embedding_dim, input_length=maxlen),

    layers.Dropout(0.5),

    layers.Conv1D(embedding_dim/2, 5, padding='valid', activation='relu', strides=1),
    layers.MaxPooling1D(pool_size=4),

        layers.Bidirectional(layers.LSTM(lstm_cells, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        layers.Bidirectional(layers.LSTM(lstm_cells, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
    layers.Bidirectional(layers.LSTM(lstm_cells, dropout=0.2, recurrent_dropout=0.2)),

    # layers.LSTM(lstm_cells,return_sequences=True),
    # layers.LSTM(lstm_cells),

#         layers.Dropout(0.5),
        
    # use ReLU in place of tanh function since they are very good alternatives of each other.
    layers.Dense(embedding_dim, activation='relu'),
        
    layers.Dropout(0.25),
        
    # layers.Dense(8),
    # layers.Dropout(0.2),
        
    # Add a Dense layer with 3 units (3 classes) and softmax activation.
    # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
    layers.Dense(3, activation='softmax')
    ])

    # Compile model, show summary
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", 
        metrics=['accuracy'])
    
    return model

def set_stopwords():
    """Snowball-Stem English stopwords, 
    list found at http://www.textfixer.com/resources/common-english-words.txt"""
    stemmer = SnowballStemmer('english')
    STOPWORDS = "a,able,about,across,after,all,almost,also,am,among,an,and,any"+\
        "are,as,at,be,because,been,but,by,can,could,dear,did,do,does,either"+\
        "else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his"+\
        "how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may"+\
        "me,might,most,must,my,neither,no,of,off,often,on,only,or,other,our"+\
        "own,rather,said,say,says,she,should,since,so,some,than,that,the,their"+\
        "them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were"+\
        "what,when,where,which,while,who,whom,why,will,with,would,yet,you,your"
    STOPWORDS = STOPWORDS.split(',') 
    stemmed_stopwords = set([stemmer.stem(word) for word in STOPWORDS])
    return stemmed_stopwords

def process_input(text, maxlen, num_words):
# Lower case, remove punctuation and stop words from X data
    text = text.lower()
    # Get stopwords
    stopwords = set_stopwords()
    stop_pat = ' | '.join(stopwords)
    
    # Remove punctuation digits, and stop words
    regex = '[^a-zA-Z\s]'
    text = re.sub(regex, ' ', text)
    text = re.sub(stop_pat, ' ', text)
    
    # Snowball stemming
    stemmer = SnowballStemmer('english')
    tokenized_text = [" ".join([stemmer.stem(word) for word in text.split()])]
    
    # # Tokenize data ######################

    # # Set tokenization params
    oov_tok = '<OOV>'
    trunc_type = 'post'
    padding_type = 'post'
    
    # Load tokenizer from pickle
    with open('src/tokenizer_50pct.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    tokenized_text = tokenizer.texts_to_sequences(tokenized_text)
        
    # Pad with zeros
    tokenized_text = pad_sequences(tokenized_text,padding=padding_type, truncating=trunc_type, maxlen=maxlen)
    tokenized_text = np.array(tokenized_text)
    # print('Tokenized text shape:', tokenized_text.shape) #DEBUG
    return tokenized_text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)