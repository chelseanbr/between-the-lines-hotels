# Docker: docker exec -it quizzical_taussig /bin/bash

from flask import Flask, render_template, request
app = Flask(__name__)

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Home page
@app.route('/')
def index():
    return render_template('jumbotron.html', title='Predict an Airbnb Rating')

# My sentiment predictor app
@app.route('/prediction', methods=['POST'])
def show_pred():
    text = str(request.form['user_input'])
    tokenized_text = process_input(text)
    # PREDICT
    return str(tokenized_text)

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

def process_input(text):
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
    maxlen = 300
    oov_tok = '<OOV>'
    num_words = 5000
    trunc_type = 'post'
    padding_type = 'post'
    
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_tok)
    tokenize = tokenizer.fit_on_texts(tokenized_text)
    tokenized_text = tokenizer.texts_to_sequences(tokenized_text)

    vocab_size=len(tokenizer.word_index)+1
        
    # Pad with zeros
    tokenized_text = pad_sequences(tokenized_text,padding=padding_type, truncating=trunc_type, maxlen=maxlen)
    tokenized_text = np.array(tokenized_text)
    # print('Tokenized text shape:', tokenized_text.shape) #DEBUG
    return tokenized_text


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)