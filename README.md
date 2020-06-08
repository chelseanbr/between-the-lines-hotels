# Between the Lines of Tripadvisor Hotel Reviews
![Image from https://www.pexels.com/photo/bedroom-door-entrance-guest-room-271639/](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/hotel.jpg?raw=true)
#### Link to Presentation: https://docs.google.com/presentation/d/1nZ9morIyqlIuJPOEAuhNwTw9m3lByksouw4KqXlmOfQ/edit?usp=sharing
_____
## Initial Project Proposal
### What are you trying to do?
What I will try to do for my project is use TripAdvisor hotel reviews with ratings per review to classify sentiment and recommend hotels to users. Being able to automatically classify sentiment from review content is important to get a sense of how customers feel and what they would like.
### How has this problem been solved before?
This problem has been solved before with techniques in natural language processing such as TF-IDF to train recommenders through matrix factorization, as described in the following links:

Multi Class Text Classification with LSTM using TensorFlow 2.0:
https://towardsdatascience.com/multi-class-text-classification-with-lstm-using-tensorflow-2-0-d88627c10a35

Text Classification Example with Keras LSTM in Python:
https://www.datatechnotes.com/2019/06/text-classification-example-with-keras.html
### What is new about your approach, why do you think it will be successful?
What is new about my approach is that I would try to use more advanced NLP techniques like Word2Vec in order to prove my sentiment classifier accuracy from 81% (capstone 2) to closer to or above 90% and then separately incorporate sentiment into recommendation. Also, I will try to scrape my own data to make sure to get a large enough data size for balancing classes and if needed, I will address the “cold start” problem for recommendation.
### Who cares? If you're successful, what will the impact be?
If I am successful, the impact will be that with my own dataset, I would have built a sentiment classifier and hotel recommender using my own combination of techniques.
### How will you present your work?
I would like people to be able to interact with my work through a flask dashboard. I want them to be able to try uploading their own hotel review text to try out my finished hotel rating predictor and see if the rating my classifier predicts matches with what rating they would give based on the review. In addition, I would provide hotel recommendations 
### What are your data sources? What is the size of your dataset, and what is your storage format?
My data sources for the Tripadvisor hotel reviews would be my scraped data in csv files stored in multiple folders. From my previous capstone, I had 500k reviews, so I will work on possibly doubling the size.
### What are potential problems with your capstone?
The potential problems with my capstone are the “cold start” problem for recommenders, because I will likely have only one or very few reviews for the majority of users in my data.
### What is the next thing you need to work on?
The next thing I need to work on is scraping more data and making sure it will meet my new needs for a recommender system.
___
## Context
### Imagine you rent out places to stay like on Airbnb.
> How can you easily know how customers feel to reach out and improve reputation?
#### Solution: Mine hotel reviews “labeled” with ratings and use them to predict sentiment.

## Summary of Process
![Tripadvisor_Logo_horizontal-lockup_registered_RGB.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/Tripadvisor_Logo_horizontal-lockup_registered_RGB.png) ![bs.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/bs.png)
1. Web-scraped TripAdvisor hotel reviews
  * 2 EC2 instances ran in parallel over 2 days
  * Set up data cleaning, EDA, and modeling pipeline while scraping
2. Split data into 80:20 train/test, then split train into 80:20 train/validation
3. Balanced training data with undersampling
4. Evaluated models on accuracy and confusion matrix

## Directory Structure
```bash
between-the-lines-hotels
├── README.md
├── eda.ipynb
├── imgs
│   └── (Images for README)
└── src
    ├── eda.py
    ├── flask_app
    │   ├── app.py
    │   ├── static
    │   │   ├── (Folders: css, fonts, js)
    │   │   └── (Images for site)
    │   └── templates
    │       ├── jumbotron.html
    │       └── predict.html
    ├── keras_lstm.py
    ├── preprocess.py
    ├── tokenizer*.pickle (3 files)
    └── scrapers
        └── tripadvisor_scraper*.py (10 files)
```
### File Descriptions and Instructions for Use:
1. README&#46;md is the file you are reading
2. eda&#46;ipynb is a jupyter notebook that contains my exploratory data analysis (EDA)
* In it, I loaded data into a pandas dataframe from either multiple CSV files contained in multiple folders or a single CSV file and then created data visualizations after understanding and processing the data
3. The src folder contains all python modules
* eda&#46;py contains the helper functions used for plotting in my EDA notebook (eda.ipynb)
* The flask_app folder contains all code to run my web app
  * app&#46;py runs the web app
  * The static folder 
  * The templates folder contain the layout of the web app home page (jumbotron.html), and prediction page (predict.html)
4. The scrapers folder contains 10 tripadvisor_scraper*.py modules that can be run to scrape Tripadvisor hotel reviews
 * Each module contains many links, each corresponding to a different hotel, to scrape from and the output of the modules are CSV files, one per link/hotel
 * A preview of the fields of the CSV files can be seen in the eda.ipynb notebook (Out[8]): https://github.com/chelseanbr/between-the-lines-hotels/blob/master/eda.ipynb

## EDA
* Whole dataset consisted of 1.2 million hotel reviews in English, each with a Tripadvisor “bubble” rating from 1 to 5

![countplot_reviews_byLocation_full.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/countplot_reviews_byLocation_full.png)

![boxplt_ratings_byLocation_full.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/boxplt_ratings_byLocation_full.png)

* Added sentiment label based on hotel rating per review

![countplot_ratings_full.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/countplot_ratings_full.png)

![pie_sentiments_initial.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/pie_sentiments_initial.png)

![sample1000_review_len_dist.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/sample1000_review_len_dist.png)

## Predictive Modeling

### Handling Imbalanced Classes
* Under-sampled train data to balance classes
* Train data qty reduced from ~300k to 94k observations
![pie_sentiments_initial.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/pie_sentiments_train_undersample.png)
* Validation set had 77k observations, test set had 96k

### NLP
* Removed English 
stop words, digits, and 
punctuation

* Tried different stemmers/
lemmatizers and TF-IDF
max features

![mnb_accuracy_over_feature_size.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/mnb_accuracy_over_feature_size.png)

* Decided to proceed with 
TF-IDF, 
WordNetLemmatizer,
and 5,000 features

## Results
![confusion_matrix_final_lr_test.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/confusion_matrix_final_lr_test.png)

* Logistic Regression (multinomial)
* Achieved after tuning C to 0.1 with GridSearch
* Solver = "newton-cg"
* 81% accuracy on validation and test data
* Did best with WordNet Lemmatized TF-TDF on 5,000 features

![wordcloud_positive.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/wordcloud_positive.png)

![wordcloud_neutral.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/wordcloud_neutral.png)

![wordcloud_negative.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/wordcloud_negative.png)

### Example of final model use on Airbnb review:
> "Street noise is noticeable at the higher floors"
* Predicted **neutral.**
32% negative, **47% neutral,** 21% positive

![confusion_matrix_dumb_model.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/confusion_matrix_dumb_model.png)


```bash
              precision    recall  f1-score   support

    negative       0.00      0.00      0.00       771
     neutral       0.00      0.00      0.00      1064
    positive       0.85      1.00      0.92     10338

    accuracy                           0.85     12173
   macro avg       0.28      0.33      0.31     12173
weighted avg       0.72      0.85      0.78     12173



Splitting data into train/test/val...
        Train: 779120, Test: 243475, Val: 194780
Taking 10pct of data - Train: 77912, Val: 19478, Test: 24347
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 550, 128)          640000
_________________________________________________________________
dropout (Dropout)            (None, 550, 128)          0
_________________________________________________________________
conv1d (Conv1D)              (None, 546, 64)           41024
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 136, 64)           0
_________________________________________________________________
bidirectional (Bidirectional (None, 136, 200)          132000
_________________________________________________________________
bidirectional_1 (Bidirection (None, 136, 200)          240800
_________________________________________________________________
bidirectional_2 (Bidirection (None, 200)               240800
_________________________________________________________________
dense (Dense)                (None, 128)               25728
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 387
=================================================================
Total params: 1,320,739
Trainable params: 1,320,739
Non-trainable params: 0
_________________________________________________________________
...
Epoch 10/10
1218/1218 [==============================] - ETA: 0s - loss: 0.2892 - accuracy: 0.8887
Epoch 00010: val_accuracy improved from 0.86380 to 0.86451, saving model to BEST_saved_models/lstm_10epochs_20200608-04:29:46/
1218/1218 [==============================] - 639s 525ms/step - loss: 0.2892 - 
accuracy: 0.8887 - val_loss: 0.3776 - val_accuracy: 0.8645

Took 6415.16s to train
        Predict train
              precision    recall  f1-score   support

    negative       0.93      0.94      0.93      4935
     neutral       0.56      0.94      0.70      6807
    positive       1.00      0.92      0.96     66170

    accuracy                           0.93     77912
   macro avg       0.83      0.94      0.86     77912
weighted avg       0.95      0.93      0.93     77912


        Predict val
              precision    recall  f1-score   support

    negative       0.68      0.58      0.63      1233
     neutral       0.37      0.66      0.47      1702
    positive       0.98      0.91      0.94     16543

    accuracy                           0.86     19478
   macro avg       0.67      0.71      0.68     19478
weighted avg       0.90      0.86      0.88     19478


        Predict test
              precision    recall  f1-score   support

    negative       0.67      0.62      0.64      1542
     neutral       0.38      0.66      0.48      2127
    positive       0.98      0.91      0.94     20678

    accuracy                           0.87     24347
   macro avg       0.67      0.73      0.69     24347
weighted avg       0.91      0.87      0.88     24347
```
![confusion_matrix_train_lstm_10epochs_20200608-04/29/46.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/neural_net_cm/confusion_matrix_train_lstm_10epochs_20200608-04/29/46.png)

![confusion_matrix_val_lstm_10epochs_20200608-04/29/46.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/neural_net_cm/confusion_matrix_val_lstm_10epochs_20200608-04/29/46.png)

![confusion_matrix_test_lstm_10epochs_20200608-04/29/46.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/neural_net_cm/confusion_matrix_test_lstm_10epochs_20200608-04/29/46.png)

![lstm_10epochs_20200608-04/29/46_loss.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/neural_net_history/lstm_10epochs_20200608-04/29/46_loss.png)

![lstm_10epochs_20200608-04/29/46_accuracy.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/neural_net_history/lstm_10epochs_20200608-04/29/46_accuracy.png)

```bash
Will save model to: saved_models/lstm_6epochs_20200608-07:34:50

2020-06-08 07:34:50.551559: E tensorflow/stream_executor/cuda/cuda_driver.cc:313] failed call to cuInit: UNKNOWN ERROR (303)


Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 550, 128)          640000
_________________________________________________________________
dropout (Dropout)            (None, 550, 128)          0
_________________________________________________________________
conv1d (Conv1D)              (None, 546, 64)           41024
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 136, 64)           0
_________________________________________________________________
bidirectional (Bidirectional (None, 136, 200)          132000
_________________________________________________________________
bidirectional_1 (Bidirection (None, 136, 200)          240800
_________________________________________________________________
bidirectional_2 (Bidirection (None, 200)               240800
_________________________________________________________________
dense (Dense)                (None, 128)               25728
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 387
=================================================================
Total params: 1,320,739
Trainable params: 1,320,739
Non-trainable params: 0
_________________________________________________________________

```
## Web App
#### Check out my Airbnb Review Sentiment Classifier: https://tinyurl.com/rating-predictor

Here is what the home page looks like:

![site_home.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/site_home.png)

Neutral prediction example:

![site_pred_neutral_default.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/setup/imgs/site_pred_neutral_default.png)

Positive prediction example:

![site_pred_pos.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/setup/imgs/site_pred_pos.png)

Negative prediction example:

![site_pred_neg.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/setup/imgs/site_pred_neg.png)

Just for fun, I tried submitting a fake review I wrote, and it was pretty funny to see the result:

![site_pred_neg_funny.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/setup/imgs/site_pred_neg_funny.png)

## Next Steps
* 