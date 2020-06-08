# Between the Lines of Tripadvisor Hotel Reviews

![Image from https://www.pexels.com/photo/bedroom-door-entrance-guest-room-271639/](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/hotel.jpg)

## Check out my Flask web app - Airbnb Review Sentiment Classifier: https://tinyurl.com/rating-predictor

#### Link to Project Presentation: https://docs.google.com/presentation/d/1nZ9morIyqlIuJPOEAuhNwTw9m3lByksouw4KqXlmOfQ/edit?usp=sharing
___
## Business Context
### Imagine you rent out places to stay like on Airbnb.
On Airbnb, unlike Tripadvisor, there is no rating per user review, so you cannot know the exact rating each user gives you. Now, let's say you have tons of reviews and cannot read every single one to determine whether they had positive (4-5 stars), neutral (3 stars), or negative (1-2 stars) sentiment.
> Problem: How can you automatically know how your customers feel in order to reach out to them promptly and properly based on their sentiments? 
#### Solution: Mine Tripadvisor hotel reviews “labeled” with ratings and use them to train a sentiment classifier.
 * Tripadvisor hotel reviews suit our needs because they are labeled data (have user rating per review) and may be similar to Airbnb reviews since they are both in written in the context of a person's experience staying of at a place.

In this project, I collected my own data through web scraping, used natural language processing, and built and evaluated different machine learning models, including logistic regressions and neural networks, in order to create a minumum viable product that solves this business problem.

![Tripadvisor_Logo_horizontal-lockup_registered_RGB.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/Tripadvisor_Logo_horizontal-lockup_registered_RGB.png) 

![bs.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/bs.png)

## Summary of the Process
1. First, I web-scraped TripAdvisor hotel reviews. I used 2-3 AWS EC2 Linux instances (t2.micro) that ran in parallel over 4 days. While scraping, I set up my data cleaning, EDA, and modeling pipelines in python modules. The data was automatically saved into CSV files, one per link/hotel scraped.
2. Next, I cleaned the data and split it into 80:20 train/test, then split the train data into 80:20 train/validation. This resulted in an overall 64/16/20 train/val/test split.
3. After, I experimented with natural language processing techniques (removing stop words, stemming/lemmatizing) on the review text data and built different machine learning models (logistic regressions, multinomial Naive Bayes, random forests, LSTM neural networks). I evaluated models on both accuracy and confusion matrices. Due to class imbalance, I undersampled the training data for non-neural network models and used class weights without undersampling for neural networks.
4. Finally, I deployed my sentiment classifier, a CNN-LSTM neural network model, as Flask web app running in a Tensorflow docker container on an AWS EC2 Linux instance (t3.large). The web app is now live for people to try out on unseen, real-world data - Airbnb reviews (or other similar reviews on places to stay).

![Tripadvisor_Logo_horizontal-lockup_registered_RGB.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/Tripadvisor_Logo_horizontal-lockup_registered_RGB.png) 
![bs.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/bs.png)

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
* keras_lstm&#46;py runs either training a new model for a specified number or epochs on specified data or loading a previously saved model from a specified path to evaluate on specified data.
* preprocess&#46;py includes all data preprocessing functions and runs 5-fold cross validation on 
* tokenizer&#46;pickle files contain the tokenizers used on data before input to neural network-based models.
4. The scrapers folder contains 10 tripadvisor_scraper*.py modules that can be run to scrape Tripadvisor hotel reviews
 * Each module contains many links, each corresponding to a different hotel, to scrape from and the output of the modules are CSV files, one per link/hotel
 * A preview of the fields of the CSV files can be seen in the eda.ipynb notebook (Out[8]): https://github.com/chelseanbr/between-the-lines-hotels/blob/master/eda.ipynb

## EDA
My final dataset consisted of 1.2 million hotel reviews in English, each with a Tripadvisor “bubble” rating from 1 to 5.
* There were 11 columns in my data: review_id, url, hotel_name, review_date, review_body, user_location, user_name, helpful_vote, rating, csv, and folder.

![countplot_reviews_byLocation_full.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/countplot_reviews_byLocation_full.png)

![boxplt_ratings_byLocation_full.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/boxplt_ratings_byLocation_full.png)

#### I added sentiment labels based on hotel rating per review. 1-2 = negative, 3 = neutral, 4-5 = positive.
* After cleaning/preprocessing the data, there were 14 columns total, 3 were added: city, loc, and sentiment (the labels/target values).

![countplot_ratings_full.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/setup/imgs/countplot_ratings_full.png)

![pie_sentiments_initial.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/pie_sentiments_initial.png)

#### The plots above show the class imbalance.

![sample1000_review_len_dist.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/sample1000_review_len_dist.png)

## Modeling
## Part 1: Non-Neural Network-Based Models
Initially, I tried non-neural network-based Models with TF-IDF (term frequency–inverse document frequency) fetaures.
### Handling Imbalanced Classes
I undersampled only the training data to balance classes. This was done to prevent my classifier from becoming biased and tend to mostly predict the majority class, which was 'positive' class. The 'negative' class was the minority, so I undersampled the bigger 'positive' and 'neutral' classes to have them match the minority class in size.

![pie_sentiments_initial.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/setup/imgs/pie_sentiments_train_undersample.png)
* The validation set and test set remained untouched.

### Natural Language Processing
1. First, I removed digits, punctuation, and English 
stop words. For stop words, I used a custom list seen in my python modules.
2. Then, I tried various stemmers/lemmatizers and different numbers of TF-IDF max features.

![mnb_accuracy_over_feature_size.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/mnb_accuracy_over_feature_size.png)

#### I decided to proceed with using the WordNetLemmatizer and 50k+ TF-IDF features. After further experimentation, I found I could reduce the TF-IDF features to 5,000 since it did not really impact scores. 

![model_val_comparisions.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/setup/imgs/model_val_comparisions.png)

## Results
![confusion_matrix_final_lr_val.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/setup/imgs/confusion_matrix_final_lr_val.png)

* Logistic Regression (multinomial)
* Achieved after tuning C to 0.1 with GridSearch
* Solver = "newton-cg"
* 81% accuracy on validation data
* Did best with WordNet Lemmatized TF-TDF on 5,000 features

![wordcloud_positive.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/wordcloud_positive.png)

![wordcloud_neutral.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/wordcloud_neutral.png)

![wordcloud_negative.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/setup/imgs/confusion_matrix_final_lr_val.png)

### Example of use on Airbnb review:
> "Street noise is noticeable at the higher floors"
* Predicted **neutral.**
32% negative, **47% neutral,** 21% positive

## Part 2: Neural Network-Based Models

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
![confusion_matrix_train_lstm_10epochs_20200608-04:29:46.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/setup/imgs/neural_net_cm/confusion_matrix_train_lstm_10epochs_20200608-04:29:46.png)

![confusion_matrix_val_lstm_10epochs_20200608-04:29:46.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/setup/imgs/neural_net_cm/confusion_matrix_val_lstm_10epochs_20200608-04:29:46.png)

![lstm_10epochs_20200608-04:29:46_loss.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/setup/imgs/neural_net_history/lstm_10epochs_20200608-04:29:46_loss.png)

![lstm_10epochs_20200608-04:29:46_accuracy.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/setup/imgs/neural_net_history/lstm_10epochs_20200608-04:29:46_accuracy.png)
```bash
Reading data/tripadvisor_reviews_1p2m.csv...
Cleaning data...

Splitting data into train/test/val...
        Train: 779120, Test: 243475, Val: 194780
Taking 50pct of data - Train: 389560, Val: 97390, Test: 121737

Removing punctuation, digits, and stop words from X_train/val/test data...

Tokenizing X_train/val/test data...

Starting modeling...

Will save model to: saved_models/lstm_6epochs_20200608-07:34:50

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
Epoch 1/6
6087/6087 [==============================] - ETA: 0s - loss: 0.5901 - accuracy
: 0.8166
Epoch 00001: val_accuracy improved from -inf to 0.86511, saving model to BEST_
saved_models/lstm_6epochs_20200608-07:34:50/
6087/6087 [==============================] - 3186s 523ms/step - loss: 0.5901 -
 accuracy: 0.8166 - val_loss: 0.3307 - val_accuracy: 0.8651
Epoch 2/6
6087/6087 [==============================] - ETA: 0s - loss: 0.5084 - accuracy
: 0.8421
Epoch 00002: val_accuracy improved from 0.86511 to 0.87598, saving mo[35/2291]
ST_saved_models/lstm_6epochs_20200608-07:34:50/
6087/6087 [==============================] - 3184s 523ms/step - loss: 0.5084 $
 accuracy: 0.8421 - val_loss: 0.2904 - val_accuracy: 0.8760
Epoch 3/6
6087/6087 [==============================] - ETA: 0s - loss: 0.4790 - accurac$
: 0.8503
Epoch 00003: val_accuracy did not improve from 0.87598
6087/6087 [==============================] - 3183s 523ms/step - loss: 0.4790 $
 accuracy: 0.8503 - val_loss: 0.2989 - val_accuracy: 0.8741
Epoch 4/6
6087/6087 [==============================] - ETA: 0s - loss: 0.4599 - accurac$
: 0.8547
Epoch 00004: val_accuracy did not improve from 0.87598
6087/6087 [==============================] - 3181s 523ms/step - loss: 0.4599 $
 accuracy: 0.8547 - val_loss: 0.3054 - val_accuracy: 0.8668
Epoch 5/6
6087/6087 [==============================] - ETA: 0s - loss: 0.4451 - accurac$
: 0.8563
Epoch 00005: val_accuracy did not improve from 0.87598
6087/6087 [==============================] - 3187s 524ms/step - loss: 0.4451 $
 accuracy: 0.8563 - val_loss: 0.3043 - val_accuracy: 0.8694
Epoch 6/6
6087/6087 [==============================] - ETA: 0s - loss: 0.4308 - accurac$
: 0.8600
Epoch 00006: val_accuracy improved from 0.87598 to 0.87695, saving model to B$
ST_saved_models/lstm_6epochs_20200608-07:34:50/
6087/6087 [==============================] - 3181s 523ms/step - loss: 0.4308 $
 accuracy: 0.8600 - val_loss: 0.2901 - val_accuracy: 0.8769

Took 19116.94s to train
12174/12174 [==============================] - 823s 68ms/step - loss: 0.2432 $
 accuracy: 0.8971
Training Accuracy:  0.8971455097198486
3044/3044 [==============================] - 204s 67ms/step - loss: 0.5526 - $
ccuracy: 0.8769
Val Accuracy:  0.876948356628418
3805/3805 [==============================] - 255s 67ms/step - loss: 0.5434 - $ccuracy: 0.8782
Test Accuracy:  0.8781553506851196

        Predict train
              precision    recall  f1-score   support

    negative       0.82      0.83      0.83     24673
     neutral       0.46      0.80      0.58     34034
    positive       0.99      0.91      0.95    330853

    accuracy                           0.90    389560
   macro avg       0.76      0.85      0.79    389560
weighted avg       0.93      0.90      0.91    389560


        Predict val
              precision    recall  f1-score   support

    negative       0.72      0.71      0.72      6168
     neutral       0.40      0.70      0.51      8508
    positive       0.98      0.91      0.94     82714

    accuracy                           0.88     97390
   macro avg       0.70      0.77      0.72     97390
weighted avg       0.92      0.88      0.89     97390


        Predict test
              precision    recall  f1-score   support

    negative       0.72      0.74      0.73      7710
     neutral       0.40      0.70      0.51     10636
    positive       0.98      0.91      0.94    103391

    accuracy                           0.88    121737
   macro avg       0.70      0.78      0.73    121737
weighted avg       0.92      0.88      0.89    121737
```

![confusion_matrix_train_lstm_6epochs_20200608-07:34:50.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/setup/imgs/neural_net_cm/confusion_matrix_train_lstm_6epochs_20200608-07:34:50.png)

![confusion_matrix_val_lstm_6epochs_20200608-07:34:50.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/setup/imgs/neural_net_cm/confusion_matrix_val_lstm_6epochs_20200608-07:34:50.png)

![confusion_matrix_test_lstm_6epochs_20200608-07:34:50.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/setup/imgs/neural_net_cm/confusion_matrix_test_lstm_6epochs_20200608-07:34:50.png)

![lstm_6epochs_20200608-07:34:50_loss.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/setup/imgs/neural_net_history/lstm_6epochs_20200608-07:34:50_loss.png)

![lstm_6epochs_20200608-07:34:50_accuracy.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/setup/imgs/neural_net_history/lstm_6epochs_20200608-07:34:50_accuracy.png)

## Web App
#### Airbnb Review Sentiment Classifier: https://tinyurl.com/rating-predictor

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

## Conclusion
* 

## Next Steps
* 

_____ 
## Initial Project Proposal
### What are you trying to do?
What I will try to do for my project is use TripAdvisor hotel reviews with ratings per review to classify sentiments as positive, neutral, or negative. Being able to automatically classify sentiment from review content is important to get a sense of how customers feel and what they would like.
### How has this problem been solved before?
This problem has been solved before with techniques in natural language processing such as TF-IDF or with LSTM neural networks. 

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
My data sources for the Tripadvisor hotel reviews is my own scraped data in csv files stored in multiple folders. From my previous capstone, I had 500k reviews, so I will work on doubling the size.
### What are potential problems with your capstone?
The potential problems with my capstone are not achieving as high accuracy due to the amount of time it takes to train neural networks.
### What is the next thing you need to work on?
The next thing I need to work on is scraping more data and making sure it will meet my new needs for a recommender system.