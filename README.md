# Between the Lines of Tripadvisor Hotel Reviews
![Image from https://www.pexels.com/photo/bedroom-door-entrance-guest-room-271639/](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/hotel.jpg?raw=true)
#### Link to Presentation: 
* To be added
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

## File Directory
```bash
between-the-lines-hotels
├── README.md
├── eda.ipynb
├── imgs
│   └── (__ images)
└── src
    ├── eda.py
    ├── flask_app
    │   ├── app.py
    │   ├── boostrap
    │   │   └── (Folders: css, fonts, js)
    │   ├── static
    │   │   ├── (Folders: css, fonts, js)
    │   │   └── (4 images)
    │   └── templates
    │       ├── jumbotron.html
    │       ├── predict.html
    ├── gridsearch.py
    ├── keras_lstm.py
    ├── preprocess.py
    └── scrapers
        └── tripadvisor_scraper.*.py (10 files)
```

## EDA
* Whole dataset consisted of 1.2 million hotel reviews in English, each with a Tripadvisor “bubble” rating from 1 to 5

![countplot_reviews_byCity_full.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/countplot_reviews_byCity_full.png)

![boxplt_ratings_byCity_full.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/boxplt_ratings_byCity_full.png)

* Added sentiment label based on hotel rating per review

![countplot_ratings_full.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/countplot_ratings_full.png)

![pie_sentiments_initial.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/pie_sentiments_initial.png)

## Predictive Modeling

### Handling Imbalanced Classes
* Under-sampled train data to balance classes
* Train data qty reduced from ~300k to 94k observations
![pie_sentiments_train_undersample.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/pie_sentiments_train_undersample.png)
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

## Web App

## Next Steps
* Try out model on other data like tweets in the context of hotels/places to stay
* Explore advanced NLP/ML methods like Word2Vec, LSTM recurrent neural networks
* Mine more data and build a hotel recommender system