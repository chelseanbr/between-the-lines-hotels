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
Previous capstone on Tripadvisor hotel recommendation using TF-IDF, kNN, k-means clustering, and matrix factorization (Surprise): https://github.com/mathilda0902/howtoescapefrombatesmotel
https://towardsdatascience.com/building-a-content-based-recommender-system-for-hotels-in-seattle-d724f0a32070
Paper on Tripadvisor hotels with sentiment introduced into recommendation: 
https://www.insight-centre.org/sites/default/files/publications/16.177_from_more_like_this_to_better_than_this_hotel_recommendations_from_user_generated_reviews_0.pdf
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

## Context
### Imagine you rent out places to stay like on Airbnb.
> How can you easily know how customers feel to reach out and improve reputation?
#### Solution: Mine hotel reviews “labeled” with ratings and use them to predict sentiment.

## Summary of Process
![Tripadvisor_Logo_horizontal-lockup_registered_RGB.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/Tripadvisor_Logo_horizontal-lockup_registered_RGB.png) ![bs.png](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/bs.png)
1. Web-scraped TripAdvisor hotel reviews
  * 2 AWS EC2 instances ran in parallel over 2 days
2. Split data into 80:20 train/test, then split train into 80:20 train/validation
3. Balanced training data by undersampling minority classes
4. Evaluated models mainly on accuracy and confusion matrix

## File Directory
```bash
between-the-lines-hotels
├── BEST_saved_models
├── README.md
├── eda.ipynb
├── imgs
│   ├── boxplt_ratings_byLocation_full.png
│   ├── countplot_ratings_full.png
│   ├── countplot_reviews_byLocation_full.png
│   ├── hotel.jpg
│   ├── pie_sentiments_initial.png
│   ├── pie_sentiments_train_undersample.png
│   └── sample1000_review_len_dist.png
├── saved_models
└── src
    ├── PerformanceVisualizationCallback.py
    ├── eda.py
    ├── flask_app
    │   └── app.py
    ├── gridsearch.py
    ├── keras_lstm.py
    ├── pickle_model.py
    ├── preprocess.py
    └── scrapers
        └── tripadvisor_scraper.*.py (10 files)
```

## EDA
* Whole dataset consisted of ______ hotel reviews in English, 
each with a Tripadvisor “bubble” rating from 1 to 5
* Added sentiment label based on hotel rating per review

![countplot_ratings_full.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/countplot_ratings_full.png)

![pie_sentiments_initial.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/pie_sentiments_initial.png)