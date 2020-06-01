# Between the Lines of Tripadvisor Hotel Reviews

![Image from https://www.pexels.com/photo/bedroom-door-entrance-guest-room-271639/](https://github.com/chelseanbr/between-the-lines/blob/final_eda_modeling/images/hotel.jpg?raw=true)
#### Link to Presentation: 
* To be added
_____
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

## EDA
* Whole dataset consisted of ______ hotel reviews in English, 
each with a Tripadvisor “bubble” rating from 1 to 5
* Added sentiment label based on hotel rating per review

![countplot_ratings_full.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/countplot_ratings_full.png)

![pie_sentiments_initial.png](https://github.com/chelseanbr/between-the-lines-hotels/blob/master/imgs/pie_sentiments_initial.png)
