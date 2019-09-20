# Machinelearning- Assignments 

Assignments solved and has been commited in respective branches

KNN branch - Has Custom- KNN algorithm compared with SKlearn algorithm 

Timeseries - It has predictions of next 30 days of HDFC stock prices , using 5 years past stockprices using time series analysis

Ted- Is consists of EDA for the TED talk show

# Custom KNN  Algorithm:
K Nearest Neighbours is one of the most commonly implemented Machine Learning classification algorithms. 
The kNN is more widely used in classification problems than for regression problems, although it can be applied for both classification and regression problems.

k-NN algorithm uses the entire training dataset as its model. For a new prediction to come out of k-NN, the algorithm will scour the entire training data to find ‘k’ points nearest to the incoming data by using Euclidean distance. Then, depending on the class of these ‘k’ points, the probable class of the incoming data is predicted.

Libraries that can be used
Standard Libraries,Pandas,Numpy,Matplotlib,Seaborn,Sklearn



#  Custom KNN Algorithm

I followed the below steps  to implement k-Nearest Neighbors in Python

Handle Data: Open the dataset from Text file , converted in to Dataframe , and added column names,
            treated  misiing values and split into test/train datasets.

Similarity: Calculate the distance between two data instances.

Neighbors: Locate k most similar data instances.

Response: Generate a response from a set of data instances.

Accuracy: Summarize the accuracy of predictions.

Compare : Compare the Acuuracy and timing with Sklearn KNN algorithm

Balance : Blancing the data using SMOTE model , re run the alorithm and comapre Custom and Sklearn algorithms peroformances 



# Time series :
Time Series Forecasting finds a lot of applications in many branches of industry or business. It allows to predict product demand (thus optimizing production and warehouse storage), forecast amount of money from sales (adjusting company’s expenses) or to predict future values of stock prices.

Before moving to sophisticated time series models Analysis here are the models used in Predictions

1. regression_on_time

2.Naive method

3.Simple Average Method

4.Moving Average Method

5.Simple exponential smoothing

6.holts method

7.Holts winter method - Additive

8.Holts winter method - Mutltiplicative

9. Auto Regression MOdel

10. ARIMA 


# TED Talk

Analysis of TED talk attributes to predict highly viewed talks. EDA done to analyze some content of the most and least viewed talks. 

The dataset contains a number of features including the speaker, related tags, related talks, number of comments, and number of view. It also contains the transcripts for each talk.

The purpose of this exploration was to try to train an algorithm to predict which talks would be highly viewed (view count above the mean view count). Feature engineering and various classification tools were used to determine the most useful parameters to train with.

Subsequent work explored EDA and the content of the most and least viewed TED talk in the dataset. This work is still ongoing.
