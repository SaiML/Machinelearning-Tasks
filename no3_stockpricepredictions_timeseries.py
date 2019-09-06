# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:48:11 2019

@author: Sainath.Reddy
"""

# =============================================================================
# importing Libraries
# =============================================================================

import pandas as pd 
import numpy as np 
from pandas import Series
from pandas import DataFrame
from math import sqrt
import matplotlib.pyplot as plt 
import seaborn as sns
from pandas import concat
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import lag_plot

from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# Loading Data 
# =============================================================================

# BSE HDFC Bank ltd - 5 years Daily stock prices Dataset
# Data has been taken from the link 
#https://www.quandl.com/data/BSE/BOM500180-HDFC-Bank-Ltd-EOD-Prices

df = pd.read_csv('BSE-BOM500180.csv', header=0)
print(df.head())
print(df.tail())
print(df.dtypes)

df['Date']=pd.to_datetime(df['Date'])
df=df.set_index('Date')
print(df.head())
print(df.describe().transpose())

df1=df[['Close']]
print(df1.head())
print(df1.tail())

# Hence it is daily data, but few stockprices values are missing for few Dates 
# so , creating a index of continous dates and filling the missed values linearly, so that we can have all dates stock prices 

idx = pd.date_range( '2014-08-27','2019-08-27')
df2=df1.reindex(idx, fill_value=0)

df2.replace(0,np.nan, inplace=True)
dfnew=df2.interpolate(method='linear')
print(dfnew.head())
print(dfnew.tail())

# =============================================================================
# # Checking Data Stationary :
# =============================================================================

#Determinig Rolling Statistics :

rollingmean=dfnew.rolling(window=365).mean()
rollingstd=dfnew.rolling(window=365).std()

orig=plt.plot(dfnew,color='blue',label='Original')
mean=plt.plot(rollingmean,color='red',label='Rolling Mean')
std=plt.plot(rollingstd,color='Orange',label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling mean and Std')
plt.show()

#Dickey- Fuller Test

from statsmodels.tsa.stattools import adfuller

print(" Results of Dickey- Fuller Test:")
dftest = adfuller(dfnew['Close'])
dfoutput=pd.Series(dftest[0:4],index=['Test statisctic','p-value','#lags used','number of observations used'])
for key,value in dftest[4].items():
    dfoutput['critical value (%s)' %key] =value
print(dfoutput)

# Data is not Stationary because rolling mean is not constant , and from  Dickey- Fuller Test P- Values is > 0.5 
# =============================================================================
# 
# # Checking Seasonality and Trend
# =============================================================================

import statsmodels.api as sm
decomp= sm.tsa.seasonal_decompose(dfnew,freq=365)
# result = sm.tsa.stattools.adfuller(dfnew)
decomp.plot()
plt.show()

# We can observe the Trend is in upward direction and it has seasonality aswell

# =============================================================================
# # Train and Test data Split
# =============================================================================

train=dfnew.loc['2014-08-27':'2019-02-26']
test=dfnew.loc['2019-02-27':'2019-08-27']

# Taking 6 months data as Testing data

train.Close.plot(figsize=(15,8), title= 'Close price', fontsize=14)
test.Close.plot(figsize=(15,8), title= 'Close price train and test data', fontsize=14)
plt.show()

# =============================================================================
# # Method 1 : REGRESSION ON TIME 
# =============================================================================

time = [i+1 for i in range(len(dfnew))]
df1 = dfnew.copy()
df1['time'] = time
df1 = df1[['time', 'Close']]
train_data=df1.loc['2014-08-27':'2019-02-26']
test_data=df1.loc['2019-02-27':'2019-08-27']
x_train = train_data.drop('Close', axis=1)
x_test = test_data.drop('Close', axis=1)
y_train = train_data[['Close']]
y_test = test_data[['Close']]

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
y_test['RegOnTime'] = predictions
plt.figure(figsize=(16,8))
plt.plot( train_data['Close'], label='Train')
plt.plot(test_data['Close'], label='Test')
plt.plot(y_test['RegOnTime'], label='Regression On Time')
plt.title("Regression On Time")
plt.legend(loc='best')
plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(test_data.Close, y_test.RegOnTime))
rmse = round(rmse, 3) 
abs_error = np.abs(y_test['Close']-y_test['RegOnTime'])
actual = y_test['Close']
mape = np.round(np.mean(abs_error/actual),3)
resultsDf = pd.DataFrame({'Method':['Rregression on time'], 'MAPE': [mape], 'RMSE': [rmse]})
resultsDf = resultsDf[['Method', 'RMSE', 'MAPE']]
resultsDf

# =============================================================================
# # Method 2 : Naive Method
# =============================================================================

df_naive = dfnew[['Close']]
df_naive['trend']=np.nan
for i in range(len(df_naive)):
    if i == 0:
        df_naive['trend'] = np.nan
    else:
        df_naive['trend'][i] = df_naive.Close[i-1]


train_data=df_naive.loc['2014-08-27':'2019-02-26']
test_data=df_naive.loc['2019-02-27':'2019-08-27']

import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plt.plot(train_data.index, train_data['Close'], label='Train')
plt.plot(test_data.index,test_data['trend'], label='Naive Forecast')
plt.plot(test_data.index,test_data['Close'], label='Test')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show()

MSE       =  mean_squared_error(test_data['Close'], test_data['trend'])
rmse      =  np.sqrt(MSE)
mape= np.mean(np.abs(test_data['Close']-test_data['trend'])/test_data['Close'])

tempResultsDf = pd.DataFrame({'Method':['Naive approach'], 'RMSE': [rmse],'MAPE': [mape]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf


# =============================================================================
# # 3.Simple Average Method
# =============================================================================

df_sa=dfnew[['Close']]
df_sa['trend']=np.nan
for i in range(len(dfnew)):
    df_sa['trend'][i] = round(dfnew.Close.mean())

train=df_sa.loc['2014-08-27':'2019-02-26']
test=df_sa.loc['2019-02-27':'2019-08-27']

import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plt.plot(train.index, train['Close'], label='Train')
plt.plot(test.index,test['Close'], label='Test')
plt.plot(test.index,test['trend'], label='Simple Average Method')
plt.legend(loc='best')
plt.title("Simple Average Method")
plt.show()

MSE       =  mean_squared_error(test['Close'], test['trend'])
rmse      =  np.sqrt(MSE)
mape= np.mean(np.abs(test['Close']-test['trend'])/test['Close'])

tempResultsDf = pd.DataFrame({'Method':['Simple Average'], 'RMSE': [rmse],'MAPE': [mape]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf

# =============================================================================
# 
# # 4.Moving Average Method
# =============================================================================

df_ma = dfnew[['Close']]
df_ma['trend'] = df_ma['Close'].rolling(15).mean()

train=df_ma.loc['2014-08-27':'2019-02-26']
test=df_ma.loc['2019-02-27':'2019-08-27']


import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plt.plot(train.index, train['Close'], label='Train')
plt.plot(test.index,test['Close'], label='Test')
plt.plot(test.index,test['trend'], label='Moving Average')
plt.legend(loc='best')
plt.title("Moving Average")
plt.show()

MSE       =  mean_squared_error(test['Close'], test['trend'])
rmse      =  np.sqrt(MSE)
mape= np.mean(np.abs(test['Close']-test['trend'])/test['Close'])

tempResultsDf = pd.DataFrame({'Method':['Moving Average'], 'RMSE': [rmse],'MAPE': [mape]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf

# =============================================================================
# # 5. Simple Exponential Smoothing
# =============================================================================

train=dfnew.loc['2014-08-27':'2019-02-26']
test=dfnew.loc['2019-02-27':'2019-08-27']

from statsmodels.tsa.api import SimpleExpSmoothing
model = SimpleExpSmoothing(train['Close'])
model_fit = model.fit(smoothing_level=0.6,optimized=False)
model_fit.params

test['trend'] = model_fit.forecast(len(test['Close']))
plt.figure(figsize=(12,8))
plt.plot(train.index, train['Close'], label='Train')
plt.plot(test.index,test['Close'], label='Test')
plt.plot(test.index,test['trend'], label='Simple Moving Average')
plt.legend(loc='best')
plt.title("simple exponential smoothing Forecast")
plt.show()

MSE       =  mean_squared_error(test['Close'], test['trend'])
rmse      =  np.sqrt(MSE)
mape= np.mean(np.abs(test['Close']-test['trend'])/test['Close'])

tempResultsDf = pd.DataFrame({'Method':['Simple exponential Smoothing'], 'RMSE': [rmse],'MAPE': [mape]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf

# =============================================================================
# # 6. Holts Method
# =============================================================================

train=dfnew.loc['2014-08-27':'2019-02-26']
test=dfnew.loc['2019-02-27':'2019-08-27']
from statsmodels.tsa.api import ExponentialSmoothing
molts_model = ExponentialSmoothing(train[['Close']],trend='add', seasonal='none').fit()
molts_model.params
test['trend'] = molts_model.forecast(len(test))

plt.figure(figsize=(12,8))
plt.plot(train.index, train['Close'], label='Train')
plt.plot(test.index,test['Close'], label='Test')
plt.plot(test.index,test['trend'], label='Holts Method')
plt.legend(loc='best')
plt.title("Holts Method")
plt.show()

MSE       =  mean_squared_error(test['Close'], test['trend'])
rmse      =  np.sqrt(MSE)
mape= np.mean(np.abs(test['Close']-test['trend'])/test['Close'])

tempResultsDf = pd.DataFrame({'Method':['Holts Method'], 'RMSE': [rmse],'MAPE': [mape]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf

# =============================================================================
# # 7. Holts Winter Method - Additive
# =============================================================================

from statsmodels.tsa.api import SimpleExpSmoothing
holts_model = ExponentialSmoothing(train[['Close']],seasonal_periods=365 ,trend='add', seasonal='add').fit()
# holts_model.params
test['trend'] = holts_model.forecast(len(test))
plt.figure(figsize=(12,8))
plt.plot(train.index, train['Close'], label='Train')
plt.plot(test.index,test['Close'], label='Test')
plt.plot(test.index,test['trend'], label='Holts Winter Forecast')
plt.legend(loc='best')
plt.title("Holts Winter Method - Additive")
plt.show()

MSE       =  mean_squared_error(test['Close'], test['trend'])
rmse      =  np.sqrt(MSE)
mape= np.mean(np.abs(test['Close']-test['trend'])/test['Close'])
tempResultsDf = pd.DataFrame({'Method':['Holts Winter - Additive'], 'RMSE': [rmse],'MAPE': [mape]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf

# =============================================================================
# # 8. Holt-Winters Method - Multiplicative
# =============================================================================

train=dfnew.loc['2014-08-27':'2019-02-26']
test=dfnew.loc['2019-02-27':'2019-08-27']
holts_model = ExponentialSmoothing(train['Close'],seasonal_periods=365 ,trend='add', seasonal='mul').fit()
test['trend'] = holts_model.forecast(len(test))
plt.figure(figsize=(12,8))
plt.plot(train.index, train['Close'], label='Train')
plt.plot(test.index,test['Close'], label='Test')
plt.plot(test.index,test['trend'], label='Holts Winter- Multiplicative')
plt.legend(loc='best')
plt.title("Holt-Winters Method - Multiplicative")
plt.show()

MSE       =  mean_squared_error(test['Close'], test['trend'])
rmse      =  np.sqrt(MSE)
mape= np.mean(np.abs(test['Close']-test['trend'])/test['Close'])
tempResultsDf = pd.DataFrame({'Method':['Holts Winter - Multiplicative'], 'RMSE': [rmse],'MAPE': [mape]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
print(resultsDf)
# Among all basic models Niave approach and moving Average are having RMSE values low

# =============================================================================
# # 9.Autoregression (AR) Method
# =============================================================================

train=dfnew.loc['2014-08-27':'2019-02-26']
test=dfnew.loc['2019-02-27':'2019-08-27']
from statsmodels.tsa.ar_model import AR
from random import random
model = AR(train)
model_fit = model.fit()
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
plt.figure(figsize=(12,8))
plt.plot(train.index, train['Close'], label='Train')
plt.plot(test.index,test['Close'], label='Test')
plt.plot(test.index,predictions, label='Autoregression (AR) Method')
plt.legend(loc='best')
plt.title("Autoregression (AR) Method")
plt.show()

rmse = np.sqrt(mean_squared_error(test, predictions))
mape='nan'
tempResultsDf = pd.DataFrame({'Method':['AR model'], 'RMSE': [rmse],'MAPE': [mape]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf

# =============================================================================
# # 10.ARIMA model
# =============================================================================

# Checking for Auto Correlation 
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot

plot_acf(dfnew['Close'], lags=36)
plt.show()
plot_pacf(dfnew, ax=pyplot.gca(),lags=30)
pyplot.show()
plt.figure(figsize=(16,8))
pd.plotting.autocorrelation_plot(dfnew['Close'])
plt.show()

# Log tranforming Data to make data Stationary 
ts_log = np.log(dfnew.Close)
plt.title('Log of the data')
plt.plot(ts_log)
plt.show()

moving_avg = ts_log.rolling(365).mean()
plt.plot(ts_log)
plt.title('12 years of Moving average')
plt.plot(moving_avg, color='blue')
plt.show()

ts_log_moving_avg_diff = ts_log - moving_avg

def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=365).mean()
    rolstd = timeseries.rolling(window=365).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)
expwighted_avg = ts_log.ewm(halflife=365).mean()
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')
plt.show()
ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)

ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)
plt.show()

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

decomp= sm.tsa.seasonal_decompose(ts_log_diff,freq=365)
decomp.plot()
plt.show()

from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.show()
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.show()
plt.tight_layout()

# Finding best comnbination of p,d,q values for our ARIMA model 
p=d=q=range(0,5)
import itertools
val = list(itertools.product(p,d,q))

from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings('ignore')
print("Combinations of p,d,p for ARIMA")
for param in val:
    try:
        model_arima = ARIMA(ts_log,order = param)
        model_arima_fit = model_arima.fit()
        print(param,model_arima_fit.aic)
    
    except:
        continue
    
model = ARIMA(ts_log, order=(0,2,0), freq=ts_log.index.inferred_freq)  
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.show()
print(results_ARIMA.summary())


# =============================================================================
# # Auto Arima , SARIMAX
# =============================================================================

from pmdarima.arima import auto_arima
stepwise_model = auto_arima(dfnew, start_p=1, start_q=1,
                           max_p=5, max_q=5, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())

print(stepwise_model.summary())

predictions = stepwise_model.predict(n_periods=30)
future_dates = pd.date_range(start='28/08/2019', freq='D', periods=len(predictions))
future_dates

# =============================================================================
# # Predictions of Stock prices for Next 30 days using SARIMA model
# =============================================================================
data = pd.DataFrame(predictions,index=future_dates)
data.columns=['Close']

future_forecast = pd.DataFrame(predictions,index=future_dates)
future_forecast.columns=['prediction']
print("Future predictions for Next 30 days \n",future_forecast)

dfh=dfnew['2019-01-01':'2019-08-27']
plt.figure(figsize=(15,7.5))
plt.title("SARIMA forecast for next 30 days")
plt.plot(dfh.Close)
plt.plot(data,color = 'red')
plt.show()
