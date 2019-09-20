# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:48:11 2019

@author: Sainath.Reddy
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import SimpleExpSmoothing,ExponentialSmoothing
from statsmodels.tsa.stattools import acf, pacf,adfuller
from pmdarima.arima import auto_arima
import statsmodels.api as sm
from math import sqrt
import warnings
warnings.filterwarnings("ignore")


Data_stocks = pd.read_csv('BSE-BOM500180.csv', header=0)
Data_stocks['Date']=pd.to_datetime(Data_stocks['Date'])
Data_stocks=Data_stocks.set_index('Date')
timeseries_df=Data_stocks[['Close']]

# Hence it is daily data, but few stockprices values are missing for few Dates 
# so , creating a index of continous dates and filling the missed values linearly, so that we can have all dates stock prices 

idx = pd.date_range( '2014-08-27','2019-08-27')
timeseries_df1=timeseries_df.reindex(idx, fill_value=0)
timeseries_df1.replace(0,np.nan, inplace=True)
timeseries_dfnew= timeseries_df1.interpolate(method='linear')

# Checking Data Stationary :
def test_statinoray(timeseries):
    #Determinig Rolling Statistics
    rollingmean=timeseries.rolling(window=365).mean()
    rollingstd=timeseries.rolling(window=365).std()
    # Plotting Rolliong stats
    orig=plt.plot(timeseries,color='blue',label='Original')
    mean=plt.plot(rollingmean,color='red',label='Rolling Mean')
    std=plt.plot(rollingstd,color='Orange',label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling mean and Std')
    plt.show()

    # Performing Dickey- Fuller Test
    print(" Results of Dickey- Fuller Test:")
    dftest = adfuller(timeseries['Close'])
    dfoutput=pd.Series(dftest[0:4],index=['Test statisctic','p-value','#lags used','number of observations used'])
    for key,value in dftest[4].items():
        dfoutput[f"critical value {key}"] =value
    print(dfoutput)

test_statinoray(timeseries_dfnew)

# Checking Seasonality and Trend
decomp= sm.tsa.seasonal_decompose(timeseries_dfnew,freq=365)
decomp.plot()

#  Train and Test data Split and Functions for Plotting Graphs and Metrics
train=timeseries_dfnew.loc['2014-08-27':'2019-02-26']
test=timeseries_dfnew.loc['2019-02-27':'2019-08-27']

# Taking 6 months data as Testing data
def plltt(train_data,test_data,y_test,model):
    plt.figure(figsize=(16,8))
    plt.plot( train_data, label='Train')
    plt.plot(test_data, label='Test')
    plt.plot(y_test, label=model)
    plt.legend(loc='best')
    return plt.show()

def newall(test_data,test_pred,model):
    rmse = np.sqrt(mean_squared_error(test_data,test_pred))
    rmse = round(rmse, 3)
    abs_error = np.abs(test_data-test_pred)
    actual = test_data
    mape = np.round(np.mean(abs_error/actual),3)
    resultsDf = pd.DataFrame({'Method':[model], 'MAPE': [mape], 'RMSE': [rmse]})
    resultsDf = resultsDf[['Method', 'RMSE', 'MAPE']]
    return resultsDf 

# =============================================================================
# Models 
# =============================================================================

def models(dfnew):
    
    # Linear regression
    time = [i+1 for i in range(len(dfnew))]
    df_linear = dfnew.copy()
    df_linear['time'] = time
    df_linear = df_linear[['time', 'Close']]
    train_data=df_linear.loc['2014-08-27':'2019-02-26']
    test_data=df_linear.loc['2019-02-27':'2019-08-27']
    x_train = train_data.drop('Close', axis=1)
    x_test = test_data.drop('Close', axis=1)
    y_train = train_data[['Close']]
    y_test = test_data[['Close']]
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    y_test['RegOnTime'] = predictions
    plltt(train_data['Close'],test_data['Close'],y_test['RegOnTime'],'Regression')
    x1=newall(test_data.Close, y_test.RegOnTime,'regression_on_time')

    # Naives method
    df_naive = dfnew[['Close']]
    df_naive['trend']=np.nan
    for i in range(len(df_naive)):
        if i == 0:
            df_naive['trend'] = np.nan
        else:
            df_naive['trend'][i] = df_naive.Close[i-1]
    train_data=df_naive.loc['2014-08-27':'2019-02-26']
    test_data=df_naive.loc['2019-02-27':'2019-08-27']
    plltt(train_data['Close'],test_data['Close'],test_data['trend'],'Naive method')
    x2=newall(test_data.Close, test_data.trend,'Naive method')
    
    # Simple Average Method

    df_sa=dfnew[['Close']]
    df_sa['trend']=np.nan
    for i in range(len(dfnew)):
        df_sa['trend'][i] = round(dfnew.Close.mean())
    train=df_sa.loc['2014-08-27':'2019-02-26']
    test=df_sa.loc['2019-02-27':'2019-08-27']
    plltt(train['Close'],test['Close'],test['trend'],'Simple Average Method')
    x3=newall(test.Close, test.trend,'Simple Average Method')
  
    # Moving Average method
    
    df_ma = dfnew[['Close']]
    df_ma['trend'] = df_ma['Close'].rolling(15).mean()
    train=df_ma.loc['2014-08-27':'2019-02-26']
    test=df_ma.loc['2019-02-27':'2019-08-27']
    plltt(train['Close'],test['Close'],test['trend'],'Moving Average Method')
    x4=newall(test.Close, test.trend,'Moving Average Method')
    
    # Simple exponential smoothing

    train=dfnew.loc['2014-08-27':'2019-02-26']
    test=dfnew.loc['2019-02-27':'2019-08-27']
    model = SimpleExpSmoothing(train['Close'])
    model_fit = model.fit(smoothing_level=0.6,optimized=False)
    test['trend'] = model_fit.forecast(len(test['Close']))
    plltt(train['Close'],test['Close'],test['trend'],'Simple exponential smoothing')
    x5=newall(test.Close, test.trend,'Simple exponential smoothing')

    # Holts method
    train=dfnew.loc['2014-08-27':'2019-02-26']
    test=dfnew.loc['2019-02-27':'2019-08-27']
    molts_model = ExponentialSmoothing(train[['Close']],trend='add', seasonal='none').fit()
    test['trend'] = molts_model.forecast(len(test))
    plltt(train['Close'],test['Close'],test['trend'],'holts method')
    x6=newall(test.Close, test.trend,'holts method')

    # Holts winter method - Additive
    train=dfnew.loc['2014-08-27':'2019-02-26']
    test=dfnew.loc['2019-02-27':'2019-08-27']
    holts_model = ExponentialSmoothing(train[['Close']],seasonal_periods=365 ,trend='add', seasonal='add').fit()
    test['trend'] = holts_model.forecast(len(test))
    plltt(train['Close'],test['Close'],test['trend'],'Holts winter method - Additive')
    x7=newall(test.Close, test.trend,'Holts winter method - Additive')

    # hotls winter method - Multiplicative
    train=dfnew.loc['2014-08-27':'2019-02-26']
    test=dfnew.loc['2019-02-27':'2019-08-27']
    holts_model = ExponentialSmoothing(train['Close'],seasonal_periods=365 ,trend='add', seasonal='mul').fit()
    test['trend'] = holts_model.forecast(len(test))
    plltt(train['Close'],test['Close'],test['trend'],'Holts winter method - Mutltiplicative')
    x8=newall(test.Close, test.trend,'Holts winter method - Mutltiplicative')
    
    # Results
    resultsDf = pd.concat([x1,x2,x3,x4,x5,x6,x7,x8])
    resultsDf=resultsDf.reset_index(drop=True)
    print(resultsDf)

models(timeseries_dfnew)


# Checking Auto Correlation - converting data in to Stationary
def Autocorr(timeseries_dfnew):
    print(plot_acf(timeseries_dfnew['Close'], lags=36))
    plot_pacf(timeseries_dfnew, ax=plt.gca(),lags=30)
    plt.figure(figsize=(16,8))
    pd.plotting.autocorrelation_plot(timeseries_dfnew['Close'])
    plt.show()

Autocorr(timeseries_dfnew)

# =============================================================================
# Log tranforming Data to make data Stationary ( DE TRENDING and de seasonalising )
# =============================================================================

timeseries_dfnew['log']=np.log(timeseries_dfnew['Close'])
timeseries_dfnew['Close_log_diff'] = timeseries_dfnew['log'] - timeseries_dfnew['log'].shift(1)
timeseries_dfnew['Close_log_diff'].dropna().plot()
plt.show()

def visualdecompeseddata(x):
   
    lag_acf = acf(x, nlags=15)
    lag_pacf = pacf(x, nlags=20, method='ols')
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(x)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(x)),linestyle='--',color='gray')
    plt.plot(lag_pacf)
    plt.title('Autocorrelation Function')
    plt.figure(figsize=(16,8))
    pd.plotting.autocorrelation_plot(timeseries_dfnew['Close_log_diff'])
    decomp= sm.tsa.seasonal_decompose(x,freq=365)
    decomp.plot()
timeseries_dfnew['Close_log_diff'].dropna(inplace=True)
visualdecompeseddata(timeseries_dfnew['Close_log_diff']) 

# Autoregression (AR) Method
timeseries_dfnew.dropna(inplace=True)
train=timeseries_dfnew[['Close_log_diff']].loc['2014-08-27':'2019-02-26']
test=timeseries_dfnew[['Close_log_diff']].loc['2019-02-27':'2019-08-27']

model = AR(train)
model_fit = model.fit()
print(f"Lag: {model_fit.k_ar}")
print(f"Coefficients: {model_fit.params}")
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
plltt(train['Close_log_diff'],test['Close_log_diff'],predictions,'Auto Regression model')
newall(test['Close_log_diff'],predictions,'AR model')


# ARIMA
p=d=q=range(0,5)
import itertools
val = list(itertools.product(p,d,q))

print("Combinations of p,d,p for ARIMA to get low AIC ")
for param in val:
    try:
        model_arima = ARIMA(test['Close_log_diff'],order = param)
        model_arima_fit = model_arima.fit()
        print(param,model_arima_fit.aic)
    
    except:
        continue

model = ARIMA(test['Close_log_diff'], order=(2,2,0), freq=test['Close_log_diff'].index.inferred_freq)  
results_ARIMA = model.fit(disp=-1)
plt.plot(test['Close_log_diff'])
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.show()
print(results_ARIMA.summary())


# AUTO Arima
stepwise_model = auto_arima(timeseries_dfnew['Close'], start_p=1, start_q=1,
                           max_p=5, max_q=5, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

predictions = stepwise_model.predict(n_periods=30)
future_dates = pd.date_range(start='28/08/2019', freq='D', periods=len(predictions))
print(stepwise_model.summary())


# stock Predictions for next 30 Days

future_forecast = pd.DataFrame(predictions,index=future_dates)
future_forecast.columns=['prediction']
data = pd.DataFrame(predictions,index=future_dates)
data.columns=['Close']
dfh=timeseries_dfnew['2019-01-01':'2019-08-27']
plt.figure(figsize=(15,7.5))
plt.title("SARIMA forecast")
plt.plot(dfh.Close)
plt.plot(data,color = 'red')

