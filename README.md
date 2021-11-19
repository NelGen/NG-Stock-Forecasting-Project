# Stock Forecasting with Time Series Models


## Business Case:

Finding stocks to invest in by using time series models to forecast their growth rates over 60 days.


## Sources:

5 years of historical stock prices downloaded from Yahoo! Finance.

Stocks analyzed include:
    1. MongoDB (MDB)
    2. American Airlines (AAL)
    3. Amazon (AMZN)
    4. General Electric (GE)
    5. Intel (INTC)
    6. Tesla (TSLA)
    

## Strategies

Fit ARIMA, Auto-ARIMA and Facebook Prophet models to the stock being analyzed.  The model with the lowest RMSE on the test data was
then fit on the entire stock price data set to forecast the next 60 days. Growth rates were recorded for each.

To begin, several helper/sub functions were created to easily replicate the modeling process on any stock price data.
These functions include:
* Preprocessing functions to log transform and/or prepare the data for fitting each model.
* Train Test Split function to divide the data specifically for time series. By default, the first 75% of the data is used to train.
* Functions to find starting p,d,q values for the base ARIMA model order.  
    - For finding the initial d value, the data was differenced and measured for significance using Dickey-Fuller test.
    - For finding the initial p value, analyzed the PACF graphs using the differenced data.
    - For finding the initial q value, analyzed the ACF graphs using the differenced data as well.
* ARIMA, Auto-ARIMA and Facebook Prophet modeling functions with functionality to plot, summarize, return RMSE or growth rate


    
## Predictions

![](Images/Table.png)
![](Images/Barplot.png)

Out of the 6 stocks analyzed, MongoDB and Tesla have the highest expected growth rates of approximately 17% each, however,
with a low degree of confidence.

## Improvements and Possible Next Steps

Introducing other models such as neural network and testing performance.  Including other variables related to a company's income and other financial statements should help obtain useful insights to the price direction.

For next steps, automating the process on a large number of stocks and narrowing in-depth analysis on the most promising stocks.