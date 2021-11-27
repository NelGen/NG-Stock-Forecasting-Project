# Stock Price Forecasting with Time Series Models


## Business/Use Case:

Streamlining stock price forecasting to build a portfolio of potentially high growth rate stocks.  Price will be forecasted over 60 days and percentage growth recorded for each.

The models trained and functions developed will be tuned specifically for stock data.  These functions will help produce results for an expanded selection of stocks.


## Sources:

5 years of historical stock prices downloaded from Yahoo! Finance.  Only the Adjusted Closing Price will be used with Volume being the exogenous variable.  Using other prices such as Opening Price for the day could introduce unwanted collinearity since the opening price for the next day will be equal to the closing price of the prior.

Stock analysis examples include:
    1. MongoDB (MDB)
        - MongoDB experienced significant growth over the last 3 years with the continued rise in the value of data. Could the models catch large spikes in growth?
    2. American Airlines (AAL)
        - The airline industry has suffered due to recent events.  Want to analyze what the next few months may look like for one major airline.
    3. Amazon (AMZN)
        - One of the most popular big tech companies which should be a safe bet to invest in.
    4. General Electric (GE)
        - GE has been on a steady decline over the years.  Using this stock as an example of potentially negative or no growth.
    5. Intel (INTC)
        - Intel has seen a lot of competition in the tech market as well.  Looking at how it may continue to be affected.
    6. Tesla (TSLA)
        - A very popular stock experiencing significant growth.  Would be interesting to analyze other data sources such as social media  and review its impact on this stock's price.
    

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

**Table of RMSE's by Model**
![](https://github.com/NelGen/NG-Stock-Forecasting-Project/blob/main/Images/Table.PNG)
![](https://github.com/NelGen/NG-Stock-Forecasting-Project/blob/main/Images/Barplot.PNG)

Out of the 6 stocks analyzed, MongoDB and Tesla have the highest expected growth rates of approximately 17% each, however,
with a low degree of confidence.

## Improvements and Possible Next Steps

Introducing other models such as neural network and testing performance.  Including other variables related to a company's income and other financial statements should help obtain useful insights to the price direction.

For next steps, automating the process on a large number of stocks and narrowing in-depth analysis on the most promising stocks.
