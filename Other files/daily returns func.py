import yfinance as yf
import pandas as pd

ticker = input("Enter stock ticker : ").upper()
start_date = input("Enter start date YYYY-MM-DD : ")
end_date = input("Enter end date YYYY-MM-DD : ")

def compute_daily_returns(ticker, start_date, end_date):
    
    '''
    Getting stock price data using yfinance api and compute the daily returns manually using formula
    
    close price data

    formula used:
        r_t = (P_t - P_{t-1}) / P_{t-1}
    
    '''
    
    '''
    variables:
        ticker (str): e.g 'NVDA'
        start_data (str): e.g '2025-01-01'
        end_data (str): e.g '2025-01-30'



    '''

    #Retrieving data from yfinance
    data = yf.download(ticker=ticker, start=start_date, end=end_date)

    #To see the general data retrieved from yfinance
    print(data)  


    #Ensure data is correct.
    if data.empty:
        print("No data found from given range or ticker ")
        
    #Extracting only closing prices from API data.
    close_prices = data['Close']

    #Computing of daily returns using formula
    #possible changing to * 100 to get percentage

    daily_returns = (close_prices - close_prices.shift(1)) / close_prices.shift(1)


    return(daily_returns)

daily_returns = compute_daily_returns(ticker, start_date, end_date)
print(daily_returns)