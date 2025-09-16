import yfinance as yf
import pandas as pd
from stock_utils import fetch_stock_data



# --- Inputs ---
ticker = input("Enter stock ticker symbol (e.g., AAPL, TSLA, MSFT): ").upper()
duration = input("Enter duration (e.g., 1mo, 3mo, 6mo, 1y, 2y, 3y): ").lower()
if duration not in ["1mo", "3mo", "6mo", "1y", "2y", "3y"]:
    print("Invalid duration! Defaulting to 3y.")
    duration = "3y"

# --- Fetch Data ---
df = fetch_stock_data(ticker=ticker, period=duration)
if df.empty:
    print("No data fetched. Please check the ticker or duration.")
    exit()



def daily_return(df):
    '''
    Getting stock price data using yfinance api and compute the daily returns manually using formula
    
    close price data

    formula used:
        r_t = (P_t - P_{t-1}) / P_{t-1}
    '''
    
    #if data not found
    if df.empty:
        print("No data found from given range or ticker ")
        return None
    
    #Extracting only closing prices from API data.
    close_prices = df['Close']
    
    #Computing of daily returns using formula
    daily_returns = (close_prices - close_prices.shift(1)) / close_prices.shift(1)
    return daily_returns

daily_returns = daily_return(df)
if daily_returns is not None:
    print(daily_returns)