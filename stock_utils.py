import yfinance as yf
import matplotlib
matplotlib.use('Agg')   # headless
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class Inputs():
    ticker: str
    duration: str
    sma_period: int

@dataclass(frozen=True)
class Runs():
    longest_up_streak: int
    longest_down_streak: int
    up_count: int
    down_count: int
    up_streaks: int
    down_streaks: int
    streaks_series: pd.Series

def collect_inputs(ticker, duration, sma_period):
    while True:
        ticker = ticker.strip().upper()
        if not ticker:
            print("Input cannot be empty. Please try again.")
            continue

        #Validate if ticker is real
        test = yf.Ticker(ticker).history(period = "1d")
        if test.empty:
            print(f"Ticker {ticker} is invalid. Please try again.")
            continue
        
        else:
            break
        
    while True:
        duration = duration.strip().lower()
        if not duration:
            print("Input cannot be empty. Please try again.")
            continue

        if duration not in ["1mo", "3mo", "6mo", "1y", "2y", "3y"]:
            print("Invalid duration! Defaulting to 3y.")
            duration = "3y"
            break

        else:
            break
    
    trading_days = { "1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "3y": 756,}
    while True:

        if not sma_period.isdigit():
            print("SMA period must be a number. Please try again.")
            continue
        
        sma_period = int(sma_period)

        if sma_period == 0:
            print("SMA period cannot be zero. Please try again.")
            continue
        
        if sma_period > trading_days[duration]:
            print(f"SMA period too large for {duration}. Defaulting to 20. ")
            sma_period = 20
            break
        break

    return Inputs(ticker, duration, sma_period)



def fetch_stock_data(ticker="AAPL", period="3y"):
    """
    Fetch historical stock data using yfinance.
    """
    df = yf.download(ticker, period=period, timeout=10, auto_adjust=True)   # time out so it wont hang/auto adjust for stock split for consistent value
    if df.empty:
        print("No data fetched. Default to AAPL")
        df = yf.download('AAPL', period=period)
        Inputs.ticker = 'AAPL'
    return df



def calculate_sma(df, period=20):
    """
    Calculate Simple Moving Average (SMA).
    Adds a new column 'SMA_<period>' to the DataFrame.
    """
    window = np.ones(period)/ period

    for ticker in df["Close"].columns:
        closeprices= df["Close"][ticker].tolist()
        SMA= np.convolve(closeprices, window, mode = "valid")
        SMA_array= np.full(len(closeprices), np.nan)
        SMA_array[period-1:] = SMA  
        df[(f"SMA_{period}",ticker)] = SMA_array

    return df



# Plot stock with SMA and buy/sell markers
def plot_stock_with_sma_and_trades(df, ticker, sma_period, transactions, closing_prices):

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract closing prices as a Series
    prices = closing_prices.values
    dates = df.index.values

    # Plot price line segments: green if up, red if down
    for i in range(1, len(prices)):
        color = "green" if prices[i] > prices[i - 1] else "red"
        ax.plot([dates[i - 1], dates[i]], [prices[i - 1], prices[i]], color=color, linewidth=1.5)

    # Plot SMA line in orange
    ax.plot(df.index, df[f"SMA_{sma_period}"], label=f"SMA {sma_period}", color="orange")

    # Plot buy/sell markers
    buy_points, sell_points = [], []
    for buy_idx, sell_idx in transactions:
        buy_date, sell_date = df.index[buy_idx], df.index[sell_idx]
        buy_price = float(closing_prices.iloc[buy_idx])
        sell_price = float(closing_prices.iloc[sell_idx])
        profit = sell_price - buy_price

        # Plot arrows for buy/sell points
        buy_marker, = ax.plot(buy_date, buy_price, marker="^", color="green", markersize=6, linestyle="")
        sell_marker, = ax.plot(sell_date, sell_price, marker="v", color="red", markersize=6, linestyle="")

        # Store marker and tooltip together
        buy_points.append((buy_marker, f"Buy on {buy_date.date()} @ ${buy_price:.2f}"))
        sell_points.append((sell_marker, f"Sell on {sell_date.date()} @ ${sell_price:.2f}\nProfit: ${profit:.2f}"))

    # Add legend entries
    ax.plot([], [], color="green", label="Uptrend")
    ax.plot([], [], color="red", label="Downtrend")
    ax.plot([], [], marker="^", color="green", label="Buy", linestyle="")
    ax.plot([], [], marker="v", color="red", label="Sell", linestyle="")
    ax.legend()

    # Labels and title
    ax.set_title(f"{ticker} Stock Price & {sma_period}-Day SMA with Trade Highlights")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")


    '''
    Generate plot as png, save in static and return filename
    '''

    # check if static path exists
    os.makedirs("static", exist_ok=True)
    
    # generate png name
    timestamp = datetime.now().strftime("%Y%m%d_%H%Mhr")
    fname = f"{ticker}_SMA{sma_period}_{timestamp}.png"
    
    # generate
    fig.tight_layout()

    # make png file save in static folder
    img_path = os.path.join("static", fname)
    fig.savefig(img_path, dpi=150)

    plt.close(fig)
    return fname
    


# Find max profit and transactions
def maxProfitWithTransactions(prices):
    """
    Find max profit and all corresponding transactions (buy, sell).
    Calculates Max profit from multiple buy/sell transactions.
    Assuming you can buy and sell multiple times, however you must sell before you buy again.
    """
    # Store total profit and all transactions
    total_profit = 0
    transactions = []

    # Go through all prices one by one
    current_day = 0
    total_days = len(prices)

    while current_day < total_days - 1:

        # Find the Buy point (minimum price)
        while current_day < total_days - 1 and prices.iat[current_day + 1] <= prices.iat[current_day]:
            current_day += 1

        # If we reached the end, stop — no more buying possible
        if current_day == total_days - 1:
            break

        buy = current_day # Mark the buy day
        current_day += 1

        # Find the Sell point (maximum price)
        while current_day < total_days and prices.iat[current_day] >= prices.iat[current_day - 1]:
            current_day += 1

        sell = current_day - 1

        # Calculate profit via daily returns
        profit_factor = 1
        for day in range(buy + 1, sell + 1):
            daily_r = daily_return(prices.iat[day], prices.iat[day - 1])
            profit_factor *= (1 + daily_r)
        profit = (profit_factor - 1) * prices.iat[buy]  
        total_profit += profit

        transactions.append((buy, sell))

    return total_profit, transactions



def close_data(df):
    try:
        df = df["Close"].squeeze()
        return df
    except AttributeError:
        return "Error: Attribute error in close_data"



#Extracting only closing prices from API data.
def daily_return(close_price, day_before_price):
    
    return (close_price - day_before_price) / day_before_price   



def upward_downward_run(close_price):
    up_streaks = 0 # number of up streaks 
    down_streaks = 0 # number of down streaks 

    direction_streaks = 0 # +ve means up streak, -ve means down streak, 0 means flat

    lst_streaks = [0] # list of streaks, +ve means up, -ve means down, start from 0 as nothing to compare with

    up_count = 0    # number of up days 
    down_count = 0  # number of down days 

    run_direction = "" # saves the previous run direction
    idx = 1   # index in series

    try:
        while idx < len(close_price):   

            # current up direction
            if (daily_return(close_price.iloc[idx], close_price.iloc[idx-1])) > 0: 
                up_count += 1
                direction_streaks += 1

                # direction switched to up
                if run_direction != "up":                       
                    up_streaks += 1
                    direction_streaks = 1
                    run_direction = "up"

                lst_streaks.append(direction_streaks)

            # current down direction
            elif (daily_return(close_price.iloc[idx], close_price.iloc[idx-1])) < 0: 
                down_count += 1
                direction_streaks -= 1

                # direction switched to down
                if run_direction != "down": 
                    down_streaks += 1
                    direction_streaks = -1
                    run_direction = "down"

                lst_streaks.append(direction_streaks)

            # flat direction if daily return = 0
            else:
                run_direction = "flat"
                lst_streaks.append(0)
                
            idx += 1
            
        streaks_series = pd.Series(lst_streaks, index = close_price.index, name = "streaks")
        
        longest_down_streak = min(lst_streaks)
        longest_up_streak = max(lst_streaks)

        return Runs(longest_up_streak, longest_down_streak, up_count, down_count, up_streaks, down_streaks, streaks_series)
    
    except TypeError:
        return "Error: Invalid input/Type Error for upward_downward_run"


# joins Series into a dataframe
def analysis_dataframe(df, closing_prices, transactions, sma_period, total_profit, streaks_series):
    """
    Create analysis dataframe 
    """
    # Create an empty dataframe indexed by trading dates
    panel = pd.DataFrame(index=closing_prices.index)

    # Pull SMA from existing dataframe(df)
    sma_col = f"SMA_{sma_period}"

    try:
        sma_series = df[sma_col]
        
    except KeyError:
        # If the SMA column isn't there, create an empty float Series (or raise)
        sma_series = pd.Series(index=panel.index, dtype=float)  # all NaN

    # Adding values into dataframe columns
    panel["Closing_Price"] = closing_prices.astype(float)        
    panel[f"SMA{sma_period}"] = sma_series.reindex(panel.index)
    panel["Daily_Return"] = closing_prices.pct_change().round(3)
    panel["Direction"] = np.where(panel["Daily_Return"].gt(0), "Up", np.where(panel["Daily_Return"].lt(0), "Down", "Flat"))
    panel["Streaks"] = streaks_series
    panel["Profit"] = np.nan  # float NaN so ffill works
    panel["Signal"] = pd.Series(index=panel.index, dtype="object")

    # If no profitable transactions is found (Fallback)
    if not transactions:
        panel["Profit"] = panel["Profit"].astype(float).ffill().fillna(0.0)
        print(panel)  
        print("\nNo profitable transactions were found for the given period.")
        print("Total transactions: 0")
        print("Total realized profit: $0.00")
        return panel

    realized = 0.0

    # Loop through each buy_index and sell index
    for buy_idx, sell_idx in transactions:

        # Converting integer positions to date labels
        buy_date  = closing_prices.index[buy_idx]
        sell_date = closing_prices.index[sell_idx]

        buy_px  = float(closing_prices.iloc[buy_idx])
        sell_px = float(closing_prices.iloc[sell_idx])

        # Mark Buy, Sell and Hold inside signal column
        panel.loc[buy_date,  "Signal"] = "Buy"
        between = (panel.index > buy_date) & (panel.index < sell_date)
        panel.loc[between & panel["Signal"].isna(), "Signal"] = "Hold"
        panel.loc[sell_date, "Signal"] = "Sell"

        # Update Profit ONLY when sell stock
        realized += (sell_px - buy_px)
        panel.loc[sell_date, "Profit"] = realized

    # Keep profit the same between sell and 0.0 before first sell
    panel.sort_index(inplace=True)
    panel["Profit"] = panel["Profit"].astype(float).ffill().fillna(0.0)

    print(panel)
    print(f"\nTotal transactions: {len(transactions)}")
    print(f"Total realized profit: ${total_profit:.2f}")
    return panel



def save_as_csv(df, ticker, duration):

    timestamp = datetime.now().strftime("%Y%m%d_%H%Mhr")
    filename = f"{ticker}_{duration}_{timestamp}_analysis.csv"
    df.to_csv(filename, index = True)
    print(f"Analysis_dataframe saved into {filename}")

