import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import pandas as pd
from dataclasses import dataclass

@dataclass(frozen=True)
class Inputs():
    ticker: str
    duration: str
    sma_period: int

def collect_inputs():
    while True:
        ticker = input("Enter stock ticker symbol (e.g., AAPL, TSLA, MSFT): ").strip().upper()
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
        duration = input("Enter duration (e.g., 1mo, 3mo, 6mo, 1y, 2y, 3y): ").strip().lower()
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
        sma_period = input("Enter SMA period (e.g., 20, 50, 200): ")

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
def plot_stock_with_sma_and_trades(df, ticker, sma_period, transactions, closing_prices, total_profit):
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

    # Enable hover tooltips only on markers
    cursor = mplcursors.cursor([p[0] for p in buy_points + sell_points], hover=True)
    @cursor.connect("add")
    def on_hover(sel):
        for marker, text in buy_points + sell_points:
            if sel.artist == marker:
                sel.annotation.set_text(text)
                sel.annotation.get_bbox_patch().set_alpha(0.9)

    # Add legend entries
    ax.plot([], [], color="green", label="Uptrend")
    ax.plot([], [], color="red", label="Downtrend")
    ax.plot([], [], color="orange", label=f"SMA {sma_period}")
    ax.plot([], [], marker="^", color="green", label="Buy", linestyle="")
    ax.plot([], [], marker="v", color="red", label="Sell", linestyle="")
    ax.legend()

    # Labels and title
    ax.set_title(f"{ticker} Stock Price & {sma_period}-Day SMA with Trade Highlights")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")

    # Add total transactions and profit as text on the plot
    if total_profit is not None:
        total_txns = len(transactions)
        ax.text(0.02, 0.95, f"Total Transactions: {total_txns}\nTotal Max Profit: ${total_profit:.2f}", 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))




    fig.tight_layout()
    plt.show()



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


'''
formula used:
r_t = (P_t - P_{t-1}) / P_{t-1}
'''
#Extracting only closing prices from API data.
def daily_return(close_price, day_before_price):
    
    return (close_price - day_before_price) / day_before_price
    # if daily_returns is not None:
    #     print("No daily return data available")
    # else:
    #     return daily_returns.round(3)    


def upward_downward_run(arr):
    longest_up_run_count = 0 # longest up streak
    longest_down_run_count = 0 # longest down streak
    up_run_count = 0 # number of up streaks, even if run is 1 day only
    down_run_count = 0 # number of down streaks, even if run is 1 day only
    up_count = 0    # number of up days 
    down_count = 0  # number of down days 
    temp = 0 # temp data to compare with longest streak
    run_direction = "" # saves the previous run direction
    idx = 1   # index in arr
    try:
        while idx < len(arr):     
            if (daily_return(arr.iloc[idx], arr.iloc[idx-1])) > 0: # current up direction
                up_count += 1
                if run_direction == "up":   # same direction
                    None
                else:                       # direction switched down to up
                    temp = 0
                    up_run_count += 1
                    run_direction = "up"

                temp += 1
                if longest_up_run_count < temp: # save temp to longest run count
                    longest_up_run_count = temp


            elif (daily_return(arr.iloc[idx], arr.iloc[idx-1])) < 0: # current down direction
                down_count += 1
                if run_direction == "down":   
                    None  
                else:
                    temp = 0
                    down_run_count += 1
                    run_direction = "down"

                temp += 1
                if longest_down_run_count < temp:
                    longest_down_run_count = temp
            else:
                run_direction = ""  # resets direction if there is no difference
                
            idx += 1

        print("upward_downward_run run successfully")
        return [longest_up_run_count, longest_down_run_count, up_count, down_count, up_run_count, down_run_count]
    except TypeError:
        return "Error: Invalid input/Type Error for upward_downward_run"

def analysis_dataframe(df, closing_prices, transactions, sma_period, total_profit):
    """
    Create analysis dataframe 
    """
    #Create an empty dataframe indexed by trading dates
    panel = pd.DataFrame(index=closing_prices.index)

    # Pull SMA from existing dataframe(df)
    sma_col = f"SMA_{sma_period}"
    try:
        sma_series = df[sma_col]
    except KeyError:
        # If the SMA column isn't there, create an empty float Series (or raise)
        sma_series = pd.Series(index=panel.index, dtype=float)  # all NaN

    #Adding values into dataframe columns
    panel["Closing_Price"] = closing_prices.astype(float)        
    panel[f"SMA{sma_period}"] = sma_series.reindex(panel.index)
    panel["Daily_Return"] = closing_prices.pct_change().round(3)
    panel["Direction"] = np.where(panel["Daily_Return"].gt(0), "Up", np.where(panel["Daily_Return"].lt(0), "Down", "Flat"))
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
    #Loop through each buy_index and sell index
    for buy_idx, sell_idx in transactions:

        #Converting integer positions to date labels
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