import yfinance as yf
import matplotlib.pyplot as plt
import mplcursors
from dataclasses import dataclass



def collect_inputs():
    @dataclass(frozen=True)
    class Inputs():
        ticker: str
        duration: str
        sma_period: int

    ticker = input("Enter stock ticker symbol (e.g., AAPL, TSLA, MSFT): ").upper()
    if yf.Lookup(ticker, timeout = 10).stock.empty:
        ticker = "AAPL" 
        print("Invalid ticker selection, default to AAPL")
    
    duration = input("Enter duration (e.g., 1mo, 3mo, 6mo, 1y, 2y, 3y): ").lower()
    if duration not in ["1mo", "3mo", "6mo", "1y", "2y", "3y"]:
        print("Invalid duration! Defaulting to 3y.")
        duration = "3y"

    sma_period = input("Enter SMA period (e.g., 20, 50, 200): ")
    if sma_period.isdigit():
        sma_period = int(sma_period)
    else:
        print("Invalid SMA period! Defaulting to 20.")
        sma_period = 20

    return Inputs(ticker, duration, sma_period)



def fetch_stock_data(ticker="AAPL", period="3y"):
    """
    Fetch historical stock data using yfinance.
    """
    df = yf.download(ticker, period=period, timeout=10, auto_adjust=True)   # time out so it wont hang/auto adjust for stock split for consistent value
    if df.empty:
        print("No data fetched. Please check the ticker. Default to AAPL")
        df = yf.download(ticker, period=period)
        return
    return df



def calculate_sma(df, period=20):
    """
    Calculate Simple Moving Average (SMA).
    Adds a new column 'SMA_<period>' to the DataFrame.
    """
    df[f"SMA_{period}"] = df["Close"].rolling(window=period).mean()
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
    fig.tight_layout()
    plt.show()



# Find max profit and transactions
def maxProfitWithTransactions(prices):
    """
    Find max profit and all corresponding transactions (buy, sell).

    This function finds all buy and sell points where the price is at a local
    minimum followed by a local maximum. This strategy assumes an unlimited
    number of transactions.
    """
    n = len(prices)
    profit = 0
    transactions = []
    i = 0
    while i < n - 1:
        # Find local minimum (buy point)
        while i < n - 1 and prices.iat[i + 1] <= prices.iat[i]:
            i += 1
        if i == n - 1:
            break
        buy = i
        i += 1
        # Find local maximum (sell point)
        while i < n and prices.iat[i] >= prices.iat[i - 1]:
            i += 1
        sell = i - 1
        # Add profit and store transaction
        profit += prices.iat[sell] - prices.iat[buy]
        transactions.append((buy, sell))
    return profit, transactions



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
