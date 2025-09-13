import yfinance as yf
import matplotlib.pyplot as plt
import mplcursors
import pandas

def fetch_stock_data(ticker="AAPL", period="3y"):
    """
    Fetch historical stock data using yfinance.
    """
    df = yf.download(ticker, period=period)
    return df

def calculate_sma(df, period=20):
    """
    Calculate Simple Moving Average (SMA).
    Adds a new column 'SMA_<period>' to the DataFrame.
    """
    df[f"SMA_{period}"] = df["Close"].rolling(window=period).mean()
    return df

def plot_stock_with_sma_and_trades(df, ticker, sma_period, transactions):
    """Plot SMA + closing price with MplCursor for buy/sell points."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Close"], label="Closing Price", alpha=0.8)
    plt.plot(df.index, df[f"SMA_{sma_period}"], label=f"SMA {sma_period}", color="orange")

    buy_points = []
    sell_points = []

    for buy_idx, sell_idx in transactions:
        buy_date = df.index[buy_idx]
        sell_date = df.index[sell_idx]
        buy_price = float(df["Close"].iloc[buy_idx])
        sell_price = float(df["Close"].iloc[sell_idx])
        profit = sell_price - buy_price

        # Plot buy/sell markers
        buy_marker, = plt.plot(buy_date, buy_price, marker="^", color="green", markersize=10, linestyle="")
        sell_marker, = plt.plot(sell_date, sell_price, marker="v", color="red", markersize=10, linestyle="")

        buy_points.append((buy_marker, f"Buy on {buy_date.date()}\n${buy_price:.2f}"))
        sell_points.append((sell_marker, f"Sell on {sell_date.date()}\n${sell_price:.2f}\nProfit: ${profit:.2f}"))

    # Make hover tooltips
    cursor = mplcursors.cursor([p[0] for p in buy_points + sell_points], hover=True)

    @cursor.connect("add")
    def on_hover(sel):
        # sel.index gives the point index
        for marker, text in buy_points + sell_points:
            if sel.artist == marker:
                sel.annotation.set_text(text)
                sel.annotation.get_bbox_patch().set_alpha(0.9)

    # Clean legend (only once)
    plt.plot([], [], marker="^", color="green", label="Buy", linestyle="")
    plt.plot([], [], marker="v", color="red", label="Sell", linestyle="")
    plt.legend()

    plt.title(f"{ticker} Stock Price & {sma_period}-Day SMA with Buy/Sell Points")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.show()


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
        # Using .iat[] for explicit single value access to avoid ambiguity
        while i < n - 1 and prices.iat[i + 1] <= prices.iat[i]:
            i += 1
        if i == n - 1:
            break
        buy = i
        i += 1

        # Find local maximum (sell point)
        # Using .iat[] for explicit single value access to avoid ambiguity
        while i < n and prices.iat[i] >= prices.iat[i - 1]:
            i += 1
        sell = i - 1

        profit += prices.iat[sell] - prices.iat[buy]
        transactions.append((buy, sell))
    
    return profit, transactions
    
def upward_downward_run(arr):
    longest_up_run_count = 0 # longest up streak
    longest_down_run_count = 0 # longest down streak
    up_run_count = 0 # number of up streaks, even if run is 1 day only
    down_run_count = 0 # number of down streaks, even if run is 1 day only
    up_count = 0
    down_count = 0

    temp = 0 # temp data to compare with longest streak
    run_direction = "" # saves the previous run direction
    i = 1
    try:
        arr = arr.iloc # iloc instead of loc because it uses index to get items.
        while i < len(arr[:,0]):        #iloc[:,0] means calling entire column 0. ignore the rows
            if (arr[i,0] - arr[i-1,0]) > 0: # current up direction
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


            elif (arr[i,0] - arr[i-1,0]) < 0: # current down direction
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
                
            i += 1
            # print (run_direction)
        print (f"longest up trend: {longest_up_run_count}")
        print (f"longest down trend: {longest_down_run_count}")
        print (f"bullish days: {up_count}")
        print (f"bearish days: {down_count}")
        print (f"up runs: {up_run_count}")
        print (f"down runs: {down_run_count}")
        return [longest_up_run_count, longest_down_run_count, up_count, down_count, up_run_count, down_run_count]
    except TypeError:
        return print("Invalid input.")
