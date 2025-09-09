import yfinance as yf
import matplotlib.pyplot as plt

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

def plot_stock_with_sma(df, ticker="AAPL", period=20):
    """
    Plot closing price with SMA overlay.
    """
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["Close"], label="Closing Price", alpha=0.8)
    plt.plot(df.index, df[f"SMA_{period}"], label=f"SMA {period}", color="orange")
    plt.title(f"{ticker} Stock Price & {period}-Day SMA")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()
    
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
    while i < len(arr):
        if (arr[i] - arr[i-1]) > 0: # current up direction
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


        elif (arr[i] - arr[i-1]) < 0: # current down direction
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
        #print (run_direction)
    print (f"longest up trend: {longest_up_run_count}")
    print (f"longest down trend: {longest_down_run_count}")
    print (f"bullish days: {up_count}")
    print (f"bearish days: {down_count}")
    print (f"up runs: {up_run_count}")
    print (f"down runs: {down_run_count}")