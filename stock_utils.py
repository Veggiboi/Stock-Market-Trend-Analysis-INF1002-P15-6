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
