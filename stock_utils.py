import yfinance as yf
import matplotlib
matplotlib.use('Agg')   # headless
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict, List


@dataclass(frozen=True)
class Inputs():
    """Validated user inputs used across the analysis pipeline.

    Attributes:
        ticker:   Normalized ticker symbol (uppercase).
        duration: Data duration string (e.g., '1mo', '3y').
        sma_period: Integer period for SMA calculation.
    """
    ticker: str
    duration: str
    sma_period: int

@dataclass(frozen=True)
class Runs():
    """Summary of up/down runs derived from closing prices.

    Attributes:
        longest_up_streak:   Max consecutive up days.
        longest_down_streak: Max consecutive down days.
        up_count:            Total up days.
        down_count:          Total down days.
        up_streaks:          Number of distinct up runs.
        down_streaks:        Number of distinct down runs.
        streaks_series:      Signed streak series (+ up, - down, 0 flat).
    """
    longest_up_streak: int
    longest_down_streak: int
    up_count: int
    down_count: int
    up_streaks: int
    down_streaks: int
    streaks_series: pd.Series

def validate_inputs(ticker_raw: Optional[str], duration_raw: Optional[str], sma_raw: Optional[str]) -> Tuple[Optional[Inputs], Dict[str, str]]:
    """Validate form inputs from Flask (single pass, no loops).

    Ensures ticker exists (using yfinance) and enforces the single rule that
    SMA period must be ≤ the number of trading days implied by the duration.

    Args:
        ticker_raw:   Raw ticker string from the form.
        duration_raw: Raw duration string from the form.
        sma_raw:      Raw SMA string from the form (dropdown -> int).

    Returns:
        A tuple of (Inputs|None, errors_dict). If errors is non-empty,
        Inputs will be None.
    """
    errors: Dict[str, str] = {}

    # Clean inputs
    ticker = (ticker_raw or "").strip().upper()
    duration = (duration_raw or "").strip().lower()
    sma = int((sma_raw or "0").strip())

    # Ticker
    if not ticker:
        errors["ticker"] = "Ticker is required."
    else:
        try:
            test = yf.Ticker(ticker).history(period="1d")
            if test.empty:
                errors["ticker"] = f"Ticker '{ticker}' not found."
        except Exception as e:
            errors["ticker"] = f"Ticker check failed: {e}"

    # SMA period
    trading_days = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "3y": 756}
    max = trading_days.get(duration)
    if sma > max:
        errors["sma"] = f"SMA period must be ≤ {max} for duration '{duration}'."

    # if errors, return None for Inputs and the error dict
    if errors:
        return None, errors

    return Inputs(ticker=ticker, duration=duration, sma_period=sma), {}



def fetch_stock_data(ticker, period)-> pd.DataFrame:
    """
    Fetch historical stock data using yfinance.

    Args:
        ticker:  Ticker symbol.
        period:  yfinance period (e.g., '1mo', '3y').

    Returns:
        A DataFrame indexed by date with OHLCV columns
    """
    df = yf.download(ticker, period=period, timeout=10, auto_adjust=True)   # time out so it wont hang/auto adjust for stock split for consistent value
    if df.empty:
        print("No data fetched. Default to AAPL")
        df = yf.download('AAPL', period=period)
        Inputs.ticker = 'AAPL'
    return df



def calculate_sma(df: pd.DataFrame, period: int)-> pd.DataFrame:
    """Compute a convolution-based SMA and add it to the DataFrame.

    Using a method involving uniform window via np.convolve, fill exisitng dataframe with NaNs before sufficient data.

    Args:
        df:     Input price DataFrame (expects df['Close']).
        period: SMA window length.

    Returns:
        The input DataFrame with a flat SMA column named 'SMA_<period>'.
    """

    window = np.ones(period)/ period

    for ticker in df["Close"].columns:
        closeprices= df["Close"][ticker].tolist()
        SMA= np.convolve(closeprices, window, mode = "valid")
        SMA_array= np.full(len(closeprices), np.nan)
        SMA_array[period-1:] = SMA  
        df[f"SMA_{period}"] = SMA_array
    return df



# Plot stock with SMA and buy/sell markers
def plot_stock_with_sma_and_trades(df: pd.DataFrame, ticker: str, sma_period: int, transactions: List[tuple[int, int]], closing_prices: pd.Series, static_dir: Optional[str] = None) -> str:
    """Render a colored-price plot with SMA and buy/sell markers; save to PNG.

    Lines are colored green/red for up/down segments between days. Buy/Sell
    markers are added using transaction indices. The image is saved as a
    timestamped PNG and the filename is returned.

    Args:
        df:              DataFrame containing price and SMA columns.
        ticker:          Ticker label for the title and filename.
        sma_period:      SMA window length used for the SMA column.
        transactions:    List of (buy_idx, sell_idx) pairs.
        closing_prices:  Series of close prices (single ticker).
        static_dir:      Directory to save the PNG; defaults to '<module>/static'.

    Returns:
        The PNG filename.
    """

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

    # check if static_dir exists, if not create it
    if static_dir is None:
        # default to a static folder next to this file as a fallback
        static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)

    fname = f"{ticker}_SMA{sma_period}_{datetime.now().strftime('%Y%m%d_%H%Mhr')}.png"
    img_path = os.path.join(static_dir, fname)
    fig.savefig(img_path, dpi=150)
    plt.close(fig)
    return fname  # return just the name; template uses url_for('static', filename=...)


# Find max profit and transactions
def maxProfitWithTransactions(prices: pd.Series) -> tuple[float, List[tuple[int, int]], pd.Series]:
    """Compute max total profit with multiple buy/sell pairs + realized P&L series.

    Logic:
      1) Scan for local minima (buy) and subsequent local maxima (sell).
      2) Profit for each trade is computed from chained daily returns.
      3) Return (total_profit, transactions, pnl_series), where pnl_series
         updates only at sell dates and is forward-filled between sells.

    Args:
        prices: Series of daily close prices.

    Returns:
        total_profit: Sum of profits across all trades.
        transactions: List of (buy_idx, sell_idx) index pairs.
        pnl:          Realized cumulative profit Series aligned to `prices.index`.
    """
    # Store total profit and all transactions (buy_index, sell_index) in a list and pnl Series
    total_profit = 0
    transactions = []
    pnl = pd.Series(np.nan, index=prices.index, name="Profit")  
    realized = 0.0                                             

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
        realized += profit                                     
        pnl.iloc[sell] = realized  

        transactions.append((buy, sell))

    # Fill forward the pnl Series to have cumulative profit on sell days, 0.0 before first sell, NaN otherwise
    pnl = pnl.astype(float).ffill().fillna(0.0)

    return total_profit, transactions, pnl

def signal_data (closing_prices: pd.Series, transactions: List[tuple[int, int]]) -> pd.Series:
    """Create a 'Signal' Series with values: 'Buy', 'Hold', 'Sell', 'Wait'.

    'Wait' is the default on all dates. For each (buy_idx, sell_idx) pair:
      - Buy date -> 'Buy'
      - Sell date -> 'Sell'
      - Dates strictly between -> 'Hold'

    Args:
        closing_prices: Price series (index used for dates).
        transactions:   List of (buy_idx, sell_idx) pairs.

    Returns:
        A Series named 'Signal' with categorical trade-state strings.
    """
    signal = pd.Series("Wait", index=closing_prices.index, dtype="object", name="Signal")
    transactions = transactions or []
    for buy_idx, sell_idx in transactions:
        buy_date  = closing_prices.index[buy_idx]
        sell_date = closing_prices.index[sell_idx]

        signal.loc[buy_date] = "Buy"
        signal.loc[sell_date] = "Sell"

        mid = (signal.index > buy_date) & (signal.index < sell_date)
        signal.loc[mid] = "Hold"
        
    return signal

def close_data(df: pd.DataFrame) -> pd.Series:
    """Extract a single-ticker close Series from the yfinance DataFrame.

    Returns:
        A pandas Series of closing prices.
    """
    try:
        df = df["Close"].squeeze()
        return df
    except AttributeError:
        return "Error: Attribute error in close_data"


#Extracting only closing prices from API data.
def daily_return(close_price: float, day_before_price: float) -> float:
    """Compute simple daily return from two price points.

    Args:
        close_price:      Today's closing price.
        day_before_price: Previous day's closing price.

    Returns:
        (close_price - day_before_price) / day_before_price
    """
    return (close_price - day_before_price) / day_before_price


def average_daily_return_pct(closing_prices: pd.Series) -> float:
    """
    Compute the average daily return in percentage from a Series of closing prices.
    
    Args:
        closing_prices: Series of closing prices.
    
    Returns:
        Average daily return as a percentage (float rounded to 2 decimals)
    """
    daily_returns = closing_prices.pct_change()  # daily returns as fraction
    avg_return = daily_returns.mean() * 100      # convert to %
    return round(avg_return, 2)


def upward_downward_run(close_price: pd.Series) -> Runs:
    """Derive up/down run statistics and streak series from close prices.

    The streak series is signed: positive counts for consecutive ups,
    negative counts for consecutive downs, and 0 for flats.

    Args:
        close_price: Single-ticker close price Series.

    Returns:
        A `Runs` dataclass containing run counts, longest streaks, and the
        streaks_series aligned to `close_price.index`.
    """
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
def analysis_dataframe(
    df: pd.DataFrame,
    closing_prices: pd.Series,
    sma_period: int,
    streaks_series: pd.Series,
    signals: pd.Series,
    profit: pd.Series,
    transactions: Optional[List[Tuple[int, int]]] = None,
    total_profit: Optional[float] = None,
) -> pd.DataFrame:
    
    """
    Assemble the analysis DataFrame (no heavy logic) and optionally print:
      - the DataFrame
      - total number of transactions
      - total realized profit

    Inputs:
      df              : original df that already contains SMA_<period> (flat) or not
      closing_prices  : single-ticker close Series (Date index)
      streaks_series  : Series of streak metric  (+ve up, -ve down, 0 flat)
      signals         : 'Signal' Series (Buy/Hold/Sell/Wait) 
      profit          : realized Profit Series (updates only on sell dates) 
      transactions    : list of (buy_idx, sell_idx) pairs, for counting
      total_profit    : optional precomputed total; falls back to profit.iloc[-1]

    Returns:
      panel DataFrame with columns:
        Closing_Price, SMA<period>, Daily_Return, Direction, Streaks, Signal, Profit
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
    panel[f"SMA_{sma_period}"] = sma_series.reindex(panel.index)
    panel["Daily_Return"] = closing_prices.pct_change().round(3)
    panel["Direction"] = np.where(panel["Daily_Return"].gt(0), "Up", np.where(panel["Daily_Return"].lt(0), "Down", "Flat"))
    panel["Streaks"] = streaks_series.reindex(panel.index)
    panel["Signal"] = signals.reindex(panel.index)

    if profit is None:
        panel["Profit"] = pd.Series(0.0, index=panel.index, dtype=float)
    else:
        p = profit.reindex(panel.index).astype(float)
        panel["Profit"] = p.ffill().fillna(0.0)
    
    print(panel)
    
    if len(transactions) == 0:
        print("\nNo profitable transactions were found for the given period.")
    
    # If total_profit wasn’t provided, fall back to the last Profit value
    if total_profit is None:
        realized_profit = float(panel["Profit"].iloc[-1]) if not panel["Profit"].empty else 0.0
    else:
        realized_profit = float(total_profit)

    print(f"Total Transactions: {len(transactions)}")
    print(f"Total realized profit: ${realized_profit:.2f}")
    return panel


def save_as_csv(df: pd.DataFrame, ticker: str, duration: str, data_dir: str | None = None) -> tuple[str, str]:
    """
    Save the analysis DataFrame to a CSV file in a project's 'data' directory.

    The function ensures the target directory exists, writes a timestamped CSV,
    and returns both the absolute file path and the bare filename.

    Args:
        df:      Analysis DataFrame to persist.
        ticker:  Stock ticker used in the analysis (e.g., 'AAPL').
        duration:Data duration (e.g., '3y').
        data_dir:Optional absolute directory to save into. If None, defaults to
                 a 'data' folder located alongside this module file.

    Returns:
        A tuple of:
            - absolute_path: Full filesystem path to the saved CSV.
            - filename:      The CSV filename (no directory), useful for links/logs.

    Side Effects:
        Creates the 'data' directory if it does not already exist and writes the CSV.

    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%Mhr")
    filename = f"{ticker}_{duration}_{timestamp}_analysis.csv"
    absolute_path = os.path.join(data_dir, filename)

    df.to_csv(absolute_path, index=True, encoding="utf-8")
    print(f"Analysis DataFrame saved to: {absolute_path}")
    return absolute_path, filename