import yfinance as yf
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
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
    """
    Compute a convolution-based SMA and add it to the DataFrame. Using a method involving 
    uniform window via np.convolve, fill exisitng dataframe with NaNs before sufficient data.

    Args:
        df:     Input price DataFrame (expects df['Close']).
        period: SMA window length.

    Returns:
        The input DataFrame with a flat SMA column named 'SMA_<period>'.
    """
    # Create the convolution window
    window = np.ones(period)/ period

    # Compute SMA for each ticker in df and add as new column
    for ticker in df["Close"].columns:
        closeprices= df["Close"][ticker].tolist()
        SMA= np.convolve(closeprices, window, mode = "valid")
        SMA_array= np.full(len(closeprices), np.nan)
        SMA_array[period-1:] = SMA  
        df[f"SMA_{period}"] = SMA_array
    return df


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
    daily_returns = closing_prices.pct_change() # daily returns as decimal
    avg_return = daily_returns.mean() * 100 # convert to percentage     
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
    # Initialize counters and state variables
    up_streaks = 0 
    down_streaks = 0 
    direction_streaks = 0 
    lst_streaks = [0] 
    up_count = 0    
    down_count = 0  
    run_direction = "" 
    idx = 1   

    # Iterate through closing prices to compute streaks
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

def build_plotly_chart(df, ticker, sma_period, transactions, closing_prices) -> str:
    """
    Build an interactive Plotly chart for html.

    Hover behavior:
      - Close line: shows Date + Close only
      - SMA line:   shows Date + Close + SMA
      - BUY/SELL markers: shows only their own info (label, Date, Close, SMA)


    Args:
        df:              DataFrame containing a flat 'SMA_<period>' column.
        ticker:          Ticker symbol to show in the title.
        sma_period:      Integer period used to compute the SMA (for labels).
        transactions:    List of (buy_idx, sell_idx) pairs (index positions in closing_prices).
        closing_prices:  Close price Series (DateTimeIndex), single ticker.

    Returns:
        HTML snippet (string) with the Plotly chart.
    """
    idx   = closing_prices.index
    close = closing_prices.astype(float)

    sma_key = f"SMA_{sma_period}"
    sma = df[sma_key].reindex(idx).astype(float) if sma_key in df.columns else pd.Series(index=idx, dtype=float)

    fig = go.Figure()

    # Close line — hover: Date + Close
    fig.add_trace(go.Scatter(
        x=idx, y=close.values, mode="lines", name="Close",
        hovertemplate=(
            "<b>Date</b>: %{x|%Y-%m-%d}<br>"
            "<b>Close</b>: $%{y:.2f}"
            "<extra></extra>"
        )
    ))

    # SMA line — hover: Date + Close + SMA
    fig.add_trace(go.Scatter(
        x=idx, y=sma.values, mode="lines", name=f"SMA {sma_period}",
        customdata=close.values,
        hovertemplate=(
            "<b>Date</b>: %{x|%Y-%m-%d}<br>"
            "<b>Close</b>: $%{customdata:.2f}<br>"
            "<b>SMA</b>: $%{y:.2f}"
            "<extra></extra>"
        )
    ))

    # Buy/Sell Markers — hover: Buy/Sell, Date, Close, SMA
    if transactions:
        buy_idx  = [i for i, _ in transactions]
        sell_idx = [j for _, j in transactions]

        buys_x   = idx[buy_idx]
        buys_y   = close.iloc[buy_idx].astype(float).values
        buys_sma = sma.iloc[buy_idx].values

        sells_x   = idx[sell_idx]
        sells_y   = close.iloc[sell_idx].astype(float).values
        sells_sma = sma.iloc[sell_idx].values

        fig.add_trace(go.Scatter(
            x=buys_x, y=buys_y, mode="markers", name="Buy",
            marker_symbol="triangle-up", marker_size=10, marker_line_width=1,
            marker_color="#22c55e",
            customdata=buys_sma,
            hovertemplate=(
                "<b>BUY</b><br>"
                "<b>Date</b>: %{x|%Y-%m-%d}<br>"
                "<b>Close</b>: $%{y:.2f}<br>"
                "<b>SMA</b>: $%{customdata:.2f}"
                "<extra></extra>"
            )
        ))
        fig.add_trace(go.Scatter(
            x=sells_x, y=sells_y, mode="markers", name="Sell",
            marker_symbol="triangle-down", marker_size=10, marker_line_width=1,
            marker_color="#ef4444",
            customdata=sells_sma,
            hovertemplate=(
                "<b>SELL</b><br>"
                "<b>Date</b>: %{x|%Y-%m-%d}<br>"
                "<b>Close</b>: $%{y:.2f}<br>"
                "<b>SMA</b>: $%{customdata:.2f}"
                "<extra></extra>"
            )
        ))

    # Styling graph and layout
    fig.update_layout(
        title=f"{ticker} — Close & SMA {sma_period}",
        xaxis_title="Date", yaxis_title="Price (USD)",
        hovermode="closest",
        height=720,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="#111827",  # panel
        plot_bgcolor="#1f2937",   # card
        font=dict(color="#e5e7eb", family="Inter, system-ui, -apple-system, Segoe UI, Roboto"),
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="#374151",
            font=dict(color="#e5e7eb", family="Inter, system-ui, -apple-system, Segoe UI, Roboto", size=13),
            namelength=-1,
        ),
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right", yanchor="bottom"),
        xaxis=dict(
            type="date",
            gridcolor="#374151", linecolor="#374151", zerolinecolor="#374151",
            tickfont=dict(color="#e5e7eb"), title_font=dict(color="#e5e7eb"),
            rangeslider=dict(visible=True, bgcolor="#1f2937", bordercolor="#374151")
        ),
        yaxis=dict(
            gridcolor="#374151", linecolor="#374151", zerolinecolor="#374151",
            tickfont=dict(color="#e5e7eb"), title_font=dict(color="#e5e7eb"),
        ),
    )

    return to_html(fig, full_html=False, include_plotlyjs="cdn")