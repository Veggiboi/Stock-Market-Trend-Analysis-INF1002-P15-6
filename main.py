from stock_utils import fetch_stock_data, calculate_sma, plot_stock_with_sma

def main():
    # Ask user for stock ticker
    ticker = input("Enter stock ticker symbol (e.g., AAPL, TSLA, MSFT): ").upper()

    # Ask user for duration (max 3 years)
    duration = input("Enter duration (e.g., 1mo, 3mo, 6mo, 1y, 2y, 3y - max 3 years): ").lower()
    if duration not in ["1mo", "3mo", "6mo", "1y", "2y", "3y"]:
        print("⚠️ Invalid duration! Defaulting to 3y.")
        duration = "3y"

    # Ask user for SMA period (any number allowed, fallback = 20)
    sma_period = input("Enter SMA period (e.g., 20, 50, 200): ")
    if sma_period.isdigit():
        sma_period = int(sma_period)
    else:
        print("⚠️ Invalid SMA period! Defaulting to 20.")
        sma_period = 20

    # Step 1: Fetch stock data
    df = fetch_stock_data(ticker=ticker, period=duration)

    if df.empty:
        print("⚠️ No data fetched. Please check the ticker or duration.")
        return

    # Step 2: Calculate SMA
    df = calculate_sma(df, period=sma_period)

    # Step 3: Plot results
    plot_stock_with_sma(df, ticker=ticker, period=sma_period)

if __name__ == "__main__":
    main()
