from stock_utils import fetch_stock_data, calculate_sma, plot_stock_with_sma_and_trades, maxProfitWithTransactions


def main():
    """Main function: SMA plot + Max Profit Analysis together."""
    print("Welcome to the Stock Analyzer! (SMA + Max Profit)")

    # --- Inputs ---
    ticker = input("Enter stock ticker symbol (e.g., AAPL, TSLA, MSFT): ").upper()
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

    # --- Fetch Data ---
    df = fetch_stock_data(ticker=ticker, period=duration)
    if df.empty:
        print("No data fetched. Please check the ticker or duration.")
        return

    # --- Add SMA ---
    df = calculate_sma(df, period=sma_period)

    # --- Max Profit Analysis ---
    closing_prices = df["Close"].squeeze()
    total_profit, transactions = maxProfitWithTransactions(closing_prices)

    # --- Print results ---
    if not transactions:
        print("\nA profitable trading strategy was not found for the given period.")
        print("This could be because the stock price only went down.")
    else:
        print(f"\n--- Max Profit Analysis for {ticker} over {duration} ---")
        print(f"{'Buy Date':<12} {'Buy Price':<12} {'Sell Date':<12} {'Sell Price':<12} {'Profit':<12}")
        print("-" * 65)
        for buy_index, sell_index in transactions:
            buy_date = str(closing_prices.index[buy_index].date())
            buy_price = float(closing_prices.iloc[buy_index])
            sell_date = str(closing_prices.index[sell_index].date())
            sell_price = float(closing_prices.iloc[sell_index])
            transaction_profit = sell_price - buy_price
            print(f"{buy_date:<12} ${buy_price:<11.2f} {sell_date:<12} ${sell_price:<11.2f} ${transaction_profit:<11.2f}")
        print("-" * 65)
        print(f"Total Transactions: {len(transactions)}")
        print(f"Total Maximum Profit: ${total_profit:.2f}")

    # --- Plot SMA + trades ---
    plot_stock_with_sma_and_trades(df, ticker, sma_period, transactions)


if __name__ == "__main__":
    main()
