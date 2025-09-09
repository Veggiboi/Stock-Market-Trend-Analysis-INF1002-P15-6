from stock_utils import fetch_stock_data, calculate_sma, plot_stock_with_sma
from maxprofitcalculationv1 import maxProfitWithTransactions

def main():
    """Main function to provide a menu for different stock analyses."""
    
    print("Welcome to the Stock Analyzer!")
    print("1. Plot stock price with Simple Moving Average (SMA)")
    print("2. Find the optimal buy and sell timing for max profit (multiple transactions)")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        # --- Plotting with SMA ---
        ticker = input("Enter stock ticker symbol (e.g., AAPL, TSLA, MSFT): ").upper()
        duration = input("Enter duration (e.g., 1mo, 3mo, 6mo, 1y, 2y, 3y - max 3 years): ").lower()
        if duration not in ["1mo", "3mo", "6mo", "1y", "2y", "3y"]:
            print("⚠️ Invalid duration! Defaulting to 3y.")
            duration = "3y"
        
        sma_period = input("Enter SMA period (e.g., 20, 50, 200): ")
        if sma_period.isdigit():
            sma_period = int(sma_period)
        else:
            print("⚠️ Invalid SMA period! Defaulting to 20.")
            sma_period = 20

        df = fetch_stock_data(ticker=ticker, period=duration)
        if df.empty:
            print("⚠️ No data fetched. Please check the ticker or duration.")
            return

        df = calculate_sma(df, period=sma_period)
        plot_stock_with_sma(df, ticker=ticker, period=sma_period)

    elif choice == "2":
        # --- Finding Max Profit with Multiple Transactions ---
        ticker = input("Enter stock ticker symbol (e.g., AAPL, TSLA, MSFT): ").upper()
        duration = input("Enter duration (e.g., 1mo, 3mo, 6mo, 1y, 2y, 3y - max 3 years): ").lower()
        if duration not in ["1mo", "3mo", "6mo", "1y", "2y", "3y"]:
            print("⚠️ Invalid duration! Defaulting to 3y.")
            duration = "3y"
            
        df = fetch_stock_data(ticker=ticker, period=duration)
        if df.empty:
            print("⚠️ No data fetched. Please check the ticker or duration.")
            return

        # Explicitly get the "Close" column as a Series
        closing_prices = df["Close"].squeeze()
        
        total_profit, transactions = maxProfitWithTransactions(closing_prices)
        
        if not transactions:
            print("\n😔 A profitable trading strategy was not found for the given period.")
            print("This could be because the stock price only went down.")
        else:
            print(f"\n--- Max Profit Analysis for {ticker} over {duration} ---")
            print(f"Total maximum possible profit: ${total_profit:.2f}\n")
            print("Optimal Transactions:")
            for buy_index, sell_index in transactions:
                buy_date = closing_prices.index[buy_index].date()
                buy_price = closing_prices.iat[buy_index]
                sell_date = closing_prices.index[sell_index].date()
                sell_price = closing_prices.iat[sell_index]
                transaction_profit = sell_price - buy_price
                print(f"  Buy: {buy_date} at ${buy_price:.2f} | Sell: {sell_date} at ${sell_price:.2f} (Profit: ${transaction_profit:.2f})")
            print("-" * 65)
            
    else:
        print("Invalid choice. Please run the script again and select 1 or 2.")

if __name__ == "__main__":
    main()
