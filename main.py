from stock_utils import fetch_stock_data, calculate_sma, plot_stock_with_sma_and_trades, maxProfitWithTransactions,upward_downward_run,close_data, collect_inputs

def main():
    
    print("Welcome to Stock Analyzer!")

    # Input/Validate Input
    Inputs = collect_inputs()

    # Fetch Stock Data
    df = fetch_stock_data(ticker = Inputs.ticker, period = Inputs.duration)
    closing_prices = close_data(df)

    # Analyze upward/downward trends 
    upward_downward_run(closing_prices)

    # Adding SMA 
    df = calculate_sma(df, period = Inputs.sma_period)

    # Max Profit Analysis 
    total_profit, transactions = maxProfitWithTransactions(closing_prices)

        # Print prompt if no profit is found
    if not transactions:
        print("\nA profitable trading strategy was not found for the given period.")
        print("This could be because the stock price only went down.")
    else:
        # Print table for buy/sell trades and profits
        print(f"\n--- Max Profit Analysis for {Inputs.ticker} stock over {Inputs.duration} ---")
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

    # Plot chart with SMA, buy/sell markers, and colored lines
    plot_stock_with_sma_and_trades(df, Inputs.ticker, Inputs.sma_period, transactions, closing_prices)


if __name__ == "__main__":
    main()
    

