from stock_utils import fetch_stock_data, calculate_sma, plot_stock_with_sma_and_trades, maxProfitWithTransactions,upward_downward_run,close_data, collect_inputs, analysis_dataframe, save_as_csv


def main():
    
    print("Welcome to Stock Analyzer!")

    # Input/Validate Input
    Inputs = collect_inputs()

    # Fetch Stock Data
    df = fetch_stock_data(ticker = Inputs.ticker, period = Inputs.duration)

    # Fetch closing price
    closing_prices = close_data(df)

    # Analyze upward/downward trends 
    Runs = upward_downward_run(closing_prices)

    # Adding SMA 
    df = calculate_sma(df, period = Inputs.sma_period)

    # --- Max Profit Analysis ---
    total_profit, transactions = maxProfitWithTransactions(closing_prices)

    # Create analysis dataframe
    output_df = analysis_dataframe(df, closing_prices, transactions, Inputs.sma_period, total_profit, Runs.streaks_series)

    # Plot chart with SMA, buy/sell markers, and colored lines
    plot_stock_with_sma_and_trades(df, Inputs.ticker, Inputs.sma_period, transactions, closing_prices, total_profit)

    # Save to CSV
    save_as_csv(output_df, Inputs.ticker, Inputs.duration)



if __name__ == "__main__":
    main()