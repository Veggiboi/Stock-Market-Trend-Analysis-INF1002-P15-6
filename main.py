from stock_utils import fetch_stock_data, calculate_sma, plot_stock_with_sma_and_trades, maxProfitWithTransactions,upward_downward_run,close_data, collect_inputs, analysis_dataframe
from datetime import datetime

def main():
    
    print("Welcome to Stock Analyzer!")

    # Input/Validate Input
    Inputs = collect_inputs()

    # Fetch Stock Data
    df = fetch_stock_data(ticker = Inputs.ticker, period = Inputs.duration)

    # Fetch closing price
    closing_prices = close_data(df)

    # Analyze upward/downward trends 
    print(upward_downward_run(closing_prices))

    # Adding SMA 
    df = calculate_sma(df, period = Inputs.sma_period)

    # --- Max Profit Analysis ---
    total_profit, transactions = maxProfitWithTransactions(closing_prices)

    # Create analysis dataframe
    analysis_df = analysis_dataframe(df, closing_prices, transactions, Inputs.sma_period, total_profit)

    # Plot chart with SMA, buy/sell markers, and colored lines
    plot_stock_with_sma_and_trades(df, Inputs.ticker, Inputs.sma_period, transactions, closing_prices, total_profit)

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%Mhr")
    filename = f"{Inputs.ticker}_{Inputs.duration}_{timestamp}_analysis.csv"
    analysis_df.to_csv(filename, index=True)
    print(f"Analysis_dataframe saved into {filename}")


if __name__ == "__main__":
    main()
    

