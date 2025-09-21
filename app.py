from stock_utils import fetch_stock_data, calculate_sma, plot_stock_with_sma_and_trades, maxProfitWithTransactions,upward_downward_run,close_data, collect_inputs, analysis_dataframe, save_as_csv
from flask import Flask, request, flash, render_template

app = Flask(__name__)
@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        # Validate Input (needs updating)
        Inputs = collect_inputs(request.form.get('ticker'), request.form.get('duration'), request.form.get('sma'))

        # --- Pipeline ---
        # Fetch Stock Data
        df = fetch_stock_data(ticker = Inputs.ticker, period = Inputs.duration)

        # Fetch closing price
        closing_prices = close_data(df)

        # Analyze upward/downward trends 
        Runs = upward_downward_run(closing_prices)

        # Adding SMA 
        df = calculate_sma(df, period = Inputs.sma_period)

        # Max Profit Analysis
        total_profit, transactions = maxProfitWithTransactions(closing_prices)

        # Create analysis dataframe
        output_df = analysis_dataframe(df, closing_prices, transactions, Inputs.sma_period, total_profit, Runs.streaks_series)

        # Plot chart with SMA, buy/sell markers, and colored lines
        img_name = plot_stock_with_sma_and_trades(df, Inputs.ticker, Inputs.sma_period, transactions, closing_prices)

        # WIP: button to save as csv in static folder 

        # # Save to CSV
        # save_as_csv(output_df, Inputs.ticker, Inputs.duration)

        return render_template('main.html', 
                                img_name = img_name,

                                ticker = Inputs.ticker, 
                                duration = Inputs.duration, 
                                sma = Inputs.sma_period, 

                                longest_up_streak = Runs.longest_up_streak,
                                longest_down_streak = Runs.longest_down_streak,
                                up_count = Runs.up_count,
                                down_count = Runs.down_count,
                                up_run_count = Runs.up_streaks,
                                down_run_count = Runs.down_streaks,
                                max_profit = total_profit
                                )
                
    # None when first generate the html
    return render_template('main.html', 
                           img_name = None,

                           ticker = None, 
                           duration = None, 
                           sma = None, 

                           longest_up_streak = None,
                           longest_down_streak = None,
                           up_count = None,
                           down_count = None,
                           up_run_count = None,
                           down_run_count = None,
                           max_profit = None
                           )

if __name__ == "__main__":
    app.run(debug=True)