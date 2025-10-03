import os, time
from stock_utils import fetch_stock_data, calculate_sma, plot_stock_with_sma_and_trades, maxProfitWithTransactions,upward_downward_run,close_data, validate_inputs, analysis_dataframe, save_as_csv, signal_data, average_daily_return_pct
from flask import Flask, request, flash, render_template, redirect, url_for, send_from_directory

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-change-me')  # use env var in prod
@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

         # Validate once; no loops
        Inputs, errors = validate_inputs(
            request.form.get("ticker"),
            request.form.get("duration"),
            request.form.get("sma"),
        )

        # If invalid, flash messages and re-render form with prior values
        if errors:
            for _, msg in errors.items():
                flash(msg, "error")
            return render_template(
                "main.html",
                img_name=None,
                ticker=request.form.get("ticker"),
                duration=request.form.get("duration"),
                sma=request.form.get("sma"),
                longest_up_streak=None,
                longest_down_streak=None,
                up_count=None,
                down_count=None,
                up_run_count=None,
                down_run_count=None,
                max_profit=None,
            )
        
        # --- Pipeline ---
        # Fetch Stock Data
        df = fetch_stock_data(ticker = Inputs.ticker, period = Inputs.duration)

        # Fetch closing price
        closing_prices = close_data(df)

        # Analyze upward/downward trends 
        Runs = upward_downward_run(closing_prices)

        # Adding SMA 
        df_with_sma = calculate_sma(df, period = Inputs.sma_period)

        # Max Profit Analysis
        total_profit, transactions, profit_data = maxProfitWithTransactions(closing_prices)

        # Generate buy/sell signals
        signals = signal_data(closing_prices, transactions)

        # Create analysis dataframe
        output_df = analysis_dataframe(df=df_with_sma, closing_prices=closing_prices, sma_period=Inputs.sma_period, streaks_series=Runs.streaks_series, signals=signals, profit=profit_data, transactions=transactions, total_profit=total_profit)

        # Plot chart with SMA, buy/sell markers, and colored lines
        img_name = plot_stock_with_sma_and_trades(df_with_sma, Inputs.ticker, Inputs.sma_period, transactions, closing_prices, static_dir=app.static_folder)

        # Flash success message
        flash(f"Analysis completed for {Inputs.ticker} ({Inputs.duration}, SMA {Inputs.sma_period}).", "success")

        # Average daily return in percentage
        daily_return_avg_pct = average_daily_return_pct(closing_prices)


        # Save to CSV (static folder)
        absolute_path, csv_name = save_as_csv(output_df, Inputs.ticker, Inputs.duration, data_dir=app.static_folder)


        return render_template('main.html', 
                                img_name = img_name,
                                cache_bust=str(int(time.time())), # prevent browser caching of image

                                ticker = Inputs.ticker, 
                                duration = Inputs.duration, 
                                sma = Inputs.sma_period, 

                                longest_up_streak = Runs.longest_up_streak,
                                longest_down_streak = Runs.longest_down_streak,
                                up_count = Runs.up_count,
                                down_count = Runs.down_count,
                                up_run_count = Runs.up_streaks,
                                down_run_count = Runs.down_streaks,
                                max_profit = f"{total_profit:.2f}",
                                daily_return_avg_pct = daily_return_avg_pct,
                                csv_name=csv_name
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