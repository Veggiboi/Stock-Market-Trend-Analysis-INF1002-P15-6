import os, datetime, secrets
from stock_utils import fetch_stock_data, calculate_sma, maxProfitWithTransactions,upward_downward_run,close_data, validate_inputs, analysis_dataframe, signal_data, average_daily_return_pct, build_plotly_chart
from flask import Flask, request, abort, make_response, flash, render_template

analysis_cache = {}
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-change-me')  # use env var in prod

@app.get("/download/csv")
def download_csv():
    """On button click: write the CSV to memory and stream it to user."""
    key = request.args.get("key", "")
    if not key or key not in analysis_cache:
        abort(400, description="No analysis available. Run analysis first.")

    payload  = analysis_cache[key]
    df       = payload["df"]
    ticker   = payload["ticker"]
    duration = payload["duration"]

    # Build CSV in memory and stream it
    csv_text = df.to_csv(index=True, encoding="utf-8")
    return_name = f"{ticker}_{duration}_{datetime.datetime.now().strftime('%Y%m%d_%H%Mhr')}_analysis.csv"

    resp = make_response(csv_text)
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = f'attachment; filename="{return_name}"'
    return resp
    
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


        # Flash success message
        flash(f"Analysis completed for {Inputs.ticker} ({Inputs.duration}, SMA {Inputs.sma_period}).", "success")

        # Average daily return in percentage
        daily_return_avg_pct = average_daily_return_pct(closing_prices)

        # Interactive chart HTML
        plot_html = build_plotly_chart(df_with_sma, Inputs.ticker, Inputs.sma_period, transactions, closing_prices)

        # Save CSV to disk and get filename
        key = secrets.token_urlsafe(16)
        analysis_cache[key] = {"df": output_df, "ticker": Inputs.ticker,"duration": Inputs.duration}

        return render_template('main.html', 
                                plot_html = plot_html,
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
                                analysis_key = key,
                                )
    
                
    # None when first generate the html
    return render_template('main.html', 

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