import os, datetime, secrets
from stock_utils import (
    fetch_stock_data,
    calculate_sma,
    maxProfitWithTransactions,
    upward_downward_run,
    close_data,
    validate_inputs,
    analysis_dataframe,
    signal_data,
    average_daily_return_pct,
    build_plotly_chart,
)
from flask import Flask, request, abort, make_response, flash, render_template, send_file, url_for
import io
import zipfile

analysis_cache = {}
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-change-me')  # use env var in prod

@app.get("/download/csv")
def download_csv():
    """On button click: write the CSV to memory and stream it to user."""
    key = request.args.get("key", "")
    if not key or key not in analysis_cache:
        abort(400, description="No analysis available. Run analysis first.")

    # Find requested ticker if provided
    payload = analysis_cache[key]
    desired_ticker = request.args.get("ticker")

    # If payload contains multiple entries
    if "entries" in payload:
        entries = payload.get("entries") or []
        if desired_ticker:
            entry = next((e for e in entries if e["ticker"].upper() == desired_ticker.upper()), None)
            if entry is None:
                abort(404, description="Requested ticker not found in this analysis bundle.")
        else:
            # default to first
            entry = entries[0]

        df = entry["df"]
        ticker = entry["ticker"]
        duration = entry.get("duration", "")
    else:
        # backwards compatibility: single-entry payload
        df = payload.get("df")
        ticker = payload.get("ticker", "data")
        duration = payload.get("duration", "")

    csv_bytes = df.to_csv(index=True, encoding="utf-8").encode("utf-8")
    return_name = f"{ticker}_{duration}_{datetime.datetime.now().strftime('%Y%m%d_%H%Mhr')}_analysis.csv"

    return send_file(
        io.BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name=return_name,
    )


@app.get("/download/zip")
def download_all_zip():
    """Download all analyses stored in the cache as a ZIP of CSV files."""
    key = request.args.get("key", "")
    if not key or key not in analysis_cache:
        abort(400, description="No analysis available. Run analysis first.")

    payload = analysis_cache[key]
    # payload for multi-ticker will contain entries: payload['entries'] -> list of {ticker, duration, df}
    entries = payload.get("entries") or []
    if not entries:
        abort(400, description="No entries to zip.")

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for entry in entries:
            csv_bytes = entry["df"].to_csv(index=True, encoding="utf-8").encode("utf-8")
            filename = f"{entry['ticker']}_{entry['duration']}_{datetime.datetime.now().strftime('%Y%m%d_%H%Mhr')}.csv"
            zf.writestr(filename, csv_bytes)

    mem.seek(0)
    return send_file(mem, mimetype="application/zip", as_attachment=True, download_name=f"analyses_{datetime.datetime.now().strftime('%Y%m%d_%H%Mhr')}.zip")
    
@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

         # Validate once; no loops
        # Allow comma-separated multiple tickers
        raw_tickers = request.form.get("ticker") or ""
        # normalize and split
        tickers_list = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]

        # For validation reuse validate_inputs but one-by-one; collect errors per ticker
        duration_input = request.form.get("duration")
        sma_input = request.form.get("sma")

        # Validate common inputs (duration and sma) by calling validate_inputs with first ticker if exists
        # We will still check existence of each ticker later via yfinance in stock_utils.fetch_stock_data
        Inputs, errors = validate_inputs(tickers_list[0] if tickers_list else "", duration_input, sma_input)

        # If initial validation failed, surface errors

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
        # --- Pipeline for multiple tickers ---
        entries = []
        plots = []
        summaries = []
        # track combined metrics only in aggregate form for summary (first ticker shown)
        first_runs = None
        first_total_profit = None
        first_daily_return = None

        for t in tickers_list:
            # fetch data
            df = fetch_stock_data(ticker=t, period=Inputs.duration)
            closing_prices = close_data(df)
            Runs = upward_downward_run(closing_prices)
            df_with_sma = calculate_sma(df, period=Inputs.sma_period)
            total_profit, transactions, profit_data = maxProfitWithTransactions(closing_prices)
            signals = signal_data(closing_prices, transactions)
            output_df = analysis_dataframe(df=df_with_sma, closing_prices=closing_prices, sma_period=Inputs.sma_period, streaks_series=Runs.streaks_series, signals=signals, profit=profit_data, transactions=transactions, total_profit=total_profit)

            # Build per-ticker interactive chart with desired colors (handled in build_plotly_chart)
            plot_html = build_plotly_chart(df_with_sma, t, Inputs.sma_period, transactions, closing_prices)

            entries.append({"ticker": t, "duration": Inputs.duration, "df": output_df})
            plots.append({"ticker": t, "plot": plot_html})

            # per-ticker summary
            summaries.append({
                "ticker": t,
                "longest_up_streak": Runs.longest_up_streak,
                "longest_down_streak": Runs.longest_down_streak,
                "up_count": Runs.up_count,
                "down_count": Runs.down_count,
                "up_run_count": Runs.up_streaks,
                "down_run_count": Runs.down_streaks,
                "max_profit": f"{total_profit:.2f}",
                "avg_daily_return": average_daily_return_pct(closing_prices),
            })

            # keep first for summary metrics in UI
            if first_runs is None:
                first_runs = Runs
                first_total_profit = total_profit
                first_daily_return = average_daily_return_pct(closing_prices)

        # Save in cache under one key as entries
        key = secrets.token_urlsafe(16)
        analysis_cache[key] = {"entries": entries}

        flash(f"Analysis completed for {', '.join(tickers_list)} ({Inputs.duration}, SMA {Inputs.sma_period}).", "success")

        # Render template with multiple plots and per-ticker download links
    return render_template('main.html', 
                plots = plots,
                summaries = summaries,
                ticker = ",".join(tickers_list), 
                duration = Inputs.duration, 
                sma = Inputs.sma_period, 

                longest_up_streak = first_runs.longest_up_streak if first_runs else None,
                longest_down_streak = first_runs.longest_down_streak if first_runs else None,
                up_count = first_runs.up_count if first_runs else None,
                down_count = first_runs.down_count if first_runs else None,
                up_run_count = first_runs.up_streaks if first_runs else None,
                down_run_count = first_runs.down_streaks if first_runs else None,
                max_profit = f"{first_total_profit:.2f}" if first_total_profit is not None else None,
                daily_return_avg_pct = first_daily_return,
                analysis_key = key,
                )
if __name__ == "__main__":
    app.run(debug=True)