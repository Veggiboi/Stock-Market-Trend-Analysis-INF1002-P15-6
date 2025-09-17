import yfinance as yf

def fetch_stock_data(ticker="AAPL", period="3y"):

    return yf.download(ticker, period=period, timeout=10, auto_adjust=True)
    # try:
    #     df = yf.download(ticker, period=period, timeout=10, auto_adjust=True)   # time out so it wont hang/auto adjust for stock split for consistent value
    #     if df.empty:
    #         print("No data fetched. Please check the ticker. Default to AAPL")
    #         df = yf.download(ticker = 'AAPL', period=period)
    #         return
        
    # except Exception as e:
    #     print(f"Error fetching data {e}\nDefault ticker=AAPL")
    #     fetch_stock_data(ticker="AAPL", period=period)


fetch_stock_data("asdfghjk","3y")