import yfinance as yf

ticker = input("Enter stock ticker symbol (e.g., AAPL, TSLA, MSFT): ").upper()
x = yf.Lookup(ticker, timeout=1)
print(x.stock)
print(x.stock.empty)
# print(type(x.stock))
# print(x.stock.columns)