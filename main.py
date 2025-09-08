import yfinance as yf
dat = yf.Ticker("MSFT").history(period="3y", interval="1d")
print(dat[0])
