from dataclasses import dataclass

def collect_inputs():
    # use label to find "close" instead of hard code column number
    # collect 
    @dataclass(frozen=True)
    class Inputs():
        ticker: str
        duration: str
        sma_period: int

    ticker = input("Enter stock ticker symbol (e.g., AAPL, TSLA, MSFT): ").upper()
    duration = input("Enter duration (e.g., 1mo, 3mo, 6mo, 1y, 2y, 3y): ").lower()
    sma_period = input("Enter SMA period (e.g., 20, 50, 200): ")
    return Inputs(ticker,duration,sma_period)

def validation_inputs(duration, sma_period):

    if duration not in ["1mo", "3mo", "6mo", "1y", "2y", "3y"]:
        print("Invalid duration! Defaulting to 3y.")
        duration = "3y"
    if sma_period.isdigit():
        sma_period = int(sma_period)
    else:
        print("Invalid SMA period! Defaulting to 20.")
        sma_period = 20
    return 

Inputs = collect_inputs()
validation_inputs(Inputs.duration, Inputs.sma_period)
