import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

def fetch_stock_data(ticker="AAPL", period="3y"):
    """
    Fetch historical stock data using yfinance.
    """
    df = yf.download(ticker, period=period)
    return df

data = fetch_stock_data("AAPL","3y").iloc

prices = [1,3,6,8,4,3,7.7,8,9,10,3,3,2,6,7,8,9,2,4,12.12,8,4,5,7,4,1,8,9,3,5,8,3,5]

def upward_downward_run(arr):
    longest_up_run_count = 0 # longest up streak
    longest_down_run_count = 0 # longest down streak
    up_run_count = 0 # number of up streaks, even if run is 1 day only
    down_run_count = 0 # number of down streaks, even if run is 1 day only
    up_count = 0
    down_count = 0
    temp = 0 # temp data to compare with longest streak
    run_direction = "" # saves the previous run direction
    i = 1
    while i < len(arr[:,0]):
        if (arr[i,0] - arr[i-1,0]) > 0: # current up direction
            up_count += 1
            if run_direction == "up":   # same direction
                 None
            else:                       # direction switched down to up
                temp = 0
                up_run_count += 1
                run_direction = "up"

            temp += 1
            if longest_up_run_count < temp: # save temp to longest run count
                longest_up_run_count = temp


        elif (arr[i,0] - arr[i-1,0]) < 0: # current down direction
            down_count += 1
            if run_direction == "down":   
                 None  
            else:
                temp = 0
                down_run_count += 1
                run_direction = "down"

            temp += 1
            if longest_down_run_count < temp:
                longest_down_run_count = temp
        else:
            run_direction = ""  # resets direction if there is no difference
            
        i += 1
        #print (run_direction)
    print (f"longest up trend: {longest_up_run_count}")
    print (f"longest down trend: {longest_down_run_count}")
    print (f"bullish days: {up_count}")
    print (f"bearish days: {down_count}")
    print (f"up runs: {up_run_count}")
    print (f"down runs: {down_run_count}")

upward_downward_run(data)
# print(len(data[:,0]))
# print (data)
# print(f"date: {data.index[0]}")
# print(f"dates: {data.index}")
# print(f"iloc: {data.iloc}")
# print(f"{data.index[0]} $ {data.iloc[0,0]:.2f}")
# print(f"{data.index[1]} $ {data.iloc[1,0]:.2f}")





