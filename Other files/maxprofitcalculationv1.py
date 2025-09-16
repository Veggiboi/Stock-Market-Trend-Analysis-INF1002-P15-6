import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime


def maxProfitWithTransactions(prices):
    """
    Find max profit and all corresponding transactions (buy, sell).

    This function finds all buy and sell points where the price is at a local
    minimum followed by a local maximum. This strategy assumes an unlimited
    number of transactions.
    """
    n = len(prices)
    profit = 0
    transactions = []
    
    i = 0
    while i < n - 1:
        # Find local minimum (buy point)
        # Using .iat[] for explicit single value access to avoid ambiguity
        while i < n - 1 and prices.iat[i + 1] <= prices.iat[i]:
            i += 1
        if i == n - 1:
            break
        buy = i
        i += 1

        # Find local maximum (sell point)
        # Using .iat[] for explicit single value access to avoid ambiguity
        while i < n and prices.iat[i] >= prices.iat[i - 1]:
            i += 1
        sell = i - 1

        profit += prices.iat[sell] - prices.iat[buy]
        transactions.append((buy, sell))
    
    return profit, transactions


