import numpy as np
import matplotlib.pyplot as plt

def generate_prices(start_price=100, days=30, mean_return=0.001, std_dev=0.02):
    """Generate synthetic stock prices from random returns."""
    returns = np.random.normal(mean_return, std_dev, days)
    prices = [start_price]
    for r in returns:
        prices.append(float(prices[-1] * (1 + r)))
    return prices

def maxProfitWithTransactions(prices):
    """Find max profit and transactions (buy, sell)."""
    n = len(prices)
    profit = 0
    transactions = []
    
    i = 0
    while i < n - 1:
        # find local minimum (buy)
        while i < n - 1 and prices[i + 1] <= prices[i]:
            i += 1
        if i == n - 1:
            break
        buy = i
        i += 1

        # find local maximum (sell)
        while i < n and prices[i] >= prices[i - 1]:
            i += 1
        sell = i - 1

        profit += prices[sell] - prices[buy]
        transactions.append((buy, sell))
    
    return profit, transactions

# Generate prices
prices = generate_prices(start_price=100, days=20)
profit, transactions = maxProfitWithTransactions(prices)

# Print results
print("Generated Prices:", [round(p, 2) for p in prices])
print("Max Profit:", round(profit, 2))
for buy, sell in transactions:
    print(f"Buy on day {buy} at {prices[buy]:.2f}, Sell on day {sell} at {prices[sell]:.2f}")

# Plot chart
plt.figure(figsize=(10, 5))
plt.plot(prices, label="Stock Price", linewidth=2)

# Mark buys and sells
for buy, sell in transactions:
    plt.scatter(buy, prices[buy], color="green", marker="^", s=100, label="Buy" if buy == transactions[0][0] else "")
    plt.scatter(sell, prices[sell], color="red", marker="v", s=100, label="Sell" if sell == transactions[0][1] else "")

plt.title("Randomly Generated Stock Prices with Buy/Sell Signals")
plt.xlabel("Day")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()