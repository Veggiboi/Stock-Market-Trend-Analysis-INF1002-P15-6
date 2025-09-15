import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()             # Create a figure containing a single Axes.
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the Axes.
ax.plot([10, 20, 30, 40], [1, 4, 2, 3])  # Plot some data on the Axes.
# ax.plot(df.index, closes)

cell_text = [
    ["Longest up",   2],
    ["Longest down", "3"],
    ["Bullish days", 4],
    ["Bearish days", 5],
    ["Up runs",      6],
    ["Down runs",    7],
]
# bbox = [x0, y0, width, height] in axes coords; x0>1 puts it outside on the right
tbl = ax.table(cellText=cell_text,
               cellLoc="left", colLoc="left",
               bbox=[1.0, -0.30, 0.1, 0.2])  # adjust position/size
plt.show()   