import pandas as pd
import numpy as np

dframe = pd.DataFrame({'A': [1, 2, 3, 0], 'Column 2': ['b', 'g', '0', 'o']})     
df = np.full(len(dframe), np.nan)
print(df)

s = pd.Series([], name="X")

lst = [1,3,5,6,8]
s = pd.Series(lst, name = "up run")
print(type(s))
frames = [dframe, s]
print(pd.concat(frames, axis=1)) 
