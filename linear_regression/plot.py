import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("Auto.data", index_col=False)
data.replace({"?": np.nan}, inplace=True)
data.dropna()
mpg = data['mpg']
horsepower = data['horsepower']
plt.scatter(horsepower, mpg, color='b')
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.show()