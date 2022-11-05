import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#loading the data
df = pd.read_csv("Spo2.csv")
#Visualizing the data
plt.xlabel('Ratio')
plt.ylabel('Spo2')
plt.scatter(df.Ratio, df.Spo2, color='red',marker='+' )

# Fitting the model to the data
reg = LinearRegression()
reg.fit(df[['Ratio']],df.Spo2)

# Predict new data
reg.predict([[0.46]])
# the y intercept
print(reg.intercept_)

#the slope
print(reg.coef_)