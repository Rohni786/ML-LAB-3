import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("advertising.csv")
TV = data['TV'].values
Radio = data['Radio'].values
Newspaper = data['Newspaper'].values
y = data['Sales'].values
TV = (TV-np.mean(TV)) / np.std(TV)
Radio = (Radio-np.mean(Radio)) / np.std(Radio)
Newspaper = (Newspaper - np.mean(Newspaper)) / np.std(Newspaper)
m = len(y)
b0 = 0
b1 = 0
b2 = 0
b3 = 0
alpha = 0.01  
epochs = 1000
for i in range(epochs):
    y_pred = b0 + b1*TV + b2*Radio + b3*Newspaper
    db0 = (1/m) * np.sum(y_pred - y)
    db1 = (1/m) * np.sum((y_pred - y) * TV)
    db2 = (1/m) * np.sum((y_pred - y) * Radio)
    db3 = (1/m) * np.sum((y_pred - y) * Newspaper)
    b0 = b0 - alpha * db0
    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2
    b3 = b3 - alpha * db3
print("Intercept:", b0)
print("TV coef:", b1)
print("Radio coef:", b2)
print("Newspaper coef:", b3)
y_pred = b0 + b1*TV + b2*Radio + b3*Newspaper
plt.scatter(y, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()