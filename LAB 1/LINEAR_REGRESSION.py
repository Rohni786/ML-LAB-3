import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wine_data = pd.read_csv("/Users/rohni./Documents/ML/LAB 1/WineQT.csv")
print("First 5 rows:")
print(wine_data.head())
print("\nShape of dataset:")
print(wine_data.shape)
print("\nMissing values:")
print(wine_data.isnull().sum())
X=wine_data['alcohol'].values
y=wine_data['quality'].values
split=int(0.8*len(X))
X_train=X[:split]
X_test=X[split:]
y_train=y[:split]
y_test=y[split:]
mean_x=np.mean(X_train)
mean_y=np.mean(y_train)
numerator=np.sum((X_train-mean_x)*(y_train-mean_y))
denominator=np.sum((X_train-mean_x)**2)
m=numerator/denominator
b=mean_y-m*mean_x
print("\nSlope (m):",m)
print("Intercept (b):",b)
y_pred=m*X_test+b
mse=np.mean((y_test-y_pred)**2)
mae=np.mean(np.abs(y_test-y_pred))
ss_total=np.sum((y_test-np.mean(y_test))**2)
ss_residual=np.sum((y_test-y_pred)**2)
r2=1-(ss_residual/ss_total)
print("\nR^2 Score:",r2)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.xlabel("Alcohol")
plt.ylabel("Wine Quality")
plt.title("Best Fit Line (From Scratch)")
plt.show()