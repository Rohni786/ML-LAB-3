import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("advertising.csv")

data['Target'] = data['Sales'].apply(lambda x: 1 if x > 10 else 0)
X = data.drop(['Sales', 'Target'], axis=1).values
y = data['Target'].values
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
n_samples, n_features = X.shape
W = np.zeros(n_features)   # weights vector
b = 0                      # bias

learning_rate = 0.01
epochs = 1000
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
for _ in range(epochs):
    z = np.dot(X, W) + b
    y_pred = sigmoid(z)
    dW = (1/n_samples) * np.dot(X.T, (y_pred - y))
    db = (1/n_samples) * np.sum(y_pred - y)
    W -= learning_rate * dW
    b -= learning_rate * db
z = np.dot(X, W) + b
y_pred = sigmoid(z)
y_pred_class = (y_pred > 0.5).astype(int)
accuracy = np.mean(y_pred_class == y)
print("Accuracy:", accuracy)