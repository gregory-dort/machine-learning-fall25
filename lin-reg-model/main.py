# Importing SciKit Learn Library and Wine Dataset, MatPlotLib and NumPy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Loading Wine Dataset
wine = load_wine()

# Feature selection and target selection
# feature selection from column 1 of data set to track malic acid data
# target selection from column 0 to predict alcohol content of wine
x = wine.data[:, 12].reshape(-1, 1)
y = wine.data[:, 0]

# Print feature and target data and sample numbers
print(f"Feature: {wine.feature_names[12]} | Target: {wine.feature_names[0]}")
print(f"Total samples: {x.shape[0]}")

# Splitting data into training and testing sets
# Training at 80% and Testing at 20% of data values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Creating and training Linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Testing Data Prediction
y_pred = model.predict(x_test)

# Model Evaluation
# Print Model Coefficient (m) and Intercept (b)
print("\n--- Model Parameters ---")
print(f"Coefficient (m): {model.coef_[0]:.4f}")
print(f"Intercept (b): {model.intercept_:.4f}")

# Print Evaluation Metrics
print("\n--- Evaluation Metrics ---")
# Calculate MSE and Covariance of the dataset and print the values
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
print(f"Covariance Score: {r2_score(y_test, y_pred):.4f}")

# Visualization using MatPlotLib
plt.figure(figsize=(10, 6))
# Test Data Points Plotted on Scatter Plot
plt.scatter(x_test, y_test, color='blue', label='Test Data Points')
# Red Regression Line Created
plt.plot(x_test, y_pred, color='red', linewidth=3, label='Regression Line')
plt.title(f'Linear Regression: {wine.feature_names[12]} vs. {wine.feature_names[0]}')
plt.xlabel(wine.feature_names[12].capitalize())
plt.ylabel(wine.feature_names[0].capitalize())
plt.legend()
plt.grid(True)
plt.show() 