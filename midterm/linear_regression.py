# importing panadas, numpy and matplotlib libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing scikit-learn models, metrics and scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# loading csv file
df = pd.read_csv("home_garden_iot_data.csv")

# Data Cleaning
# dropping unnamed columns and needs_watering column
columns_to_drop = [col for col in df.columns if 'Unnamed' in col or col == 'needs_watering']
df.drop(columns=columns_to_drop, inplace=True)

# drops empty columns
df.dropna(inplace=True)

# target and feature columns
target_column = 'soil_moisture'
feature_columns = df.columns.drop(target_column)

# print dataset shape and target column name
print(f"\nDataset shape: {df.shape}")
print(f"Target variable for prediction: '{target_column}'")

# identifying x and y variables (x = features y = target)
x = df[feature_columns]
y = df[target_column]

# setting train / test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print("-" * 50)

# normalizing data
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(f"X_train_scaled now has shape: {x_train_scaled.shape}")
print("-" * 50)

# creating linear regression model
model = LinearRegression()
model.fit(x_train_scaled, y_train)

# predictions
y_pred = model.predict(x_test_scaled)

# evaluation of model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# display evaluation and interpretation
print("Model Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print("-" * 50)

# Interpretation: A lower RMSE is better.
print(f"The RMSE of {rmse:.2f} means that, on average, the model's prediction for")
print(f"soil moisture is off by approximately {rmse:.2f} units from the actual value.")


