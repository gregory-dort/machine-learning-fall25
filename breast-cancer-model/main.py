import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

# importing train test split and logistic regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# importing evaluation metrics and confusion matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Loading breast cancer dataset
data = load_breast_cancer()

# X: Features (Inputs)
X = pd.DataFrame(data.data, columns=data.feature_names)
# y: Target (Output)
y = pd.Series(data.target)

print("--- Dataset Shape and Statistics ---")
print(f"Shape of X (Samples, Features): {X.shape}")
print(f"Shape of y (Target): {y.shape}")
print("\nBasic Statistics:")
print(X.describe().T)

# Report how many samples are malignant vs. benign
target_counts = y.value_counts()
malignant_count = target_counts.get(0, 0)
benign_count = target_counts.get(1, 0)

print("\n--- Target Counts ---")
print(f"Malignant (Class 0): {malignant_count} samples")
print(f"Benign (Class 1): {benign_count} samples")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(solver='liblinear', max_iter=200)
model.fit(X_train, y_train)

# Use the trained model to make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Model Performance Metrics ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Calculate Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n--- Confusion Matrix ---")
print("                       Predicted Malignant (0)  Predicted Benign (1)")
print(f"Actual Malignant (0):     {conf_matrix[0, 0]}                     {conf_matrix[0, 1]}")
print(f"Actual Benign (1):        {conf_matrix[1, 0]}                     {conf_matrix[1, 1]}")