import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the models
classifiers = {
    "kNN": KNeighborsClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVC (SVM)": SVC(random_state=42)
}

results = {}

for name, model in classifiers.items():
    # --- Training ---
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    train_time = end_train - start_train

    # --- Prediction ---
    start_predict = time.time()
    y_pred = model.predict(X_test)
    end_predict = time.time()
    predict_time = end_predict - start_predict

    # --- Metrics ---
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = {
        "Train Time (s)": train_time,
        "Prediction Time (s)": predict_time,
        "Accuracy": accuracy
    }

# Display results
import pandas as pd
results_df = pd.DataFrame(results).T
print(results_df)