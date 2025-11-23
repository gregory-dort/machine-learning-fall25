import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing scikit-learn models, metrics and scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# classification related imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score, roc_curve, auc, classification_report, confusion_matrix

# Data Loading
df = pd.read_csv("home_garden_iot_data.csv")

# Data Cleaning
columns_to_drop = [col for col in df.columns if 'Unnamed' in col or col == 'soil_moisture']
df.drop(columns=columns_to_drop, inplace=True)

# drops empty columns
df.dropna(inplace=True)

# target and feature columns
target_column = 'needs_watering'
feature_columns = df.columns.drop(target_column)

# print dataset shape and target column name
print(f"\nDataset shape: {df.shape}")
print(f"Target variable for prediction: '{target_column}'")

# identifying x and y variables (x = features y = target)
x = df[feature_columns]
y = df[target_column]

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,  stratify=y)

# normalizing data using standard scaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(f"\nClassification X_train shape: {x_train_scaled.shape}")
print(f"Classification X_test shape: {x_test_scaled.shape}")

# initializing models with parameters 
models = {
    "kNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "SVM": SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
}

results = {}

print("\n--- Classification Model Training and Evaluation ---")

for name, model in models.items():
    model.fit(x_train_scaled, y_train)
    
    # Predict on Test Set
    y_pred = model.predict(x_test_scaled)
     
    # pos_label=1 means 'needs watering' (the positive case)
    sensitivity = recall_score(y_test, y_pred, pos_label=1)
    # pos_label=0 means 'does not need watering' (the negative case)
    specificity = recall_score(y_test, y_pred, pos_label=0)

    results[name] = {
        "Sensitivity (Recall)": f"{sensitivity:.4f}",
        "Specificity (TNR)": f"{specificity:.4f}",
        "Model": model
    }
    
    print(f"\nModel: {name}")
    print(f"Sensitivity (Recall): {results[name]['Sensitivity (Recall)']}")
    print(f"Specificity (TNR): {results[name]['Specificity (TNR)']}")
    
# Display summary table
results_df = pd.DataFrame({
    'Model': [name for name in results.keys()],
    'Sensitivity': [res['Sensitivity (Recall)'] for res in results.values()],
    'Specificity': [res['Specificity (TNR)'] for res in results.values()]
})

print("\nSummary of Sensitivity and Specificity:")
print(results_df.to_markdown(index=False))

# Create a figure for the ROC curves
plt.figure(figsize=(8, 6))

# determines which model to use then creates plot
for name, res in results.items():
    model = res['Model']
    
    if name == "Decision Tree":
        x_test_data = x_test.values
    else:
        x_test_data = x_test_scaled

    y_proba = model.predict_proba(x_test_data)[:, 1]

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot the curve
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
    
    # Store AUC for reporting
    results[name]['AUC'] = roc_auc

# Plotting settings
plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Report AUC values
auc_report = pd.DataFrame({
    'Model': [name for name in results.keys()],
    'AUC': [res.get('AUC', 'N/A') for res in results.values()]
})
print("\n--- Area Under the Curve (AUC) Report ---")
print(auc_report.to_markdown(index=False))