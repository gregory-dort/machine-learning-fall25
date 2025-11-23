import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# load csv file
df = pd.read_csv('banks.csv')

# create target and predictor variables
target = 'Financial Condition'
predictors = ['TotLns&Lses/Assets', 'TotExp/Assets']

# creating x and y variables
x = df[predictors]
y = df[target]

# create and fit LR model
model = LogisticRegression(
    penalty='l2',
    C=1e9,
    solver='liblinear',
    random_state=42
)
model.fit(x, y)

# displaying model coefficients
coef_df = pd.DataFrame({
        'Predictor': predictors,
        'Coefficient': model.coef_[0],
        'Odds Ratio': np.exp(model.coef_[0])
    })

print(coef_df.to_string(index=False))

# coefficient interpretations
print(f"\n1. Intercept: ")
print("\nThe intercept is the log-odds of a bank being 'weak' when both predictor ratios are 0.")

print(f"\n2. TotLns&Lses/Assets and TotExp/Assets Ratios: ")
print(f"\nThe two chosen predictors perfectly or near-perfectly separate the 'weak' banks from the 'strong' banks in this sample of 20 banks")
print(f"The Higher loan-to-asset and expense-to-asset ratios are associated with the 'weak' financial condition")

