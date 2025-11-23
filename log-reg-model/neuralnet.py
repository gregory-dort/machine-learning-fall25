import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# load csv file
df = pd.read_csv('banks.csv')

# setting random seed to 42
np.random.seed(42)

# create target and predictor variables
target = 'Financial Condition'
predictors = ['TotLns&Lses/Assets', 'TotExp/Assets']

# creating x and y variables
x = df[predictors]
y = df[target]

# Feature (x) scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# neural network model creation
mlp_model = MLPClassifier(
    # Architecture
    hidden_layer_sizes=(3,), # 1 hidden layer with 3 neurons
    activation='logistic',   # This is scikit-learn's term for the Sigmoid function
    
    # Optimization/Training Parameters
    solver='adam',           # Optimizer
    alpha=0.0001,            # L2 regularization term (small by default)
    learning_rate_init=0.01, # Similar to Keras's learning rate
    max_iter=1000,           # Set a high number of epochs (scikit-learn uses max_iter)
    random_state=42,         # For reproducibility
    verbose=False            # Hide training output
)

# fit neural network model
# since entire set is being used forr trining there is no train test split needed
mlp_model.fit(x_scaled, y)
print("--- Training Results ---")
print(f"Number of Epochs/Iterations: {mlp_model.n_iter_}")
print(f"Final Training Loss: {mlp_model.loss_:.4f}")
print(f"Final Training Accuracy: {mlp_model.score(x_scaled, y):.4f}")


