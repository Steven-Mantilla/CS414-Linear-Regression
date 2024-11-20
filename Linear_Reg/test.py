import pandas as pd
from os import path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from itertools import combinations

# Set paths
PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Datasets")
DATA_FILE = path.join(DATA_DIR, "Original_data_with_more_rows.csv")

# Load the dataset
df = pd.read_csv(DATA_FILE, low_memory=False)

# Select the relevant columns for X and Y
X = df[['MathScore', 'WritingScore', 'Gender', 'EthnicGroup', 'ParentEduc', 'LunchType', 'TestPrep']]  # Features
Y = df['ReadingScore']  # Target variable

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X, drop_first=True)  # drop_first=True to avoid multicollinearity

# Generate combinations of 2 or 3 features
feature_combinations = list(combinations(X_encoded.columns, 2)) + list(combinations(X_encoded.columns, 3))

# Store the results
results = []

# Evaluate each combination of features
for features in feature_combinations:
    # Select the features for the current combination
    X_comb = X_encoded[list(features)]
    
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_comb, Y, test_size=0.2, random_state=42)
    
    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    # Make predictions
    Y_pred = model.predict(X_test)
    
    # Evaluate the model using R-squared
    r2 = r2_score(Y_test, Y_pred)
    
    # Store the result
    results.append((features, r2))

# Sort the results based on R² (in descending order)
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

# Output the best feature combination based on R-squared
best_combination = sorted_results[0]
print("\nBest Feature Combination based on R-squared:")
print(f"Features: {best_combination[0]}")
print(f"R-squared: {best_combination[1]:.4f}")
