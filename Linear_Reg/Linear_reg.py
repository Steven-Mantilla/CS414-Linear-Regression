import pandas as pd
from os import path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Set paths
PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Datasets")
DATA_FILE = path.join(DATA_DIR, "Original_data_with_more_rows.csv")

# Load the dataset
df = pd.read_csv(DATA_FILE, low_memory=False)

# Perform one-hot encoding on the 'TestPrep' column to convert categorical to binary features
df_encoded = pd.get_dummies(df, columns=['TestPrep'], drop_first=True)

# Select the features that are part of the best combination (MathScore, WritingScore, and TestPrep_none)
# Note: 'TestPrep_none' is created after one-hot encoding
X = df_encoded[['MathScore', 'WritingScore', 'TestPrep_none']]  # Features: MathScore, WritingScore, TestPrep_none
Y = df['ReadingScore']  # Target variable: ReadingScore

# Split the data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature Scaling (optional but often useful for regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, Y_train)

# Make predictions
Y_pred = model.predict(X_test_scaled)

# Display the model's intercept and coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# Plotting Actual vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='orange', label='Perfect Prediction')
plt.xlabel('Actual ReadingScore')
plt.ylabel('Predicted ReadingScore')
plt.title('Actual vs Predicted ReadingScore')
plt.legend()
plt.show()

# Plot residuals
plt.figure(figsize=(8, 6))
residuals = Y_test - Y_pred
plt.scatter(Y_pred, residuals, color='blue')
plt.axhline(y=0, color='orange', linestyle='--')
plt.xlabel('Predicted ReadingScore')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
