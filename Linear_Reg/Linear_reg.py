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

# Function to perform linear regression and generate different outputs
def linear_regression_analysis():
    # Load the dataset
    df = pd.read_csv(DATA_FILE, low_memory=False)

    # If you are using one-hot encoding for 'TestPrep' column, apply it
    df_encoded = pd.get_dummies(df, columns=['TestPrep'], drop_first=True)

    # Choose the best feature combination based on the R-squared value (as mentioned earlier)
    X = df_encoded[['MathScore', 'WritingScore', 'TestPrep_none']]  # Best combination of features
    Y = df['ReadingScore']  # Target variable: ReadingScore

    # Split the data into training and testing sets (80% for training, 20% for testing)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # Feature Scaling (optional but often useful for regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, Y_train)

    # Make predictions
    Y_pred = model.predict(X_test_scaled)

    # Store intercept and coefficients for later use
    intercept = model.intercept_
    coefficients = model.coef_

    # Evaluate the model
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    # Menu for user interaction
    while True:
        print("\nChoose an option:")
        print("1. Display Analysis (Intercept, Coefficients, Evaluation)")
        print("2. Plot Actual vs Predicted values")
        print("3. Plot Residuals")
        print("4. Display First 20 Actual vs Predicted values")
        print("5. Exit")

        choice = input("Enter the number of your choice: ")

        if choice == '1':
            print("\n--- Analysis ---")
            print(f"Intercept: {intercept}")
            print(f"Coefficients: {coefficients}")
            print(f"Mean Squared Error: {mse}")
            print(f"R-squared: {r2}")
        
        elif choice == '2':
            plt.figure(figsize=(8, 6))
            plt.scatter(Y_test, Y_pred, color='blue', label='Predicted vs Actual')
            plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='orange', label='Perfect Prediction')
            plt.xlabel('Actual ReadingScore')
            plt.ylabel('Predicted ReadingScore')
            plt.title('Actual vs Predicted ReadingScore')
            plt.legend()
            plt.show()

        elif choice == '3':
            plt.figure(figsize=(8, 6))
            residuals = Y_test - Y_pred
            plt.scatter(Y_pred, residuals, color='blue')
            plt.axhline(y=0, color='orange', linestyle='--')
            plt.xlabel('Predicted ReadingScore')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            plt.show()

        elif choice == '4':
            comparison_df = pd.DataFrame({
                'Actual ReadingScore': Y_test[:20].values,
                'Predicted ReadingScore': Y_pred[:20]
            })
            print("\nFirst 20 Actual vs Predicted Scores:")
            print(comparison_df)

        elif choice == '5':
            print("Exiting...")
            break

        else:
            print("Invalid choice, please select a valid option.")

        # Ask if the user wants to view other outputs
        next_choice = input("\nWould you like to view another output? (y/n): ").strip().lower()
        if next_choice != 'y':
            print("Exiting...")
            break

# Run the analysis
linear_regression_analysis()
