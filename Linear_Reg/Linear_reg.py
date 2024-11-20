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

# Function to perform linear regression analysis
def linear_regression_analysis():
    # Load the dataset
    df = pd.read_csv(DATA_FILE, low_memory=False)

    # Independent and target variables
    X = df[['WritingScore']]  # WritingScore as the independent variable
    Y = df['ReadingScore']  # Target variable: ReadingScore

    # Split the data into training and testing sets (80% for training, 20% for testing)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, Y_train)

    # Make predictions
    Y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    # Standard Deviation of Independent Variables (SDx for each feature)
    SDx = np.std(X_test, axis=0)  # Standard deviation for WritingScore
    SDy = np.std(Y_test)  # Standard deviation for ReadingScore

    # Pearson Correlation Coefficient
    pearson_corr = np.corrcoef(Y_test, Y_pred)[0, 1]

    # Function to display analysis results
    def display_analysis():
        print("\n--- Analysis ---")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R-squared: {r2:.4f}")
        print(f"Pearson Correlation Coefficient (R): {pearson_corr:.4f}")
        print(f"Standard Deviation of WritingScore (SDx): {SDx.iloc[0]:.4f}")
        print(f"Standard Deviation of ReadingScore (SDy): {SDy:.4f}")
        print(f"Slope (Coefficient): {model.coef_[0]:.4f}")
        print(f"Intercept: {model.intercept_:.4f}")

    # Function to display first 20 actual vs predicted values
    def display_comparison():
        output_table = pd.DataFrame({
            'X (WritingScore)': X_test['WritingScore'],
            'Y (Actual ReadingScore)': Y_test.values,
            'Predicted ReadingScore': Y_pred,
            'e (Squared)': (Y_test.values - Y_pred) ** 2  # Squared residuals
        })
        print("\n--- First 20 Actual vs Predicted Scores ---")
        print(output_table.head(20))

    # Function to plot the regression line (WritingScore vs ReadingScore)
    def plot_regression_line():
        plt.figure(figsize=(8, 6))

        # Scatter plot for actual values
        plt.scatter(X_test['WritingScore'], Y_test, color='blue', label='Actual')

        # Create X_line with WritingScore values from min to max
        X_line = np.linspace(X_test['WritingScore'].min(), X_test['WritingScore'].max(), 100).reshape(-1, 1)

        # Scale the X_line data (WritingScore)
        X_line_scaled = scaler.transform(X_line)

        # Make predictions using the scaled X_line
        Y_line = model.predict(X_line_scaled)

        # Plot the regression line (Model's predictions)
        plt.plot(X_line, Y_line, color='red', label='Regression Line')
        plt.xlabel('WritingScore')
        plt.ylabel('ReadingScore')
        plt.title('Regression Line: WritingScore vs ReadingScore')
        plt.legend()
        plt.show()

    # Function to plot Actual vs Predicted values
    def plot_actual_vs_predicted():
        plt.figure(figsize=(8, 6))
        plt.scatter(Y_test, Y_pred, color='blue', label='Predicted vs Actual')
        plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='orange', label='Perfect Prediction')
        plt.xlabel('Actual ReadingScore')
        plt.ylabel('Predicted ReadingScore')
        plt.title('Actual vs Predicted ReadingScore')
        plt.legend()
        plt.show()

    # Function to plot residuals
    def plot_residuals():
        plt.figure(figsize=(8, 6))
        residuals = Y_test - Y_pred
        plt.scatter(Y_pred, residuals, color='blue')
        plt.axhline(y=0, color='orange', linestyle='--')
        plt.xlabel('Predicted ReadingScore')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()

    # Menu for user interaction
    menu_options = {
        '1': display_analysis,
        '2': display_comparison,
        '3': plot_regression_line,
        '4': plot_actual_vs_predicted,
        '5': plot_residuals,
        '6': lambda: print("Exiting...")
    }

    while True:
        print("\nChoose an option:")
        print("1. Display Analysis (MSE, R-squared, Pearson-R, Standard Deviations, Evaluation, Slope, Intercept)")
        print("2. Display First 20 Actual vs Predicted values in table format")
        print("3. Plot Regression Line (WritingScore vs ReadingScore)")
        print("4. Plot Actual vs Predicted values")
        print("5. Plot Residuals")
        print("6. Exit")

        choice = input("Enter the number of your choice: ")
        if choice in menu_options:
            if choice == '6':  # Exit case
                menu_options[choice]()
                break
            else:
                menu_options[choice]()
        else:
            print("Invalid choice, please select a valid option.")

        # Ask if the user wants to view other outputs
        while True:
            next_choice = input("\nWould you like to view another output? (y/n): ").strip().lower()
            if next_choice == 'y':
                break  # Continue with the next choice
            elif next_choice == 'n':
                print("Exiting...")
                return  # Exit the program
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

# Run the analysis
linear_regression_analysis()
