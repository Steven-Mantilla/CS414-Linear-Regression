# CS414-Linear-Regression

## Project Overview
This project implements a **Linear Regression** algorithm on a dataset containing at least 100 rows. The dataset includes two features for prediction (e.g., `MathScore` and `WritingScore`) and a target variable (`ReadingScore`). The script performs data preprocessing, scales the features, trains the model, and provides multiple outputs for evaluation, visualization, and analysis. Additionally, the project applies the Linear Regression algorithm on a dataset of at least 100 entries with two features: one feature for the x-axis and another for the y-axis.


## Requirements
The following Python libraries are required to run the script:
- `pandas`
- `os`
- `matplotlib`
- `scikit-learn`

Install them using the following command:
```bash
pip install pandas matplotlib scikit-learn

## Dataset Requirements
- The dataset must have **at least 100 rows**.
- Required columns in the dataset:
  - `MathScore`: Feature used on the x-axis.
  - `WritingScore`: Feature used on the y-axis.
  - `ReadingScore`: The target variable.
- Place the dataset in the `Datasets/` folder with the filename `Original_data_with_more_rows.csv`.


## File Structure
- **Project Root**:
  - `Datasets/Original_data_with_more_rows.csv`: Dataset file.
  - `script_name.py`: Python script implementing linear regression.

---
# How to Run

1. Place the dataset in the `Datasets/` folder.
2. Run the script using the following command:

   ```bash
   python script_name.py
3. Use the interactive menu to choose outputs:
   - Display Analysis: Displays intercept, coefficients, Mean Squared Error (MSE), and R-squared values.
   - Plot Actual vs Predicted Values: Shows a scatter plot of actual vs. predicted values.
   - Plot Residuals: Displays a residual plot to analyze errors.
   - Display Actual vs Predicted Values: Outputs a table with the first 20 rows of actual vs. predicted values along with squared residuals.
   - Exit: Ends the program.
# Features

- **Linear Regression:**
  - Predicts "ReadingScore" based on "MathScore" and "WritingScore."
  - Scales features using StandardScaler.

- **Model Evaluation:**
  - Displays metrics: Mean Squared Error (MSE) and R-squared values.

- **Visualization:**
  - Generates scatter plots for actual vs. predicted values.
  - Displays residual plots to analyze error distributions.

- **Tabular Output:**
  - Displays a table showing actual vs. predicted scores, along with residuals.

# Example Interaction

Upon running the script, you will see an interactive menu:

Choose an option:
1. Display Analysis (Intercept, Coefficients, Evaluation)
2. Plot Actual vs Predicted values
3. Plot Residuals
4. Display First 20 Actual vs Predicted values in table format
5. Exit


