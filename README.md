# CS414-Linear-Regression

# Linear Regression Analysis Project

This project performs linear regression analysis on a dataset to predict `ReadingScore` based on `WritingScore`. It includes data preparation, model training, evaluation metrics, and various visualization options to help understand the model's performance.

## Project Overview
This project demonstrates the use of linear regression to predict one variable (`ReadingScore`) based on another (`WritingScore`). The workflow includes:
1. Preparing the data
2. Training the linear regression model
3. Evaluating model performance with various metrics
4. Generating visualizations for analysis

## Dataset
The dataset contains information on students' scores across different subjects, with `WritingScore` as the predictor variable and `ReadingScore` as the target variable.

## Features
- **Data Preparation**: Cleans and preprocesses the dataset for modeling.
- **Model Training**: Builds a simple linear regression model.
- **Evaluation Metrics**: Calculates error metrics to evaluate model accuracy.
- **Visualizations**: Generates plots to illustrate model performance and fit.

## Project Structure
- **Project Root**: Contains the main script.
- **Datasets Directory**: Stores the dataset, `Original_data_with_more_rows.csv`.

## Options Available
- **Display Analysis**: Shows Mean Squared Error, R-squared, Pearson Correlation, standard deviations, slope, and intercept.
- **Display First 20 Actual vs Predicted Values**: Displays a table comparing actual and predicted values for the first 20 instances.
- **Plot Regression Line**: Shows a scatter plot of actual values and the linear regression line.
- **Plot Actual vs Predicted Values**: Compares actual vs predicted values.
- **Plot Residuals**: Visualizes residuals to assess model accuracy.
- **Exit**: Exits the program.
## Additional Details
- **Data Scaling**: `StandardScaler` is used to scale the features.
- **Model Evaluation**: Includes Mean Squared Error (MSE), R-squared, and Pearson Correlation Coefficient.
- **Visualization**: `matplotlib` is used for plots to help evaluate model performance visually.


## Setup and Requirements and  ## How to Run the Analysis

To start the analysis, run the script. A menu will guide you through various options for data analysis and visualization.
- **Python 3.x**
- Required libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

Install dependencies with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn

```bash
python main.py


