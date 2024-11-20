# CS414-Linear-Regression

# Linear Regression for Feature Combination Evaluation

This project uses linear regression to evaluate various feature combinations for predicting students' reading scores. It tests different feature subsets and selects the best combination based on the R-squared metric. The process involves feature selection, one-hot encoding, and evaluation of the model's performance using different feature combinations.

Additionally, the project applies the Linear Regression algorithm on a dataset of at least 100 entries with two features: one feature for the x-axis and another for the y-axis.

## Project Overview

- **Objective**: Evaluate different combinations of features to predict reading scores and identify the best set of features for the regression model.
- **Model**: Linear Regression
- **Evaluation Metric**: R-squared (R²)
- **Feature Combinations**: 2-feature and 3-feature combinations
- **Dataset Size**: At least 100 rows
- **Features**: At least two features (one for the x-axis and one for the y-axis)

## Features

- **MathScore**: Student's math test score
- **WritingScore**: Student's writing test score
- **Gender**: Student's gender
- **EthnicGroup**: Student's ethnic group
- **ParentEduc**: Parent's level of education
- **LunchType**: Type of lunch (standard/reduced)
- **TestPrep**: Participation in test preparation course

## Data Preprocessing

- **One-hot Encoding**: Categorical features such as `Gender`, `EthnicGroup`, `ParentEduc`, `LunchType`, and `TestPrep` are one-hot encoded to convert them into binary variables.
- **Feature Combinations**: All possible combinations of 2 or 3 features are evaluated to identify which set of features yields the best performance in predicting the reading scores.

## Requirements

- Python 3.x
- `pandas`
- `numpy`
- `sklearn`

You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn
