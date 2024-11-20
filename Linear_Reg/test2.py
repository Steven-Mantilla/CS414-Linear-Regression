import pandas as pd
from os import path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats


# Set paths
PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Datasets")
DATA_FILE = path.join(DATA_DIR, "Original_data_with_more_rows.csv")

# Load the dataset
df = pd.read_csv(DATA_FILE, low_memory=False)

# Select the features that are part of the best combination (MathScore, WritingScore, and TestPrep_none)
# Note: 'TestPrep_none' is created after one-hot encoding
X = df['WritingScore']  # Features: MathScore, WritingScore, TestPrep_none
Y = df['ReadingScore']  # Target variable: ReadingScore

# Split the data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

slope, intercept, r, p, std_err = stats.linregress(X_train, Y_train)

def myfunc(x):
  return slope * x + intercept

import pandas as pd

print("Actual vs Predicted")
df = pd.DataFrame({'Actual (Reading Score)': Y_test, 'Predicted (Reading Score)': [myfunc(x) for x in X_test]})
print(df)

mymodel = list(map(myfunc, X_test))

plt.scatter(X_train, Y_train)
plt.plot(X_test, mymodel, color='orange')
plt.xlabel('WritingScore')
plt.ylabel('ReadingScore')
plt.show()
