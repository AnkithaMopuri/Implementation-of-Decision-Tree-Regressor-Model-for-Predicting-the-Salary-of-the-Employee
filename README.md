# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load the dataset and extract features (Level) and target (Salary).
2. Split the data into training and testing sets.
3. Train the Decision Tree Regressor model on the training data.
4. Predict the salary on the test data using the trained model.
5. Display the actual vs. predicted salaries in a tabular format and (optionally) calculate evaluation metrics (MSE, R²).

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: MOPURI ANKITHA
RegisterNumber: 212223040117 
*/
```
```
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
salary_data = pd.read_csv('Salary.csv')

# Extract features (Level) and target (Salary)
X = salary_data[['Level']]
y = salary_data['Salary']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Regressor model
dt_regressor = DecisionTreeRegressor(random_state=42)

# Train the model
dt_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_regressor.predict(X_test)

# Combine the position, level, actual salary, and predicted salary into a dataframe
output_df = pd.DataFrame({
    'Position': salary_data.iloc[X_test.index]['Position'],  # Get the corresponding positions
    'Level': X_test['Level'],
    'Actual Salary': y_test,
    'Predicted Salary': y_pred
})

# Print the tabular output
print(output_df)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the evaluation metrics
print(f"\nMean Squared Error: {mse}")
print(f"R² Score: {r2}")
```

## Output:
![image](https://github.com/user-attachments/assets/33e88cb5-7325-44a3-8a44-205570a7e5fb)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
