# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load dataset and split into features (`X`) and target (`y`).
2. Train a Decision Tree Regressor on `X` and `y`.
3. Predict salary values using the trained model.
4. Evaluate model performance using MSE and R² metrics.
5. Plot and visualize the decision tree structure.
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the dataset
file_path = 'Salary.csv'
data = pd.read_csv(file_path)

# Prepare the input and output data
X = data[['Level']]  # Independent variable (Level)
y = data['Salary']   # Dependent variable (Salary)

# Train the Decision Tree Regressor model
model = DecisionTreeRegressor(random_state=0)
model.fit(X, y)

# Predict salaries for all levels
y_pred = model.predict(X)

# Add predictions to the dataframe
data['Predicted Salary'] = y_pred

# Display sample data with predictions
# Formatting the display for a clear table output
sample = data[['Position', 'Level', 'Salary', 'Predicted Salary']]
print(sample.to_string(index=False))  # Ensures neat column alignment

# Calculate and display performance metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"\nMean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=['Level'], filled=True, rounded=True, fontsize=10)
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/7f678ad8-9516-471b-bd6b-aa727b4b0c88)
![image](https://github.com/user-attachments/assets/0cd1d086-6952-4975-8e48-ec059b57fdf9)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
