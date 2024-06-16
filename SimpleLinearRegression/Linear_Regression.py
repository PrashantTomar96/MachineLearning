import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing Data set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# Splitting the dataset into Training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Simple Linear Regression model on the Training set
# 1) Calling linear regression class from sci-kit library
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # Object of model L.R.
# fit method to train our regression model
regressor.fit(X_train, y_train)
# Predicting the Test set results by giving year of experience as input
y_pred = regressor.predict(X_test)
# Visualizing the training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Experience')
plt.ylabel('salary')
plt.show()
# Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
# regression lines need to be the same hence we use X_train
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Testing set)')
plt.xlabel('Experience')
plt.ylabel('salary')
plt.show()
