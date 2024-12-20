# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:56:10 2024

@author: Haroon
"""


#packagrs for project 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset
dataset =pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)
predict = regressor.predict([[1.4]])
print(predict)


#Visualise decision Tree
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()