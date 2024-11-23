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

#Training the linear Regression model on the whole dataset]
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Training the Polynominal RegresnomialFeaturession model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly  = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# Display model details
print("Coefficients:", lin_reg_2.coef_)
print("Intercept:", lin_reg_2.intercept_)

#model paramaters
print("Model Parameters:", lin_reg_2.get_params())


plt.scatter(X,y,color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression Result
plt.scatter(X,y,color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynimal Regression Results (for higher resolution and smoother curve)
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear Regression 
lin_pred = lin_reg.predict([[6.5]])
print(' linear prediction = ',lin_pred)

#Predicting a new result with Polynomial Regression
poly_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(' Polynomial prediction = ',lin_pred)
