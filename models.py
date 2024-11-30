# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 18:37:01 2024

@author: Haroon
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error


#Train Linear Regression
def train_linear_regression(X_train,y_train):
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)
    return regressor

def train_random_forest(X_train,y_train):
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0  ) 
    regressor.fit(X_train,y_train)
    return regressor


def train_polynomial_regression(X_train,y_train):
    #Training the Polynominal RegresnomialFeaturession model on the whole dataset
    
    poly_reg = PolynomialFeatures(degree = 2)
    X_poly  = poly_reg.fit_transform(X_train)

    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly,y_train)
    return poly_reg, lin_reg_2

    # # Display model details
    # print("Coefficients:", lin_reg_2.coef_)
    # print("Intercept:", lin_reg_2.intercept_)

    # #model paramaters
    # print("Model Parameters:", lin_reg_2.get_params())

# Evaluate Model
def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mse