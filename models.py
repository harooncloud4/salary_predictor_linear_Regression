# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 18:37:01 2024

@author: Haroon
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR


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

def train_decision_tree(X_train,y_train):
    # Training the Decision Tree Regression model on the whole dataset
    
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train,y_train)
    return regressor

# Train Support Vector Regression (SVR)
def train_svr(X_train, y_train):
   
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train, y_train)
    return svr

# Train Gradient Boosting (XGBoost)
def train_xgboost(X_train, y_train):
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
    xgb.fit(X_train, y_train)
    return xgb

# Evaluate Model
def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mse