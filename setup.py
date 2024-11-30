# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:39:21 2024

@author: Haroon
"""
from data_preprocessing import load_and_split_dataset
from models import train_linear_regression, train_random_forest, evaluate_model,train_polynomial_regression,train_decision_tree


#Load and split the dataset
X_train, X_test,y_train,y_test = load_and_split_dataset()


# Train Linear Regression
linear_regressor =  train_linear_regression(X_train,y_train)
y_pred_linear = linear_regressor.predict(X_test)


# Train Random Forest
random_forest_regressor = train_random_forest(X_train,y_train)
y_pred_rf = random_forest_regressor.predict(X_test)

# Train Polynomial Regression 
poly_regressor, poly_lin_reg = train_polynomial_regression(X_train, y_train)
# Transform X_test using the polynomial features (same as in training)
X_test_poly = poly_regressor.transform(X_test)
y_pred_poly = poly_lin_reg.predict(X_test_poly)


#Train Decision Tree
decision_tree_regressor = train_decision_tree(X_train,y_train)
y_pred_decision_tree = decision_tree_regressor.predict(X_test)


# Evaluate Models
r2_linear, mse_linear = evaluate_model(y_test, y_pred_linear)
r2_rf, mse_rf = evaluate_model(y_test, y_pred_rf)
r2_poly, mse_poly = evaluate_model(y_test, y_pred_poly)
r2_decision_tree, mse_decision_tree = evaluate_model(y_test, y_pred_decision_tree)



# Print Evaluation Results
print(f"Linear Regression R²: {r2_linear:.2f}, MSE: {mse_linear:.2f}")
print(f"Random Forest R²: {r2_rf:.2f}, MSE: {mse_rf:.2f}")
print(f"Polynomial Regression R²: {r2_poly:.2f}, MSE: {mse_poly:.2f}")
print(f"Decision Tree R²: {r2_decision_tree:.2f}, MSE: {mse_decision_tree:.2f}")


