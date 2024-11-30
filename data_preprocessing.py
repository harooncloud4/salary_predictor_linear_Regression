# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:42:42 2024

@author: Haroon
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split_dataset():
    #importing the dataset
    dataset =pd.read_csv('Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    #Splitting the dataset into the Training set and test set 
    
    X_train, X_test,y_train,y_test =train_test_split (X,y,test_size = 1/3, random_state = 0)
    return X_train, X_test,y_train,y_test