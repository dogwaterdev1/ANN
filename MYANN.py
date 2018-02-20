#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 15:47:43 2018

@author: blake
"""

#what to learn:
    #stochastic gradient descent again - udemy reference
    #backpropagation - udemy reference
    #Going over the data preprocessing section:
        #specifically dummy variable trap
        #encoders
    
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values #last column is excluded, really taking columns 1-12
y = dataset.iloc[:, 13].values #however, because selecting one, it is 12

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
#we have 2 categorical values, country and gender so need to do following for each
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1]) #creating dummy variables for countries
#we don't create them for gender because there's only two elements, needs to be > 2. known as dummy variable trap
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
#we need to scale all of our variables now
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#import keras libraries/packages
import keras #import keras
from keras.models import Sequential#modules importing need these two
from keras.layers import Dense

#initialize neural net, defining initial layer(s)
#by sequence of layers or defining a graph. Using sequential layers though
classifier = Sequential() #making an obj

#input layer + first hidden layer
#need input_dim as well
classifier.add(Dense(output_dim=6, init='uniform',activation='relu', input_dim=11)) #dense is really adding the hidden layer. it's art, we don't know entirely.
#tip is to take average of number of nodes in input + output layer, otherwise experiment with parameter tuning.

#adding new hidden layers
classifier.add(Dense(output_dim=6, init='uniform',activation='relu')) 

#output layer, only one output layer #uniform is inputs from second hidden layer, need to replace w/ sigmoid. 
classifier.add(Dense(output_dim=1, init='uniform',activation='sigmoid')) #output dim is also categories of dependent variable
#would need softmax + num output dims if more than one classifier
#predicting test results
y_pred = classifier.predict(X_test)

# step 1 - randomly initialize the weights of each of the nodes to small numbers close to zero. using dense to do this

# step 2 - first observation of dataset goes into input layer, each feature is one input node. this will be 11 nodes.

#step 3 - the neurons fire an activate activation function. using signmoid for output layer, as it uses probability, rectifier for hidden layer

#step 4 - compare, generate error,

# step 5 - back propagation

#step 6 - repeat 1-5 for each observation

#step 7 - epoch completed, do more if want. (Using stochastic gradient descent)
