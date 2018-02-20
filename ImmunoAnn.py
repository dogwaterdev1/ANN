#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:56:20 2018

@author: Blake
The dataset is solely the work of UC Irvine Machine Learning Repository, Center For Machine Learning and Intelligent Systems
at https://archive.ics.uci.edu/ml/index.php link for the dataset can be found at https://archive.ics.uci.edu/ml/datasets/Immunotherapy+Dataset#
My implementation is just a rough scratch of what I hope to accomplish, and is completely educational/ experimental to advance
my knowledge of the field. I renamed the datafile as well so it could be in CSV format. The Citation requests are as follows:

Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

-----------------------------------------------
Relevant Papers:

1. F. Khozeimeh, R. Alizadehsani, M. Roshanzamir, A. Khosravi, P. Layegh, and S. Nahavandi, 'An expert system for selecting wart treatment method,' Computers in Biology and Medicine, vol. 81, pp. 167-175, 2/1/ 2017. 
2. F. Khozeimeh, F. Jabbari Azad, Y. Mahboubi Oskouei, M. Jafari, S. Tehranian, R. Alizadehsani, et al., 'Intralesional immunotherapy compared to cryotherapy in the treatment of warts,' International Journal of Dermatology, 2017, DOI: 10.1111/ijd.13535 
3. Intralesional immunotherapy with Candida antigen compared to cryotherapy in the treatment of warts. M Teimoorian, F Khozeimeh, P Layegh, R Alizadehsani 
American Academy of Dermatology, 2016 

Citation Request:

1. F. Khozeimeh, R. Alizadehsani, M. Roshanzamir, A. Khosravi, P. Layegh, and S. Nahavandi, 'An expert system for selecting wart treatment method,' Computers in Biology and Medicine, vol. 81, pp. 167-175, 2/1/ 2017. 
2. F. Khozeimeh, F. Jabbari Azad, Y. Mahboubi Oskouei, M. Jafari, S. Tehranian, R. Alizadehsani, et al., 'Intralesional immunotherapy compared to cryotherapy in the treatment of warts,' International Journal of Dermatology, 2017, DOI: 10.1111/ijd.13535
-----------------------------------------------


"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#loading and separating the data
dataset = pd.read_csv("Immuno.csv");
actual_data = dataset.iloc[:,0:7 ]
labeled_data = dataset.iloc[:,7]
#there's no categorical data, so don't have to worry about it

#split the data into training and test sets
#in bigger dbs need evenly random samples
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(actual_data, labeled_data, test_size = 0.2, random_state=0)

#scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) #scaling the x training data
x_test = scaler.fit_transform(x_test)






#creating the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
#initialize
classifier = Sequential()

#7 input variables,
# 4 hidden layers
#1 output layer. 
#binary classification result of 1 or 0. don't have to change much of anything
classifier.add(Dense(output_dim=4, init='uniform',activation='relu',input_dim=7))
#classifier.add(Dense(output_dim=1, init='uniform', activation='relu'))
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid')) #sigmoid will give the probabilites of the values
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
#now actually tying the ANN to the rest of the data, run this to see it training
classifier.fit(x_train,y_train,batch_size=2,nb_epoch=100)
#now the model is trained, using the model, make predictions
y_predictions = classifier.predict(x_test)


#now making a confusion matrix
from sklearn.metrics import confusion_matrix
#saying that all values > 0.75 are generally true. because it's sigmoid
y_predictions = (y_predictions > 0.5)
confusium_matrix = confusion_matrix(y_test, y_predictions) #comparing the values
#first round now great, only about 62% accuracy