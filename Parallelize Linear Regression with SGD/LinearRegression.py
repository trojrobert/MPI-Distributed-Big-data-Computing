# -*- coding: utf-8 -*-
"""
Created on Tue May  8 18:39:03 2018
@author: John Robert
"""

import sklearn
from sklearn.datasets import load_svmlight_file
import os
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

from mpi4py import MPI                     
from decimal import Decimal, ROUND_HALF_UP

import math

def GetDataFromDataset():
    features_X = []
    expected_target_Y = []
    
    #Getting access to the path where we have the dataset
    path ="C:/Users/John Robert/Documents/Summer 2018/Distributed Data Analysis/Solutions/Exercise 4/Dataset_1/"
    files= os.listdir(path)
    os.chdir(path)
    
    # To read the data in the file one aftwer the other then combine them as 1 
    for file in files:  
        x_axis, y_axis = load_svmlight_file(file)
        features_X.append(pd.DataFrame(x_axis.todense()))
        expected_target_Y.append(y_axis)
        
    #To clean the dataset by removing the nan values 
    features_X = pd.concat(features_X).fillna(0)
    
    #Normalizating to rescaling real valued numeric attributes into the range 0 and 1.
    features_X = pd.DataFrame(sklearn.preprocessing.normalize(features_X))
    expected_target_Y = pd.DataFrame(np.concatenate(np.array(expected_target_Y)))

    return features_X, expected_target_Y

#Multiply each coefficient with each feature to get the predict value of y
# yhat = b0 + b1 * x1 + b2 * x2 .....
def predicted_target_Y(features, coefficient):
    yhat_result = np.zeros(features.shape[0])
    for j in range(0, features.shape[1]):
        yhat_result += features[j]*coefficient[j]
    return pd.DataFrame(yhat_result)  


def estimating_new_coefficient(X, Y, predicted_Y, coefficient,learning_rate, number_of_instance = 2000):
   for i in range(0,number_of_instance):
        #error = predicted value of Y - expected value of Y 
        error = (predicted_Y.iloc[i] - Y.iloc[i])
        #loop through each feature
        for j in range(0, X.shape[1]):
            
            #new coefficient  = old coefficient - learning_rate * error * X
            coefficient[j] = (coefficient[j] - (learning_rate * error * X.iloc[i][j]))
#        pred_Y = predictY(X, coef)
       # print(mean_squared_error(pred_Y, Y))
   return coefficient

def dividing_data(x_train, y_train, size_of_workers):
    #Divide the data among the workers
    slice_for_each_worker = int(Decimal(x_train.shape[0]/size_of_workers).quantize(Decimal('1.'), rounding = ROUND_HALF_UP))      
    print('Slice of data for each worker: {}'.format(slice_for_each_worker))
    x_data_for_worker = []
    y_data_for_worker = []
    for i in range(0,size_of_workers):
        if i < size_of_workers - 1:
            x_data_for_worker.append(x_train[slice_for_each_worker*i:slice_for_each_worker*(i+1)])
            y_data_for_worker.append(y_train[slice_for_each_worker*i:slice_for_each_worker*(i+1)])
        else:
            x_data_for_worker.append(x_train[slice_for_each_worker*i:])
            y_data_for_worker.append(y_train[slice_for_each_worker*i:])
    return x_data_for_worker, y_data_for_worker


comm = MPI.COMM_WORLD                       # Initialize communicator
rank=comm.Get_rank()                        # ID of the cureent worker 
status = MPI.Status()                       # Rank ID of sender
size = comm.Get_size()                      # Number odf workers 
root = 0  
number_of_epochs = 10
   
sliced_features_X_train = []
sliced_expected_target_Y_train = []
t_start = 0
t_diff = 0
if rank == root:
    
    t_start = MPI.Wtime()
    # Read data from the dataset
    features_X, expected_target_Y = GetDataFromDataset()

    # Divide the data into 70% train set and 30% test set 
    #Randomly pick 70% 0f the data 
    set_of_data = np.random.rand(len(features_X)) <= 0.7
    features_X_train = features_X[set_of_data]
    expected_target_Y_train = expected_target_Y[set_of_data]
    #the remaining 30% is for the test set
    features_X_test  = features_X[~set_of_data]
    expected_target_Y_test  = expected_target_Y[~set_of_data]
    
    #Call the function to divide the traning data among the worker
    sliced_features_X_train, sliced_expected_target_Y_train = dividing_data(features_X_train, expected_target_Y_train, size)
    
    #coefficient = np.zeros(features_X_training.shape[1])
    predicted_Y_test = predicted_target_Y(features_X_test, np.zeros(features_X_test.shape[1]))
    print("Initial Error: ", mean_squared_error(expected_target_Y_test,  predicted_Y_test))
    
    
else:
    
    slice_features_X_train = None
    slice_expected_target_Y_train = None
    
    wt = MPI.Wtime()

# Send the slice of data to work on to each worker    
sliced_features_X_train = comm.scatter(sliced_features_X_train, root = root)
sliced_expected_target_Y_train = comm.scatter(sliced_expected_target_Y_train, root = root)

# Predict Y
coefficient_sliced_X = np.zeros(sliced_features_X_train.shape[1])

for e in range(0, number_of_epochs):
    predicted_y_sliced = predicted_target_Y(sliced_features_X_train, coefficient_sliced_X)
    
    # estimating new coefficient
    new_coefficients =  estimating_new_coefficient(sliced_features_X_train, sliced_expected_target_Y_train, predicted_y_sliced, coefficient_sliced_X, learning_rate = 0.001, number_of_instance = 100)
    # Gather the new coeffiecient for each slice of the training data
    gather_new_coefficients = pd.DataFrame(comm.gather(new_coefficients, root=0))

    comm.barrier()

    if rank == root:
        coef = gather_new_coefficients.mean()
        predicted_y = predicted_target_Y(features_X_test, coef)
   
        print("Test set error(RMSE) for {} epoch is {}".format(e+1, math.sqrt(mean_squared_error(expected_target_Y_test, predicted_y))))
        print("Train set error(RMSE) for {} epoch is {}" .format(e+1, math.sqrt(mean_squared_error(sliced_expected_target_Y_train, predicted_y_sliced))))
        
        #print("Test set error(RMSE) for {} epoch is {}" .format(e+1, math.sqrt(mean_squared_error(expected_target_Y_test, predicted_y))))
        #print("Train set error(RMSE) for {} epoch is {}" .format(e+1, math.sqrt(mean_squared_error(expected_target_Y_train, predicted_y_sliced))))
t_diff = MPI.Wtime() - t_start
print('Process {}: {} secs.' .format(rank,t_diff))




