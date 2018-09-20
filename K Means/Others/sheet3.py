# -*- coding: utf-8 -*-
"""
Created on Tue May  1 17:40:32 2018

@author: Shaba
"""

from mpi4py import MPI
import numpy
import csv
import copy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
k = 8
scatter_dataset = []
centroids=[]
previous_centroid=[]
current_centroids = []
dataset_recv = []
distance_matrix = []
membership_list = []
local_centroid = []
list_lcentroid = []
converge = 0

def Calculate_Distance(a,b):
    distance = 0
    for i in range(0,len(a)):
        distance = distance + ((int(a[i]) - int(b[i]))**2)
    return numpy.sqrt(distance)

def assign_membership():    
    
    for i in range(0,len(distance_matrix)):
        minValue = distance_matrix [i][0]
        minMember = 0
        for j in range(0,len(distance_matrix[i])):
            if (distance_matrix[i][j] < minValue):
                minMember = j
                minValue = distance_matrix [i][j]
        membership_list.append(minMember)

def compute_localMean():
    
    data_per_lcentroid =  numpy.random.randint(0,1,size = (len(centroids)))
    data_per_lcentroid.fill(0)
    
    for m in range(0,len(membership_list)):        
        local_centroid[membership_list[m]] = (local_centroid[membership_list[m]] + dataset_recv[m])
        data_per_lcentroid[membership_list[m]] = data_per_lcentroid[membership_list[m]] + 1
       
    for n in range(0,len(local_centroid)):
        if data_per_lcentroid[n] != 0:
            local_centroid[n] = local_centroid[n] * (1/data_per_lcentroid[n])
    return local_centroid

def compute_globalMean():
    
    
    for i in range(0,len(list_lcentroid)):
        for j in range(0,len(list_lcentroid[i])):
            current_centroids[j] = (current_centroids[j] + list_lcentroid[i][j])
            
    for n in range(0,len(current_centroids)):
        for m in range(0,len(current_centroids[n])):
            current_centroids[n][m] = numpy.around((current_centroids[n][m]/size),decimals = 3)

    
if rank == 0:
    data =[]
    with open('C:/Users/chamu/Documents/Python Scripts/DDA Lab/Sheet 3/Absenteeism_at_work.csv') as csvfile:
        readcsv = csv.reader(csvfile, delimiter = ';')    
        for row in (readcsv):
           data.append(row)
    data = numpy.delete(data,(0), axis = 1)
    data = numpy.delete(data,(0), axis = 0)
    dataset_size = len(data)
    
    data_size = round(dataset_size/size) #determine how many rows of pixels to send to each process
    
    start_pointer = 0 #the start index of the image row for the first process
    
    #slice the images and put each slice in an array
    for i in range(0,size):     #for each process, create a list of start index and end index        
        #if it's not the last process, number of rows is data_size
        if i != (size-1):
            end_pointer = (i+1)*data_size
        else:   #if it is the last process, end_pointer is the last row of pixels
            end_pointer = dataset_size
        
        scatter_dataset.append(numpy.array(data[start_pointer:end_pointer],numpy.float_))    #append the start index and end index
        
        start_pointer = end_pointer#set index = end index for next process
    current_centroids = numpy.array(data[0:k],numpy.float_)
    previous_centroid = numpy.array(data[1:k+1],numpy.float_)
    
dataset_recv = comm.scatter(scatter_dataset, root = 0) #scatter data set


while (converge == 0):
  
    centroids = comm.bcast(current_centroids, root = 0)
    comm.barrier()
    distance_matrix = numpy.random.randint(0,1,size = (len(dataset_recv),len(centroids)))
    
    for i in range(0, len(dataset_recv)):
        for j in range(0,k):
            distance_matrix[i][j] = (Calculate_Distance(dataset_recv[i],centroids[j]))

    membership_list = []       
    assign_membership()

    local_centroid = []
    local_centroid = numpy.random.randint(0,1,size = (len(centroids),len(centroids[0])))
    local_centroid = compute_localMean()    
   
    list_lcentroid = comm.gather(local_centroid, root = 0)    
    
    
    if rank == 0:
        
        previous_centroid = copy.deepcopy(current_centroids)
        current_centroids.fill(0)
        compute_globalMean()
#        print("current")
#        print(current_centroids)
#        print("previous")
#        print(previous_centroid)
        
        converge = numpy.array_equal(previous_centroid,current_centroids)
    
    converge = comm.bcast(converge,root = 0)
    comm.barrier()
if rank == 0:
    print("Global Centroid computed")
    
    for i in range(0,len(current_centroids)):
        print("Centroid {}".format(i))
        print(current_centroids[i])