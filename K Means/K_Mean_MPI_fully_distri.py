
"""
Created on Thu May  3 15:20:42 2018

@author: John Robert
"""
#import libraries
from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plot 
import pandas as pd 

import copy
from scipy.spatial import distance_matrix 
from sklearn.cluster import KMeans
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN

#get the details of the processes 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# The number of clusters 
k = 5

centroids =[]
dataToScatter =[]
previous_centroid=[]
current_centroids = []
dataset_recv = []
distance_matrix = []
membership_list = []
local_mean = []
list_centroid = []
converge = 0

#import dataset
df= pd.read_csv("Absenteeism_at_work.csv", delimiter= ";")
#print(dataset)


dataset =df.values
dataset = dataset.astype(int)

#get the number of rows and column
nrow = dataset.shape[0]
ncol = dataset.shape[1] 


def Calculate_Distance(a,b):    #compute euclidean distance between two vectors
    distance = 0
    for i in range(0,len(a)):
        distance = distance + ((int(a[i]) - int(b[i]))**2)
    return np.sqrt(distance)

def assign_membership(): #find the min distance and the centroid and assign membership to that centroid   
    
    for i in range(0,len(distance_matrix)): #loop through distance matrix
        minValue = distance_matrix [i][0]   #min value is the first value in distance matrix
        minMember = 0   #centroid with min distance to dataset
        for j in range(0,len(distance_matrix[i])):  #find minimum distance and assign memsbership
            if (distance_matrix[i][j] < minValue):
                minMember = j
                minValue = distance_matrix [i][j]
        membership_list.append(minMember)

def compute_localMean():    #compute local mean
    
    data_per_lcentroid =  np.random.randint(0,1,size = (len(centroids)))
    data_per_lcentroid.fill(0)
    
    for m in range(0,len(membership_list)):     #based on membership matrix, add the centroids values       
        local_mean[membership_list[m]] = (local_mean[membership_list[m]] + dataset_recv[m]) #increment values for centroid
        data_per_lcentroid[membership_list[m]] = data_per_lcentroid[membership_list[m]] + 1 #count number of datasets per centroid
       
    for n in range(0,len(local_mean)):  #perform mean of addition of centroids
        if data_per_lcentroid[n] != 0:
            local_mean[n] = local_mean[n] * (1/data_per_lcentroid[n]) #divide addtion of centroid values by number of datasets per centroid
    return local_mean

### Main 

if rank == 0:
    start_time = MPI.Wtime()    #start a counter

    #create intial centroids
    #Pick 5(number of K) random rows as the centroid 
    initial_centroids = np.random.choice(nrow,k,replace=False)
    
    #Get the details of each rows we pick as centroid
    current_centroids = dataset[initial_centroids]
    
    #Get the size of the data meant for each process/worker - number of rows meant for each process 
    #Note the root worker divide the data and 
    #send the data to other workers for computation 
    slice = int(Decimal(nrow / (size)).quantize(Decimal('1.'), rounding= ROUND_HALF_UP))
    print("Number of rows or Size of data meant for each process is {} \n".format(slice))
    
    #slice the initial data and put each slice in an array
    for i in range(0,size):
        
        startPoint = slice * (i)
        
        #Checking if it is the last worker
        #Give remaining data to the last worker,in case the last slice is not equal to slice
        if i == (size - 1):
            endPoint = nrow
        else:
            endPoint = startPoint + slice    
        
        #make each slice an element of an array 
        dataToScatter.append(dataset[startPoint:endPoint,:])
comm.barrier()

#Send slices to all other workers          
dataset_recv = comm.scatter(dataToScatter, root = 0)
 
#Create a temp variable to store old centroid and old cluster for each point(rows)
previous_centroid = np.zeros((k,ncol))
cluster_assignments = np.zeros(nrow)

while (converge == 0): 
  
    centroids = comm.bcast(current_centroids, root = 0) 
    comm.barrier()      #to make sure that all process have reached this barrier
    
    #distance matrix for each dataset recv and k centroids
    distance_matrix = np.random.randint(0,1,size = (len(dataset_recv),len(centroids)))   
    
    for i in range(0, len(dataset_recv)):   #for each dataset, compute distance to k centroids
        for j in range(0,k):
            distance_matrix[i][j] = (Calculate_Distance(dataset_recv[i],centroids[j]))

    membership_list = []    #contains membership list for each dataset   
    assign_membership() #assisgn membership to each dataset rows based on min distance

    local_mean = []  #contains local mean for each worker
    local_mean = np.random.randint(0,1,size = (len(centroids),len(centroids[0])))
    
    #compute the local mean for each P
    local_mean = compute_localMean()    
    print("local_mean {}".format(local_mean))
    #reduce the sum all local mean at root
    list_centroid = comm.reduce(local_mean, root = 0, op = MPI.SUM)     
    print("list_centroid {}".format(list_centroid))

    #if rank == 0, compute global mean and broadcast to each process
    if rank == 0: 
        
        #current centroid is now the previous centroid
        previous_centroid = copy.deepcopy(current_centroids)
        #global mean centroids
        current_centroids = np.around(list_centroid/size, decimals = 1) 
        #print(current_centroids)
       
        #checks if the program has converged. 
        #program is considered as converged when the previous centroids sent is the same as new centroid
        converge = np.array_equal(previous_centroid,current_centroids)   
    
    
    converge = comm.bcast(converge,root = 0)
    comm.barrier() 
    
if rank == 0:
    print("Global Centroid computed")
    
    for i in range(0,len(current_centroids)):
        print("Centroid {}".format(i))
        #print(current_centroids[i])
        
    end_time = MPI.Wtime()  #end timer
    processing_time = end_time - start_time
    print("Processing time: {}".format(processing_time)) #print processing time


