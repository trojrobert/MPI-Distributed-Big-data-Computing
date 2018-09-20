# -*- coding: utf-8 -*-
"""
Created on Tue May  1 23:25:37 2018

@author: John Robert
"""

from mpi4py import MPI
import numpy as np
from scipy.spatial import distance_matrix 
import matplotlib.pyplot as plot 
import pandas as pd 
from sklearn.cluster import KMeans

K = 6

df= pd.read_csv("Absenteeism_at_work.csv", delimiter= ";")
#print(dataset)

    
dataset =df.values
dataset = dataset.astype(int)

def k_means(X,K):
     
    t_start = MPI.Wtime()
    
    nrow = X.shape[0]
    ncol = X.shape[1]
    
    #Pick 4 random rows as the centroid 
    initial_centroids = np.random.choice(nrow,K,replace=False)
    #print("initial_centroids: {} \n".format(initial_centroids))
    
    #Get the details of each rows we pick as centroid
    centroids = X[initial_centroids]
    print("Fist Centroid centroids: \n {} \n".format(centroids))
    
    #Create a temp variable to store old centroid and old cluster for each point(rows)
    centroids_old = np.zeros((K,ncol))
    cluster_assignments = np.zeros(nrow)
    
    
    while (centroids_old != centroids).any():
        
        #To keep track of the change in centroid because the present centroid will be replaced
        centroids_old = centroids.copy()
        print("Present centroid: {} \n".format(centroids_old))
        #Compute distance between data points and centroids
        dist_matrix = distance_matrix(X,centroids, p=2)
        #print("dist_matrix: {} \n".format(dist_matrix))
        print("The distance of each centriod to each point \n {} \n".format(dist_matrix ))
        
        for i in np.arange(nrow):
            
            #To calculate the closest centriod for each row or point
            d = dist_matrix[i]
            closest_centroid = (np.where(d == np.min(d)))[0][0]
            
            cluster_assignments[i] = closest_centroid
            #print("Point {} is closest to {} to it is in cluster {}\n".format(i,closest_centroid,closest_centroid ))
            
            
        for k in np.arange(K):
            #Pick all the point in each K 
            Xk = dataset[ cluster_assignments == k]
            
            print("List of all point in cluster {} \n {} \n".format(k,Xk ))
            
            
            #Find the mean of all the point in each cluster 
            centroids[k] = np.apply_along_axis(np.mean,axis = 0, arr=Xk)
            #print("Xk: {} \n".format(Xk))
            
    t_diff = MPI.Wtime() - t_start
    return(centroids,cluster_assignments,t_diff)
    
k_means_result = k_means(dataset,K)

centriods = k_means_result[0]
cluster_assignemnts = (k_means_result[1]).tolist()
time_spent = k_means_result[2]
print("Final Centroid \n {} \n".format(centriods))
 
print("Final Cluster \n {} \n".format(cluster_assignemnts)) 

print("Total time taken {} \n".format(time_spent))
