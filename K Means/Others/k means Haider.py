# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 15:57:16 2018

@author: haider
"""

from mpi4py import MPI
import pandas as pd
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
t_start = MPI.Wtime()
Master = 0


colum = 21
#row for centroid
crow = 8
#row for data
#drow = 16

if rank == Master:
#    dataSet = np.random.randint(10,size=[drow,colum])
#    #print(matrix)
#    slice1 = drow//size
    cantroid = np.random.randint(10,size=[crow,colum])
    
    df1 = pd.read_csv("Absenteeism_at_work.csv",delimiter = ";")
    dataSet = df1.values
    slice1 = dataSet.shape[0]//size
    
    for i in range(1,size):
        Sindx = i*slice1
        Eindx = (i+1)*slice1
        #for last complete send
        if i == size-1:
            chunkkk = dataSet[Sindx:dataSet.shape[0],:]
            comm.send(chunkkk,dest = i)
            
        else:
            
            chunkkk = dataSet[Sindx:Eindx,:]
            comm.send(chunkkk,dest = i)
        comm.send(cantroid,dest = i)
    #data at master postion
    recv1 = dataSet[0:slice1,:]
    recv2 = cantroid   
        

#print(slice)
#print(df.shape)
    
    
else:

#for data recv other than master   
    recv1 = comm.recv(source=Master)
#for centroid recv
    recv2 = comm.recv(source=Master)


while True:    
    result = np.zeros(shape = (recv1.shape[0],recv2.shape[0]), dtype = int)
    #column = 4
#loop for data
    for m in range (0,recv1.shape[0]):
#loof for centroid
        for q in range (0,crow):
            #loop for every column at one row
            for p in range (0,colum):
                result[m,q] = result[m,q] + (recv2[q,p] - recv1[m,p])**2
            result[m,q] = np.sqrt(result[m,q])
        
#distance
    print("distance results")
    print(result)

#for membership matrix
    membr = np.zeros (shape = (recv1.shape[0],1), dtype = int)
    for n in range (0,recv1.shape[0]):
        #for minimum index finding for a valu
        membr[n,0] = np.argmin(result[n,:])
#    print("membership array")
#    print(membr)
        
    
    fResult= np.zeros (shape = (recv2.shape[0],colum), dtype = int)
    for localMean in range (0,recv1.shape[0]):
        
        fResult[membr[localMean]] = fResult[membr[localMean]] + recv1[localMean]
    
    print("addition of all members")
    print(fResult)

    if rank == Master:
        Globalmean= np.zeros (shape = (recv2.shape[0],colum), dtype = int)
    else:
        Globalmean = None
    
    comm.Reduce([fResult , MPI.INT],[Globalmean , MPI.INT])
    akhatyMembr = comm.gather(membr)
    if rank == Master:
        akhatyMembr = np.vstack(akhatyMembr)
    
    #count gather members for taking mean
        clusterSize = np.zeros (shape = (crow,1), dtype = int)
        for countgather in range (0,dataSet.shape[0]):
            clusterSize[akhatyMembr[countgather]] = clusterSize[akhatyMembr[countgather]] + 1
    
    #for centroid find
        Globalmeanfinal= np.zeros (shape = (recv2.shape[0],colum), dtype = int)
        for centroid in range (0,crow):
            #for avoiding the devision with 0
            if clusterSize[centroid] != 0:
                Globalmeanfinal[centroid] = Globalmean[centroid] /clusterSize[centroid]
    else:
        Globalmeanfinal = None

    temporary = recv2.copy()
    recv2 = comm.bcast(Globalmeanfinal)
    print("differnce")
    #print(temporary, recv2)
    print ("previous")
    print (temporary)
    print ("final")
    print (recv2)
    if np.array_equal(temporary, recv2):
        if rank == Master:
            end_time = MPI.Wtime()
            total_time = end_time - t_start
            print("total time")
            print(total_time)
        break
        
    
