"""
Created on Wed Apr 18 09:29:02 2018
@author: John Robert
"""

from mpi4py import MPI
import numpy as np
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
processorName = MPI.Get_processor_name()
root = 0

#Initailize the size of the Matrix and vector 
N = 10000

#if the number of worker is 1
#The worker do all the tack 
if size == 1:
    t_start = MPI.Wtime()
    #Generating a Matrix and a Vector with random numbers
    matrixA = np.random.randint(1,100, size=(N,N))
    print("Matrix A: \n {} \n".format(matrixA))
    vectorB = np.random.randint(1,100, size=(N,1))
    print("Vector B: \n {} \n".format(vectorB))
    
    
    #multiply matrix A and vector B
    mulResult = np.matmul(matrixA,vectorB)
    
    t_diff = MPI.Wtime() - t_start
    print("The multiplication of Matrix A and vector B is \n {} \n".format(mulResult))
    print("When N = 10000")
    print("The total time taken multiply Matrix A and Vector B {}s \n".format(t_diff))
else:
    #The number of workers is greater than 1 then you need to divide the task among the workers
    t_start = MPI.Wtime()
    
    #check if it is the root worker
    if rank == 0:
        
        #Generating a Matrix and a Vector with random numbers
        matrixA = np.random.randint(1,100, size=(N,N))
        print("Matrix A: \n {} \n".format(matrixA))
        vectorB = np.random.randint(1,100, size=(N,1))
        print("Vector B: \n {} \n".format(vectorB))
        
        #Getting the size of the data meant for each worker
        #Note the root worker divide and send the data for computation 
        dataSize4Process = int(Decimal(N / (size - 1)).quantize(Decimal('1.'), rounding= ROUND_HALF_UP))
        print("Size of data for each process is {} \n".format(dataSize4Process))
        
        #Dividing the initial data and send to workers for computation
        for i in range(1,size):
            
            startSlice = dataSize4Process * (i-1)
            
            #Check if it is the last worker
            #Give remaining data to the last worker, in case all slice don't have equal slice
            if i == size:
                endSlice = N
            else:
                endSlice = startSlice + dataSize4Process     
            
            #Getting the slice of data to send 
            dataA = matrixA[startSlice:endSlice,:]
            #print ("data A to send: {} \n".format(dataA ))
            dataB = vectorB
            #print ("data B to send: {} \n".format(dataB))
        
            #Send data to each worker
            comm.send(dataA,dest=i, tag = 1)
            comm.send(dataB,dest=i, tag = 2)
            #print("Data sent \n")
        
        multResult = []
        
        for i in range(1,size):
            #Receive the computed data from each worker
            recComputedValue = comm.recv(source= i)
            
            #Gather the the computed data receeive from each process
            #organise the final result
            for n in range(0,len(recComputedValue)):
                multResult.append(recComputedValue[n][0])
            
        t_diff = MPI.Wtime() - t_start
        multResult = np.reshape(multResult,(len(multResult), 1))
        print("The multiplication of matrix A and Vector B is \n {} \n".format(multResult))  
        print("When N = 10000")
        print("The total time taken is  {} \n".format(t_diff))    
    if rank != 0:
        #Receive the data to compute from the root worker
        recDataA = comm.recv(source= root, tag= 1)
        recDataB = comm.recv(source= root, tag= 2)
        #print("Data Received \n")
        
        #compute the multiplication of the data sent
        computedValue = np.matmul(recDataA,recDataB)
        
        #print ("Data computed by process: {} \n".format(rank + 1))
        #print ("Computed Value: {} \n".format(computedValue))
        
        #send the computed values back to the root 
        #print("Sending Computed Data to root process \n")
        comm.send(computedValue, dest=0)
        #print("Computed data sent \n")