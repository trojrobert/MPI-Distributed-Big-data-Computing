"""
@author: John Robert
"""

from mpi4py import MPI        #Load MPI Library
import numpy as np
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN

comm = MPI.COMM_WORLD             # Initialize communicator
size = comm.Get_size()            # Get total number of running workers
rank = comm.Get_rank()            # Current worker
processorName = MPI.Get_processor_name()
root = 0                          # Setup root node

#Initailize the size of the vectors
N = 100000

#if the number of workers is 1
#The worker do all the tack 
if size == 1:
    
    t_start = MPI.Wtime()
    #Generating Vectors with random numbers
    vectorA = np.random.randint(1,100, size=(N,1))
    print("Vector A: \n {} \n".format(vectorA))
    vectorB = np.random.randint(1,100, size=(N,1))
    print("Vector B: \n {} \n".format(vectorB))
     
    #getting the sum of Vector A and vector B
    sumResult = np.add(vectorA,vectorB)
    
    t_diff = MPI.Wtime() - t_start
    
    print("The summation of vector A and Vector B is \n {} \n".format(sumResult))
    print("When N = {}".format(N))
    print("The total time taken to sum Vector A and Vector B {}s \n".format(t_diff))
else:
    #The number of workers is greater than 1 then you need to divide the task among the workers
    #check if it is the root worker
    if rank == 0:
         
        t_start = MPI.Wtime()
        
        #Generating Vectors with random numbers
        vectorA = np.random.randint(1,100, size=(N,1))
        print ("Vector A: \n {} \n".format(vectorA))
        vectorB = np.random.randint(1,100, size=(N,1))
        print("Vector B: \n {} \n".format(vectorB))
        
        #Getting the size of the data meant for each worker
        #Note the root worker divide and send the data to other workers for computation 
        dataSize4Worker = int(Decimal(N / (size - 1)).quantize(Decimal('1.'), rounding= ROUND_HALF_UP))
        print("Size of data for each worker is {} \n".format(dataSize4Worker))
        
        #Dividing  the initial vectors among the workers and sending it to them for computation
        for i in range(1,size):
            
            startSlice = dataSize4Worker * (i-1)
            
            #Checking if it is the last worker
            #Give remaining data to the last worker, in case all slice don't have equal size
            if i == size:
                endSlice = N
            else:
                endSlice = startSlice + dataSize4Worker     
            
            #Getting the slice of data to send 
            dataA = vectorA[startSlice:endSlice,:]
            #print ("data A to send: {} \n".format(dataA ))
            dataB = vectorB[startSlice:endSlice,:]
            #print ("data B to send: {} \n".format(dataB))
        
            #Send raw data to each worker apart from the root worker
            comm.send(dataA,dest=i, tag = 1)
            comm.send(dataB,dest=i, tag = 2)
            #print("Data sent \n")
        
        sumResult = []
        
        for i in range(1,size):
            #Receive the computed data from each worker
            recComputedValue = comm.recv(source= i)  
            
            #Gather the the computed data receeive from each worker
            #organise the final result
            for n in range(0,len(recComputedValue)):
                sumResult.append(recComputedValue[n][0])
        
        t_diff = MPI.Wtime() - t_start
        
        sumResult = np.reshape(sumResult,(len(sumResult), 1))
        print("The summation of vector A and Vector B is \n {} \n".format(sumResult))
        print("When N = 100000")
        print("The total time taken is sum Vector A and Vector B {}s \n".format(t_diff))
   
    if rank != 0:
        #Receive the raw data for compututation from the root worker
        recDataA = comm.recv(source= root, tag= 1)
        recDataB = comm.recv(source= root, tag= 2)
        #print("Data Received \n")
        
        #compute the summation of the data sent
        computedValue = np.add(recDataA,recDataB)
        
        #print ("Data computed by process: {} \n".format(rank + 1))
        #print ("Computed Value: {} \n".format(computedValue))
        
        #send the computed values back to the root process
        #print("Sending Computed Data to root process \n")
        comm.send(computedValue, dest=0)
        #print("Computed data sent \n")