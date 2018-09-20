"""
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

#Initailize the size of the vector
N = 100000

#if the number of worker is 1
#The worker do all the work 
if size == 1:
    t_start = MPI.Wtime()
    
    #Generating a Vector with random numbers
    vectorA = np.random.randint(1,100, size=(N,1))
    print("Vector A: \n {} \n".format(vectorA))
    
    
    #getting the avaerage of Vector A 
    avgResult = np.mean(vectorA)
    
    t_diff = MPI.Wtime() - t_start
    print("The average of vector A  is \n {} \n".format(avgResult))
    print("When N = 100000")
    print("The total time taken to get the average of Vector A  {}s \n".format(t_diff))
else:
    #The number of workers is greater than 1 then you need to divide the work among the workers
    t_start = MPI.Wtime()

    if rank == 0:
        #check if it is the root worker
        
        #Generating a Vector with random numbers
        vectorA = np.random.randint(1,100, size=(N,1))
        print ("Vector A: \n {} \n".format(vectorA))
        
        
        #Getting the size of the data meant for each process
        #Note the root worker divide the vector and send the data to other workers for computation 
        dataSize4Process = int(Decimal(N / (size - 1)).quantize(Decimal('1.'), rounding= ROUND_HALF_UP))
        print("Size of data for each worker is {} \n".format(dataSize4Process))
        
        #Dividing the initial vector and sending to workers for computation
        for i in range(1,size):
            
            startSlice = dataSize4Process * (i-1)
            
            #Check if it is the last worker
            #Give remaining data to the last worker, in case all slice don't have equal slice
            if i == size:
                endSlice = N
            else:
                endSlice = startSlice + dataSize4Process     
            #print ("Process {} \n".format(i))
            
            #Getting the slice of data to send 
            dataA = vectorA[startSlice:endSlice,:]
            print ("data A to send: {} \n".format(dataA ))
            
            #Send raw sliced data to each worker
            comm.send(dataA,dest=i, tag = 1)
            #print("Data sent \n")
        
        avgResult = []
        
        for i in range(1,size):
            #Receive the computed data from each process
            recComputedValue = comm.recv(source= i)  
            
            #Gather the the computed data receeive from each process
            #organise the final result
            avgResult.append(recComputedValue)
        
        t_diff = MPI.Wtime() - t_start
        
        avgResult = np.mean(avgResult)
        print("The average of vector A  is \n {} \n".format(avgResult))
        print("When N = 100000")
        print("The total time taken is calculate the average of  Vector A {}s \n".format(t_diff))
   
    if rank != 0:
        #Receive the raw data for compututation from the root worker
        recDataA = comm.recv(source= root, tag= 1)
        #print("Data Received \n")
        
        #compute the average of the data sent
        computedValue = np.mean(recDataA)
        
        #print ("Data computed by process: {} \n".format(rank + 1))
        #print ("Computed Value: {} \n".format(computedValue))
        
        #send the computed values back to the root process
        #print("Sending Computed Data to root process \n")
        comm.send(computedValue, dest=0)
        #print("Computed data sent \n")