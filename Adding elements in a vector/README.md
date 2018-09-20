# Adding-elements-in-a-vector-with-mpi
Paralleling the adding of elements in a vector

Parallelization strategy
1. Intialize the size of the vector 
2. Generate a vector A
3. Check if the number of workers is greater than 1
4. If the  number of workers is equal to 1, then the worker should not divide the vectors, instead the worker should do the work by getting average of all the values in the vector  
5. If the number of workers is greater than 1 then worker 1 to assign the root worker(organiser)
6. Worker 1 divide vector A into slices, then send the slice vectors to other workers to compute the average 
7. Other workers receive the sliced from worker 1, compute the average of the sliced vectors sent to them, then send result back to the worker 1
8. Worker 1, collect all the results from different workers, organise the result, then display the organise results 