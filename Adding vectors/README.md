# adding-vectors-with-mpi
Add two vectors and store results in a third vector.

Parallelizing addition of vectors with python mpi4py

Parallelization strategy
1. Intialize the size of the vectors 
2. Generate two vector A and B with random numbers
3. Check if the number of workers is greater than 1
4. If the  number of workers is equal to 1, then the worker should not divide the vectors, instead the worker should do the work of summing vector A and B
5. If the number of workers is greater than 1 then worker 1 to assign the root worker(organiser)
6. Worker 1 divide vector A and vector B by the number of workers, then send the slice vectors to other(respective) workers to do the work of summation 
7. Other other workers receive the sliced vectors, do the summation of the sliced vectors sent to them, then send the result back to the worker 1
8. Worker 1, collect all the results from different workers, organise the result, then display the organise results