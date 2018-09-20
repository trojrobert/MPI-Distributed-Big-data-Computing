# matrix-vector-multiplication-using-MPI

Parallel matrix vector multiplication using MPI point-to-point communication i.e. Send
and Recv.

Parallelization strategy
1. Intialize the size of the N for matrix A and Vector B  
2. Generate matrix A and vector B with random numbers
3. Check if the number of workers is greater than 1
4. If the  number of workers is equal to 1, then the worker should not divide the matrix, instead the worker should do the work by multiplying the matrix by the Vector  
5. If the number of workers is greater than 1 then worker 1 to assign the root worker(organiser)
6. Worker 1 divide Matrix A, then send the slice matrix and vector B to other workers for multiplication 
7. Other workers receive the sliced matrix and vector B, do the multiplication of the sliced matrix and vector then sent the result back to the worker 1
8. Worker 1, collect all the results from different workers, organise the result, then display the organise results 
