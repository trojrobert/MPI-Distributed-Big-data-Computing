# Parallel linear regression with MPI
I will discuss about the implementation of a parallel Linear regression program which uses the Parallel Stochastic Gradient Descent (PSGD)learning algorithm. The program has been designed to work for any number for processes

### Parallelizing 
A process is also called a worker 
**Processs 0 is the root/master process**
Strategy
1. When it is the root worker 
2. Load the file in the directory and store them in a dataframe 
3. Divide the data, given 70% to the train set and 30% to the test set
4. Divide the training set among the workers 
5. Initialize the first coefficients as zero 
6. Send slices of the training set(the features data X and the expected target data Y) to ever woker including the root worker 
7. or each epoch 
8. Every worker should get the predict target Y(yhat) for each slice
9. Get the new coefficient of each intance in a slice 
10. Gather the new coefficient from each worker 
11. Calculate the root mean square error for the test set 
12. Calculate the root mean square error for the test set 