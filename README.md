# MPI - Distributed Big data programming 
# Parallel Computing
## Overview 
The Message Passing Interface (MPI) is a standardized and portable message-passing system designed to function on a wide variety of
parallel computers.

+ A parallel program is decomposed into processes, called ranks.
+ Each rank holds a portion of the program's data into its private
memory.
+ Communication among ranks is made explicit through messages.
+ All process are launched simultaneously.

## About MPI for Python
+ mpi4py is the MPI for Python.
+ mpi4py provides bindings of the MPI standard for the Python programming language, allowing any Python program to exploit
multiple processors.
+ mpi4py package can be found [here](http://mpi4py.readthedocs.io/en/stable/)
+ One can follow the package installation at [here](http://mpi4py.readthedocs.io/en/stable/install.html)

## Installation 
Install Anaconda, Python 3.5 [link](https://www.continuum.io/downloads)

Install pip package [link](https://anaconda.org/anaconda/pip)

Install Microsoft MPI [link](https://www.microsoft.com/en-us/download/details.aspx?id=54607)

    You need to run both files msmpisdk.msi and MSMpiSetup.exe
    Add $PATH$ in the Environment Variables, e.g. C:nProgram Files (x86)nMicrosoft SDKsnMPI
Install mpi4py package by conda install mpi4py

Run a python program by command

    mpiexec -n N python your file.py
 N is the number of copies in parallel.
 
## Project solved Using MPI
+ Kmean
+ Vector Multiplication
+ Vector Addition
