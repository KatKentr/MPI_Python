#Execution: mpiexec -n <number_of_processes> python back_sub_mpi.py
#number of processes equals to problem size


import sys, time
from mpi4py import MPI
import numpy as np

#Backward substitution problem: Attempt for a pipeline: Number of processes equals problem size?

def main():

    comm=MPI.COMM_WORLD
    # MPI-related data
    rank=comm.Get_rank()
    nprocs=comm.Get_size()

    n=nprocs             #problem size equals number of processes (Can we specify the number of processes according to problem size?)

    x = np.zeros(n)

    if rank==0:         # initialize tables

       a = np.random.random((n,n))
       a*= np.tri(*a.shape)
       b = np.random.random(n)

       start_time=time.time()            #start timing

       # print("a is: ",a)  debugging purposes
       # print("b is: ",b)
       # print("x is: ",x)

    else:

        a=np.zeros((n,n))
        b=np.zeros(n)

    comm.Bcast(a,root=0)    #seems inefficient though, many communications (Could we pack both arrays in a buffer somehow( in C: MPI_datatype) and brodcast the buffer?)
    comm.Bcast(b,root=0)

    # print(" in rank",rank," array a,b is: ",a,b)
    sum=0.0

    if rank!=0:          #Recieve arrays with Xs

        comm.Recv(x,source=rank-1,tag=22)
        # print("Recieved x from process ",rank-1,"sending to process ",rank+1)

    for j in range (0, rank, 1):     #computation
        sum+= x[rank]*a[rank, j]
    x[rank] = (b[rank] - sum) / a[rank,rank]   #x calculated in this rank

    if (rank!=nprocs-1):   #if the process is not the last one

        comm.Send(x,dest=rank+1,tag=22)

    if (rank==nprocs-1):

        # print(" in process ",rank, "x is: ",x)
        comm.Send(x,dest=0,tag=11)        #send final result back to process 0

    if (rank==0):

        comm.Recv(x,source=nprocs-1,tag=11)
        tot_time=time.time() -start_time
        # print("process ",rank," x is: ",x)
        print("Time %s sec " % tot_time)




if __name__ == '__main__':
    main()



