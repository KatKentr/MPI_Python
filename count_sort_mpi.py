#Execution: mpiexec -n <number_of_processes> python count_sort_mpi.py <size_of_array>
#Poor performance, slower than the sequential execution

import random, sys, time
from mpi4py import MPI
import numpy as np


def main(argv):
    if len(argv) != 2:
        print('Usage: {} <list size>' .format(argv[0]))
        exit(1)
    else:
        try:
            arrSize = int(argv[1])
        except ValueError as e:
            print('Integer convertion error: {}' .format(e))
            exit(2)

    if arrSize <= 0:
        print('Steps cannot be non-positive.')
        exit(3)

    comm=MPI.COMM_WORLD
    # MPI-related data
    rank=comm.Get_rank()
    nprocs=comm.Get_size()


    if rank==0:

        x=np.random.randint(1,high=100,size=arrSize)   #create a random array to be processed
        result_y=np.zeros(arrSize,dtype=int)          #initialize an array to store the sorted array
        # print("initial array: ",x)
        start_time=time.time()
    else:
        x=np.zeros(arrSize,dtype=int)         #x needs to be initialized on the worker processes before Bcast is called
        result_y=None

    comm.Bcast(x,root=0)       #broadcasting a buffer like object x from the master process to all the worker processes

    # print('Process {} has data:'.format(rank), x)


    # determine the size of array block that each process will sort
    ave, res = divmod (x.size, nprocs)

    # determine the starting and ending array index for each process
    start = rank*ave
    stop = (rank+1)*ave if rank < nprocs-1 else (rank+1)*ave+res

    local_y=np.zeros(x.size,dtype=int)  #local array to store the results of each process

    for j in range (start,stop):
        my_num = x[j]
        my_place = 0
        for i in range (x.size):
            if  ((my_num > x[i]) or (my_num == x[i]) and (i < j)):   #each process scans the whole array and sorts its block of elements to the right position
                my_place += 1
        local_y[my_place] = my_num

    # print('Process {} has local_y:'.format(rank), local_y)

    comm.Reduce(local_y,result_y,op=MPI.SUM,root=0)          #combine the results to a global array (the ith element from each local array (local_y) are summed into the ith element in result array (result_y) of process 0
    # comm.Reduce((local_y,1,MPI.INT),(result_y,1,MPI.INT),root=0)  same performance

    if rank == 0:
        # print('After Reduce, sorted array on process 0 is:', result_y)
        print('array sorted in {:.3f} sec'.format(time.time() - start_time))



if __name__ == '__main__':
    main(sys.argv)
