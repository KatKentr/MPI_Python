#Execution: mpiexec -n <number_of_processes> python mergesort_sort_mpi.py <size_of_array>

import random
import sys
import fileinput
from mpi4py import MPI
import numpy as np
from mergesort import *

def mergesort_mpi(argv):

    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    if rank==0:

        if len(argv) != 2:
            print('Usage: {} <size of array>'.format(argv[0]))
            exit(1)
        else:
            try:
                arrSize = int(argv[1])        #size of the array


            except ValueError as e:
                print('Integer convertion error: {}'.format(e))
                exit(2)

        if arrSize <= 0:
            print('Size cannot be non-positive.')
            exit(3)

        arr=np.zeros(arrSize, dtype=np.int)        #initialize array of type int
        size_chunk=np.zeros(nprocs,dtype=np.int)   #holds the size of each subarray assigned to each process
        displ=np.zeros(nprocs,dtype=np.int)        #holds the start index(displacement) of each subarray assigned to each process

        for i in range (arrSize):                  #generate array of random integers
            arr[i]=random.randint(10, 100)
        # print(arr)

        start_time=time.time()             #start timing

        # determine the size of each sub-task
        ave, res = divmod(arrSize, nprocs)

        for i in range(nprocs):

            # determine the starting and ending index of each sub-task
            start = i*ave
            stop = (i+1)*ave if i< nprocs-1 else (i+1)*ave+res
            size_chunk[i]=stop-start     #size of each subtask
            displ[i]=start


        # print("size_chuck is: ",size_chunk)
        # print("displ is: ",displ)
    else:
        arr=None
        size_chunk=np.zeros(nprocs,dtype=np.int)
        displ=None

    comm.Bcast(size_chunk,root=0)    #broadcast size_chunk array to all processes since local_array(recieve buffer) needs to be initialized in each process

    #allocate memory space for recieving buffer in each process
    local_array=np.zeros(size_chunk[rank],dtype=np.int)

    comm.Scatterv([arr,size_chunk,displ,MPI.INT],local_array,root=0)   #Scatter subarrays to processess
    # print('After Scatterv, process {} has data:'.format(rank), local_array)


    local_array=mergeSort_np(local_array)   #perform the mergesort on each process


    #gather the sorted subarrays

    recvbuf2 = np.zeros(sum(size_chunk),dtype=np.int)
    comm.Gatherv(local_array, [recvbuf2, size_chunk, displ, MPI.INT], root=0)

    #make the final mergesort call
    if rank==0:

        recvbuf2=mergeSort_np(recvbuf2)
        # print(recvbuf2)
        # print(recvbuf2.size)

        stop_time=time.time()-start_time
        print("Time %s sec " % stop_time)

        # f = open("mergesort-Output.txt", "a")          #writes output in a new line
        # text=['mpi program','with',str(nprocs),'processes','input',str(arrSize),'time:',str(stop_time),"\n"]
        # s=' '.join(text)
        # f.write(s)
        # f.close()


if __name__ == '__main__':
    mergesort_mpi(sys.argv)