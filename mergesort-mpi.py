import random

from mpi4py import MPI
import numpy as np
from mergesort import *

def mergesort_mpi():
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    if rank==0:

        size=10
        arr=np.zeros(size, dtype=np.int)
        size_chunk=np.zeros(nprocs,dtype=np.int)   #holds the size of each subarray assigned to each process
        displ=np.zeros(nprocs,dtype=np.int)        #holds the start index(displacement) of each subarray assigned to each process

        for i in range (size):
            arr[i]=random.randint(10, 100)
        print(arr)

        # determine the size of each sub-task
        ave, res = divmod(size, nprocs)

        for i in range(nprocs):

            # determine the starting and ending of each sub-task
            start = i*ave
            stop = (i+1)*ave if i< nprocs-1 else (i+1)*ave+res
            size_chunk[i]=stop-start     #size of each subtask
            displ[i]=start


        print("size_chuck is: ",size_chunk)
        print("displ is: ",displ)
    else:
        arr=None
        size_chunk=np.zeros(nprocs,dtype=np.int)
        displ=None

    comm.Bcast(size_chunk,root=0)    #broadcast size_chunk array to all processes since local_array(recieve buffer) needs to be initialized in each process

    #allocate space for recbuf in each process
    local_array=np.zeros(size_chunk[rank],dtype=np.int)

    comm.Scatterv([arr,size_chunk,displ,MPI.INT],local_array,root=0)   #Scatterv works too with exactly the same arguments
    print('After Scatterv, process {} has data:'.format(rank), local_array)

    local_list=local_array.tolist()     #convert array to list
    mergeSort(local_list)               #perform the mergesort in each process



    #gather the sorted subarrays
    sendbuf2 = np.array(local_array)          #convert list to numpy array
    recvbuf2 = np.zeros(sum(size_chunk),dtype=np.int)
    comm.Gatherv(sendbuf2, [recvbuf2, size_chunk, displ, MPI.INT], root=0)

    #make the final mergesort call
    if rank==0:
        recvbuf2=recvbuf2.tolist()
        mergeSort(recvbuf2)
        printList(recvbuf2)
        print(recvbuf2)
        print(len(recvbuf2))

if __name__ == '__main__':
    mergesort_mpi()