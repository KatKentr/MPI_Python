#Execution: mpiexec -n <number_of_processes> python jakobi_heat_mpi.py <problem_size>
# Notes: For the current implementation input size should be evenly divisable by the number of processes. Attempts to scatter the matrix among process with Scatterv resulted in incorrect results
# Further investigation: 1) how to scatter and gather uneven 2D matrices,2) implement non-blocking communication

import sys,time
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

def main(argv):

    comm=MPI.COMM_WORLD
    # MPI-related data
    rank=comm.Get_rank()
    nprocs=comm.Get_size()

    if rank==0:         #process input

        if len(argv) != 2:
            print('Usage: {} <problem size>'.format(argv[0]))
            exit(1)
        else:
            try:
                tableSize = int(argv[1])        #dimension of nxn table
                # print("steps: ",steps)
                start_time=time.time()            #start timing
            except ValueError as e:
                print('Integer convertion error: {}'.format(e))
                exit(2)

        if tableSize <= 0:
            print('Steps cannot be non-positive.')
            exit(3)
    else:

        tableSize=None


    tableSize=comm.bcast(tableSize,root=0)         #broadcast tableSize to all processes

    iterations = 200       #timesteps
    accuracy = 5

    ave, res = divmod(tableSize, nprocs)                    #each process will calculate a submatrix of the original matrix of size: myRows x tableSize

   #determine the starting and ending row index of each submatrix assigned to processs
    start=rank*ave
    stop=(rank+1)*ave if rank<nprocs-1 else (rank+1)*ave+res
    myRows=stop-start
    localTable1=np.zeros((myRows+2,tableSize))            #initialize submatrices for each process. Number of rows=myRows+ 2 extra rows that will store ghost points(the values of the border region cells, which belong to the neighboring process and are necessary for the compuattion of each process
    localTable2=np.zeros((myRows+2,tableSize))

    # print('process {} has data:'.format(rank), localTable1,"shape: ",localTable1.shape)

    i_first = 1;                #variables that store the indices of the inner points of each process(excluding ghost rows)
    i_last  = myRows;
    # if (rank == 0) :       i_first=i_first+1;    #in case we had an upper and lower boundary condition
    # if (rank == nprocs - 1) : i_last=i_last+1;

    ROW =myRows//2           #position of heat source(somewhere in the middle)
    COL = tableSize//2
    START = 200


    for i in range(0,iterations):

        #send and recieve border region of local submatrix
        if rank>0:

         comm.Send(localTable1[1,:],dest=rank-1,tag=11)
         # comm.Isend(localTable1[1,:],dest=rank-1,tag=11)    #non-blocking send, may be not correct though
         comm.Recv(localTable1[0,:],source=rank-1,tag=22)

        if rank<nprocs-1:

         comm.Recv(localTable1[myRows+1,:],source=rank+1,tag=11)
         comm.Send(localTable1[myRows,:],dest=rank+1,tag=22)
         # comm.Isend(localTable1[myRows,:],dest=rank+1,tag=22)     #non-blocking send

        if rank==nprocs//2:          #heat Source

            localTable1[ROW][COL]=START

        localDiff=0


        # Perform calculations
        for i in range(i_first, i_last+1):
           for j in range(1, tableSize-1):
               localTable2[i, j] = 0.25 * (localTable1[i-1, j] +
                                   localTable1[i+1, j] + localTable1[i, j-1] + localTable1[i, j+1])
               localDiff += (localTable2[i, j]-localTable1[i, j])*(localTable2[i, j]-localTable1[i, j])


        for i in range(i_first, i_last+1):
            for j in range(1,tableSize-1):
                localTable1[i][j]=localTable2[i][j].copy()

        globDiff=comm.allreduce(localDiff,op=MPI.SUM)   #sum the local diff calculated in each process in globaldiff.
        globDiff = np.sqrt(globDiff)

        # if rank==0:
        #     print(" At iteration: ",t," diff is: ",globDiff)

        if globDiff<accuracy:
            break

    recvbuf = np.zeros((tableSize,tableSize))
    comm.Gatherv(localTable1[i_first:i_last+1],recvbuf,root=0)      #gather the result submatrices to a global matrix in process 0

    if rank==0:

        endTime=time.time() - start_time
        print('solution calculated in in {:.3f} sec'.format(endTime))
        # plt.imshow(recvbuf)        #plot result
        # plt.show()



if __name__ == '__main__':
    main(sys.argv)




