#Execution: mpiexec -n <number_of_processes> python histogram_mpi.py <text_file>

import sys,time
from mpi4py import MPI
import numpy as np

def main(argv):

  comm=MPI.COMM_WORLD
    # MPI-related data
  rank=comm.Get_rank()
  nprocs=comm.Get_size()

  if rank==0:               #process input

    if len(argv) != 2:
        print('Usage: {} <file name>' .format(argv[0]))
        exit(1)
    else:
        filename = argv[1]

    print("file is: ",filename)
    text_file = open(filename, "r")
    # chars=text_file.read()      caused error
    # print("number of chars: ",len(chars))
    # print(type(chars))

    n=0
    buff=[]
    while True:
        char = text_file.read(1)
        if char:
            buff.append(char)
            n += 1
        else:
            break

    text_file.close()

    print("number of characters: ",len(buff))
    text_arr = np.array(buff)            #convert list to numpy array
    # print(text_arr.dtype)


    size_chunk=np.zeros(nprocs,dtype=np.int)   #holds the size of each subarray assigned to each process
    displ=np.zeros(nprocs,dtype=np.int)        #holds the start index(displacement) of each subarray assigned to each process

    start_time=time.time()

    # determine the size of each sub-task
    ave, res = divmod(text_arr.size, nprocs)

    for i in range(nprocs):

        # determine the starting and ending of each sub-task
        start = i*ave
        stop = (i+1)*ave if i< nprocs-1 else (i+1)*ave+res
        size_chunk[i]=stop-start     #size of each subtask
        displ[i]=start

    # list_new=np.array_split(text_arr,nprocs)  #its a list of arrays. Array is splitted in nprocs arrays. Each process will recieve a chuck of the initial array
    # arr_new=np.array(list_new)                #converts list to a numpy array with nprocs elements
    # print(len(arr_new[0]))

  else:

      text_arr=None
      size_chunk=np.zeros(nprocs,dtype=np.int)
      displ=None


  comm.Bcast(size_chunk,root=0)

  data=np.empty(size_chunk[rank],dtype='U')
  comm.Scatterv([text_arr,size_chunk,displ,MPI.INT],data,root=0)   #Scatter subarrays to processess


  # recbuf=np.empty(len(arr_new[rank]))   #(Investigation, if we could use something like this) allocating space for the recieving buffer array
  # print("rank ",rank,"my recbuf is: ",recbuf)
  # comm.Scatter(data, recvbuf, root=0)

  #scatter data
  # data=comm.scatter(arr_new,root=0)      #Each process will recieve an element(a subarray) of the array arr_new

  # print("my rank: ",rank,data.size)

  alphabetsize = 256
  local_hist=np.zeros(alphabetsize,dtype=int)   #initialize local histogram arrays

  for i in range(data.size):                     #computation for each process, results stored in local arrays local hist
      local_hist[ord(data[i])] += 1

  # print ("my rank", rank, "my local hist: ",local_hist)

  global_hist=None

  if rank==0:

    global_hist=np.empty(alphabetsize,dtype=int)   #allocate space for global histogram

  if rank==0:

      global_hist=local_hist
      comm.Reduce(MPI.IN_PLACE,global_hist,op=MPI.SUM,root=0)

  else:
      comm.Reduce(local_hist,global_hist,op=MPI.SUM,root=0)     #add result of the local histogram of each process to global histogram

  if rank==0:

      # print("in process: ",rank,"histogram of characters calculated: ",global_hist)
      print('histogram calculated in {:.3f} sec'.format(time.time() - start_time))


if __name__ == '__main__':
    main(sys.argv)