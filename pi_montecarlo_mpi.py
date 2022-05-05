#Execution: mpiexec -n <number_of_processes> python pi_monte_carlo_mpi.py <number_of_steps>
#Notes: cyclic distribution resulted in wrong result-->investigation (something wrong in the foor loop)
import fileinput

from mpi4py import MPI
import time
import math
import random
import sys
import numpy as np

def main(argv):

    comm=MPI.COMM_WORLD
    # MPI-related data
    rank=comm.Get_rank()
    nprocs=comm.Get_size()

    if rank==0:         #process input

        if len(argv) != 2:
            print('Usage: {} <number of steps>'.format(argv[0]))
            exit(1)
        else:
            try:
                steps = int(argv[1])        #number of steps(tosses)
                # print("steps: ",steps)
                start_time=time.time()            #start timing
            except ValueError as e:
                print('Integer convertion error: {}'.format(e))
                exit(2)

        if steps <= 0:
            print('Steps cannot be non-positive.')
            exit(3)
    else:

        steps=None


    steps=comm.bcast(steps,root=0)         #broadcast steps to all processes

    my_count=0.0   #count of each process

    # compute partial contribution to pi on each process
    # determine the size of each sub-task
    ave, res = divmod (steps, nprocs)

    # determine the starting and ending of each sub-task
    start = rank*ave
    stop = (rank+1)*ave if rank < nprocs-1 else (rank+1)*ave+res

    # np.random.seed()   tried too, resulted in time increase

    for i in range(start,stop):
        # Randomly generated x and y values from a
        # uniform distribution
        # Rannge of x and y values is -1 to 1
        rand_x= random.uniform(-1, 1)
        # rand_x= random.random()      #results in even better performance
        rand_y= random.uniform(-1, 1)
        # rand_y= random.random()

        # Distance between (x, y) from the origin
        origin_dist= rand_x**2 + rand_y**2

        # Checking if (x, y) lies inside the circle
        if origin_dist<= 1:
            my_count+= 1

    # print("my rank is",rank," my count is: ",my_count)
    global_count=comm.reduce(my_count, op=MPI.SUM, root=0)    #collecting the results of each process in a global count

    if rank==0:
    # Estimating value of pi,
    # pi= 4*(no. of points generated inside the
    # circle)/ (no. of points generated inside the square)

       pi=4* global_count/steps

       stop_time=time.time()-start_time
       print('pi is: ',pi)

       print("Time %s sec " % stop_time)

       # f = open("monteCarlo-Output.txt", "a")          #writes output in a new line
       # text=['mpi program','with',str(nprocs),'processes','input',str(steps),'time:',str(stop_time),"\n"]
       # s=' '.join(text)
       # f.write(s)
       # f.close()


if __name__ == '__main__':
    main(sys.argv)
