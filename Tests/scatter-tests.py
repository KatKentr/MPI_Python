from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf = None
if rank == 0:
    sendbuf = np.empty([size, 10], dtype='i')
    print("sendbuf is: ",sendbuf)
    sendbuf.T[:,:] = range(size)
recvbuf = np.empty(10, dtype='i')
comm.Scatter(sendbuf, recvbuf, root=0)
print("my rank is",rank,"recv buf is: ",recvbuf)
assert np.allclose(recvbuf, rank)