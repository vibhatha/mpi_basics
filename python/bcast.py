from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Example data
data = None
if rank == 0:
    data = np.arange(size)

# MPI_Bcast: Broadcasting data from root process to all other processes
data = comm.bcast(data, root=0)
print(f"Rank {rank}: Broadcasted data: {data}")
