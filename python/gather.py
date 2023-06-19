from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Data to gather
local_data = np.arange(rank * 2, (rank + 1) * 2)  # Generate local data
print(f"Data in Rank {rank} : {local_data}")

# Gather data from all processes
gathered_data = None
if rank == 0:
    gathered_data = np.empty(size * 2, dtype=int)  # Allocate array for gathered data

comm.Gather(local_data, gathered_data, root=0)

# Print the gathered data on the root process
if rank == 0:
    print(f"Rank {rank}: Gathered data: {gathered_data}")
