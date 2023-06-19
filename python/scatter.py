from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Data to scatter
if rank == 0:
    data = np.arange(size * 2)  # Generate data on root process
    print(f"Data to scatter {data}")
else:
    data = None

# Scatter the data to all processes
local_data = np.empty(2, dtype=int)
comm.Scatter(data, local_data, root=0)

# Print the received data on each process
print(f"Rank {rank}: Received data: {local_data}")
