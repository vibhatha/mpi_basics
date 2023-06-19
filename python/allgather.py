from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Data to allgather
local_data = np.array([rank] * 2, dtype=int)  # Generate local data
print(f"Data in Rank {rank} : {local_data}")

# Allgather data from all processes
allgathered_data = np.empty(size * 2, dtype=int)
comm.Allgather(local_data, allgathered_data)

# Print the allgathered data on each process
print(f"Rank {rank}: Allgathered data: {allgathered_data}")
