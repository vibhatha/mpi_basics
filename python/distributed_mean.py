from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data_points = 8
features = 4

# Data to scatter
if rank == 0:
    data = np.arange(data_points * features)  # Generate data on root process
    data = np.reshape(data, (data_points, features))
    print(f"Data to scatter {data}")
else:
    data = None

# Scatter the data to all processes
local_data = np.empty((data_points//size, features), dtype=int)
comm.Scatter(data, local_data, root=0)

# Print the received data on each process
print(f"Rank {rank}: Received data: {local_data}")

local_mean = np.mean(local_data, axis=0)
global_mean = comm.allreduce(local_mean, op=MPI.SUM) / size

# Calculate Global Mean
print(f"Rank {rank}: Received Global Mean: {global_mean}")
