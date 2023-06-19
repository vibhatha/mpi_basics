from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

local_data = np.array([10, 30]) * (rank + 1)
print(f"Data in Rank {rank} : {local_data}")
# Calculate the local average
local_average = np.mean(local_data)
print(f"Local Average in Rank {rank}: >> {local_average}")

# Reduce all local averages to the root process (process `0`)
global_average = comm.reduce(local_average, op=MPI.SUM, root=0)

# Root process calculates the final average
if rank == 0:
    final_average = global_average / size
    print("Final average:", final_average)
    # local verification
    rank1_data = np.array([10, 20, 30, 40, 50])
    rank2_data = np.array([10, 20, 30, 40, 50]) * 2
