from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Each process loads its chunk of 10 records
start_idx = rank * 10
end_idx = start_idx + 10
data = np.load("shuffled_numbers.npy")[start_idx:end_idx]

print(f"Initial Data on Rank {rank}:", data)

# Now, let's shuffle data among processes
ranges = [(0, 10), (10, 20), (20, 30), (30, 40)]

# Extract data that belongs to other processes and send it to them
for i, r in enumerate(ranges):
    if i == rank:
        continue

    mask = np.logical_and(data >= r[0], data < r[1])
    to_send = data[mask]
    data = data[~mask]
    
    comm.send(to_send, dest=i)

# Receive data from other processes
for i, r in enumerate(ranges):
    if i == rank:
        continue

    received_data = comm.recv(source=i)
    data = np.concatenate((data, received_data))

print(f"Partitioned Data on Rank {rank}:", data)

# Make sure you run the script using mpirun with 4 processes
