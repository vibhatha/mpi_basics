from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()  # Number of processes

# Load the data for each process
start_idx = rank * 10
end_idx = start_idx + 10
data = np.load("shuffled_numbers.npy")[start_idx:end_idx]

print(f"Initial Data on Rank {rank}:", data)

# Determine which numbers will go to which process in a round-robin fashion
to_send = {i: [] for i in range(size)}
for idx, num in enumerate(data):
    target_rank = (rank + idx) % size
    to_send[target_rank].append(num)

# Send count of numbers to each process and receive counts from other processes
send_counts = {i: len(to_send[i]) for i in range(size)}
recv_counts = comm.alltoall(send_counts.values())

print(f"Send Counts {rank} => {send_counts}")
print(f"Recv Counts {rank} => {recv_counts}")

# Send chunks of numbers to their target processes
for target_rank, numbers in to_send.items():
    if numbers:  # check if there are numbers to send
        comm.send(numbers, dest=target_rank)

# Receive numbers based on received counts
received_numbers = []
for src_rank, count in enumerate(recv_counts):
    if count > 0:  # check if there are numbers to receive
        numbers_from_src = comm.recv(source=src_rank)
        received_numbers.extend(numbers_from_src)

print(f"Round Robin Shuffled Data on Rank {rank}:", received_numbers)

# Make sure you run the script using mpiexec with the appropriate number of processes
