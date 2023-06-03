from mpi4py import MPI

def main():
    # Initialize the MPI environment
    comm = MPI.COMM_WORLD

    # Get the rank of the process and the size of the MPI_COMM_WORLD communicator
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Print a hello world message from each process
    print(f"Hello from process {rank} out of {size}")

if __name__ == "__main__":
    main()
