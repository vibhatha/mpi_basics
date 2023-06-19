from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = numpy.array([0, 1, 2, 3, 4], dtype="i")
    print(f"Sending numpy data with MPI Types: {data}")
    comm.Send([data, MPI.INT], dest=1, tag=11)
elif rank == 1:
    data = numpy.empty(5, dtype="i")
    comm.Recv([data, MPI.INT], source=0, tag=11)
    print(f"Receiving data with MPI type {data}")

# automatic MPI datatype discovery
if rank == 0:
    data = numpy.array([10, 20, 30, 40], dtype=numpy.float64)
    print(f"Sending numpy data with numpy Types: {data}")
    comm.Send(data, dest=1, tag=15)
elif rank == 1:
    data = numpy.empty(4, dtype=numpy.float64)
    comm.Recv(data, source=0, tag=15)
    print(f"Receiving data with numpy type {data}")
