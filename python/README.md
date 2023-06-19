# MPI with Python

In this module we will be using `mpi4py` library to write MPI programmes in Python. 

## Example 1

A simple hello world programme, 

```bash
mpirun -n 4 python3 hello_world.py
```

## Example 2

Send data from one process to the other

```bash
mpirun -n 2 python3 simple_send_recv.py
```

### Example 3

Send data from one process to the other using Numpy

```bash
mpirun -n 2 python3 numpy_with_mpi.py
```
