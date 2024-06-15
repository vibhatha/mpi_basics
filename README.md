# mpi_basics

Dedicated to MPI Basics 

Make sure you have installed MPI in your system.  We will be using OpenMPI to write our examples.

## C++

```bash
cd cpp
mkdir build
cd build
cmake ../
make
```

## Python

Create a virtual environment

```bash
python3 -m venv mpi_env
source mpi_env/bin/activate
pip install mpi4py
```

## Run Examples

### C++

```bash
cd cpp\build
mpirun -n 4 ./hello_world
```

```bash
cd cpp\build
mpirun -n 4 ./send_recv <program-num>

Program Numbers:
1. Send Number 
2. Send Array
3. Send vector
```

### Python

```bash
mpirun -n 4 python3 hello_world.py
```
