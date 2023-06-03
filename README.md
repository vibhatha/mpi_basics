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
cd cpp
mpirun -n 4 ./hello_world
```

### Python

```bash
mpirun -n 4 python3 hello_world.py
```