# Writing Hybrid Applications

## Stage 1: Sequential Application

First, let's write a basic sequential application to sum an array.

```cpp
// sequential_sum.cpp
#include <iostream>
#include <vector>

long long sumArray(const std::vector<int>& array) {
    long long sum = 0;
    for (auto val : array) {
        sum += val;
    }
    return sum;
}

int main() {
    std::vector<int> array(100000000, 1);  // Array with 100 million 1's
    long long sum = sumArray(array);
    std::cout << "Sum: " << sum << std::endl;
    return 0;
}
```

Compile and run:

```bash
g++ sequential_sum.cpp -o sequential_sum
./sequential_sum
```

## Stage 2: MPI Application

Now, let's parallelize this using MPI.

```cpp
// mpi_sum.cpp
#include <iostream>
#include <vector>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 100000000;
    const int local_N = N / size;
    std::vector<int> local_array(local_N, 1);

    long long local_sum = 0;
    for (auto val : local_array) {
        local_sum += val;
    }

    long long global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "MPI Sum: " << global_sum << std::endl;
    }

    MPI_Finalize();
    return 0;
}
```

Compile and run:

```bash
mpic++ mpi_sum.cpp -o mpi_sum
mpirun -n 4 ./mpi_sum
```

## Stage 3: MPI + OpenMP Hybrid Application

Finally, let's make it a hybrid MPI+OpenMP application.

```cpp
// hybrid_sum.cpp
#include <iostream>
#include <vector>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 100000000;
    const int local_N = N / size;
    std::vector<int> local_array(local_N, 1);

    long long local_sum = 0;

    #pragma omp parallel for reduction(+:local_sum)
    for (int i = 0; i < local_N; ++i) {
        local_sum += local_array[i];
    }

    long long global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Hybrid MPI+OpenMP Sum: " << global_sum << std::endl;
    }

    MPI_Finalize();
    return 0;
}
```

Compile and run:

```bash
mpic++ -fopenmp hybrid_sum.cpp -o hybrid_sum
mpirun -n 4 ./hybrid_sum
```

## Performance Benchmark

You can measure the performance of these three applications by using timing mechanisms like `std::chrono` in C++ for the sequential version and `MPI_Wtime()` for MPI and MPI+OpenMP versions. Typically, the hybrid application will show the best performance by leveraging both data parallelism (across MPI nodes) and thread-level parallelism (within each node using OpenMP).

Here is how you can add timing to the MPI and hybrid MPI+OpenMP applications. Place the following lines before and after the computation:

```cpp
double start_time = MPI_Wtime();
// ... computation ...
double end_time = MPI_Wtime();
if (rank == 0) {
    std::cout << "Time taken: " << end_time - start_time << " seconds" << std::endl;
}
```

For the sequential version, you can use `std::chrono` as in the previous examples to measure the elapsed time.


```bash
$ ./mpi_sum 
MPI Sum: 100000000
Time taken: 0.510314 seconds
$ ./hybrid_sum 
Hybrid MPI+OpenMP Sum: 100000000
Time taken: 0.0266457 seconds
```