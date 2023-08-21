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

    double start_time = MPI_Wtime();

    long long local_sum = 0;
    for (auto val : local_array) {
        local_sum += val;
    }

    long long global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        std::cout << "MPI Sum: " << global_sum << std::endl;
    }

    if (rank == 0) {
        std::cout << "Time taken: " << end_time - start_time << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
