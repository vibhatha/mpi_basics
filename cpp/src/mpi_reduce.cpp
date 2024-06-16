#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int array_size = 4;
    int numbers[array_size];
    for (int i = 0; i < array_size; ++i) {
        numbers[i] = world_rank * 10 + i;
    }

    int local_sum = 0;
    for (int num : numbers) {
        local_sum += num;
    }

    int global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int global_min;
    MPI_Reduce(&local_sum, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    int global_max;
    MPI_Reduce(&local_sum, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << "Global sum = " << global_sum << std::endl;
        std::cout << "Minimum rank sum = " << global_min << std::endl;
        std::cout << "Maximum rank sum = " << global_max << std::endl;
    }

    std::cout << "Local sum = " << local_sum << " for the rank = " << world_rank << std::endl;

    MPI_Finalize();
    return 0;
}