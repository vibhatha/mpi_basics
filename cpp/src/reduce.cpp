#include <mpi.h>
#include <iostream>
#include <limits>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int array_size = 4;
    int numbers[array_size];
    for (int i = 0; i < array_size; ++i) {
        numbers[i] = world_rank * 10 + i;
    }

    int local_sum = 0;
    int local_min = std::numeric_limits<int>::max();
    int local_max = std::numeric_limits<int>::min();
    for (int num : numbers) {
        local_sum += num;
        local_min = local_min >= num ? num : local_min;
        local_max = local_max >= num ? local_max : num;
    }

    int global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int global_min;
    MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    int global_max;
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << "Global sum = " << global_sum << " | Global min = " << global_min << " | Global max = " << global_max << std::endl;
    }

    std::cout << "Local sum = " << local_sum << " | Local min = " << local_min << " | Local max = " << local_max << " for the rank = " << world_rank << std::endl;

    MPI_Finalize();
    return 0;
}
