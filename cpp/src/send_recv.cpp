#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    // Find out rank and size
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        std::cerr << "At least 2 processes should be provided." << std::endl;
        return EXIT_FAILURE;
    }

    int number;
    if (my_rank == 0) {
        number = 12345;
        // Send the number to process 1
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        std::cout << "Process 0 sent number " << number << " to process 1\n";
    } else if (my_rank == 1) {
        // Receive the number from process 0
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Process 1 received number " << number << " from process 0\n";
    }

    MPI_Finalize();
}
