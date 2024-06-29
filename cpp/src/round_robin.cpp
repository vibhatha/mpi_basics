#include <iostream>
#include <random>

#include "mpi.h"

void mpi_send_number(int number_to_send, int destination, int rank);

void mpi_receive(int *receive, int source, int rank);

int main(int argc, char **argv) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> int_distribution(1, 100);

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        std::cerr << "At least 2 processes should be provided." << std::endl;
        return EXIT_FAILURE;
    }

    int next_rank = (rank + 1) % world_size;
    int previous_rank = (rank - 1 + world_size) % world_size;
    int random_int = int_distribution(generator);
    int receive_val;


    mpi_send_number(random_int, next_rank, rank);
    mpi_receive(&receive_val, previous_rank, rank);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

void mpi_send_number(int number_to_send, int destination, int rank) {
    std::cout << rank << ". Sending to rank  " << destination << std::endl;
    MPI_Send(&number_to_send, 1, MPI_INT, destination, 0, MPI_COMM_WORLD);
    std::cout << rank << ". sent number " << number_to_send << " to rank " << destination << std::endl;
}

void mpi_receive(int *receive, int source, int rank) {
    std::cout << rank << ". Receiving from rank " << source << std::endl;
    MPI_Recv(receive, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::cout << rank << ". received number " << *receive << " from rank " << source << std::endl;
}
