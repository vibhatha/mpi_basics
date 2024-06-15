#include <iostream>
#include <random>
#include <format>

#include "mpi.h"

using namespace std;

void mpi_send_number(int number_to_send, int destination, int rank);

void mpi_receive(int *receive, int source, int rank);

int main(int argc, char **argv) {
    random_device rd;
    mt19937 generator(rd());
    uniform_int_distribution<int> int_distribution(1, 100);

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        cerr << "At least 2 processes should be provided." << std::endl;
        return EXIT_FAILURE;
    }

    auto next_rank = (rank + 1) % world_size;
    auto previous_rank = (rank - 1 + world_size) % world_size;
    auto random_int = int_distribution(generator);
    int receive_val;


    mpi_send_number(random_int, next_rank, rank);
    mpi_receive(&receive_val, previous_rank, rank);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

void mpi_send_number(int number_to_send, int destination, int rank) {
    cout << format("{}. Sending to rank {} ", rank, destination) << endl;
    MPI_Send(&number_to_send, 1, MPI_INT, destination, 0, MPI_COMM_WORLD);
    cout << format("{}. sent number {} to rank {}", rank, number_to_send, destination) << endl;
}

void mpi_receive(int *receive, int source, int rank) {
    cout << format("{}. Receiving from rank {}", rank, source) << endl;
    MPI_Recv(receive, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    cout << format("{}. received number {} from rank {}", rank, *receive, source) << endl;
}
