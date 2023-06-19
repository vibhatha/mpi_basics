#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    const int INPUT = 100;
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(world_rank % 2 == 0) {
        // even
        auto val = INPUT * 2;
        std::cout << "Times 2: " << val << std::endl; 
    } else {
        // odd
        auto val = INPUT * 4;
        std::cout << "Times 4: " << val << std::endl;
    }

    // Print off a hello world message
    

    // Finalize the MPI environment.
    MPI_Finalize();
}
