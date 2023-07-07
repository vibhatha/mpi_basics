#include <mpi.h>
#include <iostream>

#include "util.h"

int send_recv_array();
int send_recv_vector();
int simple_send_recv();

int main(int argc, char** argv) {
    if(argc > 0) {
        int arg1 = std::stoi(argv[1]);
        if (arg1 == 1) {
            simple_send_recv();
        } else if (arg1 == 2) {
            send_recv_array();
        } else if (arg1 == 3) {
            send_recv_vector();
        } else {
            std::cerr << "Unsupported option " << arg1 << std::endl;
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "No command-line argument provided.\n";
    }

    
}

int simple_send_recv() {
    MPI_Init(NULL, NULL);

    // Find out rank and size
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        std::cerr << "At least 2 processes should be provided." << std::endl;
        return EXIT_FAILURE;
    }

    int number;
    if (rank == 0) {
        number = 12345;
        // Send the number to process 1
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        std::cout << "Process 0 sent number " << number << " to process 1\n";
    } else if (rank == 1) {
        // Receive the number from process 0
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Process 1 received number " << number << " from process 0\n";
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

int send_recv_array() {
    MPI_Init(NULL, NULL);

    const int SOURCE =0;
    const int DEST = 1;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        std::cerr << "At least 2 processes should be provided." << std::endl;
        return EXIT_FAILURE;
    }

    if(rank == 0) {
        std::cout << "\nSending and Receving Standard Array" << std::endl;
    }
    const int SIZE = 5;
    if (rank==SOURCE) {
        int* data = new int[SIZE];
        for(int i=0; i < SIZE; i++) {
            data[i] = i * 10;
        }
        std::cout << "Sending Data : ";
        print_int_array(data, SIZE);
        std::cout << "From " << SOURCE << " to " << DEST << std::endl;
        MPI_Send(data, SIZE, MPI_INT, DEST, 0, MPI_COMM_WORLD);
    }

    else if (rank==DEST) {
        int* recv_data = new int[SIZE];
        MPI_Recv(recv_data, SIZE, MPI_INT, SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Receiving Data : ";
        print_int_array(recv_data, SIZE);
        std::cout << "From " << SOURCE << " to " << DEST << std::endl;
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

int send_recv_vector() {
    MPI_Init(NULL, NULL);
    const int SOURCE =0;
    const int DEST = 1;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        std::cerr << "At least 2 processes should be provided." << std::endl;
        return EXIT_FAILURE;
    }

    if(rank == 0) {
        std::cout << "\nSending and Receving Standard Vector" << std::endl;
    }
    const int SIZE = 5;
    if (rank==SOURCE) {
        std::vector<int> data= {0, 10, 20, 30, 40};
        std::cout << "Sending Data : ";
        print_int_vector(data);
        std::cout << "From " << SOURCE << " to " << DEST << std::endl;
        MPI_Send(data.data(), SIZE, MPI_INT, DEST, 0, MPI_COMM_WORLD);
    }

    else if (rank==DEST) {
        int* recv_data = new int[SIZE];
        MPI_Recv(recv_data, SIZE, MPI_INT, SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Receiving Data : ";
        print_int_array(recv_data, SIZE);
        std::cout << "From " << SOURCE << " to " << DEST << std::endl;
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
