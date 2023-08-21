#include <iostream>
#include <omp.h>

int main() {
    // Set the number of threads to 4
    omp_set_num_threads(4);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num(); // Get the thread ID
        int num_threads = omp_get_num_threads(); // Get the total number of threads

        std::cout << "Hello, World! from thread " << thread_id << " of " << num_threads << std::endl;
    }

    return 0;
}

