#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

// Function to sum up elements in a vector
long long sumVectorParallel(const std::vector<int>& vec) {
    long long sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < vec.size(); i++) {
        sum += vec[i];
    }
    
    return sum;
}

int main() {
    const int SIZE = 100000000;
    std::vector<int> vec(SIZE, 1); // Initialize a vector of size SIZE with all elements as 1

    omp_set_num_threads(4);

    // Measure time taken by parallel sum
    auto start = std::chrono::high_resolution_clock::now();
    long long sum = sumVectorParallel(vec);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Parallel sum: " << sum << std::endl;
    std::cout << "Time taken by parallel code: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}

