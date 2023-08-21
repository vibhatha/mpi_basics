#include <iostream>
#include <vector>
#include <chrono>

// Function to sum up elements in a vector
long long sumVector(const std::vector<int>& vec) {
    long long sum = 0;
    for(int num : vec) {
        sum += num;
    }
    return sum;
}

int main() {
    const int SIZE = 100000000;
    std::vector<int> vec(SIZE, 1); // Initialize a vector of size SIZE with all elements as 1

    // Measure time taken by sequential sum
    auto start = std::chrono::high_resolution_clock::now();
    long long sum = sumVector(vec);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Sequential sum: " << sum << std::endl;
    std::cout << "Time taken by sequential code: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}

