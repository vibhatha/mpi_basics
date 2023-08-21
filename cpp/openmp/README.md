# Introduction to OpenMP

## Hello World

```c++
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
```

Compile your code

```bash
g++ -fopenmp hello_world_openmp.cpp -o hello_world_openmp
```

Run your code

```bash
./hello_world_openmp
```

Output

```bash
Hello, World! from thread 0 of 4
Hello, World! from thread 1 of 4
Hello, World! from thread 2 of 4
Hello, World! from thread 3 of 4

```

## Sum Vector 

```c++
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
```


```bash
g++ sum_vector.cpp -o sum_vector
```

## Parallel Sum Vector using OpenMP

```c++
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
```

```bash
g++ -fopenmp parallel_sum_vector.cpp -o parallel_sum_vector
```

```bash
./parallel_sum
```

Output (Sample) after running both sequential and parallel version

```bash
Sequential sum: 100000000
Time taken by sequential code: 564 milliseconds

Parallel sum: 100000000
Time taken by parallel code: 44 milliseconds
```

