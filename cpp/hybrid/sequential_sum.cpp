// sequential_sum.cpp
#include <iostream>
#include <vector>

long long sumArray(const std::vector<int>& array) {
    long long sum = 0;
    for (auto val : array) {
        sum += val;
    }
    return sum;
}

int main() {
    std::vector<int> array(100000000, 1);  // Array with 100 million 1's
    long long sum = sumArray(array);
    std::cout << "Sum: " << sum << std::endl;
    return 0;
}
