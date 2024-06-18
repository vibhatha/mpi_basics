#include "util.h"
#include <iostream>

void print_int_array(const int* array, int size) {
    for (int i=0; i < size; i++) {
        if(i != size - 1) {
            std::cout << array[i] << ", ";
        } else {
            std::cout << array[i];
        }
    }
    std::cout << std::endl;
}

void print_int_vector(const std::vector<int>& data) {
    size_t vec_size = data.size();
    for(size_t i=0; i < vec_size; i++) {
        if(i != vec_size - 1) {
            std::cout << data.at(i) << ", ";
        } else {
            std::cout << data.at(i);
        }
    }
    std::cout << std::endl;
}