#include <stdio.h>

// CUDA Kernel to double an integer value
__global__ void doubleValue(int *value) {
    *value *= 2;
}

int main() {
    int h_value = 10; // host value
    int *d_value; // device value pointer

    // Allocate device memory
    cudaMalloc((void**)&d_value, sizeof(int));

    // Copy host value to device value
    cudaMemcpy(d_value, &h_value, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to double the value
    doubleValue<<<1, 1>>>(d_value);

    // Copy result back to host
    cudaMemcpy(&h_value, d_value, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the doubled value
    printf("Doubled value is: %d\n", h_value);

    // Free device memory
    cudaFree(d_value);

    return 0;
}

