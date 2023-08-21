#include <stdio.h>

// CUDA Kernel to double the values of an array
__global__ void doubleArray(int *array, int arraySize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < arraySize) {
        array[idx] = __mul24(array[idx], 2);
    }
}

int main() {
    const int arraySize = 10;
    int h_array[arraySize]; // host array
    int *d_array; // device array pointer

    // Initialize host array with values 1 to 10
    for (int i = 0; i < arraySize; ++i) {
        h_array[i] = i + 1;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_array, arraySize * sizeof(int));

    // Copy host array to device array
    cudaMemcpy(d_array, h_array, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to double the array elements
    // Using 1 block with arraySize threads for simplicity
    doubleArray<<<1, arraySize>>>(d_array, arraySize);

    // Copy result back to host
    cudaMemcpy(h_array, d_array, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the doubled values
    printf("Doubled array values are: ");
    for (int i = 0; i < arraySize; ++i) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_array);

    return 0;
}

