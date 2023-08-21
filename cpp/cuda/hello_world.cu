#include <stdio.h>

// GPU Kernel function
__global__ void helloWorldFromGPU() {
    printf("Hello World from GPU!\n");
}

int main() {
    // Hello World from CPU
    printf("Hello World from CPU!\n");

    // Launch the kernel
    helloWorldFromGPU<<<1, 10>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    return 0;
}

