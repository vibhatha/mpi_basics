# CUDA Programming

## Hello World


```c++
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
```

```bash
nvcc hello_world.cu -o hello_world
```

```bash
./hello_world
```

## Calculate on GPU Simple Example


## Array multiplication on GPU 


```c++
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
```

### Explanation

The code example demonstrates how to take an array of integers, copy it to a GPU, double each element using CUDA's internal math function, and then copy the array back to the CPU. Let's go through it part by part:

### Header Files

```c
#include <stdio.h>
```

This includes the standard I/O library, which allows us to use functions like `printf()` for printing to the console.

### CUDA Kernel

```c
__global__ void doubleArray(int *array, int arraySize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < arraySize) {
        array[idx] = __mul24(array[idx], 2);
    }
}
```

- `__global__` indicates that this is a CUDA kernel function, which is executable on the GPU.
- `doubleArray(int *array, int arraySize)` is the function that takes a pointer to an integer array and the size of that array as arguments.
- `int idx = threadIdx.x + blockIdx.x * blockDim.x;` calculates the unique index for each thread in the entire grid. This index is used to access the array elements.
- `if (idx < arraySize)` ensures that the thread index does not go out of array bounds.
- `__mul24(array[idx], 2);` doubles the element at `array[idx]`. The function `__mul24()` performs a 24-bit integer multiplication, which can be faster than standard multiplication on some GPUs.

### Main Function

```c
int main() {
    const int arraySize = 10;
    int h_array[arraySize]; // host array
    int *d_array; // device array pointer
```

- `const int arraySize = 10;` sets the size of the array to 10.
- `int h_array[arraySize];` declares an integer array `h_array` on the host (CPU).
- `int *d_array;` declares an integer pointer `d_array` for device (GPU) memory.

```c
for (int i = 0; i < arraySize; ++i) {
    h_array[i] = i + 1;
}
```

This loop initializes the elements of `h_array` with values from 1 to 10.

```c
cudaMalloc((void**)&d_array, arraySize * sizeof(int));
```

`cudaMalloc()` allocates memory on the device. Here, we allocate enough space to hold 10 integers.

```c
cudaMemcpy(d_array, h_array, arraySize * sizeof(int), cudaMemcpyHostToDevice);
```

`cudaMemcpy()` copies the array from the host to the device. The last argument specifies the direction of the copy, which is from host to device here.

```c
doubleArray<<<1, arraySize>>>(d_array, arraySize);
```

This launches the `doubleArray` kernel on the GPU. Here, we use 1 block with `arraySize` threads. Each thread will handle one array element.

```c
cudaMemcpy(h_array, d_array, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
```

This copies the modified array back from the device to the host after the kernel execution.

```c
printf("Doubled array values are: ");
for (int i = 0; i < arraySize; ++i) {
    printf("%d ", h_array[i]);
}
printf("\n");
```

This prints the doubled array values to the console.

```c
cudaFree(d_array);
```

This frees the device memory that we allocated earlier.

```c
return 0;
}
```

This indicates successful execution of the program.

The program uses some of the core CUDA API functions (`cudaMalloc`, `cudaMemcpy`, and `cudaFree`) and launches a kernel (`doubleArray`) to demonstrate basic GPU programming. It's a straightforward example, but it covers some essential aspects of CUDA programming, such as memory management and kernel launching.

## NCCL for GPU Programming

Like OpenMPI for CPU distributed computing, we have NCCL for distributed computing with NVIDIA GPU devices. 


```bash
#include <stdio.h>
#include <nccl.h>

int main(int argc, char* argv[])
{
    // Initialize NCCL
    ncclComm_t comm;
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclCommInitRank(&comm, 1, id, 0);

    // Allocate and initialize host and device arrays
    const int N = 10;
    float *h_array1 = new float[N];
    float *h_array2 = new float[N];

    for (int i = 0; i < N; ++i) {
        h_array1[i] = 1.0f;
        h_array2[i] = 2.0f;
    }

    float *d_array1, *d_array2;
    cudaMalloc(&d_array1, N * sizeof(float));
    cudaMalloc(&d_array2, N * sizeof(float));
    cudaMemcpy(d_array1, h_array1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, h_array2, N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform reduce operation using NCCL
    ncclReduce(d_array1, d_array2, N, ncclFloat, ncclSum, 0, comm, cudaStreamDefault);

    // Copy result back to host
    cudaMemcpy(h_array2, d_array2, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < N; ++i) {
        printf("%f ", h_array2[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_array1);
    cudaFree(d_array2);
    delete[] h_array1;
    delete[] h_array2;
    ncclCommDestroy(comm);

    return 0;
}
```


```bash
nvcc nccl_hello_world.cpp -o nccl_hello_world -lnccl
```


```bash
./nccl_hello_world
```

Output


```bash
3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000

```
