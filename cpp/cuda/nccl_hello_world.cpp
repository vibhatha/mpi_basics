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

