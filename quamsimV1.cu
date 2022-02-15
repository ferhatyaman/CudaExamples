#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
evalGate(const float *A, const float *gate, float *B, int two_to_t)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int div = (int) (i / (two_to_t));
    int mod = i % (two_to_t);

    int first = div * (two_to_t << 1) + mod;
    int second = first + (two_to_t);

    B[first] = A[first] * gate[0] + A[second] * gate[1];
    B[second] = A[first] * gate[2] + A[second] * gate[3];

}



int main(int argc, char* argv[]) {
    FILE *FP;               // File handler
    char *trace_file;       // Variable that holds trace file name;

    clock_t start, end;
    float cpu_time_used;

    if (argc != 2)
    {
        printf("Error: Wrong number of inputs:%d\n", argc-1);
        exit(EXIT_FAILURE);
    }



    start = clock();


    trace_file          = argv[1];
    // Open trace_file in read mode
    FP = fopen(trace_file, "r");
    if(FP == NULL)
    {
        // Throw error and exit if fopen() failed
        printf("Error: Unable to open file %s\n", trace_file);
        exit(EXIT_FAILURE);
    }
    int file_size = 0, N = 0, N_copy;
    char line[15];
    while(fscanf(FP, "%s", line) != EOF)
        file_size++;
    int n = 0, t = 0, two_to_t = 0, i = 0;

    N = file_size - 5;
    // Calculate log2
    N_copy = N>>1;
    for (;N_copy > 0 ; n++)
        N_copy = N_copy >> 1;

    size_t size = N * sizeof(float);

    float *h_gate = (float *)malloc(4 * sizeof(float));
    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);
    // Allocate the host output vector B
    float *h_B = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_gate == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    fseek(FP, 0,SEEK_SET);
    fscanf(FP, "%f %f", &h_gate[0], &h_gate[1]);
    fscanf(FP, "%f %f", &h_gate[2], &h_gate[3]);
    for (; i < N; ++i)
        fscanf(FP, "%f", &h_A[i]);

    fscanf(FP, "%ul", &t);
    two_to_t = 1 << t;
//
//
//    i = 0;
//    for (; i < 4; ++i)
//        printf("%.3f\n", h_gate[i]);
//
//    i = 0;
//    for (; i < N; ++i)
//        printf("%.3f\n", h_A[i]);




    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector gate
    float *d_gate = NULL;
    err = cudaMalloc((void **)&d_gate, 4 * sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector gate (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
//    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_gate, h_gate, 4 * sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    int numElements = N >> 1;
    // Launch the Eval Gate CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
//    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    evalGate<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_gate, d_B, two_to_t);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch Eval Gate kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
//    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    i = 0;
    for (; i < N; ++i)
        printf("%.3f\n", h_B[i]);

//    FP = fopen(argv[2], "r");
//    if(FP == NULL)
//    {
//        // Throw error and exit if fopen() failed
//        printf("Error: Unable to open file %s\n", trace_file);
//        exit(EXIT_FAILURE);
//    }

//    float temp;
//
//    // Verify that the result vector is correct
//    for (int i = 0; i < N; ++i)
//    {
//        fscanf(FP, "%f", &temp);
//        if (fabs(h_B[i] - temp) > 1e-4)
//        {
//            fprintf(stderr, "fabs = %.10f, error = %.10f\n", fabs(h_B[i] - temp), 1e-4);
//            fprintf(stderr, "h_B[%d] = %.10f, temp = %.10f\n", i, h_B[i], temp);
//            fprintf(stderr, "Result verification failed at element %d!\n", i);
//            exit(EXIT_FAILURE);
//        }
//    }
//
//    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_gate);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector gate (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_gate);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

//    printf("Done\n");

    end = clock();
    cpu_time_used = ((float) (end - start)) / CLOCKS_PER_SEC;

    printf("Total time = %.3f\n", cpu_time_used);

    return 0;
}
