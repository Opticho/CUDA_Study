
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define  ARRAYSIZE  100
#define NUM_BLOCK 1
#define NUM_THREAD 128


void innerC(int* in, int* in2, int* out)
{
    int result = 0;
    for (int i = 0; i < ARRAYSIZE; i++)
    {
          result += in[i] * in2[i];
     }
        out[0] = result;
        
}

__global__ void KernelinnerCUDA(int* in, int* in2, int* out)
{   
    
    int idx = threadIdx.x;
    int jdx = blockIdx.x;
    int location, temp;
    location = jdx * NUM_THREAD + idx;
    // printf("[%d][%d] = %d \n", jdx, idx, location);
    out[location] = in[location]* in2[location];
    
}
/*
__global__ void KernelinnerCUDA(int* in)
{
    int idx = threadIdx.x;
    int jdx = blockIdx.x;
    int location, temp;
    location = jdx * NUM_THREAD + idx;
    printf("[%d][%d] = %d \n", jdx, idx, location);
    __shared__ int array[128];
    __shared__ int array2[128];
    array[location] = in[location];

    __syncthreads();

    if (location < ARRAYSIZE - 1) {
        temp = array[location + 1];
        __syncthreads();
        array[location] = temp;
        __syncthreads();
    }

    in[location] = array[location];
}
*/


cudaError_t innerCUDA(int* in, int* in2, int* outCUDA);

int main()
{
    int a[ARRAYSIZE], b[ARRAYSIZE], bC[ARRAYSIZE], bCUDA[ARRAYSIZE];
    int out2=0;


    printf("BEFORE \n\n\n");


    for (int i = 0; i < ARRAYSIZE; i++)
    {
        a[i] = 1;
        b[i] = i;
        bC[i] = 0;
    }



    // Swap in parallel.
    cudaError_t cudaStatus = innerCUDA(a, b, bCUDA);
    for (int i = 0; i < ARRAYSIZE; i++)
        out2 += bCUDA[i];
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "kernelSap failed!");
        return 1;
    }

    innerC(a, b, bC);
    printf("AFTER \n\n\n");
    for (int i = 0; i < ARRAYSIZE; i++) {
        printf("bCUDA[%d]=%d  bC[%d]=%d \n", i, bCUDA[i], i, bC[i]);
    }
    printf("bCUDA=%d  bC=%d \n", out2, bC[0]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t innerCUDA(int* a, int* b, int* outCUDA)
{
    int* dev_a = 0, * dev_b = 0, *dev_outCUDA;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_a, ARRAYSIZE * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_b, ARRAYSIZE * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_outCUDA, ARRAYSIZE * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, ARRAYSIZE * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, ARRAYSIZE * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    KernelinnerCUDA << <NUM_BLOCK, NUM_THREAD >> > (dev_a, dev_b, dev_outCUDA);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(outCUDA, dev_outCUDA, ARRAYSIZE * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:

    cudaFree(dev_a);


    return cudaStatus;
}
