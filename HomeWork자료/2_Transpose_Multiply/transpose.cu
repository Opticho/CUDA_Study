
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

#include <stdio.h>


cudaError_t transpose8WithCuda(unsigned char* input, unsigned char* output, int xSize, int ySize);

int verify(unsigned char* input, unsigned char* output, int xSize, int ySize);

int verify(unsigned char* GoldInput, unsigned char* CudaInput, int xSize, int ySize) {
	for (int i = 0; i < xSize * ySize; i++) {
		if (GoldInput[i] != CudaInput[i]) {
			printf("GoldInput[%d] = %d CInput[%d]=%d \n", i, GoldInput[i], i, CudaInput[i]);
			return(1);
		}
	}
	return(0);
}


void transpose8WithC(unsigned char* in, unsigned char* out, int x_size, int y_size) {
	size_t x, y;
	for (y = 0; y < y_size; y += 1) {
		for (x = 0; x < x_size; x += 1) {
			size_t in_pos = y * x_size + x;
			size_t out_pos = x * y_size + y;
			*(out + out_pos) = *(in + in_pos);
		}
	}
}


__global__ void kernelTranspose8(unsigned char* input, unsigned char* output, int size) {
	int xWidth = blockDim.x * gridDim.x;
	int xLoc = (blockIdx.x * blockDim.x + threadIdx.x);
	int yLoc = blockIdx.y * blockDim.y + threadIdx.y;

	int index = xLoc + yLoc * xWidth;

	if (index < (size)) {
		output[yLoc * xWidth + xLoc] = input[index];
	}
}

int main() {
	unsigned char* input, * CudaOutput, * GoldOutput;
	int xSize, ySize;

	xSize = 512;
	ySize = 512;
	input = new unsigned char[xSize * ySize];
	CudaOutput = new unsigned char[xSize * ySize];
	GoldOutput = new unsigned char[xSize * ySize];
	int i, j;
	printf("xSize=%d ySize=%d \n", xSize, ySize);

	FILE* fp;

	//fp = fopen("barbara_gray.raw", "rb");

	//fread(input, xSize, ySize, fp);

	for (int i = 0; i < ySize; i++)
		for (int j = 0; j < xSize; j++)
			input[i * xSize + j] = (i * j) % 255;

	transpose8WithC(input, GoldOutput, xSize, ySize);
	// Add vectors in parallel.
	cudaError_t cudaStatus = transpose8WithCuda(input, CudaOutput, xSize, ySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "invert8WithCuda failed!");
		return 1;
	}

	int error = verify(GoldOutput, CudaOutput, xSize, ySize);

	if (error != 0)
		printf("Verify Failed \n");
	else
		printf("Verify Successful \n");

	fp = fopen("COutput.raw", "wb");
	fwrite(GoldOutput, xSize, ySize, fp);
	fclose(fp);

	fp = fopen("CudaOutput.raw", "wb");
	fwrite(CudaOutput, xSize, ySize, fp);
	fclose(fp);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	delete[] CudaOutput;
	delete[] GoldOutput;
	delete[] input;

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t transpose8WithCuda(unsigned char* input, unsigned char* output, int xSize, int ySize) {
	unsigned char* dev_input = 0;
	unsigned char* dev_output = 0;

	//	cudaProfilerInitialize();
	unsigned int xysize = xSize * ySize;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.

	cudaDeviceProp prop;
	int count;

	dim3 blocks, threads;

	threads.x = 512;
	threads.y = 1;
	//512x512 : along X 512/8 = 64 thread blocks Alon gY 64 blocks
	blocks.x = (xSize + threads.x - 1) / (threads.x); //1
	blocks.y = (ySize + threads.y - 1) / (threads.y); //512
	printf("blocks.x = %d blocks.y=%d \n", blocks.x, blocks.y);
	printf("threads.x = %d threads.y=%d \n", threads.x, threads.y);


	cudaGetDeviceCount(&count);
	printf("Count =  %d\n", count);

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaEventRecord(start, 0);
	// Allocate GPU buffers for two input     .
	cudaStatus = cudaMalloc((void**)&dev_input, xysize * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_output, xysize * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_input, input, xysize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaProfilerStart();

	// Launch a kernel on the GPU with one thread for each element.
	kernelTranspose8 __dim__(blocks, threads) (dev_input, dev_output, xysize);

	cudaProfilerStop();
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching invert8Kernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(output, dev_output, xysize * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);


	float cudaElapsedTime;
	cudaEventElapsedTime(&cudaElapsedTime, start, stop);
	printf("Time for execution = %3.1f ms \n", cudaElapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

Error:
	cudaFree(dev_input);
	cudaFree(dev_output);

	return cudaStatus;
}


