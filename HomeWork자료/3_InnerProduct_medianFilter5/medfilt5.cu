
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include <stdio.h>


cudaError_t median5Cuda(unsigned char* input, unsigned char* output, int xSize, int ySize);

void median5C(unsigned char* input, unsigned char* output, int xSize, int ySize);

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


void median5C(unsigned char* input, unsigned char* output, int xSize, int ySize)
{
	int n[5];
	int temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
	int i, j;
	int xi, xj;
	int x_count;
	int y_count;
	for (y_count = 0; y_count < ySize - 4; y_count++)
	{
		for (x_count = 0; x_count < xSize; x_count++)
		{

			n[0] = *(input + (y_count)*xSize + x_count + 0);
			n[1] = *(input + (y_count)*xSize + x_count + 1);
			n[2] = *(input + (y_count)*xSize + x_count + 2);
			n[3] = *(input + (y_count)*xSize + x_count + 3);
			n[4] = *(input + (y_count)*xSize + x_count + 4);

			unsigned char temp = 0;
			int j, k;
			for (j = 0; j < 5; j++)
			{
				for (k = 0; k < 5 - j - 1; k++)
				{
					if (n[k] > n[k + 1])
					{
						temp = n[k];
						n[k] = n[k + 1];
						n[k + 1] = temp;
					}
				}
			}
			*(output + y_count * xSize + x_count) = n[2];



			/*temp0 = (n[0] < n[1]) ? n[0] : n[1];
			temp1 = (n[2] < n[3]) ? n[2] : n[3];
			temp2 = (temp0 > temp1) ? temp0 : temp1;

			temp3 = (n[0] > n[1]) ? n[0] : n[1];
			temp4 = (n[2] > n[3]) ? n[2] : n[3];
			temp5 = (temp3 < temp4) ? temp3 : temp4;

			temp6 = (n[4] < temp2) ? n[4] : temp2;
			temp7 = (n[4] > temp2) ? n[4] : temp2;
			temp8 = (temp5 < temp7) ? temp5 : temp7;
			temp9 = (temp6 > temp8) ? temp6 : temp8;

			*(output + y_count * xSize + x_count) = temp9;*/

		}
	}
}

__global__ void kernelmedian5Cuda(unsigned char* input, unsigned char* output, int size)
{
	int xWidth = blockDim.x * gridDim.x;
	int yWidth = blockDim.y * gridDim.y;
	//printf("blockDim.x = %d gridDim.x=%d \n", blockDim.x, gridDim.x);
	int xLoc = (blockIdx.x * blockDim.x + threadIdx.x);
	int yLoc = blockIdx.y * blockDim.y + threadIdx.y;
	int index = xLoc + yLoc * xWidth;
	unsigned char value0, value1, value2, value3, value4;
	// share memory? 
	//__shared__ unsigned char value[5];
	if (index + 4 < size)
	{
		value0 = input[index + 0];
		value1 = input[index + 1];
		value2 = input[index + 2];
		value3 = input[index + 3];
		value4 = input[index + 4];

		//sort
		unsigned char  temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
		temp0 = (value0 < value1) ? value0 : value1;
		temp1 = (value2 < value3) ? value2 : value3;
		temp2 = (temp0 > temp1) ? temp0 : temp1;

		temp3 = (value0 > value1) ? value0 : value1;
		temp4 = (value2 > value3) ? value2 : value3;
		temp5 = (temp3 < temp4) ? temp3 : temp4;

		temp6 = (value4 < temp2) ? value4 : temp2;
		temp7 = (value4 > temp2) ? value4 : temp2;
		temp8 = (temp5 < temp7) ? temp5 : temp7;
		temp9 = (temp6 > temp8) ? temp6 : temp8;

		output[index] = temp9;


	}
}
int main()
{
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

	fp = fopen("barbara_gray.raw", "rb");

	fread(input, xSize, ySize, fp);

	/*for (int i = 0; i < ySize; i++)
		for (int j = 0; j < xSize; j++)
			input[i * xSize + j] = (i * j) % 255;
*/
	median5C(input, GoldOutput, xSize, ySize);
	// Add vectors in parallel.
	cudaError_t cudaStatus = median5Cuda(input, CudaOutput, xSize, ySize);
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

	fp = fopen("CudaOutput.raw", "w");
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
cudaError_t median5Cuda(unsigned char* input, unsigned char* output, int xSize, int ySize)
{
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
	kernelmedian5Cuda << <blocks, threads >> > (dev_input, dev_output, xysize);

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


