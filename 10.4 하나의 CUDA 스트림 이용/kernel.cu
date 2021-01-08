
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "book.h"

#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)

__global__ void kernel(int* a, int* b, int* c) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < N) {
		int idx1 = (idx + 1) % 256;
		int idx2 = (idx + 2) % 256;
		float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
		float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
		c[idx] = (as + bs) / 2;
	}
}

int main(void) {
	cudaDeviceProp prop;
	int whichDevice;
	HANDLE_ERROR(cudaGetDevice(&whichDevice));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
	if (!prop.deviceOverlap){
		printf("Device will not handle overlaps, so no speed up from streams\n");
	}
	else {
		printf("Deprecated. Use instead asyncEngineCount\n");
	}

	if (prop.asyncEngineCount) {
		printf("Number of asynchronous engines %d\n", prop.asyncEngineCount);
	}

	cudaEvent_t start, stop;
	float elapsedTime;

	// 타이머를 작동시킨다.
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	// 스트림을 초기화한다.
	cudaStream_t stream;
	HANDLE_ERROR(cudaStreamCreate(&stream));

	int* host_a, * host_b, * host_c;
	int* dev_a, * dev_b, * dev_c;

	// GPU 메모리를 할당한다.
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// 스트림에서 이용하기 위해 잠긴 페이지의 메모리를 할당한다.
	HANDLE_ERROR(cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));

	for (int i = 0; i < FULL_DATA_SIZE; ++i) {
		host_a[i] = rand();
		host_b[i] = rand();
	}

	// 모든 데이터에 대해서 부분 크기 간격으로 루프를 돈다.
	for (int i = 0; i < FULL_DATA_SIZE; i += N) {
		// 비동기적으로 잠긴 페이지의 메모리를 디바이스로 복사한다.
		HANDLE_ERROR(cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream));
		HANDLE_ERROR(cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream));

		kernel << < N / 256, 256, 0, stream >> > (dev_a, dev_b, dev_c);

		//디바이스에서 잠긴 페이지의 메모리로 데이터를 복사한다.
		HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream));
	}

	// 잠긴 페이지의 메모리 버퍼에서 전체 버퍼로 부분의 결과를 복사한다.
	HANDLE_ERROR(cudaStreamSynchronize(stream));

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("Time taken: %3.1f ms\n", elapsedTime);

	// 스트림과 메모리를 해제한다.
	HANDLE_ERROR(cudaFreeHost(host_a));
	HANDLE_ERROR(cudaFreeHost(host_b));
	HANDLE_ERROR(cudaFreeHost(host_c));
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));

	HANDLE_ERROR(cudaStreamDestroy(stream));

	return 0;
}