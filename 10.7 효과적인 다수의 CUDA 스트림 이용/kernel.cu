
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
		float as = (a[idx] + a[idx1] + a[idx2]) / 3.f;
		float bs = (b[idx] + b[idx1] + b[idx2]) / 3.f;
		c[idx] = (as + bs) / 2;
	}
}

int main(void) {
	cudaDeviceProp prop;
	int whichDevice;
	HANDLE_ERROR(cudaGetDevice(&whichDevice));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
	if (!prop.deviceOverlap) {
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
	cudaStream_t stream0, stream1;
	HANDLE_ERROR(cudaStreamCreate(&stream0));
	HANDLE_ERROR(cudaStreamCreate(&stream1));

	int* host_a, * host_b, * host_c;
	int* dev_a0, * dev_b0, * dev_c0;
	int* dev_a1, * dev_b1, * dev_c1;

	// GPU 메모리를 할당한다.
	HANDLE_ERROR(cudaMalloc((void**)&dev_a0, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b0, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c0, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_a1, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b1, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c1, N * sizeof(int)));

	// 스트림에서 이용하기 위해 잠긴 페이지의 메모리를 할당한다.
	HANDLE_ERROR(cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));

	for (int i = 0; i < FULL_DATA_SIZE; ++i) {
		host_a[i] = rand();
		host_b[i] = rand();
	}

	// 모든 데이터에 대해서 부분 크기 간격으로 루프를 돈다.
	for (int i = 0; i < FULL_DATA_SIZE; i += N * 2) {
		// a 복사 작업을 스트림 0과 스트림 1의 큐에 추가한다.
		HANDLE_ERROR(cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
		HANDLE_ERROR(cudaMemcpyAsync(dev_a1, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream1));

		// b 복사 작업을 스트림 0과 스트림 1의 큐에 추가한다.
		HANDLE_ERROR(cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
		HANDLE_ERROR(cudaMemcpyAsync(dev_b1, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream1));

		// 커널을 스트림 0과 스트림 1의 큐에 추가한다.
		kernel << < N / 256, 256, 0, stream0 >> > (dev_a0, dev_b0, dev_c0);
		kernel << < N / 256, 256, 0, stream1 >> > (dev_a1, dev_b1, dev_c1);

		//디바이스에서 잠긴 페이지의 메모리로 c 복사하는 작업을 큐에 추가한다.
		HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0));
		HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1));
	}

	// 잠긴 페이지의 메모리 버퍼에서 전체 버퍼로 부분의 결과를 복사한다.
	HANDLE_ERROR(cudaStreamSynchronize(stream0));
	HANDLE_ERROR(cudaStreamSynchronize(stream1));

	HANDLE_ERROR(cudaEventRecord(stop, 0));

	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("Time taken: %3.1f ms\n", elapsedTime);

	// 스트림과 메모리를 해제한다.
	HANDLE_ERROR(cudaFreeHost(host_a));
	HANDLE_ERROR(cudaFreeHost(host_b));
	HANDLE_ERROR(cudaFreeHost(host_c));
	HANDLE_ERROR(cudaFree(dev_a0));
	HANDLE_ERROR(cudaFree(dev_b0));
	HANDLE_ERROR(cudaFree(dev_c0));
	HANDLE_ERROR(cudaFree(dev_a1));
	HANDLE_ERROR(cudaFree(dev_b1));
	HANDLE_ERROR(cudaFree(dev_c1));
	HANDLE_ERROR(cudaStreamDestroy(stream0));
	HANDLE_ERROR(cudaStreamDestroy(stream1));

	return 0;
}