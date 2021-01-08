
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "book.h"

#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

float malloc_test(int size);
float cuda_host_alloc_test(int size);

int main(void) {
	cudaDeviceProp prop;
	int whichDevice;
	HANDLE_ERROR(cudaGetDevice(&whichDevice));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
	if (prop.canMapHostMemory != 1) {
		printf("Device cannot map memory.\n");
		return 0;
	}

	HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));

	float elapsedTime = malloc_test(N);
	printf("Time using cudaMalloc: %3.1f ms\n", elapsedTime);

	elapsedTime = cuda_host_alloc_test(N);
	printf("Time using cudaHostAlloc: %3.1f ms\n", elapsedTime);
}

__global__ void dot(int size, float* a, float* b, float* c) {
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	float temp = 0;
	while (tid < size) {
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	// 캐시 값들을 설정한다.
	cache[cacheIndex] = temp;

	// 이 블록의 스레드들을 동기화한다.
	__syncthreads();

	// 다음 코드 때문에 리덕션을 위해서는 threadsPerBlock은 2의 멱수어야 한다.
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}

float malloc_test(int size) {
	cudaEvent_t start, stop;
	float* a, * b, c, * partial_c;
	float* dev_a, * dev_b, * dev_partial_c;
	float elapsedTime;

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	// CPU쪽에서 메모리를 할당한다.
	a = (float*)malloc(size * sizeof(float));
	b = (float*)malloc(size * sizeof(float));
	partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

	// GPU 메모리를 할당한다.
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float)));

	// 호스트 메모리에 데이터를 채운다.
	for (int i = 0; i < size; ++i) {
		a[i] = i;
		b[i] = i * 2;
	}

	HANDLE_ERROR(cudaEventRecord(start, 0));
	// 배열'a'와 'b'를 GPU로 복사한다.
	HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice));

	dot << <blocksPerGrid, threadsPerBlock >> > (size, dev_a, dev_b, dev_partial_c);

	// 배열 'c'를 GPU에서 다시 CPU로 복사한다.
	HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	// CPU쪽에서 마무리를 짓는다.
	c = 0;
	for (int i = 0; i < blocksPerGrid; ++i) {
		c += partial_c[i];
	}

	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_partial_c));

	// CPU쪽에서 메모리를 해제한다.
	free(a);
	free(b);
	free(partial_c);

	// 이벤트를 해제한다.
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

	printf("Value calculated: %d\n", c);

	return elapsedTime;
}

float cuda_host_alloc_test(int size) {
	cudaEvent_t start, stop;
	float* a, * b, c, * partial_c;
	float* dev_a, * dev_b, * dev_partial_c;
	float elapsedTime;

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	// CPU쪽에서 메모리를 할당한다.
	HANDLE_ERROR(cudaHostAlloc((void**)&a, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&b, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&partial_c, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped));

	// 호스트 메모리에 데이터를 채운다.
	for (int i = 0; i < size; ++i) {
		a[i] = i;
		b[i] = i * 2;
	}

	HANDLE_ERROR(cudaHostGetDevicePointer(&dev_a, a, 0));
	HANDLE_ERROR(cudaHostGetDevicePointer(&dev_b, b, 0));
	HANDLE_ERROR(cudaHostGetDevicePointer(&dev_partial_c, partial_c, 0));

	HANDLE_ERROR(cudaEventRecord(start, 0));

	dot << <blocksPerGrid, threadsPerBlock >> > (size, dev_a, dev_b, dev_partial_c);

	HANDLE_ERROR(cudaThreadSynchronize());

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	// CPU쪽에서 마무리를 짓는다.
	c = 0;
	for (int i = 0; i < blocksPerGrid; ++i) {
		c += partial_c[i];
	}

	HANDLE_ERROR(cudaFreeHost(a));
	HANDLE_ERROR(cudaFreeHost(b));
	HANDLE_ERROR(cudaFreeHost(partial_c));

	// 이벤트를 해제한다.
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

	printf("Value calculated: %d\n", c);

	return elapsedTime;
}