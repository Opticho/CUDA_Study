
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "book.h"
#include "lock.h"

#include <stdio.h>

#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(Lock lock, float* a, float* b, float* c) {
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float temp = 0;
	while (tid < N) {
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	// 캐시 값들을 설정한다.
	cache[cacheIndex] = temp;

	__syncthreads();

	// 다음 코드 때문에 리덕션을 위해서는 threadsPerBlock은 2의 멱수여야 한다.
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i)
		{
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0)
	{
		lock.lock();
		*c += cache[0];
		lock.unlock();
	}
}

int main(void) {
	Lock lock;

	float* a, * b, c = 0;
	float* dev_a, * dev_b, * dev_c;

	// CPU 측의 메모리를 할당한다.
	a = new float[N];
	b = new float[N];

	// GPU 메모리를 할당한다.
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(float)));

	// 호스트 메모리에 데이터를 채운다.
	for (int i = 0; i < N; ++i) {
		a[i] = i;
		b[i] = i * 2;
	}

	// 배열 'a'와 'b'를 GPU로 복사한다.
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_c, &c, sizeof(float), cudaMemcpyHostToDevice));

	dot << <blocksPerGrid, threadsPerBlock >> > (lock, dev_a, dev_b, dev_c);

	// 배열 'C'를 GPU에서 CPU로 복사한다.
	HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(float), cudaMemcpyDeviceToHost));

#define sum_squares(x) (x*(x+1)*(2*x+1)/6)
	printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares((float)(N - 1)));

	// GPU에 헬당된 메모리를 해제한다.
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	// CPU쪽에서 메모리를 해제한다.
	delete[] a;
	delete[] b;
}