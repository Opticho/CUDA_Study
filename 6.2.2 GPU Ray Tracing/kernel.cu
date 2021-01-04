
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "book.h"
#include "cpu_bitmap.h"

#include <stdio.h>
#include <math.h>

#define INF 2e10f
#define rnd(x) (x * rand() / RAND_MAX)
#define SPHERES 20
#define DIM 1024

struct Sphere {
	float r, b, g;
	float radius;
	float x, y, z;

	__device__ float hit(float ox, float oy, float* n) {
		float dx = ox - x;
		float dy = oy - y;
		if (dx * dx + dy * dy < radius * radius) {
			float dz = sqrtf(radius * radius - dx * dx - dy * dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}
		return INF;
	}
};

// Sphere *s;
__constant__ Sphere s[SPHERES];

__global__ void kernel(unsigned char* ptr) {
	// threadIdx/blockIdx로 픽셀 위치를 결정한다.
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float ox = (x - DIM / 2);
	float oy = (y - DIM / 2);

	float r = 0, g = 0, b = 0;
	float minz = INF;
	for (int i = 0; i < SPHERES; ++i) {
		float n;
		float t = s[i].hit(ox, oy, &n);
		if (t < minz) {
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
		}
	}
	ptr[offset * 4 + 0] = (int)(r * 255);
	ptr[offset * 4 + 1] = (int)(g * 255);
	ptr[offset * 4 + 2] = (int)(b * 255);
	ptr[offset * 4 + 3] = 255;
}

int main(void) {
	// 시작 시간을 캡처(capture)한다.
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_bitmap;

	// 출력될 비트맵을 위해 GPU 메모리를 할당한다.
	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
	
	// 구 데이터를 위한 메모리를 할당한다.
	HANDLE_ERROR(cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES));

	// 임시 메모리를 할당하고 그것을 초기화 한 후에
	// GPU의 메모리로 복사한 후 임시 메모리를 해제한다.
	Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
	for (int i = 0; i < SPHERES; ++i) {
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(1000.0f) - 500;
		temp_s[i].y = rnd(1000.0f) - 500;
		temp_s[i].z = rnd(1000.0f) - 500;
		temp_s[i].radius = rnd(100.0f) + 20;
	}

	//HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES));
	HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES));

	free(temp_s);

	// 구 데이터로부터 비트맵을 생성한다.
	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel << <grids, threads >> > (dev_bitmap);

	// 화면에 출력하기 위해 비트맵을 GPU로부터 다시 복사한다.
	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	// 정지 시간을 얻은 후에 시간 측정 결과를 화면에 출력한다.
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));

	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("Time to generate: %3.1f ms\n", elapsedTime);

	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

	// 화면에 출력한다.
	//bitmap.display_and_exit();

	// 메모리를 해제한다.
	cudaFree(dev_bitmap);
	cudaFree(s);
}