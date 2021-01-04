
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "book.h"
#include "cpu_anim.h"

#include <stdio.h>

#define DIM 1024

struct DataBlock {
	unsigned char* dev_bitmap;
	CPUAnimBitmap* bitmap;
};

// GPU에 할당한 메모리를 해제한다.
void cleanup(DataBlock* d) {
	cudaFree(d->dev_bitmap);
}

__global__ void kernel(unsigned char* ptr, int ticks) {
	// threadIdx/blockIdx로 픽셀 위치를 결정한다.
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// 이제 해당 위치의 값을 계산한다.
	float fx = x - DIM / 2;
	float fy = y - DIM / 2;
	float d = sqrtf(fx * fx + fy * fy);
	unsigned char grey = (unsigned char)(128.0f + 127.f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));

	ptr[offset * 4 + 0] = grey;
	ptr[offset * 4 + 1] = grey;
	ptr[offset * 4 + 2] = grey;
	ptr[offset * 4 + 3] = 255;
}

void generate_frame(DataBlock* d, int ticks) {
	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel << <blocks, threads >> > (d->dev_bitmap, ticks);

	HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost));
}

int main(void) {
	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.bitmap = &bitmap;
	HANDLE_ERROR(cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size()));
	
	bitmap.anim_and_exit((void(*)(void*, int))generate_frame, (void(*)(void*))cleanup);
}