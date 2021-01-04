
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "book.h"
#include "cpu_bitmap.h"

#include <stdio.h>

static const int DIM = 1000;

struct cuComplex {
    float r;
    float i;

    //cuComplex(float a, float b) : r(a), i(b){}

    //float magnitude2(void) { return r * r + i * i; }
    //cuComplex operator*(const cuComplex& a) {
    //    return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    //}
    //cuComplex operator+(const cuComplex& a) {
    //    return cuComplex(r + a.r, i + a.i);
    //}

    // 생성자도 __device__ 키워드를 넣어줘야한다.
    __device__ cuComplex(float a, float b) : r(a), i(b){}

    __device__ float magnitude2(void) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r + a.r, i + a.i);
    }
};

//int julia_cpu(int x, int y) {
//    const float scale = 1.5f;
//    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
//    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);
//
//    cuComplex c(-0.8f, 0.156f);
//    cuComplex a(jx, jy);
//
//    int i = 0;
//    for (i = 0; i < 200; ++i)
//    {
//        a = a * a + c;
//        if (a.magnitude2() > 1000)
//            return 0;
//    }
//
//    return 1;
//}

__device__ int julia(int x, int y) {
    const float scale = 1.5f;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8f, 0.156f);
    cuComplex a(jx, jy);

    int i = 0;
    for (i = 0; i < 200; ++i) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

//void kernel_cpu(unsigned char* ptr) {
//    for (int y = 0; y < DIM; ++y) {
//        for (int x = 0; x < DIM; ++x) {
//            int offset = x + y * DIM;
//
//            int juliaValue = julia_cpu(x, y);
//            ptr[offset * 4 + 0] = 255 * juliaValue;
//            ptr[offset * 4 + 1] = 0;
//            ptr[offset * 4 + 2] = 0;
//            ptr[offset * 4 + 3] = 255;
//        }
//    }
//}

__global__ void kernel(unsigned char* ptr) {
    // threadIdx/blockIdx로 픽셀 위치를 결정한다.
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // 이제 해당 위치의 값을 계산한다.
    int juliaValue = julia(x, y);
    ptr[offset * 4 + 0] = 255 * juliaValue;
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

int main()
{
    CPUBitmap bitmap(DIM, DIM);

    //unsigned char* ptr = bitmap.get_ptr();
    //kernel_cpu(ptr);

    unsigned char* dev_bitmap;

    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

    dim3 grid(DIM, DIM);
    kernel << <grid, 1 >> > (dev_bitmap);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

    bitmap.display_and_exit();

    cudaFree(dev_bitmap);

    return 0;
}
