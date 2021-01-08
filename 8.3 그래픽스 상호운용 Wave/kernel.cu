#define GL_GLEXT_PROTOTYPES
#include "GL\glew.h"
#include "GL\glut.h"

#include "cuda.h"
#include "cuda_gl_interop.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "book.h"

#include <math.h>

#define DIM 512

static void draw_func(void) {
	glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}

static void key_func(unsigned char key, int x, int y);

struct GPUAnimBitmap;

GPUAnimBitmap* static_bitmap;
static GPUAnimBitmap** get_bitmap_ptr() {
	return &static_bitmap;
}

struct DataBlock {
	unsigned char* dev_bitmap;
	GPUAnimBitmap* bitmap;
};

// GLUT 콜백을 위한 정적 함수 사용
static void idle_func(void);

// GPU에 할당한 메모리를 해제한다.
void cleanup(DataBlock* d);

struct GPUAnimBitmap {
	GLuint bufferObj;
	cudaGraphicsResource* resource;
	int width, height;
	DataBlock* dataBlock;
	void (*fAnim)(uchar4*, void*, int);
	void (*animExit)(void*);
	void (*clickDrag)(void*, int, int, int, int);
	int dragStartX, dragStartY;

	GPUAnimBitmap(int w, int h, DataBlock* d) {
		width = w;
		height = h;
		dataBlock = d;
		clickDrag = NULL;

		// 우선 하나의 CUDA 디바이스를 찾은 후, 그래픽 상호운용을 설정한다.
		cudaDeviceProp prop;
		int dev;
		memset(&prop, 0, sizeof(cudaDeviceProp));
		prop.major = 1;
		prop.minor = 0;
		HANDLE_ERROR(cudaChooseDevice(&dev, &prop));

		HANDLE_ERROR(cudaGLSetGLDevice(dev));

		int c = 1;
		char* foo = "name";
		glutInit(&c, &foo);

		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
		glutInitWindowSize(width, height);
		glutCreateWindow("bitmap");

		glewInit();

		glGenBuffers(1, &bufferObj);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);

		glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * 4, NULL, GL_DYNAMIC_DRAW_ARB);

		HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone));

		dataBlock = new DataBlock();
		dataBlock->bitmap = this;
		cudaMalloc((void**)&dataBlock->dev_bitmap, width * height * 4);
	}

	~GPUAnimBitmap() {
		cleanup(dataBlock);
	}

	void anim_and_exit(void (*f)(uchar4*, void*, int), void(*e)(void*)) {
		fAnim = f;

		static_bitmap = this;

		// GLUT를 설정하고 주 루프를 시작한다.
		glutKeyboardFunc(key_func);
		glutDisplayFunc(draw_func);
		glutIdleFunc(idle_func);
		glutMainLoop();
	}
};

static void key_func(unsigned char key, int x, int y) {
	switch (key) {
	case 27:
		// OpenGL과 CUDA를 정리한다.
		GPUAnimBitmap * bitmap = *(get_bitmap_ptr());
		HANDLE_ERROR(cudaGraphicsUnregisterResource(bitmap->resource));
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffers(1, &bitmap->bufferObj);
		exit(0);
	}
}

static void idle_func(void) {
	static int ticks = 1;
	GPUAnimBitmap* bitmap = *(get_bitmap_ptr());
	uchar4* devPtr;
	size_t size;
	HANDLE_ERROR(cudaGraphicsMapResources(1, &(bitmap->resource), NULL));
	HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, bitmap->resource));

	bitmap->fAnim(devPtr, bitmap->dataBlock, ticks++);

	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &(bitmap->resource), NULL));

	glutPostRedisplay();
}

// GPU에 할당한 메모리를 해제한다.
void cleanup(DataBlock* d) {
	cudaFree(d->dev_bitmap);
}

__global__ void kernel(uchar4* ptr, int ticks) {
	// threadIdx/blockIdx로 픽셀 위치를 결정한다.
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// 이제 해당 위치의 값을 계산한다.
	float fx = x - DIM / 2;
	float fy = y - DIM / 2;
	float d = sqrtf(fx * fx + fy * fy);
	unsigned char grey = (unsigned char)(128.0f + 127.f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));

	ptr[offset].x = grey;
	ptr[offset].y = grey;
	ptr[offset].z = grey;
	ptr[offset].w = 255;
}

void generate_frame(uchar4* pixels, DataBlock* d, int ticks) {
	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel << <blocks, threads >> > (pixels, ticks);
}

int main(void) {
	GPUAnimBitmap bitmap(DIM, DIM, NULL);

	bitmap.anim_and_exit((void(*)(uchar4*, void*, int))generate_frame, NULL);
}