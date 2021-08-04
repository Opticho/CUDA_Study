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

GLuint bufferObj;
cudaGraphicsResource* resource;

// 물결 코드를 기반으로 하지만 그래픽 상호운용에서 사용하는 데이터 타입인 uchar4를 이용한다.
__global__ void kernel(uchar4* ptr) {
	// threadIdx,blockIdx로 픽셀 위치를 결정한다.
	int x = threadIdx.x + blockIdx.x * blockDim.x; 
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// 이제 해당 위치의 값을 계산한다.
	float fx = x / (float)DIM - 0.5f;
	float fy = y / (float)DIM - 0.5f;
	unsigned char green = 128 + 127 * sin(fabs(fx * 100) - fabs(fy * 100));

	// unsigned char*와 대조되는 uchar4에 접근한다.
	ptr[offset].x = 0;
	ptr[offset].y = green;
	ptr[offset].z = 0;
	ptr[offset].w = 255;
}

static void draw_func(void) {
	glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}

static void key_func(unsigned char key, int x, int y) {
	switch (key) {
	case 27:
		// OpenGL과 CUDA를 정리한다.
		HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffers(1, &bufferObj);
		exit(0);
	}
}

int main(int argc, char** argv) {
	cudaDeviceProp prop;
	int dev;

	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	HANDLE_ERROR(cudaChooseDevice(&dev, &prop));

	HANDLE_ERROR(cudaGLSetGLDevice(dev));

	// GL API들을 호출하기 전에 다음의 GLUT API들을 호출해야 한다.
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("bitmap");

	glewInit();

	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB);

	HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone));

	uchar4* devPtr;
	size_t size;
	HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, NULL));
	HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource));

	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel << <grids, threads >> > (devPtr);

	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));

	// GLUT를 설정하고 주 루프를 시작한다.
	glutKeyboardFunc(key_func);
	glutDisplayFunc(draw_func);
	glutMainLoop();
}