#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct Matrix
{
	int width;
	int height;
	double *elements;
};

__device__ float getElement(Matrix *A, int row, int col)
{
	return A->elements[row * A->width + col];
}

__device__ void setElement(Matrix *A, int row, int col, double value)
{
	A->elements[row * A->width + col] = value;
}

__global__ void matMulKernel(Matrix *A, Matrix *B, Matrix *C)
{
	float Cvalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = 0; i < A->width; ++i)
	{
		Cvalue += getElement(A, row, i) * getElement(B, i, col);
	}
	setElement(C, row, col, Cvalue);
}

int main()
{
 //   cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 128);
	int width = 1 << 10;
	int height = 1 << 10;
	Matrix *A, *B, *C, *D, *E, *F, *G;
	// 申请托管内存
	cudaError_t a = cudaMallocManaged((void**)&A, sizeof(Matrix));
	cudaError_t b = cudaMallocManaged((void**)&B, sizeof(Matrix));
	cudaError_t c = cudaMallocManaged((void**)&C, sizeof(Matrix));
    cudaMallocManaged((void**)&D, sizeof(Matrix));
    cudaMallocManaged((void**)&E, sizeof(Matrix));
    cudaMallocManaged((void**)&F, sizeof(Matrix));
    cudaMallocManaged((void**)&G, sizeof(Matrix));
	std::cout << cudaGetErrorString(a) << std::endl << cudaGetErrorString(b) << std::endl << cudaGetErrorString(c) << std::endl;
	int nBytes = width * height * sizeof(double);
	cudaError_t d = cudaMallocManaged((void**)&A->elements, nBytes);
	cudaError_t e = cudaMallocManaged((void**)&B->elements, nBytes);
	cudaError_t f = cudaMallocManaged((void**)&C->elements, nBytes);
	std::cout << cudaGetErrorString(d) << std::endl << cudaGetErrorString(e) << std::endl << cudaGetErrorString(f) << std::endl;
    cudaError_t d1 = cudaMallocManaged((void**)&D->elements, nBytes);
    std::cout << cudaGetErrorString(d1) << std::endl;
    cudaError_t e1 = cudaMallocManaged((void**)&E->elements, nBytes);
    std::cout << cudaGetErrorString(e1) << std::endl;
    cudaError_t f1 = cudaMallocManaged((void**)&F->elements, nBytes);
    std::cout << cudaGetErrorString(f1) << std::endl;
    cudaError_t g1 = cudaMallocManaged((void**)&G->elements, nBytes);
    std::cout << cudaGetErrorString(g1) << std::endl;
	// 初始化数据
	A->height = height;
	A->width = width;
	B->height = height;
	B->width = width;
	C->height = height;
	C->width = width;
	D->height = height;
	D->width = width;
	E->height = height;
	E->width = width;
	F->height = height;
	F->width = width;
	for (int i = 0; i < width * height; ++i)
	{
		A->elements[i] = 1.0;
		B->elements[i] = 2.0;
		D->elements[i] = 1.0;
		E->elements[i] = 2.0;
	}

	// 定义kernel的执行配置
	dim3 blockSize(32, 32);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y);
	// 执行kernel
	matMulKernel << < gridSize, blockSize >> > (A, B, C);
	matMulKernel << < gridSize, blockSize >> > (D, E, F);

	// 同步device 保证结果能正确访问
	cudaDeviceSynchronize();
	// 检查执行结果
	float maxError = 0.0;
	for (int i = 0; i < width * height; ++i)
		maxError = fmax(maxError, fabs(C->elements[i] - 2 * width));
		for (int i = 0; i < width * height; ++i)
		maxError = fmax(maxError, fabs(F->elements[i] - 2 * width));
	
	std::cout << "Error Number: " << maxError << std::endl;

	return 0;
}