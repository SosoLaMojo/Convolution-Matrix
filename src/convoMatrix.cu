#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <filesystem>
#include "kernels.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define pi 3.14159265359

/*
imgX, imgY = Image dimensions
fname= filename (char)
img = float vector containing the mage pixels
 */

unsigned char* LoadImage (const std::string &path, int width, int height, int channelNumber)
{
	return stbi_load(path.c_str(), &width, &height, &channelNumber, 0);
}

void WriteImage(const std::string& path, unsigned char* data, int width, int height, int channelNumber)
{
	stbi_write_png(path.c_str(), width, height, channelNumber, data, channelNumber * width);
}

__global__ void conv_img_gpu(unsigned char* img, int* kernel, unsigned char* imgf, int imgX, int imgY, int kernel_size)
{
	int threadId = threadIdx.x;
	int iy = blockIdx.x + (kernel_size - 1) / 2;
	int ix = threadIdx.x + (kernel_size - 1) / 2;
	int idx = iy * imgX + ix;
	int K2 = kernel_size * kernel_size;
	int center = (kernel_size - 1) / 2;
	int ii, jj;
	float sum = 0.0f;

	extern __shared__ float sdata[];

	if (threadId < K2)
		sdata[threadId] = kernel[threadId];

	__syncthreads();

	if(idx<imgX*imgY)
	{
		for (int ki = 0; ki<kernel_size;ki++)
			for (int kj = 0; kj<kernel_size;kj++)
			{
				ii = kj + ix - center;
				jj = ki + iy - center;
				sum+=img[jj*imgX+ii] * sdata[ki*kernel_size + kj];
			}
		imgf[idx] = (unsigned char)(sum);
	}
}

int main()
{
	cudaEvent_t start;
	cudaEvent_t stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float milliseconds = 0;
	int Nx = 512;
	int Ny = 512;
	int Nkernel = 0;
	int kernelSize = 0;
	int* kernel;

	std::cin >> Nkernel;
	
	switch (Nkernel)
	{
	case 0:
		kernel = kIdentity;
		kernelSize = 3;
		break;

	case 1:
		kernel = kBlur1;
		kernelSize = 5;
		break;
		
	case 2:
		kernel = kBlur2;
		kernelSize = 5;
		break;
		
	case 3:
		kernel = kMotionBlur;
		kernelSize = 9;
		break;
		
	case 4:
		kernel = kEdgesDetect;
		kernelSize = 3;
		break;
		
	case 5:
		kernel = kEdgesEnhance;
		kernelSize = 3;
		break;
		
	case 6:
		kernel = kHorizontalEdges;
		kernelSize = 5;
		break;
		
	case 7:
		kernel = kVerticalEdges;
		kernelSize = 5;
		break;
		
	case 8:
		kernel = kAllEdges;
		kernelSize = 3;
		break;
		
	case 9:
		kernel = kSharpen;
		kernelSize = 3;
		break;
		
	case 10:
		kernel = kSuperSharpen;
		kernelSize = 3;
		break;
		
	case 11:
		kernel = kEmboss;
		kernelSize = 3;
		break;
		
	case 12:
		kernel = kBoxFilter;
		kernelSize = 19;
		break;
		
	case 13:
		kernel = kGaussianBlur;
		kernelSize = 5;
		break;
	}

	unsigned char* img = (unsigned char*)malloc(Nx * Ny * sizeof(unsigned char));
	unsigned char* imgf = (unsigned char*)malloc(Nx * Ny * sizeof(unsigned char));

	unsigned char* d_img;
	unsigned char* d_imgf;
	int* d_kernel;

	cudaMalloc(&d_img, Nx * Ny * sizeof(unsigned char));
	cudaMalloc(&d_imgf, Nx * Ny * sizeof(unsigned char));
	cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(int));

	img = LoadImage("../data/lena.png", Nx, Ny, 0);

	cudaMemcpy(d_img, img, Nx * Ny * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * sizeof(int), cudaMemcpyHostToDevice);

	int Nblocks = Ny - (kernelSize - 1);
	int Nthreads = Nx - (kernelSize - 1);

	cudaEventRecord(start);
	conv_img_gpu <<< Nblocks, Nthreads, kernelSize* kernelSize * sizeof(int) >>> (d_img, d_kernel, d_imgf, Nx, Ny, kernelSize);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaMemcpy(imgf, d_imgf, Nx * Ny * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	WriteImage("../data/lena2.png", imgf, 512, 512, 0);

	std::cout << "Convolution complete. Elapsed time (GPU): " << milliseconds << std::endl;

	free(img);
	free(imgf);

	cudaFree(d_img);
	cudaFree(d_imgf);
	cudaFree(d_kernel);
	
	return 0;
}