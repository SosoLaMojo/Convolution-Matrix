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
	stbi_write_bmp(path.c_str(), width, height, channelNumber, data);
}

__global__ void conv_img_gpu(unsigned char* img, float* kernel, unsigned char* imgf, std::size_t imgX, std::size_t imgY, int kernel_size)
{
	//int iy = blockIdx.x + (kernel_size - 1) / 2;
	//int ix = threadIdx.x + (kernel_size - 1) / 2;
	std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int center = (kernel_size - 1) / 2;
	int stride = blockDim.x * gridDim.x;

	for (std::size_t px = idx; px < imgX * imgY; px += stride)
	{
		float sum[3] = { 0.0f, 0.0f, 0.0f };
		
		const std::size_t x = px % imgX;
		const std::size_t y = px / imgX;
		for (int ki = 0; ki < kernel_size; ki++)
		{
			for (int kj = 0; kj < kernel_size; kj++)
			{
				std::size_t ii = ki + x - center;
				std::size_t jj = kj + y - center;
				std::size_t kdx = (jj * imgX + ii) * 3;
				if (kdx > imgX * imgY * 3)
					continue;
				
				for (int color = 0; color < 3; ++color)
				{
					float image = ((float)img[kdx + color]) / 255.0f;
					sum[color] +=  image * kernel[ki * kernel_size + kj];
				}
			}
		}

		for (int color = 0; color < 3; ++color)
		{
			sum[color] = sum[color] < 0.0f ? 0.0f : sum[color] > 1.0f ? 1.0f : sum[color];
			imgf[px * 3 + color] = (unsigned char)(sum[color] * 255.0f);
		}
	}
}

int main()
{
	float milliseconds = 0;
	std::size_t Nx = 512;
	std::size_t Ny = 512;
	int Nkernel = 0;
	int kernelSize = 0;
	float* kernel;

	std::cout << "Choose a kernel:\n 0 = Identity\n 1 = Blur type 1\n 2 = Blur type 2\n 3 = Motion Blur\n 4 = Edges Detection\n"
		<< "5 = Edges Enhance\n 6 = Horizontal Edges\n 7 = Vertical Edges\n 8 = All Edges\n 9 = Sharpen\n 10 = Super Sharpen\n"
		<< "11 = Emboss\n 12 = Box Filter\n 13 = Gaussian Blur\n" << std::endl;
	
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

	default:
		kernel = kIdentity;
		kernelSize = 3;
		break;
	}

	unsigned char* img;
	unsigned char* imgf = (unsigned char*)malloc(Nx * Ny * 3 * sizeof(unsigned char));

	unsigned char* d_img;
	unsigned char* d_imgf;
	float* d_kernel;

	cudaMalloc(&d_img, Nx * Ny * 3 * sizeof(unsigned char));
	cudaMalloc(&d_imgf, Nx * Ny * 3 * sizeof(unsigned char));
	cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));

	img = LoadImage("../data/lena.png", Nx, Ny, 3);

	cudaMemcpy(d_img, img, Nx * Ny * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

	int Nthreads = 256;
	int Nblocks = (Nx * Ny + Nthreads - 1) / Nthreads;

	auto start = std::chrono::high_resolution_clock::now();
	conv_img_gpu <<< Nblocks, Nthreads>>> (d_img, d_kernel, d_imgf, Nx, Ny, kernelSize);
	cudaDeviceSynchronize();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	cudaMemcpy(imgf, d_imgf, Nx * Ny * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	
	WriteImage("../data/lena2.bmp", imgf, Nx, Ny, 3);

	std::cout << "Convolution complete. Elapsed time (GPU): " << duration.count() << std::endl;

	free(img);
	free(imgf);

	cudaFree(d_img);
	cudaFree(d_imgf);
	cudaFree(d_kernel);
	
	return 0;
}