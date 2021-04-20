#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>

#define pi 3.14159265359

/*
imgX, imgY = Image dimensions
fname= filename (char)
img = float vector containing the mage pixels
 */

void LoadImage (char *fname, int imgX, int imgY, float *img)
{
	FILE* fp;
	fp = fopen(fname, "r");

	for (int i = 0; i < imgY; i++)
	{
		for (int j = 0; j < imgX; j++)
			fscanf(fp, "%f ", &img[i * imgX + j]);
		fscanf(fp, "\n");
	}
	fclose(fp);
}

void SaveImage(char *fname, int imgX, int imgY, float *img)
{
	FILE* fp;
	fp = fopen(fname, "w");

	for (int i = 0; i < imgY; i++)
	{
		for (int j = 0; j < imgX; j++)
			fprintf(fp, "%10.3f ", img[i * imgX + j]);
		fprintf(fp, "\n");
	}
	fclose(fp);
}

//__global__ void conv_img_gpu(float* img, float* kernel, float* imgf, int imgX, int imgY, int kernel_size)
//{
//	int threadId = threadIdx.x;
//	int iy = blockIdx.x + (kernel_size - 1) / 2;
//	int ix = threadIdx.x + (kernel_size - 1) / 2;
//	int idx = iy * imgX + ix;
//	int K2 = kernel_size * kernel_size;
//	int center = (kernel_size - 1) / 2;
//	int ii, jj;
//	float sum = 0.0f;
//
//	extern __shared__ float sdata[];
//
//	if (tid < K2)
//		sdata[tid] = kernel[tid];
//
//	__syncthreads();
//
//	if(idx<imgX*imgY)
//	{
//		for (int ki = 0; ki<kernel_size;ki++)
//			for (int kj = 0; kj<kernel_size;kj++)
//			{
//				ii = kj + ix - center;
//				jj = ki + iy - center;
//				sum+=img[jj*imgX+ii] * sdata[ki*kernel_size + kj];
//			}
//		imgf[idx] = sum;
//	}
//}

//int main()
//{
//	
//}