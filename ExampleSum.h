#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <cmath>
#include <iostream>

#include "CudaError.h"
#include "Indexer.h"

namespace Kernels{
	__global__ void Sum_Kernel(int* data, int stride){
		if(blockIdx.x > 0 || threadIdx.x > 0){
			return;
		}
	
		int size0 = 10000;
		int thread_size = 2;
	
		unsigned long long grid_index = Indexer::FlatIndex2(threadIdx.x, blockIdx.x, blockDim.x);
		unsigned long long arr_index = grid_index * stride;
	
		int unit_sum = 0;
		
		for(int i = 0; i < thread_size; i++){
			unit_sum += data[arr_index + i];
		}
		printf("unit:%d\n", unit_sum);
		data[arr_index] = unit_sum;
		
		__syncthreads();
		
		if(threadIdx.x > 0){
			return;
		}
		
		int thread_sum = 0;
		for(int i = 0; i < blockDim.x; i++){
			int i_strided = i * stride;
			
			thread_sum += data[arr_index + i_strided];
		}
		
		data[arr_index] = thread_sum;
	}
}

struct ExampleSum{
	void Sum(){
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		int size0 = 10000;
		int thread_size = 2;
		int* g_data;
		int* c_data = new int[size0];
		
		for(int i = 0; i < size0; i++){
			c_data[i] = 1;
		}
		
		cudaMalloc(&g_data, (size_t)(size0 * sizeof(int)));
		cudaMemcpy(g_data, c_data, size0 * sizeof(int), cudaMemcpyHostToDevice);

		dim3 threads_per_block(32, 1, 1);

		int step_count = std::ceil(std::log(size0) / std::log(threads_per_block.x * thread_size));
		int stride;

		for(int i = 0; i < step_count; i++){
			stride = pow(thread_size * threads_per_block.x, i);
			
			if(i == 0){
				stride = 2;
			}
		
			int block_count = std::ceil(size0 / (float)(stride * thread_size * threads_per_block.x));
			dim3 blocks_per_grid(block_count, 1, 1);

			printf("[Step: %d] %d %d %d\n", i, block_count, step_count, stride);

			Kernels::Sum_Kernel<<<blocks_per_grid, threads_per_block>>>(g_data, stride);
		
			cudaMemcpy(c_data, g_data, size0 * sizeof(int), cudaMemcpyDeviceToHost);	
			
			CudaError::CheckError((cudaError_enum)cudaPeekAtLastError(), __FILE__, __LINE__);
			CudaError::CheckError((cudaError_enum)cudaDeviceSynchronize(), __FILE__, __LINE__);
			
			printf("[Final]: %d\n\n", c_data[0]);
			
			if (i == 2){
				break;
			}
		}

		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		int time_lapsed = (int)std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
		printf("\n        Done!\n");
		printf("            Time Elapsed: %d s\n", time_lapsed);
		
		
	}
};
