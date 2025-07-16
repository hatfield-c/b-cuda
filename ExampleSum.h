#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <cmath>
#include <iostream>

#include "CudaError.h"

namespace Kernels{
	__global__ void Sum_Kernel(int* data, int stride){

	}
}

struct ExampleSum{
	void Sum(){
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		int size0 = 10000;
		int asize = 10000;
		int thread_size = 2;
		int* data;
		int* cpu_data = new int[size0];

		dim3 threads_per_block(32, 1, 1);

		int step_count = std::ceil(std::log(size0) / std::log(threads_per_block.x * thread_size));
		int stride = 0;

		CudaError::CheckError((cudaError_enum)cudaMalloc(&data, size0 * sizeof(int)), __FILE__, __LINE__);

		for(int i = 0; i < step_count; i++){
			int block_count = std::ceil(size0 / (float)(threads_per_block.x * thread_size * (i + 1)));
			dim3 blocks_per_grid(block_count, 1, 1);

			printf("[Step: %d] %d %d %d\n", i, block_count, step_count, stride);

			Kernels::Sum_Kernel<<<blocks_per_grid, threads_per_block>>>(data, stride);
			stride = (i + 1) * thread_size * threads_per_block.x;
		}

		cudaMemcpy(cpu_data, data, size0 * sizeof(int), cudaMemcpyDeviceToHost);

		CudaError::CheckError((cudaError_enum)cudaPeekAtLastError(), __FILE__, __LINE__);
		CudaError::CheckError((cudaError_enum)cudaDeviceSynchronize(), __FILE__, __LINE__);

		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		int time_lapsed = (int)std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
		printf("\n        Done!\n");
		printf("            Time Elapsed: %d s\n", time_lapsed);
	}
};
