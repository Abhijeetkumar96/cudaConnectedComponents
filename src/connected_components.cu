#include <set>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

#include "connected_components.cuh"

__global__
void initialise(int* parent, int n) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < n) {
		parent[tid] = tid;
	}
}

__global__ 
void hooking(long numEdges, int* d_source, int* d_destination, int* d_rep, int* d_flag, int itr_no) 
{
	long tid = blockDim.x * blockIdx.x + threadIdx.x;
	#ifdef DEBUG
		if(tid == 0) {
			printf("\nIteration number = %d", itr_no);
			printf("\nFlag inside kernel before start of iteration = %d", *d_flag);
		}
	#endif

	if(tid < numEdges) {
		
		int edge_u = d_source[tid];
		int edge_v = d_destination[tid];

		int comp_u = d_rep[edge_u];
		int comp_v = d_rep[edge_v];

		if(comp_u != comp_v) 
		{
			*d_flag = 1;
			int max = (comp_u > comp_v) ? comp_u : comp_v;
			int min = (comp_u < comp_v) ? comp_u : comp_v;

			if(itr_no%2) {
				d_rep[min] = max;
			}
			else { 
				d_rep[max] = min;
			}
		}
	}
}

__global__ 
void short_cutting(int n, int* d_parent) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < n) {
		if(d_parent[tid] != tid) {
			d_parent[tid] = d_parent[d_parent[tid]];
		}
	}	
}

void connected_comp(long numEdges, int* u_arr, int* v_arr, int numVert) {

	std::vector<int> host_rep(numVert);

	checkCudaError(cudaFree(0), "Unable to setup device");
	cudaDeviceProp prop;

    checkCudaError(cudaGetDeviceProperties(&prop, 1), "Unable to get deviceProp");
    const long numThreads = prop.maxThreadsPerBlock;
    int numBlocks = (numVert + numThreads - 1) / numThreads;

	int* d_flag;
	checkCudaError(cudaMalloc(&d_flag, sizeof(int)), "Unable to allocate flag value");
	auto start = std::chrono::high_resolution_clock::now();
	int* d_rep;
	checkCudaError(cudaMalloc(&d_rep, numVert*sizeof(int)), "Unable to allocate rep array");

	initialise<<<numBlocks, numThreads>>>(d_rep, numVert);
	cudaError_t err = cudaGetLastError();
	checkCudaError(err, "Error in launching initialise kernel");

	int flag = 1;
	int iteration = 0;

	const long numBlocks_hooking = (numEdges + numThreads - 1) / numThreads;
	const long numBlocks_updating_parent = (numVert + numThreads - 1) / numThreads;

	while(flag) {
		flag = 0;
		iteration++;
		checkCudaError(cudaMemcpy(d_flag, &flag, sizeof(int),cudaMemcpyHostToDevice), "Unable to copy the flag to device");

		hooking<<<numBlocks_hooking, numThreads>>> (numEdges, u_arr, v_arr, d_rep, d_flag, iteration);
		err = cudaGetLastError();
		checkCudaError(err, "Error in launching hooking kernel");
		
		#ifdef DEBUG
			cudaMemcpy(host_rep.data(), d_rep, numVert * sizeof(int), cudaMemcpyDeviceToHost);
			// Printing the data
			std::cout << "\niteration num : "<< iteration << std::endl;
			std::cout << "d_rep : ";
			for (int i = 0; i < numVert; i++) {
			    std::cout << host_rep[i] << " ";
			}
			std::cout << std::endl;
		#endif

		for(int i = 0; i < std::ceil(std::log2(numVert)); ++i) {
			short_cutting<<<numBlocks_updating_parent, numThreads>>> (numVert, d_rep);
			err = cudaGetLastError();
			checkCudaError(err, "Error in launching short_cutting kernel");
		}

		checkCudaError(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost), "Unable to copy back flag to host");
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout <<"cc took " << duration << " ms." << std::endl;

	checkCudaError(cudaMemcpy(host_rep.data(), d_rep, numVert * sizeof(int), cudaMemcpyDeviceToHost), "Unable to copy back rep array");
	std::set<int> num_comp(host_rep.begin(), host_rep.end());

	std::cout <<"numComp = " << num_comp.size() << std::endl;
}
