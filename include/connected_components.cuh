/**
 * CUDA Connected Components Finder
 * 
 * This header file contains the declaration of the connected components finding algorithm using CUDA, 
 * inspired by the work described in "Fast GPU Algorithms for Graph Connectivity" by Jyothish Soman, 
 * K. Kothapalli, and P. J. Narayanan, presented at the Large Scale Parallel Processing Workshop in IPDPS, 2010.
 * 
 * For more details, please refer original paper: https://github.com/jyosoman/GpuConnectedComponents
 *
 * The implementation computes the connected components in a graph using CUDA.
 * 
 * Parameters:
 *    int *d_uArr: Device pointer to an array of 'u' vertices of edges. 
 *    int *d_vArr: Device pointer to an array of 'v' vertices of edges.
 * 
 *    long numEdges: The number of edges in the graph.
 *    int numVert: The number of vertices in the graph.
 */

 #include <string>
 #include <iostream>
 #include <cuda_runtime.h>

#ifndef CONNECTED_COMPONENTS_H
#define CONNECTED_COMPONENTS_H

inline void checkCudaError(cudaError_t err, const std::string& msg) {
	if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
void connected_comp(long numEdges, int* d_uArr, int* d_vArr, int numVert);

#endif //CONNECTED_COMPONENTS_H
