# cuda-Connected_Components

## Introduction
This repository contains a CUDA implementation of the connected components finding algorithm as described in the paper: "Fast GPU Algorithms for Graph Connectivity" by Jyothish Soman, K. Kothapalli, and P. J. Narayanan, presented at the Large Scale Parallel Processing Workshop in IPDPS, 2010.

## Original Work
The original implementation by the authors can be found at: GpuConnectedComponents on GitHub.

## Installation
```shell
    git clone https://github.com/Abhijeetkumar96/cudaConnectedComponents.git
    cd cc
    make
```

## Usage
To run the CUDA Connected Components Finder as a stand-alone module:
```shell
    ./cc <filepath>
```
To integrate this module into your own project:
```cpp
    #include "connected_components"
```
- Headers are located in include/ directory, with sources in src/.
Header files of all the above contain an explanation of input parameters along with a simple input and output.

You may wish to update Makefile variables: CUDA, NVCC and you GPU's computing capability (NVCCSM) to match your system before building.

## Contributions
Contributions to this project are welcome. Please fork the repository and submit a pull request.

## Acknowledgements
This project is an implementation of the algorithm described in the paper: "Fast GPU Algorithms for Graph Connectivity" by Jyothish Soman, K. Kothapalli, and P. J. Narayanan. We acknowledge the authors for their groundbreaking work which inspired this implementation. The original implementation can be found at: https://github.com/jyosoman/GpuConnectedComponents. 
