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

std::string get_file_extension(const std::string& filename) {
	size_t result;
    std::string fileExtension;
    result = filename.rfind('.', filename.size() - 1);
    if(result != std::string::npos)
        fileExtension = filename.substr(result + 1);
    return fileExtension;
}

void read_edges(const std::string& filename, std::vector<int>& u_arr, std::vector<int>& v_arr, int& numVert) {
	std::ifstream inputFile(filename);
	if(!inputFile) {
		std::cerr <<"Unable to open file for reading." << std::endl;
		return;
	}
	long numEdges;
	int u, v;
	inputFile >> numVert >> numEdges;
	for(long i = 0; i < numEdges; ++i) {
		inputFile >> u >> v;
		u_arr.push_back(u);
		v_arr.push_back(v);
	}
}

void read_mtx(const std::string& filename, std::vector<int>& u_arr, std::vector<int>& v_arr, int& numVert) {
	std::ifstream inputFile(filename);
	if(!inputFile) {
		std::cerr <<"Unable to open file for reading." << std::endl;
		return;
	}
	std::string line;
	std::getline(inputFile, line);
	std::istringstream iss(line);
	std::string word;
	std::vector<std::string> words;
	
	while(iss >> word)
		words.push_back(word);

	if(words.size() != 5) {
		std::cerr <<"ERROR: mtx header is missing" << std::endl;
		return;
	}
    else 
        std::cout <<"MTX header read correctly.\n";

    std::getline(inputFile, line);
    while(line.size() > 0 && line[0]=='%') 
    	std::getline(inputFile,line);

    // Clear any error state flags
    iss.clear();

    // Set the new string for the istringstream
    iss.str(line);

    int row, col, num_of_entries;
    iss >> row >> col >> num_of_entries;
    numVert = row;
    if(row != col) {
        std::cerr<<"* ERROR: This is not a square matrix."<< std::endl;
        return;
    }
    int entry_counter = 0;
    int ridx, cidx, value;

    // std::cout << "Num entries = " << num_of_entries << std::endl;
    // int batchSize = 10000000;
    // int totalBatches = num_of_entries / batchSize;
    // std::cout <<"Batch size = " << batchSize << std::endl; 
    // std::cout <<"Total Batches: " << totalBatches << std::endl;
    // int completedBatches = 0;

    while(!inputFile.eof() && entry_counter < num_of_entries) {
        getline(inputFile, line);
        entry_counter++;

        if (!line.empty()) {
            iss.clear();
            iss.str(line);
            iss >> ridx >> cidx >> value;
            ridx--;
            cidx--;

            if (ridx < 0 || ridx >= row)  
            	std::cout << "sym-mtx error: " << ridx << " row " << row << std::endl;

            if (cidx < 0 || cidx >= col)  
            	std::cout << "sym-mtx error: " << cidx << " col " << col << std::endl;

            if (ridx != cidx) {
                u_arr.push_back(ridx);
                v_arr.push_back(cidx);
            }
        }
        // if ((entry_counter % batchSize) == 0) {
        //     completedBatches++;
        //     std::cout << "Batches read: " << completedBatches << ", Remaining Batches: " << totalBatches - completedBatches << std::endl;
        //     // Display the progress bar
        //     displayProgressBar(totalBatches, completedBatches);
        // }
    }
}

void read_graph(const std::string& filename, std::vector<int>& u_arr, std::vector<int>& v_arr, int& numVert) {
	std::string ext = get_file_extension(filename);
	if(ext == "txt") {
		std::cout <<"Reading txt graph\n";
		read_edges(filename, u_arr, v_arr, numVert);
	}
	else if (ext == "mtx") {
		std::cout <<"Reading mtx graph\n";
		read_mtx(filename, u_arr, v_arr, numVert);
	}
	else
		std::cerr <<"Unsupported file format." << std::endl;
}

int main(int argc, char* argv[]) {
	std::ios_base::sync_with_stdio(false);
	if(argc < 2) {
		std::cerr <<"Usage : " << argv[0] <<" <filename> " << std::endl;
		return EXIT_FAILURE;
	}
	std::string filename = argv[1];
	std::vector<int> u_arr, v_arr;
	int numVert;
	read_graph(filename, u_arr, v_arr, numVert);
	std::cout << "num edges : " << u_arr.size() << std::endl;
	size_t size = u_arr.size() * sizeof(int);
	
	std::cout << size <<" " << std::endl;

	int *d_uArr, *d_vArr;
	
	checkCudaError(cudaMalloc(&d_uArr, size), "Unable to allocate u_arr ");
	checkCudaError(cudaMalloc(&d_vArr, size), "Unable to allocate v_arr ");

	checkCudaError(cudaMemcpy(d_uArr, u_arr.data(), size, cudaMemcpyHostToDevice), "Unable to copy u_arr to device");
	checkCudaError(cudaMemcpy(d_vArr, v_arr.data(), size, cudaMemcpyHostToDevice), "Unable to copy v_arr to device");
	
	long numEdges = u_arr.size();
	connected_comp(numEdges, d_uArr, d_vArr, numVert);

	return EXIT_SUCCESS;
}