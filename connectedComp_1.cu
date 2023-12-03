#include <set>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

class timer {
	private:
	    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
	    std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
	    bool running = false;
	public:
	    timer() {
	        start_time = std::chrono::high_resolution_clock::now();
	        running = true;
	    }

	    void stop(const std::string& str) {
	        end_time = std::chrono::high_resolution_clock::now();
	        running = false;
	        elapsedMilliseconds(str);
	    }

	    void elapsedMilliseconds(const std::string& str) {
	        std::chrono::time_point<std::chrono::high_resolution_clock> end;
	        if(running) {
	            end = std::chrono::high_resolution_clock::now();
	        } else {
	            end = end_time;
	        }
	        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_time).count();
	        std::cout << str <<" took " << duration << " ms." << std::endl;
	    }
};

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
	timer t1;
	cudaFree(0);
	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 1);

    const long numThreads = prop.maxThreadsPerBlock;
    int numBlocks = (numVert + numThreads - 1) / numThreads;

	t1.stop("Initial Setup");
	timer module_timer_t2;
	int* d_flag;
	cudaMalloc(&d_flag, sizeof(int));
	auto start = std::chrono::high_resolution_clock::now();
	int* d_rep;
	cudaMalloc(&d_rep, numVert*sizeof(int));

	initialise<<<numBlocks, numThreads>>>(d_rep, numVert);
	int flag = 1;
	int iteration = 0;

	const long numBlocks_hooking = (numEdges + numThreads - 1) / numThreads;
	const long numBlocks_updating_parent = (numVert + numThreads - 1) / numThreads;

	while(flag) {
		flag = 0;
		iteration++;
		cudaMemcpy(d_flag, &flag, sizeof(int),cudaMemcpyHostToDevice);

		hooking<<<numBlocks_hooking, numThreads>>> (numEdges, u_arr, v_arr, d_rep, d_flag, iteration);
		
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

		for(int i = 0; i < std::ceil(std::log2(numVert)); ++i)
			short_cutting<<<numBlocks_updating_parent, numThreads>>> (numVert, d_rep);

		cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
		// std::cout <<"Flag = " << flag << std::endl;
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout <<"cc took " << duration << " ms." << std::endl;
	// module_timer_t2.stop("connected_comp");

	cudaMemcpy(host_rep.data(), d_rep, numVert * sizeof(int), cudaMemcpyDeviceToHost);
	std::set<int> num_comp(host_rep.begin(), host_rep.end());

	std::cout <<"numComp = " << num_comp.size() << std::endl;
}

// Function to display the progress bar
void displayProgressBar(int totalBatches, int completedBatches) {
    const int barWidth = 50; // Width of the progress bar

    float progress = (float)completedBatches / totalBatches;
    int pos = barWidth * progress;

    std::cout << "Progress: [";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

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
	int numEdges;
	int u, v;
	inputFile >> numVert >> numEdges;
	for(int i = 0; i < numEdges; ++i) {
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

// Function to check CUDA errors
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
	std::ios_base::sync_with_stdio(false);
	if(argc < 2) {
		std::cerr <<"Usage : " << argv[0] <<"<filename> " << std::endl;
		return EXIT_FAILURE;
	}
	std::string filename = argv[1];
	std::vector<int> u_arr, v_arr;
	int numVert;
	timer t;
	read_graph(filename, u_arr, v_arr, numVert);
	t.stop("reading the graph");
	std::cout << "num edges : " << u_arr.size() << std::endl;
	size_t size = u_arr.size() * sizeof(int);
	
	std::cout << size <<" " << std::endl;

	int *d_uArr, *d_vArr;
	
	cudaMalloc(&d_uArr, size);
	cudaMalloc(&d_vArr, size);

	cudaMemcpy(d_uArr, u_arr.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vArr, v_arr.data(), size, cudaMemcpyHostToDevice);
	
	long numEdges = u_arr.size();
	connected_comp(numEdges, d_uArr, d_vArr, numVert);

	return EXIT_SUCCESS;
}