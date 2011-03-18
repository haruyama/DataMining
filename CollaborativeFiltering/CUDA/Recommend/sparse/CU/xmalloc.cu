#include "xmalloc.hpp"

#include <iostream>

void xmallocHost(void** ptr, const size_t count) {
    cudaError_t err = cudaMallocHost(ptr, count);
    if (err != cudaSuccess) {
        cerr << "xmallocHost error" << endl;
        exit(1);
    }
}


void xmallocDevice(void** ptr, const size_t count) {
    cudaError_t err = cudaMalloc(ptr, count);
    if (err != cudaSuccess) {
        cerr << "xmallocDevice error" << endl;
        exit(1);
    }
}
