#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>

const unsigned int UNIT = 1024;

template <unsigned int blockSize>
__global__ void reduce5( int *g_idata, int *g_odata );

int main( int argc, char *argv[] )
{
	int *g_idata, *g_odata,  *c_num, *ans;

    int n = 2000;
    int size = (int)(ceil(n * 1.0 / UNIT)) * UNIT;
    printf ("size: %d\n", size);

	//GPUの初期化
	CUT_DEVICE_INIT(argc, argv);

	//CPU側のメモリ確保
	cudaMallocHost((void**)&c_num, size * sizeof(int));
	cudaMallocHost((void**)&ans, size * sizeof(int));

    for (int i = 0; i < size ; ++i ) {
        c_num[i] = 0;
    }
    c_num[100] = 7;
    c_num[1100] = 5;



	//GPU側のメモリ確保
	cudaMalloc((void**)&g_idata, size * sizeof(int));
	cudaMalloc((void**)&g_odata, size * sizeof(int));

	//CPU→GPUにデータ転送
	cudaMemcpy( g_idata, c_num, size*sizeof(int), cudaMemcpyHostToDevice);

	//コピー終了まで待つ
	cudaThreadSynchronize();

    /*
    int count = 1000;
    unsigned int timer;
    cutCreateTimer(&timer);
    cutResetTimer(timer);
    cutStartTimer(timer);
    */


    dim3  grid(size/UNIT, 1),  block(512, 1);

    /*
    for(i = 0; i < count; i++) { 
        //reduce3<<< grid, block, sizeof(int)*512>>>( g_idata, g_odata );
        reduce5<512><<<grid, block, sizeof(int)*512>>>( g_idata, g_odata );
    }
    */
    //TODO

    reduce5<512><<<grid, block, sizeof(int)*512>>>( g_idata, g_odata );

	//計算終了まで待つ
	cudaThreadSynchronize();

	//GPU→CPUにデータ転送
	cudaMemcpy( ans, g_odata, size*sizeof(int), cudaMemcpyDeviceToHost);

	//コピー終了まで待つ
	cudaThreadSynchronize();

    printf("aaaaa\n");
    printf("%d\n", ans[0]);
    printf("%d\n", ans[1]);
    printf("%d\n", ans[2]);

	cudaFree(c_num);
	cudaFree(ans);
	cudaFree(g_idata);
	cudaFree(g_odata);

	CUT_EXIT(argc,argv);
    return 0;
}

template <unsigned int blockSize>
__global__ void reduce5( int *g_idata, int *g_odata )
{
	extern __shared__ int sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; }
        __syncthreads();
    }

    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
        if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
        if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
        if (blockSize >=  2) sdata[tid] += sdata[tid +  1];

    }
	
	if (tid == 0) {
		g_odata[blockIdx.x] = sdata[0];
	}
}
