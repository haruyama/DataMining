#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>

#define MAX (65536*256)

__global__ void reduce0( int *g_idata, int *g_odata );
__global__ void reduce3( int *g_idata, int *g_odata );

template <unsigned int blockSize>
__global__ void reduce5( int *g_idata, int *g_odata );

int main( int argc, char *argv[] )
{
	int *g_idata, *g_odata,  *c_num, *ans;
	int i;

	//GPUの初期化
	CUT_DEVICE_INIT(argc, argv);

	//CPU側のメモリ確保
	cudaMallocHost((void**)&c_num, MAX * sizeof(int));
	cudaMallocHost((void**)&ans, MAX * sizeof(int));

	for( i = 0; i < MAX ; i++ )
	{
		c_num[i]=1;
		ans[i]=-1;
	}

	//GPU側のメモリ確保
	cudaMalloc((void**)&g_idata, MAX * sizeof(int));
	cudaMalloc((void**)&g_odata, MAX * sizeof(int));

	//CPU→GPUにデータ転送
	cudaMemcpy( g_idata, c_num, MAX*sizeof(int), cudaMemcpyHostToDevice);

	//コピー終了まで待つ
	cudaThreadSynchronize();

    int count = 1000;
    unsigned int timer;
    cutCreateTimer(&timer);
    cutResetTimer(timer);
    cutStartTimer(timer);


    dim3  grid(MAX/1024, 1),  block(512, 1);

    for(i = 0; i < count; i++) { 
        //reduce3<<< grid, block, sizeof(int)*512>>>( g_idata, g_odata );
        reduce5<512><<<grid, block, sizeof(int)*512>>>( g_idata, g_odata );
    }
	//計算終了まで待つ
	cudaThreadSynchronize();

	//GPU→CPUにデータ転送
	cudaMemcpy( ans, g_odata, MAX*sizeof(int), cudaMemcpyDeviceToHost);

	//コピー終了まで待つ
	cudaThreadSynchronize();

    double elapsed_time = cutGetTimerValue(timer)*1.0e-03;
    printf("\nData size=%4.1f[MB]  Elapsed time=%f[sec]    Transfer rate=%f[GB/sec]\n",
                    (float)MAX*4.0*1.0e-06, elapsed_time,
                    (float)(MAX*4*3)/elapsed_time*1.0e-09*(float)count);

	//答えの表示
	for( i = 0 ; i < 10; i++ )
	{
		printf("ans[%d] = %d\n",i, ans[i]);
	}
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
__global__ void reduce3( int *g_idata, int *g_odata )
{
	extern __shared__ int sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>=1) {
        if (tid < s) {
			sdata[tid] += sdata[tid + s];
        }
        __syncthreads();

    }

    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid +  8];
        sdata[tid] += sdata[tid +  4];
        sdata[tid] += sdata[tid +  2];
        sdata[tid] += sdata[tid +  1];

    }
	
	if (tid == 0) {
		g_odata[blockIdx.x] = sdata[0];
	}
}
__global__ void reduce0( int *g_idata, int *g_odata )
{
	extern __shared__ int sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = g_idata[i];
	__syncthreads();
	
    /*
	for (unsigned int s=1; s < blockDim.x ; s *= 2) {
        int index = 2*s*tid;
        if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
        }

		__syncthreads();
	}
     */

    for (unsigned int s = blockDim.x / 2; s > 0; s >>=1) {
        if (tid < s) {
			sdata[tid] += sdata[tid + s];
        }
        __syncthreads();

    }
	
	if (tid == 0) {
		g_odata[blockIdx.x] = sdata[0];
	}
}
