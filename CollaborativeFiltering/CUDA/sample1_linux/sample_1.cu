// ---------------------------------------------------------------
//      Measurement of Data-transfer rate
// ---------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>

#define  N   (2048*512)


__global__  void  vec_add(float *, float *, float *);


int  main(int argc, char *argv[])
{
    unsigned  int   i,    n = N,   timer;
    float     *A_h,   *B_h,   *C_h,    *A_d,   *B_d,   *C_d;

    A_h = (float *) malloc(n*sizeof(float));  for(i = 0; i < n; i++) A_h[i] = 1.0;
    B_h = (float *) malloc(n*sizeof(float));  for(i = 0; i < n; i++) B_h[i] = 2.0;
    C_h = (float *) malloc(n*sizeof(float));  for(i = 0; i < n; i++) C_h[i] = 0.0;

    cudaMalloc( (void**) &A_d, n*sizeof(float) );
    cudaMalloc( (void**) &B_d, n*sizeof(float) );
    cudaMalloc( (void**) &C_d, n*sizeof(float) );

    cudaMemcpy( A_d, A_h, n*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( B_d, B_h, n*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( C_d, C_h, n*sizeof(float), cudaMemcpyHostToDevice );

    cudaThreadSynchronize();

    int   count = 1000;
    cutCreateTimer(&timer);
    cutResetTimer(timer);
    cutStartTimer(timer);

    dim3  grid(n/512, 1),  block(512, 1);

    for(i = 0; i < count; i++) vec_add<<< grid, block>>>(A_d, B_d, C_d);

    cudaThreadSynchronize();

    cutStopTimer(timer);

    cudaMemcpy( C_h, C_d, n*sizeof(float), cudaMemcpyDeviceToHost );

    double  ave = 0.0;
    for(i = 0; i < n; i++) ave += C_h[i];
    ave /= (float)n;

    double elapsed_time = cutGetTimerValue(timer)*1.0e-03;
    printf("\nData size=%4.1f[MB]  Elapsed time=%f[sec]    Transfer rate=%f[GB/sec]\n",
            (float)n*4.0*1.0e-06, elapsed_time,
            (float)(n*4*3)/elapsed_time*1.0e-09*(float)count);
    printf("  Average = %8.6f\n",ave);

    return 0;
}


__global__  
void  vec_add
// ===============================================================
(
    float  *A,         // array pointer of the global memory
    float  *B,         // array pointer of the global memory
    float  *C          // array pointer of the global memory
)
// ---------------------------------------------------------------
{
    int    i = blockDim.x*blockIdx.x + threadIdx.x;

#ifdef __DEVICE_EMULATION__
    if(blockIdx.x == 3) printf("threadIdx.x = %d\n",threadIdx.x);
#endif

    C[i] = A[i] + B[i] * 2.0f + 3.0f;
}
