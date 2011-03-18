#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>
#include <math.h>
#include <string.h>
#include <json/json.h>

typedef float data_type;

const unsigned int REDUCE_UNIT = 1024;
const unsigned int CALC_UNIT = 512;

template <unsigned int blockSize>
__global__ void reduce2d(int user_ceil_length, data_type* g_idata, data_type *g_odata )
{
	extern __shared__ data_type sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.y * user_ceil_length + blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();

#ifdef __DEVICE_EMULATION__
    if (sdata[tid] > 0.0f) {
        printf("reduce: %d %d %.2f\n", tid, i, sdata[tid]); 
    }
#endif

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
#ifdef __DEVICE_EMULATION__
        printf("reduced: %d %.2f\n", base, sdata[0]);
#endif
		g_odata[user_ceil_length* blockIdx.y / REDUCE_UNIT + blockIdx.x] = sdata[0];
	}
}
template <unsigned int blockSize>
__global__ void reduce5(int base, data_type* g_idata, data_type *g_odata )
{
	extern __shared__ data_type sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = base + blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();

#ifdef __DEVICE_EMULATION__
    if (sdata[tid] > 0.0f) {
        printf("reduce: %d %d %.2f\n", tid, i, sdata[tid]); 
    }
#endif

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
#ifdef __DEVICE_EMULATION__
        printf("reduced: %d %.2f\n", base, sdata[0]);
#endif
		g_odata[blockIdx.x] = sdata[0];
	}
}

void xcudaMallocHost(void** ptr, size_t count) {

    cudaError_t err = cudaMallocHost(ptr, count);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost error\n");
        exit(1);
    }
}


void xcudaMalloc(void** ptr, size_t count) {

    cudaError_t err = cudaMalloc(ptr, count);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error\n");
        exit(1);
    }
}

static int parse(char* filename, int& item_length, int& user_ceil_length, char**& item_names, data_type*& data) {

    json_object* root = json_object_from_file(filename);

    int length = json_object_array_length(root);
    item_length = json_object_get_int(json_object_array_get_idx(root, 0));

    if (length != (item_length + 2)) {
        fprintf(stderr, "length error\n");
        return 0;
    }

    item_names = (char**)malloc(sizeof(char*)*item_length);

    if (!item_names) {
        fprintf(stderr, "item_names malloc error\n");
        return 0;
    }

    int user_length = json_object_get_int(json_object_array_get_idx(root, 1));

    user_ceil_length =  (int)(ceil(user_length * 1.0 / REDUCE_UNIT)) * REDUCE_UNIT;

    xcudaMallocHost((void**)&data, sizeof(data_type)*user_ceil_length*item_length);

    for (int i =0; i < user_ceil_length*item_length; ++i) {
        data[i] = 0.0f;
    }
    
    for (int i = 0; i < item_length; ++i) {
        json_object* array_object = json_object_array_get_idx(root, i+2);
        int array_length = json_object_array_length(array_object);
        char* name = strdup(json_object_get_string(json_object_array_get_idx(array_object, 0)));
        item_names[i] = name;
        int base = i * user_ceil_length;
        for (int j = 1; j < array_length; ++j) {
            int user = json_object_get_int(json_object_array_get_idx(array_object, j)); 
            data[base + user] = 1.0f;
        }
    }

    return 1;
}

__global__
void calc(int base_i, int base_j, data_type* data_d, data_type* similarity_d) {
	unsigned int user = blockIdx.x * blockDim.x + threadIdx.x;

    if (data_d[base_i + user] > 0.0f && data_d[base_j + user] > 0.0f) {
        similarity_d[user] = 1.0f;
    } else {
        similarity_d[user] = 0.0f;
    }
}

__global__
void calc2d(int base_i, int user_ceil_length, data_type* data_d, data_type* similarity_d) {
    unsigned int base_j = blockIdx.y * user_ceil_length; 
	unsigned int user = blockIdx.x * blockDim.x + threadIdx.x;

    if (base_j != base_i &&
            data_d[base_i + user] > 0.0f && data_d[base_j + user] > 0.0f) {
        similarity_d[base_j + user] = 1.0f;
    } else {
        similarity_d[base_j + user] = 0.0f;
    }
}


int main( int argc, char *argv[] )
{

	//CUT_DEVICE_INIT(argc, argv);

    char** item_names;
    data_type* data;
    data_type* result;
    data_type* data_d;
    data_type* similarity_d;

    int item_length;
    int user_ceil_length;


    if (!parse(argv[1], item_length, user_ceil_length, item_names, data)) {
        fprintf(stderr, "parse error!\n");
        exit(1);
    }



    int data_size = item_length*user_ceil_length;
    int result_size = item_length*item_length;


	result = (data_type*)malloc(sizeof(data_type)*result_size);

    if (!result) {
        fprintf(stderr, "result malloc error\n");
        exit(1);
    }


	xcudaMalloc((void**)&similarity_d, sizeof(data_type)*data_size);
	xcudaMalloc((void**)&data_d, sizeof(data_type)*data_size);

	cudaMemcpy( data_d, data, sizeof(data_type)*data_size, 
            cudaMemcpyHostToDevice);

	cudaThreadSynchronize();

    data_type* reduce = (data_type*)malloc(sizeof(data_type)*item_length);

    if (!reduce) {
        exit(1);
    }

    dim3 block(512, 1);
    dim3 calc_grid(user_ceil_length/CALC_UNIT, 1);
    dim3 calc2d_grid(user_ceil_length/CALC_UNIT, item_length); 
    dim3 reduce_grid(user_ceil_length/REDUCE_UNIT, 1);
    dim3 reduce2d_grid(user_ceil_length/REDUCE_UNIT, item_length);

    data_type* reduce2d_h;
    data_type* reduce2d_d;
	xcudaMallocHost((void**)&reduce2d_h, sizeof(data_type)*data_size/REDUCE_UNIT);
	xcudaMalloc((void**)&reduce2d_d, sizeof(data_type)*data_size/REDUCE_UNIT);

    reduce2d<512><<<reduce2d_grid, block, sizeof(data_type)*512>>>(user_ceil_length, data_d, reduce2d_d);
    cudaMemcpy(reduce2d_h, reduce2d_d, sizeof(data_type)*data_size / REDUCE_UNIT, cudaMemcpyDeviceToHost);

    for (int i = 0; i < item_length; ++i) {
        data_type sum = 0.0f;
        for (int k = 0; k < user_ceil_length/REDUCE_UNIT; ++k) {
            sum += reduce2d_h[i * user_ceil_length / REDUCE_UNIT + k];
        }
        reduce[i] = sum;
    }

    for (int i = 0; i < item_length; ++i) {
        int base_i = user_ceil_length*i;
        calc2d<<<calc2d_grid, block>>>(base_i, user_ceil_length, data_d, similarity_d);

        reduce2d<512><<<reduce2d_grid, block, sizeof(data_type)*512>>>(user_ceil_length, similarity_d, reduce2d_d);

        cudaMemcpy(reduce2d_h, reduce2d_d, sizeof(data_type)*data_size / REDUCE_UNIT , cudaMemcpyDeviceToHost);
        for (int j = i + 1; j < item_length; ++j) {
            data_type sum_s = 0.0f;
            for (int k = 0; k < user_ceil_length / REDUCE_UNIT; ++k) {
                sum_s += reduce2d_h[user_ceil_length * j / REDUCE_UNIT+ k];
            }
            data_type sim = 0.0f;
            if (sum_s > 0.0f) {
                sim = (sum_s / (reduce[i] + reduce[j] - sum_s));
            }
            result[i * item_length + j] = sim;
            result[j * item_length + i] = sim;
        }
        result[i * item_length + i] = 0.0f;
    }


/*
    for (int i = 0; i < item_length; ++i) {
        int base_i = user_ceil_length*i;
        for (int j = i + 1; j < item_length; ++j) {
            int base_j = user_ceil_length*j;
            calc<<<calc_grid, block>>>(base_i, base_j, data_d, similarity_d);
            reduce5<512><<<reduce_grid, block, sizeof(data_type)*512>>>(0, similarity_d, reduce_s_d);

            cudaMemcpy(reduce_s, reduce_s_d, sizeof(data_type)*user_ceil_length, cudaMemcpyDeviceToHost);


            data_type sum_s = 0.0f;
            for (int k = 0; k < user_ceil_length/REDUCE_UNIT; ++k) {
                sum_s += reduce_s[k];
            }


            data_type sim = 0.0f;
            if (sum_s > 0.0f) {
                sim = (sum_s / (reduce[i] + reduce[j] - sum_s));
            }
            result[i * item_length + j] = sim;
            result[j * item_length + i] = sim;
        }
        result[i * item_length + i] = 0.0f;
    }
    */

    for (int i = 0; i < item_length; ++i) {
        int base_i = i * item_length;
        printf("%s:", item_names[i]);
        for (int j = 0; j < item_length; ++j) {
            if (result[base_i + j] > 0.0) {
                printf("%.3f|%s,", result[base_i+j], item_names[j]);
            }
        }
        printf("\n");
    }


	//CUT_EXIT(argc,argv);
    return 0;
}

