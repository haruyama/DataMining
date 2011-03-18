#include <math.h>
#include "recommend.hpp"
#include "parse.hpp"
#include "xmalloc.hpp"

#ifdef _GLIBCXX_ATOMIC_BUILTINS_4
#undef _GLIBCXX_ATOMIC_BUILTINS_4
#endif

#ifdef _GLIBCXX_ATOMIC_BUILTINS 
#undef _GLIBCXX_ATOMIC_BUILTINS
#endif

__global__
void calc(int i, int item_length, int* item_indexes_d, int* item_user_lengths_d, data_type* data_d, result_type* result_d) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= j || j >= item_length) {
        return;
    }

    int i_index = item_indexes_d[i];
    int j_index = item_indexes_d[j];
    int i_length = item_user_lengths_d[i];
    int j_length = item_user_lengths_d[j];

    int inter = 0;
    for(int ii = 0, jj = 0; (ii < i_length && jj < j_length);) {
        int tmp = data_d[i_index + ii] - data_d[j_index + jj];
        if (tmp == 0) {
            ++inter;
            ++ii;
            ++jj;
        } else if (tmp > 0) {
            ++jj;
        } else {
            ++ii;
        }
    }
    result_type ret = 0.0f;
    if (inter) {
        ret = (inter * 1.0f) / (i_length + j_length - inter);
    }
    result_d[j] = ret;
}


int main(int argc, char** argv) {
    char** item_names;

    int* item_indexes;
    int* item_user_lengths;
    data_type* data;
    result_type* result;

    int* item_indexes_d;
    int* item_user_lengths_d;
    data_type* data_d;
    result_type* result_i_d;

    int item_length;
    int user_length;
    int data_length;

    if (!parse(argv[1], item_length, user_length, data_length 
                , item_names, item_indexes, item_user_lengths, data)) {
        cerr << "parse error!" << endl;
        exit(1);
    }

    xmallocHost((void**)&result, sizeof(result_type) * item_length * item_length);
    //xmallocHost((void**)&result_i, sizeof(result_type) * item_length);

    xmallocDevice((void**)&item_indexes_d, sizeof(int) * item_length);
    xmallocDevice((void**)&item_user_lengths_d, sizeof(int) * item_length);
    
    xmallocDevice((void**)&data_d, sizeof(data_type) * data_length);

    //xmallocDevice((void**)&result_d, sizeof(result_type) * item_length * item_length);
    xmallocDevice((void**)&result_i_d, sizeof(result_type) * item_length);

    int number_of_grid_x =  (int)(ceil(item_length * 1.0 / BLOCK_SIZE));


    cudaMemcpy(item_indexes_d, item_indexes, sizeof(int) * item_length, 
            cudaMemcpyHostToDevice);

    cudaMemcpy(item_user_lengths_d, item_user_lengths, sizeof(int) * item_length, 
            cudaMemcpyHostToDevice);

    cudaMemcpy(data_d, data, sizeof(data_type) * data_length, 
            cudaMemcpyHostToDevice);

    cudaThreadSynchronize();

    dim3 block(BLOCK_SIZE, 1);
    dim3 calc2d_grid(number_of_grid_x, 1);

    for (int i = 0; i < item_length; ++i) {
        calc<<<calc2d_grid, block>>>(i, item_length, item_indexes_d, item_user_lengths_d, data_d, result_i_d);
        cudaMemcpy(result + (i * item_length) , result_i_d, sizeof(result_type) * item_length,
                cudaMemcpyDeviceToHost);
    }

    cudaThreadSynchronize();

    cout.precision(3);
    for (int i = 0; i < item_length; ++i) {
        int base_i = i * item_length;
        cout << item_names[i] << ":";
        for (int j = 0; j < i; ++j) {
            int base_j = j * item_length;
            if (result[base_j + i] > 0.0) {
                cout << result[base_j + i] << "|"  << item_names[j] << ",";
            }
        }
        for (int j = i + 1; j < item_length; ++j) {
            if (result[base_i+j] > 0.0) {
                cout << result[base_i + j] << "|"  << item_names[j] << ",";
            }
        }
        cout << endl;
    }


    return 0;
}
