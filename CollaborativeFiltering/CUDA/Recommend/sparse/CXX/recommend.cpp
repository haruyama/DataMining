#include <math.h>
#include "recommend.hpp"
#include "parse.hpp"
#include "xmalloc.hpp"

static void calc(int i, int j, int item_length, const int* item_indexes, const int* item_user_lengths, const data_type* data, result_type* result) {
    int inter = 0;
    for(int ii = 0, jj = 0; (ii < item_user_lengths[i] && jj < item_user_lengths[j]);) {
        int tmp = data[item_indexes[i] + ii] - data[item_indexes[j] + jj];
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
        ret = (inter * 1.0f) / (item_user_lengths[i] + item_user_lengths[j] - inter);
    }
    result[i * item_length + j] = ret;
    result[j * item_length + i] = ret;
}


int main(int argc, char** argv) {
    char** item_names;
    int* item_indexes;
    int* item_user_lengths;
    data_type* data;

    result_type* result;

    int item_length;
    int user_length;
    int data_length;

    if (!parse(argv[1], item_length, user_length, data_length 
                , item_names, item_indexes, item_user_lengths, data)) {
        cerr << "parse error!" << endl;
        exit(1);
    }

    xmallocHost((void**)&result, sizeof(result_type) * item_length * item_length);

    for (int i = 0; i < item_length; ++i) {
        for (int j = i + 1; j < item_length; ++j) {
            calc(i, j, item_length, item_indexes, item_user_lengths, data, result);
        }
        result[i * item_length + i] = 0.0f;
    }


    cout.precision(3);
    for (int i = 0; i < item_length; ++i) {
        int base_i = i * item_length;
        cout << item_names[i] << ":";
        for (int j = 0; j < item_length; ++j) {
            if (result[base_i+j] > 0.0) {
                cout << result[base_i + j] << "|"  << item_names[j] << ",";
            }
        }
        cout << endl;
    }


    return 0;
}
