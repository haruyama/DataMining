#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <json/json.h>

const int UNIT = 1024;

//typedef double data_type;
//typedef double result_type;
typedef int data_type;
typedef float result_type;


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

    user_ceil_length =  (int)(ceil(user_length * 1.0 / UNIT)) * UNIT;

    data = (data_type*)calloc(sizeof(data_type),user_ceil_length*item_length);

    if (!data) {
        fprintf(stderr, "data malloc error\n");
        return 0;
    }

    for (int i = 0; i < item_length; ++i) {
        json_object* array_object = json_object_array_get_idx(root, i+2);
        int array_length = json_object_array_length(array_object);
        char* name = strdup(json_object_get_string(json_object_array_get_idx(array_object, 0)));
        item_names[i] = name;
        int base = i*user_ceil_length;
        for (int j = 1; j < array_length; ++j) {
            int user = json_object_get_int(json_object_array_get_idx(array_object, j)); 
            data[base + user] = 1;
        }
    }

    return 1;
}

static result_type reduce(int base, int length, data_type* data) {
    result_type ret = 0.0;
    for (int i = 0; i < length; ++i) {
        ret += data[base+i];
    }
    return ret;
}


static result_type calc2(int item_length, int user_ceil_length, int i, int j, data_type* data, data_type* similarity, data_type* reduce_cache) {

    int base_i = user_ceil_length * i;
    int base_j = user_ceil_length * j;
    for (int k = 0; k < user_ceil_length; ++k) {
        similarity[k] = data[base_i + k] & data[base_j + k];
    }
    result_type sim = reduce(0, user_ceil_length, similarity);

    result_type ret = 0.0;
    if (sim > 0.0) {
        ret = (sim / (reduce_cache[i] + reduce_cache[j] - sim));
    }

    /*
    if (sim > 0.0) {
        printf("ret: %f\n", ret);
        printf("reduce_i: %f\n", reduce_i);
        printf("reduce_j: %f\n", reduce_j);
        printf("sim: %f\n", sim);

    }
    */

    return ret;
}
   

static int calc(int item_length, int user_ceil_length, data_type* data, result_type* result) {

    data_type* similarity = (data_type*)malloc(sizeof(data_type)*user_ceil_length);
    data_type* reduce_cache = (data_type*)malloc(sizeof(data_type)*item_length);

    for (int i = 0; i < item_length; ++i) {
        int base_i = user_ceil_length * i;
        reduce_cache[i] = reduce(base_i, user_ceil_length, data);
    }

    if (!similarity) {
        fprintf(stderr, "similarity malloc error\n");
        exit(1);
    }

    for(int i = 0; i < item_length; ++i) {
        int base_i = item_length * i;
        for(int j = i + 1; j < item_length; ++j) {
            int base_j = item_length * j;
            result_type sim = calc2(item_length, user_ceil_length, i, j, data, similarity, reduce_cache);
            result[base_i + j] = sim;
            result[base_j + i] = sim;
        }
        result[base_i + i] = 0.0;
    }

    free(similarity);
    free(reduce_cache);
    return 1;
}

int main(int argc, char** argv) {
    char** item_names = NULL;
    data_type* data = NULL;

    int item_length;
    int user_ceil_length;

    if (!parse(argv[1], item_length, user_ceil_length, item_names, data)) {
        fprintf(stderr, "parse errof!\n");
        exit(1);
    }

    result_type* result = (result_type*)malloc(sizeof(result_type)*item_length*item_length);
    if (!calc(item_length, user_ceil_length, data, result)) {
        fprintf(stderr, "calc errof!\n");
        exit(1);
    }

    for (int i = 0; i < item_length; ++i) {
        int base_i = i * item_length;
        printf("%s:", item_names[i]);
        for (int j = 0; j < item_length; ++j) {
            if (result[base_i+j] > 0.0) {
                printf("%.3f|%s,", result[base_i+j], item_names[j]);
            }
        }
        printf("\n");
    }

    free(data);
    free(result);
    free(item_names);


    return 0;
}
