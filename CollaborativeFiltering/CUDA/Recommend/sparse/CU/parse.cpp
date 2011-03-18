#include <json/json.h>
#include "parse.hpp"
#include "xmalloc.hpp"

int parse(char* filename, int& item_length, int& user_length, int& data_length, 
        char**& item_names, 
        int*& item_indexes,
        int*& item_user_lengths,
        data_type*& data) {

    json_object* root = json_object_from_file(filename);

    int length = json_object_array_length(root);
    item_length = json_object_get_int(json_object_array_get_idx(root, 0));

    if (length != (item_length + 3)) {
        cerr << "length error" << endl;
        return 0;
    }

    xmallocHost((void**)&item_names, sizeof(char*)*item_length);

    xmallocHost((void**)&item_indexes, sizeof(int)*item_length);

    xmallocHost((void**)&item_user_lengths, sizeof(int)*item_length);

    user_length = json_object_get_int(json_object_array_get_idx(root, 1));

    data_length = json_object_get_int(json_object_array_get_idx(root, 2));

    xmallocHost((void**)&data, sizeof(data_type) * data_length);

    int data_index = 0;
    for (int i = 0; i < item_length; ++i) {
        json_object* array_object = json_object_array_get_idx(root, i+3);
        int array_length = json_object_array_length(array_object);
        item_user_lengths[i] = array_length - 1;
        char* name = strdup(json_object_get_string(json_object_array_get_idx(array_object, 0)));
        if (!name) {
            cerr << "name strdup error" << endl;
            exit(1);
        }
        item_names[i] = name;
        item_indexes[i] = data_index;
        for (int j = 1; j < array_length; ++j) {
            data[data_index++] = json_object_get_int(json_object_array_get_idx(array_object, j));
        }
    }

    return 1;
}
