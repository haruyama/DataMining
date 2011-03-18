#pragma once
#include "recommend.hpp"
int parse(char* filename, int& item_length, int& user_length, int& entry_length, 
        char**& item_names, 
        int*& item_indexes,
        int*& item_user_lengths,
        data_type*& data);
