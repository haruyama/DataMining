#pragma once
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;


typedef int data_type;
typedef float result_type;

const int BLOCK_SIZE = 512;

#ifdef _GLIBCXX_ATOMIC_BUILTINS_4
#undef _GLIBCXX_ATOMIC_BUILTINS_4
#endif

#ifdef _GLIBCXX_ATOMIC_BUILTINS 
#undef _GLIBCXX_ATOMIC_BUILTINS
#endif
