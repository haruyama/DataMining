#include "xmalloc.hpp"

#include <iostream>

void xmallocHost(void** ptr, const size_t count) {
    *ptr = malloc(count);
    if (!ptr) {
        cerr << "xmallocHost error" << endl;
        exit(1);
    }
}

void xmallocDevice(void** ptr, const size_t count) {
    xmallocHost(ptr, count);
}
