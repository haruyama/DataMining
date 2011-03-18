#include <stdio.h>

int main(void) {
    printf("%d\n", sizeof(char));
    printf("%d\n", sizeof(short));
    printf("%d\n", sizeof(int));
    printf("%d\n", sizeof(long));
    printf("%d\n", sizeof(long long));

    printf("%d\n", sizeof(float));
    printf("%d\n", sizeof(double));
    printf("%f\n", 2000 * 1.0/1024);

    return 0;
}
