#include <stdio.h>

// examples for each conditional macro
// #if
// #ifdef
// #ifndef
// #elif
// #else
// #endif

#define PI 3.14159
#define AREA(r) (PI * r * r)

#ifndef radius
#define radius 7
#endif

// if elif else logic
// we can only use integer constants in #if and #elif
#if radius > 10
#define radius 10
#elif radius < 5
#define radius 5
#else
#define radius 7
#endif


int main() {
    printf("Area of circle with radius %d: %f\n", radius, AREA(radius)); 
    printf("Area of circle with radius %f: %f\n", radius, AREA(radius)); 
    // 以下语句
    // 单独运行时，你翻到的是操作系统的垃圾桶，里面是随机的地址碎片。
    // 连续运行时，你翻到的是上一个 printf 刚倒掉的垃圾桶，里面是一个确定的副产物 0
    printf("Area of circle with radius %d: %d\n", radius, AREA(radius)); 
}