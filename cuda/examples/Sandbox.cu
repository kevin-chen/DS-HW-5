#include "../common/book.h"

__global__ void kernel() {    
    int width = 500;
    int height = 500;
    printf("Inside Kernel\n");
    // int tid = blockIdx.x;
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    printf("x: %d, y: %d\n", x, y);
    while ( x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // printf("x: %d, y: %d\n", x, y);
        printf("Within Boundary\n");
    }
}

int main( void ) {
    int width = 500;
    int height = 500;

    dim3 block(16, 16, 1);
    dim3 grid(width / block.x, height / block.y, 1); // 20/16 , 20/16 , 1

    // int grid = 1;
    // int block = 2;

    printf("Invoking Kernel\n");
    // kernel invocation code
    kernel<<<grid, block>>>();
    return 0;
}
