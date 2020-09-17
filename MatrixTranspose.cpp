#include <iostream>

#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#define WIDTH 1024

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

// Device (Kernel) function, it must be void
__global__ void
__attribute__((visibility("default")))
matrixTranspose(float* out, float* in, const int width) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    out[y * width + x] = in[x * width + y];

}


void exec(float* gpuTransposeMatrix, float* gpuMatrix, const int width) {
    // Lauching kernel from host
    hipLaunchKernelGGL(
        matrixTranspose,
        dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0,
        gpuTransposeMatrix, gpuMatrix, WIDTH);
}
