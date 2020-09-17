#include <iostream>

//#define __HIP_PLATFORM_HCC__

// hip header file
#include <xmmintrin.h>
#include "hip/hip_runtime.h"
#include "transpose.cuh"
#include <chrono>


#define WIDTH 1024


#define NUM (WIDTH * WIDTH)

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input, const unsigned int width) {
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            output[i * width + j] = input[j * width + i];
        }
    }
}

void matrixTransposeCPUSSE(float* output, float* input, const unsigned int width) {
    __m128i pmaxB = _mm_set1_epi8(0x0);
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            __m128 p0 = _mm_loadu_ps((float *) &input[j * width + i]);
            _mm_store_ss((float *)&output[i * width + j], p0);
        }
    }
}

inline int64_t clockCounter()
{
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

inline int64_t clockFrequency()
{
    return std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;
}


int main() {
    float* Matrix;
    float* TransposeMatrix;
    float* cpuTransposeMatrix;

    float* gpuMatrix;
    float* gpuTransposeMatrix;
    int64_t freq = clockFrequency(), t0, t1;


    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    int i;
    int errors;

    Matrix = (float*)malloc(NUM * sizeof(float));
    TransposeMatrix = (float*)malloc(NUM * sizeof(float));
    cpuTransposeMatrix = (float*)malloc(NUM * sizeof(float));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        Matrix[i] = (float)i * 10.0f;
    }

    // allocate the memory on the device side
    hipMalloc((void**)&gpuMatrix, NUM * sizeof(float));
    hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));

    // Memory transfer from host to device
    hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice);

    exec(gpuTransposeMatrix, gpuMatrix, WIDTH);     // first time 
    // Lauching kernel from host
    t0 = clockCounter();
    int N = 100;
    for(int i = 0; i < N; i++) {
      exec(gpuTransposeMatrix, gpuMatrix, WIDTH);
    }
    t1 = clockCounter();
    printf("OK: hipLaunchKernel() took %.3f msec (average over %d iterations)\n", (float)(t1-t0)*1000.0f/(float)freq/(float)N, N);

    // Memory transfer from device to host
    hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float),    hipMemcpyDeviceToHost);
    // CPU MatrixTranspose computation
    matrixTransposeCPUSSE(cpuTransposeMatrix, Matrix, WIDTH);

    // verify the results
    errors = 0;
    double eps = 1.0E-6;
    for (i = 0; i < NUM; i++) {
        if (std::abs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
            errors++;
            printf("Error at %d %f(ref) != %f\n", i, TransposeMatrix[i], cpuTransposeMatrix[i]);
        }
    }
    if (errors != 0) {
        printf("FAILED: %d errors\n", errors);
    } else {
        printf("PASSED!\n");
    }

    // free the resources on device side
    hipFree(gpuMatrix);
    hipFree(gpuTransposeMatrix);

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);

    return errors;
}
