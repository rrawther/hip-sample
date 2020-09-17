/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
/* HIT_START
 * BUILD: %t %s ../test_common.cpp LINK_OPTIONS hiprtc EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */


#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <iostream>
#include <iterator>
#include <vector>
#include <chrono>

//#include "test_common.h"

static constexpr auto NUM_THREADS{1024};
static constexpr auto NUM_BLOCKS{128};
#define WIDTH 1024
#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

static constexpr auto matrixTranspose{
R"(
#include <hip/hip_runtime.h>
extern "C"
__global__ void
matrixTranspose(float* out, float* in, const int width) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    out[y * width + x] = in[x * width + y];

}
)"};

static constexpr auto saxpy{
R"(
#include <hip/hip_runtime.h>
extern "C"
__global__
void saxpy(float a, float* x, float* y, float* out, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
       out[tid] = a * x[tid] + y[tid];
    }
}
)"};


inline int64_t clockCounter()
{
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

inline int64_t clockFrequency()
{
    return std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;
}

#if 0
int main()
{
    using namespace std;
    int64_t freq = clockFrequency(), t0, t1;
    t0 = clockCounter();

    hiprtcProgram prog;
    hiprtcCreateProgram(&prog,      // prog
                        saxpy,      // buffer
                        "saxpy.cu", // name
                        0,          // numHeaders
                        nullptr,    // headers
                        nullptr);   // includeNames

    hipDeviceProp_t props;
    int device = 0;
    hipGetDeviceProperties(&props, device);
    std::string gfxName = "gfx" + std::to_string(props.gcnArch);
    std::string sarg = "--gpu-architecture=" + gfxName;
    const char* options[] = {
        sarg.c_str()
    };

    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, options)};

    size_t logSize;
    hiprtcGetProgramLogSize(prog, &logSize);

    if (logSize) {
        string log(logSize, '\0');
        hiprtcGetProgramLog(prog, &log[0]);

        cout << log << '\n';
    }

    if (compileResult != HIPRTC_SUCCESS) { printf("Compilation failed."); }

    size_t codeSize;
    hiprtcGetCodeSize(prog, &codeSize);

    vector<char> code(codeSize);
    hiprtcGetCode(prog, code.data());

    hiprtcDestroyProgram(&prog);

    hipModule_t module;
    hipFunction_t kernel;

    hipModuleLoadData(&module, code.data());
    hipModuleGetFunction(&kernel, module, "saxpy");

    size_t n = NUM_THREADS * NUM_BLOCKS;
    size_t bufferSize = n * sizeof(float);

    float a = 5.1f;
    unique_ptr<float[]> hX{new float[n]};
    unique_ptr<float[]> hY{new float[n]};
    unique_ptr<float[]> hOut{new float[n]};

    for (size_t i = 0; i < n; ++i) {
        hX[i] = static_cast<float>(i);
        hY[i] = static_cast<float>(i * 2);
    }
    t1 = clockCounter();
    printf("OK: Hip compile code took %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);


    hipDeviceptr_t dX, dY, dOut;
    hipMalloc(&dX, bufferSize);
    hipMalloc(&dY, bufferSize);
    hipMalloc(&dOut, bufferSize);
    hipMemcpyHtoD(dX, hX.get(), bufferSize);
    hipMemcpyHtoD(dY, hY.get(), bufferSize);

    struct {
        float a_;
        hipDeviceptr_t b_;
        hipDeviceptr_t c_;
        hipDeviceptr_t d_;
        size_t e_;
    } args{a, dX, dY, dOut, n};

    auto size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};
    t0 = clockCounter();
    int N = 100;
    for(int i = 0; i < N; i++) {
        hipModuleLaunchKernel(kernel, NUM_BLOCKS, 1, 1, NUM_THREADS, 1, 1,
                              0, nullptr, nullptr, config);
    }
    t1 = clockCounter();
    printf("OK: hipModuleLaunchKernel() took %.3f msec (average over %d iterations)\n", (float)(t1-t0)*1000.0f/(float)freq/(float)N, N);


    hipMemcpyDtoH(hOut.get(), dOut, bufferSize);

    for (size_t i = 0; i < n; ++i) {
       if (a * hX[i] + hY[i] != hOut[i]) { printf("Validation failed."); }
    }

    hipFree(dX);
    hipFree(dY);
    hipFree(dOut);

    hipModuleUnload(module);

    printf("Hiprtc success\n");
}
#else
int main()
{
    using namespace std;
    int64_t freq = clockFrequency(), t0, t1;
    t0 = clockCounter();

    hiprtcProgram prog;
    hiprtcCreateProgram(&prog,      // prog
                        matrixTranspose,      // buffer
                        "matrixTranspose.cu", // name
                        0,          // numHeaders
                        nullptr,    // headers
                        nullptr);   // includeNames

    hipDeviceProp_t props;
    int device = 0;
    hipGetDeviceProperties(&props, device);
    std::string gfxName = "gfx" + std::to_string(props.gcnArch);
    std::string sarg = "--gpu-architecture=" + gfxName;
    const char* options[] = {
        sarg.c_str()
    };

    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, options)};

    size_t logSize;
    hiprtcGetProgramLogSize(prog, &logSize);

    if (logSize) {
        string log(logSize, '\0');
        hiprtcGetProgramLog(prog, &log[0]);

        cout << log << '\n';
    }

    if (compileResult != HIPRTC_SUCCESS) { printf("Compilation failed."); }

    size_t codeSize;
    hiprtcGetCodeSize(prog, &codeSize);

    vector<char> code(codeSize);
    hiprtcGetCode(prog, code.data());

    hiprtcDestroyProgram(&prog);

    hipModule_t module;
    hipFunction_t kernel;

    hipModuleLoadData(&module, code.data());
    hipModuleGetFunction(&kernel, module, "matrixTranspose");

    size_t n = WIDTH * WIDTH;
    size_t bufferSize = n * sizeof(float);

    //float a = 5.1f;
    unique_ptr<float[]> hX{new float[n]};
    //unique_ptr<float[]> hY{new float[n]};
    unique_ptr<float[]> hOut{new float[n]};

    for (size_t i = 0; i < n; ++i) {
        hX[i] = static_cast<float>(i)*10.f;
        //hY[i] = static_cast<float>(i * 2);
    }
    t1 = clockCounter();
    printf("OK: Hip compile code took %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);


    hipDeviceptr_t dX, dOut;
    hipMalloc(&dX, bufferSize);
    //hipMalloc(&dY, bufferSize);
    hipMalloc(&dOut, bufferSize);
    hipMemcpyHtoD(dX, hX.get(), bufferSize);
    //hipMemcpyHtoD(dY, hY.get(), bufferSize);

    struct {
        //float a_;
        hipDeviceptr_t b_;
        hipDeviceptr_t c_;
//        hipDeviceptr_t d_;
        int e_;
    } args{dOut, dX, WIDTH};

    auto size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};
    hipModuleLaunchKernel(kernel, WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y, 1, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1,
                          0, nullptr, nullptr, config);
    t0 = clockCounter();
    int N = 100;
    for(int i = 0; i < N; i++) {
        hipModuleLaunchKernel(kernel, WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y, 1, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1,
                              0, nullptr, nullptr, config);
    }
    t1 = clockCounter();
    printf("OK: hipModuleLaunchKernel() took %.3f msec (average over %d iterations)\n", (float)(t1-t0)*1000.0f/(float)freq/(float)N, N);


    hipMemcpyDtoH(hOut.get(), dOut, bufferSize);

   // for (size_t i = 0; i < n; ++i) {
   //    if (a * hX[i] + hY[i] != hOut[i]) { printf("Validation failed."); }
   // }

    hipFree(dX);
   // hipFree(dY);
    hipFree(dOut);

    hipModuleUnload(module);

    printf("Hiprtc success\n");
}
#endif