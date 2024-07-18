#include "CalculatingPi_gpu.cuh"
#include "utils.h"

#include <numeric>

static void HandleError(const cudaError_t err, const char *file, const int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))


// using float_type = long double;  // Error #20208-D: 'long double' is treated as 'double' in device code
using float_type = double;


__global__ void pi_numerical_integral(float_type *result, const unsigned long int iterations) {
    const unsigned int n_threads = gridDim.x * blockDim.x;
    const unsigned long int chunk_size = (iterations + n_threads - 1) / n_threads;

    const unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned long int start = index * chunk_size;
    const unsigned long int end = start + chunk_size - 1 > iterations ? iterations : start + chunk_size - 1;

    float_type sum = 0;
    const float_type width = 2.0 / iterations;
    double x;

    for (unsigned long int i = start; i < end; i++) {
        x = -1.0 + (i + 0.5) * width;
        sum += sqrt(1 - x * x) * width;
    }

    result[index] = sum;
}

long double pi_num_integration_gpu(const std::string &name, const unsigned long long iterations) {
    // check for GPU
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
        return 0;
    }

    auto chronometer = StopWatch("Inside start: " + name);

    // https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    int blockSize; // The launch configurator returned block size
    int gridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    HANDLE_ERROR(cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, pi_numerical_integral, 0, 0));

    const int n_threads = gridSize * blockSize;
    float_type result[n_threads];
    float_type *dev_result;

    HANDLE_ERROR(cudaMalloc((void **) &dev_result, n_threads * sizeof(float_type)));
    pi_numerical_integral<<<gridSize, blockSize>>>(dev_result, iterations);
    HANDLE_ERROR(cudaMemcpy(result, dev_result, n_threads * sizeof(float_type), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dev_result));

    // result array has only a few thousand items. It's not necessary to use:
    // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    // https://github.com/mark-poscablo/gpu-sum-reduction
    long double pi = std::reduce(result, result + n_threads, static_cast<float_type>(0));
    pi *= 2;

    chronometer.elapsed("Inside end: " + name + " Duration:");
    return pi;
}



__global__ void pi_leibniz(float_type *result, const unsigned long int iterations) {
    const unsigned int n_threads = gridDim.x * blockDim.x;
    const unsigned long int chunk_size = (iterations + n_threads - 1) / n_threads;

    const unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned long int start = (chunk_size * index) + 1;
    const unsigned long int end = start + chunk_size - 1 > iterations ? iterations : start + chunk_size - 1;

    float_type s = start == 1 ? 1: 0;
    float_type k = 2.0 * start + 1;

    for (unsigned long long i = start; i <= end; i++){
        // s += pow(-1, i) / (2 * i + 1);
        if (i % 2 == 0)  // much faster
            s += 1.0 / k;
        else
            s -= 1.0 / k;
        k += 2;
    }

    result[index] = s;
}

long double pi_leibniz_gpu(const std::string &name, const unsigned long long iterations) {
    // check for GPU
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
        return 0;
    }

    auto chronometer = StopWatch("Inside start: " + name);

    // https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    int blockSize; // The launch configurator returned block size
    int gridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    HANDLE_ERROR(cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, pi_leibniz, 0, 0));

    const int n_threads = gridSize * blockSize;
    float_type result[n_threads];
    float_type *dev_result;

    HANDLE_ERROR(cudaMalloc((void **) &dev_result, n_threads * sizeof(float_type)));
    pi_leibniz<<<gridSize, blockSize>>>(dev_result, iterations);
    HANDLE_ERROR(cudaMemcpy(result, dev_result, n_threads * sizeof(float_type), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dev_result));

    // result array has only a few thousand items. It's not necessary to use:
    // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    // https://github.com/mark-poscablo/gpu-sum-reduction
    long double pi = std::reduce(result, result + n_threads, static_cast<float_type>(0));
    pi *= 4;

    chronometer.elapsed("Inside end: " + name + " Duration:");
    return pi;
}
