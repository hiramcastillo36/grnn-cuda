#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>

struct GRNNParams {
    float sigma;
    int n_train;
    int n_features;
};

__global__ void patternLayerKernel(
    const float* X_train,
    const float* y_train,
    const float* X_test,
    float* numerator,
    float* denominator,
    const GRNNParams params,
    const int n_test
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_test) {
        float num = 0.0f;
        float den = 0.0f;

        for(int i = 0; i < params.n_train; i++) {
            float dist = 0.0f;

            for(int j = 0; j < params.n_features; j++) {
                float diff = X_test[tid * params.n_features + j] -
                            X_train[i * params.n_features + j];
                dist += diff * diff;
            }

            float weight = exp(-dist / (2.0f * params.sigma * params.sigma));

            num += weight * y_train[i];
            den += weight;
        }

        numerator[tid] = num;
        denominator[tid] = den;
    }
}

__global__ void summationLayerKernel(
    float* numerator,
    float* denominator,
    float* output,
    const int n_test
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_test) {
        if (denominator[tid] > 1e-10f) {
            output[tid] = numerator[tid] / denominator[tid];
        } else {
            output[tid] = 0.0f;
        }
    }
}


void predictGRNN(
    const float* X_train,
    const float* y_train,
    const float* X_test,
    float* output,
    const GRNNParams params,
    const int n_test
) {

    float *d_X_train, *d_y_train, *d_X_test;
    float *d_numerator, *d_denominator, *d_output;

    cudaMalloc(&d_X_train, params.n_train * params.n_features * sizeof(float));
    cudaMalloc(&d_y_train, params.n_train * sizeof(float));
    cudaMalloc(&d_X_test, n_test * params.n_features * sizeof(float));
    cudaMalloc(&d_numerator, n_test * sizeof(float));
    cudaMalloc(&d_denominator, n_test * sizeof(float));
    cudaMalloc(&d_output, n_test * sizeof(float));

    cudaMemcpy(d_X_train, X_train, params.n_train * params.n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_train, y_train, params.n_train * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X_test, X_test, n_test * params.n_features * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n_test + blockSize - 1) / blockSize;

    patternLayerKernel<<<numBlocks, blockSize>>>(
        d_X_train, d_y_train, d_X_test,
        d_numerator, d_denominator, params, n_test
    );

    summationLayerKernel<<<numBlocks, blockSize>>>(
        d_numerator, d_denominator, d_output, n_test
    );

    cudaMemcpy(output, d_output, n_test * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_X_train);
    cudaFree(d_y_train);
    cudaFree(d_X_test);
    cudaFree(d_numerator);
    cudaFree(d_denominator);
    cudaFree(d_output);
}
