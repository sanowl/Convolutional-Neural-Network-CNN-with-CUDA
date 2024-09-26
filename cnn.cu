#include "cnn.h"
#include "cuda_kernels.h"
#include <cstdio>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

CNN::CNN() {
    CUDA_CHECK(cublasCreate(&cublasHandle));
    CUDA_CHECK(curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT));
    CUDA_CHECK(curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL));

    allocateMemory();
    initializeWeights();
}

CNN::~CNN() {
    freeMemory();
    CUDA_CHECK(cublasDestroy(cublasHandle));
    CUDA_CHECK(curandDestroyGenerator(curandGenerator));
}

void CNN::forward(float* input) {
    CUDA_CHECK(cudaMemcpy(d_input, input, INPUT_SIZE * INPUT_SIZE * INPUT_CHANNELS * sizeof(float), cudaMemcpyHostToDevice));

    // Conv1
    convolutionForward<<<dim3(32, 32), dim3(16, 16)>>>(d_input, d_conv1_weights, d_conv1_bias, d_conv1, 
        INPUT_SIZE, INPUT_CHANNELS, CONV1_SIZE, CONV1_FILTERS);
    reluForward<<<dim3(32, 32), 256>>>(d_conv1, INPUT_SIZE * INPUT_SIZE * CONV1_FILTERS);

    // Pool1
    maxPoolForward<<<dim3(16, 16), dim3(16, 16)>>>(d_conv1, d_pool1, INPUT_SIZE, CONV1_FILTERS, POOL1_SIZE);

    // Conv2
    convolutionForward<<<dim3(16, 16), dim3(16, 16)>>>(d_pool1, d_conv2_weights, d_conv2_bias, d_conv2, 
        INPUT_SIZE / 2, CONV1_FILTERS, CONV2_SIZE, CONV2_FILTERS);
    reluForward<<<dim3(16, 16), 256>>>(d_conv2, (INPUT_SIZE / 2) * (INPUT_SIZE / 2) * CONV2_FILTERS);

    // Pool2
    maxPoolForward<<<dim3(8, 8), dim3(16, 16)>>>(d_conv2, d_pool2, INPUT_SIZE / 2, CONV2_FILTERS, POOL2_SIZE);

    // FC1
    int fc1_input_size = (INPUT_SIZE / 4) * (INPUT_SIZE / 4) * CONV2_FILTERS;
    fullyConnectedForward<<<dim3(32), dim3(32)>>>(d_pool2, d_fc1_weights, d_fc1_bias, d_fc1, fc1_input_size, FC1_SIZE);
    reluForward<<<dim3(32), 32>>>(d_fc1, FC1_SIZE);

    // FC2
    fullyConnectedForward<<<dim3(32), dim3(32)>>>(d_fc1, d_fc2_weights, d_fc2_bias, d_fc2, FC1_SIZE, FC2_SIZE);
    reluForward<<<dim3(32), 32>>>(d_fc2, FC2_SIZE);

    // Output
    fullyConnectedForward<<<dim3(1), dim3(256)>>>(d_fc2, d_fc2_weights, d_fc2_bias, d_output, FC2_SIZE, NUM_CLASSES);
    softmaxForward<<<dim3(1), dim3(256)>>>(d_output, NUM_CLASSES);
}

void CNN::backward(float* target) {
    // Output layer
    softmaxBackward<<<dim3(1), dim3(256)>>>(d_output, target, d_fc2_grad, NUM_CLASSES);

    // FC2
    fullyConnectedBackward<<<dim3(32), dim3(32)>>>(d_fc2, d_fc2_grad, d_fc2_weights, d_fc1_grad, FC2_SIZE, FC1_SIZE);
    reluBackward<<<dim3(32), 32>>>(d_fc2, d_fc2_grad, FC2_SIZE);

    // FC1
    int fc1_input_size = (INPUT_SIZE / 4) * (INPUT_SIZE / 4) * CONV2_FILTERS;
    fullyConnectedBackward<<<dim3(32), dim3(32)>>>(d_fc1, d_fc1_grad, d_fc1_weights, d_pool2, FC1_SIZE, fc1_input_size);
    reluBackward<<<dim3(32), 32>>>(d_fc1, d_fc1_grad, FC1_SIZE);

    // Pool2
    maxPoolBackward<<<dim3(8, 8), dim3(16, 16)>>>(d_conv2, d_pool2, d_conv2_grad, INPUT_SIZE / 2, CONV2_FILTERS, POOL2_SIZE);

    // Conv2
    convolutionBackward<<<dim3(16, 16), dim3(16, 16)>>>(d_pool1, d_conv2_grad, d_conv2_weights, d_conv1_grad, 
        INPUT_SIZE / 2, CONV1_FILTERS, CONV2_SIZE, CONV2_FILTERS);

    // Pool1
    maxPoolBackward<<<dim3(16, 16), dim3(16, 16)>>>(d_conv1, d_pool1, d_conv1_grad, INPUT_SIZE, CONV1_FILTERS, POOL1_SIZE);

    // Conv1
    convolutionBackward<<<dim3(32, 32), dim3(16, 16)>>>(d_input, d_conv1_grad, d_conv1_weights, nullptr, 
        INPUT_SIZE, INPUT_CHANNELS, CONV1_SIZE, CONV1_FILTERS);
}

void CNN::update(float learning_rate) {
    updateWeights<<<dim3(32), dim3(32)>>>(d_conv1_weights, d_conv1_grad, learning_rate, 
        CONV1_FILTERS * INPUT_CHANNELS * CONV1_SIZE * CONV1_SIZE);
    updateWeights<<<dim3(32), dim3(32)>>>(d_conv2_weights, d_conv2_grad, learning_rate, 
        CONV2_FILTERS * CONV1_FILTERS * CONV2_SIZE * CONV2_SIZE);
    updateWeights<<<dim3(32), dim3(32)>>>(d_fc1_weights, d_fc1_grad, learning_rate, 
        FC1_SIZE * ((INPUT_SIZE / 4) * (INPUT_SIZE / 4) * CONV2_FILTERS));
    updateWeights<<<dim3(32), dim3(32)>>>(d_fc2_weights, d_fc2_grad, learning_rate, 
        FC2_SIZE * FC1_SIZE);

    updateBias<<<dim3(1), dim3(256)>>>(d_conv1_bias, d_conv1_bias_grad, learning_rate, CONV1_FILTERS);
    updateBias<<<dim3(1), dim3(256)>>>(d_conv2_bias, d_conv2_bias_grad, learning_rate, CONV2_FILTERS);
    updateBias<<<dim3(4), dim3(256)>>>(d_fc1_bias, d_fc1_bias_grad, learning_rate, FC1_SIZE);
    updateBias<<<dim3(1), dim3(256)>>>(d_fc2_bias, d_fc2_bias_grad, learning_rate, FC2_SIZE);
}

float* CNN::getOutput() {
    float* output = new float[NUM_CLASSES];
    CUDA_CHECK(cudaMemcpy(output, d_output, NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost));
    return output;
}

void CNN::initializeWeights() {
    curandGenerateNormal(curandGenerator, d_conv1_weights, CONV1_FILTERS * INPUT_CHANNELS * CONV1_SIZE * CONV1_SIZE, 0.0f, 0.01f);
    curandGenerateNormal(curandGenerator, d_conv2_weights, CONV2_FILTERS * CONV1_FILTERS * CONV2_SIZE * CONV2_SIZE, 0.0f, 0.01f);
    curandGenerateNormal(curandGenerator, d_fc1_weights, FC1_SIZE * ((INPUT_SIZE / 4) * (INPUT_SIZE / 4) * CONV2_FILTERS), 0.0f, 0.01f);
    curandGenerateNormal(curandGenerator, d_fc2_weights, FC2_SIZE * FC1_SIZE, 0.0f, 0.01f);

    CUDA_CHECK(cudaMemset(d_conv1_bias, 0, CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_conv2_bias, 0, CONV2_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fc1_bias, 0, FC1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fc2_bias, 0, FC2_SIZE * sizeof(float)));
}

void CNN::allocateMemory() {
    CUDA_CHECK(cudaMalloc(&d_input, INPUT_SIZE * INPUT_SIZE * INPUT_CHANNELS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1, INPUT_SIZE * INPUT_SIZE * CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool1, (INPUT_SIZE / 2) * (INPUT_SIZE / 2) * CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2, (INPUT_SIZE / 2) * (INPUT_SIZE / 2) * CONV2_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool2, (INPUT_SIZE / 4) * (INPUT_SIZE / 4) * CONV2_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc1, FC1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc2, FC2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, NUM_CLASSES * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_conv1_weights, CONV1_FILTERS * INPUT_CHANNELS * CONV1_SIZE * CONV1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_weights, CONV2_FILTERS * CONV1_FILTERS * CONV2_SIZE * CONV2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc1_weights, FC1_SIZE * ((INPUT_SIZE / 4) * (INPUT_SIZE / 4) * CONV2_FILTERS) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc2_weights, FC2_SIZE * FC1_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_conv1_bias, CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_bias, CONV2_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc1_bias, FC1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc2_bias, FC2_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_conv1_grad, INPUT_SIZE * INPUT_SIZE * CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_grad, (INPUT_SIZE / 2) * (INPUT_SIZE / 2) * CONV2_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc1_grad, FC1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc2_grad, FC2_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_conv1_bias_grad, CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_bias_grad, CONV2_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc1_bias_grad, FC1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc2_bias_grad, FC2_SIZE * sizeof(float)));
}

void CNN::freeMemory() {
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_conv1));
    CUDA_CHECK(cudaFree(d_pool1));
    CUDA_CHECK(cudaFree(d_conv2));
    CUDA_CHECK(cudaFree(d_pool2));
    CUDA_CHECK(cudaFree(d_fc1));
    CUDA_CHECK(cudaFree(d_fc2));
    CUDA_CHECK(cudaFree(d_output));

    CUDA_CHECK(cudaFree(d_conv1_weights));
    CUDA_CHECK(cudaFree(d_conv2_weights));
    CUDA_CHECK(cudaFree(d_fc1_weights));
    CUDA_CHECK(cudaFree(d_fc2_weights));

    CUDA_CHECK(cudaFree(d_conv1_bias));
    CUDA_CHECK(cudaFree(d_conv2_bias));
    CUDA_CHECK(cudaFree(d_fc1_bias));
    CUDA_CHECK(cudaFree(d_fc2_bias));

    CUDA_CHECK(cudaFree(d_conv1_grad));
    CUDA_CHECK(cudaFree(d_conv2_grad));
    CUDA_CHECK(cudaFree(d_fc1_grad));
    CUDA_CHECK(cudaFree(d_fc2_grad));

    CUDA_CHECK(cudaFree(d_conv1_bias_grad));
    CUDA_CHECK(cudaFree(d_conv2_bias_grad));
    CUDA_CHECK(cudaFree(d_fc1_bias_grad));
    CUDA_CHECK(cudaFree(d_fc2_bias_grad));
}
