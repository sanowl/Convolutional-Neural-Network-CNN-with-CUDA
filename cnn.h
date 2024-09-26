#ifndef CNN_H
#define CNN_H

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <vector>

#define INPUT_SIZE 32
#define INPUT_CHANNELS 3
#define CONV1_FILTERS 32
#define CONV1_SIZE 5
#define POOL1_SIZE 2
#define CONV2_FILTERS 64
#define CONV2_SIZE 5
#define POOL2_SIZE 2
#define FC1_SIZE 1024
#define FC2_SIZE 84
#define NUM_CLASSES 10

class CNN {
public:
    CNN();
    ~CNN();
    void forward(float* input);
    void backward(float* target);
    void update(float learning_rate);
    float* getOutput();

private:
    // Layers
    float *d_input, *d_conv1, *d_pool1, *d_conv2, *d_pool2, *d_fc1, *d_fc2, *d_output;
    float *d_conv1_weights, *d_conv2_weights, *d_fc1_weights, *d_fc2_weights;
    float *d_conv1_bias, *d_conv2_bias, *d_fc1_bias, *d_fc2_bias;

    // Gradients
    float *d_conv1_grad, *d_conv2_grad, *d_fc1_grad, *d_fc2_grad;
    float *d_conv1_bias_grad, *d_conv2_bias_grad, *d_fc1_bias_grad, *d_fc2_bias_grad;

    // Helper functions
    void initializeWeights();
    void allocateMemory();
    void freeMemory();

    // CUDA handles
    cublasHandle_t cublasHandle;
    curandGenerator_t curandGenerator;
};

#endif // CNN_H