#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

__global__ void convolutionForward(float* input, float* weights, float* bias, float* output,
                                   int input_size, int input_channels, int kernel_size, int num_filters);

__global__ void maxPoolForward(float* input, float* output, int input_size, int num_channels, int pool_size);

__global__ void reluForward(float* input, int size);

__global__ void fullyConnectedForward(float* input, float* weights, float* bias, float* output,
                                      int input_size, int output_size);

__global__ void softmaxForward(float* input, int size);

__global__ void convolutionBackward(float* input, float* output_grad, float* weights, float* input_grad,
                                    int input_size, int input_channels, int kernel_size, int num_filters);

__global__ void maxPoolBackward(float* input, float* output, float* output_grad, int input_size, int num_channels, int pool_size);

__global__ void reluBackward(float* input, float* output_grad, int size);

__global__ void fullyConnectedBackward(float* input, float* output_grad, float* weights, float* input_grad,
                                       int input_size, int output_size);

__global__ void softmaxBackward(float* output, float* target, float* input_grad, int size);

__global__ void updateWeights(float* weights, float* gradients, float learning_rate, int size);

__global__ void updateBias(float* bias, float* gradients, float learning_rate, int size);

#endif // CUDA_KERNELS_H
