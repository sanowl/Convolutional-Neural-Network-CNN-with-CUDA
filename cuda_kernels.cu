#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void convolutionForward(float* input, float* weights, float* bias, float* output,
                                   int input_size, int input_channels, int kernel_size, int num_filters) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z;

    if (x < input_size && y < input_size && f < num_filters) {
        float sum = 0.0f;
        for (int c = 0; c < input_channels; ++c) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int ix = x + kx - kernel_size / 2;
                    int iy = y + ky - kernel_size / 2;
                    if (ix >= 0 && ix < input_size && iy >= 0 && iy < input_size) {
                        sum += input[(c * input_size + iy) * input_size + ix] *
                               weights[(f * input_channels * kernel_size * kernel_size) +
                                       (c * kernel_size * kernel_size) +
                                       (ky * kernel_size) + kx];
                    }
                }
            }
        }
        output[(f * input_size + y) * input_size + x] = sum + bias[f];
    }
}

__global__ void maxPoolForward(float* input, float* output, int input_size, int num_channels, int pool_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    int output_size = input_size / pool_size;

    if (x < output_size && y < output_size && c < num_channels) {
        float max_val = -INFINITY;
        for (int py = 0; py < pool_size; ++py) {
            for (int px = 0; px < pool_size; ++px) {
                int ix = x * pool_size + px;
                int iy = y * pool_size + py;
                float val = input[(c * input_size + iy) * input_size + ix];
                max_val = fmaxf(max_val, val);
            }
        }
        output[(c * output_size + y) * output_size + x] = max_val;
    }
}

__global__ void reluForward(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void fullyConnectedForward(float* input, float* weights, float* bias, float* output,
                                      int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weights[idx * input_size + i];
        }
        output[idx] = sum + bias[idx];
    }
}

__global__ void softmaxForward(float* input, int size) {
    __shared__ float max_val;
    __shared__ float sum;

    // Find maximum value
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, input[i]);
    }
    atomicMax(&max_val, thread_max);
    __syncthreads();

    // Compute exponentials and sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float exp_val = expf(input[i] - max_val);
        input[i] = exp_val;
        thread_sum += exp_val;
    }
    atomicAdd(&sum, thread_sum);
    __syncthreads();

    // Normalize
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        input[i] /= sum;
    }
}

__global__ void convolutionBackward(float* input, float* output_grad, float* weights, float* input_grad,
                                    int input_size, int input_channels, int kernel_size, int num_filters) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x < input_size && y < input_size && c < input_channels) {
        float sum = 0.0f;
        for (int f = 0; f < num_filters; ++f) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int ox = x - kx + kernel_size / 2;
                    int oy = y - ky + kernel_size / 2;
                    if (ox >= 0 && ox < input_size && oy >= 0 && oy < input_size) {
                        sum += output_grad[(f * input_size + oy) * input_size + ox] *
                               weights[(f * input_channels * kernel_size * kernel_size) +
                                       (c * kernel_size * kernel_size) +
                                       (ky * kernel_size) + kx];
                    }
                }
            }
        }
        if (input_grad != nullptr) {
            input_grad[(c * input_size + y) * input_size + x] = sum;
        }
    }
}

__global__ void maxPoolBackward(float* input, float* output, float* output_grad, int input_size, int num_channels, int pool_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    int output_size = input_size / pool_size;

    if (x < output_size && y < output_size && c < num_channels) {
        int max_idx_x = 0, max_idx_y = 0;
        float max_val = -INFINITY;
        for (int py = 0; py < pool_size; ++py) {
            for (int px = 0; px < pool_size; ++px) {
                int ix = x * pool_size + px;
                int iy = y * pool_size + py;
                float val = input[(c * input_size + iy) * input_size + ix];
                if (val > max_val) {
                    max_val = val;
                    max_idx_x = ix;
                    max_idx_y = iy;
                }
            }
        }
        float grad = output_grad[(c * output_size + y) * output_size + x];
        atomicAdd(&input[(c * input_size + max_idx_y) * input_size + max_idx_x], grad);
    }
}

__global__ void reluBackward(float* input, float* output_grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output_grad[idx] = (input[idx] > 0) ? output_grad[idx] : 0;
    }
}

__global__ void fullyConnectedBackward(float* input, float* output_grad, float* weights, float* input_grad,
                                       int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size) {
        float sum = 0.0f;
        for (int i = 0; i < output_size; ++i) {
            sum += output_grad[i] * weights[i * input_size + idx];
        }
        input_grad[idx] = sum;
    }
}

__global__ void softmaxBackward(float* output, float* target, float* input_grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input_grad[idx] = output[idx] - target[idx];
    }
}

__global__ void updateWeights(float* weights, float* gradients, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

__global__ void updateBias(float* bias, float* gradients, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        bias[idx] -= learning_rate * gradients[idx];
    }
}