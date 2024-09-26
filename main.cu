// File: main.cu
#include "cnn.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

// Function to generate random data for testing
std::vector<float> generateRandomData(int size) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

int main() {
    CNN cnn;

    // Generate random input data
    int input_size = INPUT_SIZE * INPUT_SIZE * INPUT_CHANNELS;
    std::vector<float> input_data = generateRandomData(input_size);

    // Generate random target data
    std::vector<float> target_data(NUM_CLASSES, 0.0f);
    target_data[std::rand() % NUM_CLASSES] = 1.0f;

    // Training loop
    int num_epochs = 100;
    float learning_rate = 0.01f;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Forward pass
        cnn.forward(input_data.data());

        // Backward pass
        cnn.backward(target_data.data());

        // Update weights
        cnn.update(learning_rate);

        // Print loss (you need to implement a loss calculation function)
        if (epoch % 10 == 0) {
            float* output = cnn.getOutput();
            float loss = 0.0f;
            for (int i = 0; i < NUM_CLASSES; ++i) {
                loss += -target_data[i] * std::log(output[i]);
            }
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
            delete[] output;
        }
    }

    std::cout << "Training completed." << std::endl;

    return 0;
}