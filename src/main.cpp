#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "operations.hpp"
#include "optimizers.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include <cmath>

std::pair<std::vector<std::shared_ptr<Tensor>>, std::vector<std::shared_ptr<Tensor>>>
createTrainingVectors(int samples);
int generateActualClassification(float x1, float x2);

int main(int argc, char* argv[]) {
    // Print the current working directory
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

    // network definition
    auto w1 = std::make_shared<Tensor>(std::vector<ssize_t>{2, 8});
    w1->randomize();
    auto matMul1 = std::make_shared<MatMul>();
    auto b1 = std::make_shared<Tensor>(std::vector<ssize_t>{1, 8});
    b1->randomize();
    auto tenAdd1 = std::make_shared<TenAdd>();
    auto relu1 = std::make_shared<ReLU>();
    auto w2 = std::make_shared<Tensor>(std::vector<ssize_t>{8, 8});
    w2->randomize();
    auto matMul2 = std::make_shared<MatMul>();
    auto b2 = std::make_shared<Tensor>(std::vector<ssize_t>{1, 8});
    b2->randomize();
    auto tenAdd2 = std::make_shared<TenAdd>();
    auto relu2 = std::make_shared<ReLU>();
    auto w3 = std::make_shared<Tensor>(std::vector<ssize_t>{8, 2});
    w3->randomize();
    auto matMul3 = std::make_shared<MatMul>();
    auto b3 = std::make_shared<Tensor>(std::vector<ssize_t>{1, 2});
    b3->randomize();
    auto tenAdd3 = std::make_shared<TenAdd>();

    auto loss_function_ce = std::make_shared<CrossEntropy>();
    auto gradient_descent = std::make_shared<GradientDescent>(0.01f);

    // sample generation
    int samples = 10000;
    auto training_data = createTrainingVectors(samples);

    //=========================================================================== pre training

    // Open a CSV file for output
    std::ofstream csv_file("pre-training.csv");
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open output.csv for writing.\n";
        return 1;
    }

    // Write the CSV header
    csv_file << "Input1,Input2,PredictedClass\n";

    // first lets see the predicted classes before training
    for (int i = 0; i < training_data.first.size(); i++) {
        auto intermediate_1 = matMul1->mul(training_data.first[i], w1);  // shape should be (1, 2)
        auto intermediate2 = tenAdd1->add(intermediate_1, b1);           // shape should be (1, 2)
        auto intermediate3 = relu1->relu(intermediate2);
        auto intermediate4 = matMul2->mul(intermediate3, w2);
        auto intermediate5 = tenAdd2->add(intermediate4, b2);
        auto intermediate6 = relu2->relu(intermediate5);
        auto intermediate7 = matMul3->mul(intermediate6, w3);
        auto logits = tenAdd3->add(intermediate7, b3);

        auto max_it = std::max_element(logits->data.begin(), logits->data.end());

        // Retrieve the input vector
        const std::vector<float>& input_vector = training_data.first[i]->data;
        int predicted_class = std::distance(logits->data.begin(), max_it);

        // Write the input vector and predicted class to the CSV file
        for (size_t j = 0; j < input_vector.size(); j++) {
            csv_file << input_vector[j];
            if (j < input_vector.size() - 1) {
                csv_file << ",";
            }
        }
        csv_file << "," << predicted_class << "\n";
    }
    // Close the CSV file
    csv_file.close();
    std::cout << "Data written to pre-training.csv\n";

    //=========================================================================== training
    // first lets see the predicted classes before training
    for (int i = 0; i < training_data.first.size(); i++) {
        auto intermediate_1 = matMul1->mul(training_data.first[i], w1);  // shape should be (1, 2)
        auto intermediate2 = tenAdd1->add(intermediate_1, b1);           // shape should be (1, 2)
        auto intermediate3 = relu1->relu(intermediate2);
        auto intermediate4 = matMul2->mul(intermediate3, w2);
        auto intermediate5 = tenAdd2->add(intermediate4, b2);
        auto intermediate6 = relu2->relu(intermediate5);
        auto intermediate7 = matMul3->mul(intermediate6, w3);
        auto logits = tenAdd3->add(intermediate7, b3);

        auto result_tensor = loss_function_ce->calculate(logits, training_data.second[i]);

        result_tensor->backwards();

        gradient_descent->step(result_tensor);
    }
    //=========================================================================== post training

    // Open a CSV file for output
    std::ofstream csv_filepost("post-training.csv");
    if (!csv_filepost.is_open()) {
        std::cerr << "Failed to open output.csv for writing.\n";
        return 1;
    }

    // Write the CSV header
    csv_filepost << "Input1,Input2,PredictedClass\n";

    // first lets see the predicted classes before training
    for (int i = 0; i < training_data.first.size(); i++) {
        auto intermediate_1 = matMul1->mul(training_data.first[i], w1);  // shape should be (1, 2)
        auto intermediate2 = tenAdd1->add(intermediate_1, b1);           // shape should be (1, 2)
        auto intermediate3 = relu1->relu(intermediate2);
        auto intermediate4 = matMul2->mul(intermediate3, w2);
        auto intermediate5 = tenAdd2->add(intermediate4, b2);
        auto intermediate6 = relu2->relu(intermediate5);
        auto intermediate7 = matMul3->mul(intermediate6, w3);
        auto logits = tenAdd3->add(intermediate7, b3);

        auto max_it = std::max_element(logits->data.begin(), logits->data.end());

        // Retrieve the input vector
        const std::vector<float>& input_vector = training_data.first[i]->data;
        int predicted_class = std::distance(logits->data.begin(), max_it);

        // Write the input vector and predicted class to the CSV file
        for (size_t j = 0; j < input_vector.size(); j++) {
            csv_filepost << input_vector[j];
            if (j < input_vector.size() - 1) {
                csv_filepost << ",";
            }
        }
        csv_filepost << "," << predicted_class << "\n";
    }
    // Close the CSV file
    csv_filepost.close();
    std::cout << "Data written to post-training.csv\n";

    return 0;
}

std::pair<std::vector<std::shared_ptr<Tensor>>, std::vector<std::shared_ptr<Tensor>>>
createTrainingVectors(int samples) {
    // Create two vectors that can hold "samples" amount of elements.
    std::vector<std::shared_ptr<Tensor>> inputs(samples);
    std::vector<std::shared_ptr<Tensor>> y_actual(samples);

    // The elements are created like this: Tensor tensor({1, 2}, 0.0f);
    for (int i = 0; i < samples; i++) {
        inputs[i] = std::make_shared<Tensor>(std::initializer_list<ssize_t>{1, 2}, 0.0f, false);
        y_actual[i] = std::make_shared<Tensor>(std::initializer_list<ssize_t>{1, 2}, 0.0f, false);
    }

    float upper = 5.0f;
    float lower = -5.0f;  // Adjusted range to allow more diversity for `x1` and `x2`

    for (int i = 0; i < samples; i++) {
        // Generate random values for x1 and x2
        auto x1 = generateRandomFloat(lower, upper);
        auto x2 = generateRandomFloat(lower, upper);

        // Generate the actual classification for this input
        auto actual_class = generateActualClassification(x1, x2);

        // Set the ith input tensor->data[0] and [1] to x1 and x2 respectively
        inputs[i]->data[0] = x1;
        inputs[i]->data[1] = x2;

        // Set the ith output tensor->data[actual_class] to 1.0
        y_actual[i]->data[actual_class] = 1.0f;
    }

    return std::make_pair(inputs, y_actual);  // Combine the two vectors into a pair
}

// int generateActualClassification(float x1, float x2) {
//     // y=0.7x+2
//     // if x2 is below/equal y class 0
//     // if x2 is above y class 1
//     float border = 0.7f * x1 + 2.0f;
//     if (border >= x2) {
//         return 0;
//     }
//     return 1;
// }

// int generateActualClassification(float x1, float x2) {
//     // y=x^2
//     // if x2 is below/equal y class 0
//     // if x2 is above y class 1
//     float border = 0.7f * (x1 * x1) + 1.0f;
//     if (border >= x2) {
//         return 0;
//     }
//     return 1;
// }

// int generateActualClassification(float x1, float x2) {
//     // High-order polynomial boundary
//     float border = 0.3f * pow(x1, 5)
//                  - 0.2f * pow(x1, 4)
//                  + 1.5f * pow(x1, 3)
//                  - 0.5f * pow(x1, 2)
//                  + 2.0f * x1
//                  + 1.0f;
//
//     // Classification logic
//     if (x2 <= border) {
//         return 0; // Class 0 for points below or on the boundary
//     }
//     return 1; // Class 1 for points above the boundary
// }

int generateActualClassification(float x1, float x2) {
    // High-frequency boundary function
    float border = 0.5f * pow(x1, 3)    // Cubic base shape
                 - 0.9f * pow(x1, 2)    // Quadratic component
                 + sin(3 * x1)         // High-frequency oscillations
                 + 0.5f * cos(3 * x1); // Even higher frequency

    // Classification logic
    if (x2 <= border) {
        return 0; // Class 0 for points below or on the boundary
    }
    return 1; // Class 1 for points above the boundary
}
