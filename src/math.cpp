//
// Created by robert on 1/18/25.
//
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <tensor.hpp>
#include <vector>

// Function to compute the softmax
std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor> inputs) {
    if (inputs->dimensions.size() != 2 || inputs->dimensions[0] != 1) {
        throw std::invalid_argument("Must be a tensor/row vector of dimension (1,N)");
    }

    std::vector<float> exp_values(inputs->size());
    float max_input =
        *std::max_element(inputs->data.begin(), inputs->data.end());  // For numerical stability

    for (size_t i = 0; i < inputs->size(); ++i) {
        exp_values[i] = std::exp(inputs->data[i] - max_input);
    }

    float sum_exp = std::accumulate(exp_values.begin(), exp_values.end(), 0.0f);

    // Normalize to get probabilities
    for (size_t i = 0; i < exp_values.size(); ++i) {
        exp_values[i] /= sum_exp;
    }

    auto result = std::make_shared<Tensor>(inputs->dimensions, 1.0f);
    result->data = exp_values;

    return result;
}

std::shared_ptr<Tensor> subtract_tensors(const std::shared_ptr<Tensor> a,
                                         const std::shared_ptr<Tensor> b) {
    if (a->get_dimensions() != b->get_dimensions()) {
        throw std::invalid_argument(
            "Vectors must have the same size for element-wise subtraction.");
    }

    auto result = std::make_shared<Tensor>(a->dimensions, 1.0f);

    for (size_t i = 0; i < a->size(); ++i) {
        result->data[i] = a->data[i] - b->data[i];
    }

    return result;
}

std::shared_ptr<Tensor> matrix_multiply(const std::shared_ptr<Tensor> lhs,
                                        const std::shared_ptr<Tensor> rhs) {
    const auto& lhsDims = lhs->get_dimensions();
    const auto& rhsDims = rhs->get_dimensions();

    if (lhsDims.size() != 2 || rhsDims.size() != 2) {
        throw std::runtime_error(
            "matrix_multiply - Both tensors must be 2D (second-order) for matrix multiplication.");
    }

    ssize_t N = lhsDims[0];
    ssize_t M = lhsDims[1];
    ssize_t M_rhs = rhsDims[0];
    ssize_t K = rhsDims[1];

    if (M != M_rhs) {
        throw std::runtime_error(
            "matrix_multiply - Inner dimensions do not match (lhsDims[1] != rhsDims[0]).");
    }

    auto result = std::make_shared<Tensor>(std::vector<ssize_t>{N, K}, 0.0f);

    for (ssize_t i = 0; i < N; ++i) {
        for (ssize_t j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (ssize_t k = 0; k < M; ++k) {
                sum += lhs->operator()({i, k}) * rhs->operator()({k, j});
            }
            result->operator()({i, j}) = sum;
        }
    }

    return result;
}

namespace math {
std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& vec) {
    auto result = std::make_shared<Tensor>(vec->get_dimensions(), 1.0f);
    for (int i = 0; i < vec->data.size(); ++i) {
        float r;
        if (vec->data[i] > 0.0f) {
            r = vec->data[i];
        } else {
            r = 0;
        }
        result->data[i] = r;
    }

    return result;
}
}
