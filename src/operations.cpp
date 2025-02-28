#include "operations.hpp"
#include <memory>
#include <stdexcept>

#include <cmath>
#include "math.hpp"

#include "tensor.hpp"

TenAdd::TenAdd() {
    inputs.resize(2);
}

std::shared_ptr<Tensor> TenAdd::add(std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs) {
    if (lhs->get_dimensions() != rhs->get_dimensions()) {
        throw std::invalid_argument("Tensors must have the same dimensions for addition.");
    }

    inputs[0] = lhs;
    inputs[1] = rhs;

    auto result = std::make_shared<Tensor>(lhs->get_dimensions(), 0.0f);

    for (ssize_t i = 0; i < lhs->size(); ++i) {
        result->data[i] = lhs->data[i] + rhs->data[i];
    }

    result->created_by = shared_from_this();

    return result;
}

std::shared_ptr<Tensor> TenAdd::compute_error(std::shared_ptr<Tensor> error,
                                              std::shared_ptr<Tensor> output) {
    return this->inputs[0]->gradients;
};

void TenAdd::compute_gradients(std::shared_ptr<Tensor> error, std::shared_ptr<Tensor> output) {
    for (auto input : inputs) {
        if (input->requires_grad) {
            input->gradients = error;
        }
    }
}

MatMul::MatMul() {
    inputs.resize(2);
}

std::shared_ptr<Tensor> MatMul::mul(std::shared_ptr<Tensor> vec, std::shared_ptr<Tensor> mat) {
    inputs[0] = vec;
    inputs[1] = mat;

    auto result = matrix_multiply(vec, mat);

    result->created_by = shared_from_this();

    return result;
}

std::shared_ptr<Tensor> MatMul::compute_error(std::shared_ptr<Tensor> error,
                                              std::shared_ptr<Tensor> output) {
    return this->inputs[0]->gradients;
};
void MatMul::compute_gradients(std::shared_ptr<Tensor> error, std::shared_ptr<Tensor> output) {
    auto x = this->inputs[0];
    auto w = this->inputs[1];

    auto w_T = w->transpose();
    auto x_T = x->transpose();

    auto grad_x = matrix_multiply(error, w_T);
    auto grad_w = matrix_multiply(x_T, error);

    this->inputs[0]->gradients = grad_x;
    this->inputs[1]->gradients = grad_w;
}

ReLU::ReLU() {
    inputs.resize(1);
}

std::shared_ptr<Tensor> ReLU::relu(std::shared_ptr<Tensor> vec) {
    inputs[0] = vec;

    auto result = math::relu(vec);

    result->created_by = shared_from_this();
    return result;
}


void ReLU::compute_gradients(std::shared_ptr<Tensor> error,
                                     std::shared_ptr<Tensor> output) {
    auto x = this->inputs[0];

    auto x_grad = std::make_shared<Tensor>(error->get_dimensions(), 0.0f);
    for (int i = 0; i < x->data.size(); i++) {
        float g;
        if (x->data[i] > 0.0f) {
            g = 1;
        } else {
            g = 0;
        }
        x_grad->data[i] = error->data[i] * g;
    }

    x->gradients = x_grad;
}

std::shared_ptr<Tensor> ReLU::compute_error(std::shared_ptr<Tensor> error,
                                                    std::shared_ptr<Tensor> output) {
    auto x = this->inputs[0];
    return x->gradients;
};


CrossEntropy::CrossEntropy() {
    inputs.resize(2);
}

std::shared_ptr<Tensor> CrossEntropy::calculate(std::shared_ptr<Tensor> y_logits,
                                                std::shared_ptr<Tensor> y_actual) {
    if (y_logits->get_dimensions() != y_actual->get_dimensions()) {
        throw std::invalid_argument("Predicted must have the same dimensions as actual");
    }

    auto y_hat = softmax(y_logits);

    // simplifed non batched cross entropy
    const auto& predictions = y_hat->data;
    const auto& labels = y_actual->data;

    if (predictions.size() != labels.size()) {
        throw std::invalid_argument("Prediction and label sizes do not match");
    }

    inputs[0] = (y_logits);
    inputs[1] = (y_actual);

    float loss = 0.0f;
    const size_t N = predictions.size();  // number of samples
    for (size_t i = 0; i < N; ++i) {
        if (labels[i] > 0.0f) {
            loss -= labels[i] * std::log(predictions[i] + 1e-9);  // Avoid log(0)
        }
    }
    auto result = std::make_shared<Tensor>(std::vector<ssize_t>{1, 1}, loss);
    result->created_by = shared_from_this();
    return result;
}

std::shared_ptr<Tensor> CrossEntropy::compute_error(std::shared_ptr<Tensor> error,
                                                    std::shared_ptr<Tensor> output) {
    auto logits = this->inputs[0];
    return logits->gradients;
};
void CrossEntropy::compute_gradients(std::shared_ptr<Tensor> error,
                                     std::shared_ptr<Tensor> output) {
    auto logits = this->inputs[0];
    auto y_actual = this->inputs[1];
    auto s_l = softmax(logits);
    auto d = subtract_tensors(s_l, y_actual);

    logits->gradients = d;
    // We dont compute gradients for the y_actualfor the loss function
}
