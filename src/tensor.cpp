#include "tensor.hpp"
#include <numeric>
#include "operations.hpp"
#include <stdexcept>
#include "utils.hpp"

// Constructor
Tensor::Tensor(const std::vector<ssize_t>& dims, float initialValue, bool requires_grad)
    : dimensions(dims), data(calculate_size(dims), initialValue), requires_grad(requires_grad) {}

float& Tensor::operator()(const std::vector<ssize_t>& indices) {
    return data[calculate_flat_index(indices)];
}

const float& Tensor::operator()(const std::vector<ssize_t>& indices) const {
    return data[calculate_flat_index(indices)];
}

void Tensor::fill(float value) {
    std::fill(data.begin(), data.end(), value);
}

const std::vector<ssize_t>& Tensor::get_dimensions() const {
    return dimensions;
}

ssize_t Tensor::size() const {
    return data.size();
}

ssize_t Tensor::calculate_size(const std::vector<ssize_t>& dims) const {
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<ssize_t>());
}

ssize_t Tensor::calculate_flat_index(const std::vector<ssize_t>& indices) const {
    if (indices.size() != dimensions.size()) {
        throw std::invalid_argument("Number of indices must match tensor dimensions.");
    }

    ssize_t flatIndex = 0;
    ssize_t stride = 1;

    for (ssize_t i = dimensions.size(); i > 0; --i) {
        ssize_t index = indices[i - 1];
        if (index < 0 || index >= dimensions[i - 1]) {
            throw std::out_of_range("Index " + std::to_string(index) +
                                    " is out of bounds for dimension " + std::to_string(i - 1) +
                                    ".");
        }
        flatIndex += index * stride;
        stride *= dimensions[i - 1];
    }

    return flatIndex;
}

void Tensor::backwards() {
    auto created_by = this->created_by;
    if (!created_by) {
        return;
    }
    // First in the chain doesnt yet have an error. first operation has to initilize the error based
    // on its input/output The tensor that initilizes the backwards step is the first output in the
    // backprop chain
    std::shared_ptr<Tensor> nullTensor;
    this->created_by->backwards(nullTensor, this->shared_from_this());
}

std::shared_ptr<Tensor> Tensor::transpose() const {
    if (this->dimensions.size() != 2) {
        throw std::invalid_argument("Number of dimensions must be 2 (matrix)");
    }
    auto transposed_dims = this->dimensions;
    transposed_dims[0] = this->dimensions[1];
    transposed_dims[1] = this->dimensions[0];
    auto transposed = std::make_shared<Tensor>(transposed_dims, 0.0f, false);
    for (ssize_t j = 0; j < this->dimensions[0]; j++) {
        for (ssize_t i = 0; i < this->dimensions[1]; i++) {
            transposed->operator()({i, j}) = this->operator()({j, i});
        }
    }

    return transposed;
}

void Tensor::randomize() {
    // enforcing between -1.0f and 1.0f
    for (int i = 0; i < this->data.size(); ++i) {
        this->data[i] = generateRandomFloat(-1.0f, 1.0f);
    }
}