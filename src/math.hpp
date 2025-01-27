//
// Created by robert on 1/18/25.
//

#pragma once
#include <tensor.hpp>
#include <vector>

std::shared_ptr<Tensor> softmax(std::shared_ptr<Tensor> inputs);

std::shared_ptr<Tensor> subtract_tensors(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

std::shared_ptr<Tensor> matrix_multiply(const std::shared_ptr<Tensor> lhs, const std::shared_ptr<Tensor> rhs);

namespace math {

std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& vec);

}