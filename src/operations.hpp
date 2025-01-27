#pragma once

#include <memory>
#include <vector>
#include "tensor.hpp"

class IOperation : public std::enable_shared_from_this<IOperation> {
public:
    std::vector<std::shared_ptr<Tensor>> inputs;
    virtual ~IOperation() = default;
    void backwards(std::shared_ptr<Tensor> error, std::shared_ptr<Tensor> output) {
        // TODO what do we do for the first if error is null, i think let the specific impl handle this
        this->compute_gradients(error, output);
        // actually we should not compute one error, but the error per input of the operation.
        // in my case we only ever have to worry about the input x tensor error path, not for the bias path or other paths (cuz there are none) so its fine for now.
        // however this approach requires the operations to know which of the input tensors is the one that propagates the error down the computation graph.
        // its a less clean but simpler to implement approach for now.
        auto next_error = this->compute_error(error, output);
        for (auto input : this->inputs) {
            auto created_by = input->created_by;
            if (!created_by) {
                continue;
            }
            created_by->backwards(next_error, input);
        }
    }
    virtual void compute_gradients(std::shared_ptr<Tensor> error, std::shared_ptr<Tensor> output) = 0;
    virtual std::shared_ptr<Tensor> compute_error(std::shared_ptr<Tensor> error, std::shared_ptr<Tensor> output) = 0;
};

class TenAdd : public IOperation {
public:
    TenAdd();

    std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs);

    void compute_gradients(std::shared_ptr<Tensor> error, std::shared_ptr<Tensor> output) override;
    std::shared_ptr<Tensor> compute_error(std::shared_ptr<Tensor> error, std::shared_ptr<Tensor> output) override;
};

class MatMul : public IOperation {
public:
    MatMul();

    std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> vec, std::shared_ptr<Tensor> mat);

    void compute_gradients(std::shared_ptr<Tensor> error, std::shared_ptr<Tensor> output) override;
    std::shared_ptr<Tensor> compute_error(std::shared_ptr<Tensor> error, std::shared_ptr<Tensor> output) override;
};


class ReLU : public IOperation {
public:
    ReLU();

    std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> vec);

    void compute_gradients(std::shared_ptr<Tensor> error, std::shared_ptr<Tensor> output) override;
    std::shared_ptr<Tensor> compute_error(std::shared_ptr<Tensor> error, std::shared_ptr<Tensor> output) override;
};

class CrossEntropy : public IOperation {
public:
    CrossEntropy();

    std::shared_ptr<Tensor> calculate(std::shared_ptr<Tensor> y_logits,
                                      std::shared_ptr<Tensor> y_actual);
    void compute_gradients(std::shared_ptr<Tensor> error, std::shared_ptr<Tensor> output) override;
    std::shared_ptr<Tensor> compute_error(std::shared_ptr<Tensor> error, std::shared_ptr<Tensor> output) override;
};