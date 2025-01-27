//
// Created by robert on 1/25/25.
//

#include "optimizers.hpp"

#include <operations.hpp>

GradientDescent::GradientDescent(float learning_rate) : learning_rate(learning_rate) {
}

void GradientDescent::step(std::shared_ptr<Tensor> t) {
    this->do_step(t);
    if (t->created_by == nullptr) {return;}
    for (auto tensor : t->created_by->inputs) {
        this->step(tensor);
    }
}

void GradientDescent::do_step(std::shared_ptr<Tensor> t) {
    if (t->gradients == nullptr) {
        return;
    }
    for (int i = 0; i < t->gradients->size(); i++) {
        t->data[i] -= learning_rate * t->gradients->data[i];
    }
}
