//
// Created by robert on 1/25/25.
//

#include <gtest/gtest.h>
#include "tensor.hpp"
#include "operations.hpp"
#include "optimizers.hpp"

TEST(OptimizerTest, LinearNetBackwardsOptimizer) {

    auto x = std::make_shared<Tensor>(std::vector<ssize_t>{1, 3}); // shape (1,2)
    auto w = std::make_shared<Tensor>(std::vector<ssize_t>{3, 2}); // shape (2,2)
    auto b = std::make_shared<Tensor>(std::vector<ssize_t>{1, 2}); // shape (1,2)
    auto y_actual = std::make_shared<Tensor>(std::vector<ssize_t>{1, 2}); // One-hot encoded labels

    // | 1 2 3 |
    //
    // [ [ 1 2 3 ] ]
    x->operator()({0, 0}) = 1.0f;
    x->operator()({0, 1}) = 2.0f;
    x->operator()({0, 2}) = 3.0f;

    // | 7   8  |
    // | 9   10 |
    // | 11  12 |
    //
    // [ [ 7 8 ], [ 9 10 ], [ 11 12 ] ]
    w->operator()({0, 0}) = 7.0f;
    w->operator()({0, 1}) = 8.0f;
    w->operator()({1, 0}) = 9.0f;
    w->operator()({1, 1}) = 10.0f;
    w->operator()({2, 0}) = 11.0f;
    w->operator()({2, 1}) = 12.0f;

    // | 9 10 |
    //
    // [ [ 9 10 ] ]
    b->operator()({0, 0}) = 9.0f;
    b->operator()({0, 1}) = 10.0f;

    // Set one-hot encoded labels (class 0 is the correct class)
    y_actual->operator()({0, 0}) = 1.0f;
    y_actual->operator()({0, 1}) = 0.0f;

    //======
    //== run neural net
    //======

    // | 1 2 3 |
    // x
    // | 7   8  |
    // | 9   10 |
    // | 11  12 |
    // =
    // | 1x7 + 2x9 + 3x11   1x8 + 2x10 + 3x12 |
    // =
    // | 7 + 18 + 33   8 + 20 + 36 |
    // =
    // | 58 64 |
    auto matMul = std::make_shared<MatMul>(); // Use std::shared_ptr
    auto intermediate_1 = matMul->mul(x, w); // shape should be (1, 2)

    // | 58 64 |
    // +
    // | 9 10 |
    // =
    // | 67 74 |
    // [ [ 67  74 ] ] dim(1,2)
    auto tenAdd = std::make_shared<TenAdd>();
    auto intermediate_2 = tenAdd->add(intermediate_1, b); // shape should be (1, 2)

    auto ce = std::make_shared<CrossEntropy>();
    auto result_tensor = ce->calculate(intermediate_2, y_actual);

    result_tensor->backwards();

    auto gradient_descent = std::make_shared<GradientDescent>(0.0005);
    gradient_descent->step(result_tensor);

    EXPECT_NEAR(b->operator()({0, 0}), 9.000499f, 1e-4);
    EXPECT_NEAR(b->operator()({0, 1}), 9.999500f, 1e-4);

    EXPECT_NEAR(w->operator()({0, 0}), 7.00049973f, 1e-4);
    EXPECT_NEAR(w->operator()({0, 1}), 7.999500f, 1e-4);
    EXPECT_NEAR(w->operator()({1, 0}), 9.000999f, 1e-4);
    EXPECT_NEAR(w->operator()({1, 1}), 9.999000f, 1e-4);
    EXPECT_NEAR(w->operator()({2, 0}), 11.0014982f, 1e-4);
    EXPECT_NEAR(w->operator()({2, 1}), 11.998501f, 1e-4);

}
