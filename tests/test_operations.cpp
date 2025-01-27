#include <gtest/gtest.h>
#include "tensor.hpp"
#include "operations.hpp"

//===================================================
//=== TenAdd
//===================================================
TEST(TenAddTest, ForwardPass) {
    auto lhs = std::make_shared<Tensor>(std::vector<ssize_t>{1, 3});
    auto rhs = std::make_shared<Tensor>(std::vector<ssize_t>{1, 3});

    // Set values for lhs and rhs tensors
    lhs->operator()({0, 0}) = 1.0f;
    lhs->operator()({0, 1}) = 2.0f;
    lhs->operator()({0, 2}) = 3.0f;

    rhs->operator()({0, 0}) = 4.0f;
    rhs->operator()({0, 1}) = 5.0f;
    rhs->operator()({0, 2}) = 6.0f;

    auto tenAdd = std::make_shared<TenAdd>();
    auto result = tenAdd->add(lhs, rhs);

    // Verify the resulting tensor dimensions
    ASSERT_EQ(result->get_dimensions(), lhs->get_dimensions());

    // Verify the resulting data
    EXPECT_FLOAT_EQ(result->data[0], 5.0f); // 1 + 4
    EXPECT_FLOAT_EQ(result->data[1], 7.0f); // 2 + 5
    EXPECT_FLOAT_EQ(result->data[2], 9.0f); // 3 + 6
}

TEST(TenAddTest, BackwardPass) {
    auto lhs = std::make_shared<Tensor>(std::vector<ssize_t>{1, 3});
    auto rhs = std::make_shared<Tensor>(std::vector<ssize_t>{1, 3});

    // Set values for lhs and rhs tensors
    lhs->operator()({0, 0}) = 1.0f;
    lhs->operator()({0, 1}) = 2.0f;
    lhs->operator()({0, 2}) = 3.0f;
    lhs->requires_grad = true;

    rhs->operator()({0, 0}) = 4.0f;
    rhs->operator()({0, 1}) = 5.0f;
    rhs->operator()({0, 2}) = 6.0f;
    rhs->requires_grad = true;

    auto tenAdd = std::make_shared<TenAdd>();
    auto output = tenAdd->add(lhs, rhs);

    // Create an error tensor for backpropagation
    auto error = std::make_shared<Tensor>(std::vector<ssize_t>{1, 3});
    error->operator()({0, 0}) = 0.1f;
    error->operator()({0, 1}) = 0.2f;
    error->operator()({0, 2}) = 0.3f;

    // Perform backward pass
    tenAdd->compute_gradients(error, output);

    // Verify gradients for lhs and rhs
    ASSERT_TRUE(lhs->gradients != nullptr);
    ASSERT_TRUE(rhs->gradients != nullptr);

    EXPECT_FLOAT_EQ(lhs->gradients->operator()({0, 0}), 0.1f);
    EXPECT_FLOAT_EQ(lhs->gradients->operator()({0, 1}), 0.2f);
    EXPECT_FLOAT_EQ(lhs->gradients->operator()({0, 2}), 0.3f);

    EXPECT_FLOAT_EQ(rhs->gradients->operator()({0, 0}), 0.1f);
    EXPECT_FLOAT_EQ(rhs->gradients->operator()({0, 1}), 0.2f);
    EXPECT_FLOAT_EQ(rhs->gradients->operator()({0, 2}), 0.3f);
}

//===================================================
//=== MatMul
//===================================================
TEST(MatMulTest, ForwardPassVxM) {
    auto vec = std::make_shared<Tensor>(std::vector<ssize_t>{1, 2}); // shape (1, 3)
    auto mat = std::make_shared<Tensor>(std::vector<ssize_t>{2, 2}); // shape (2, 2)

    // vec = [ [1, 2, 3] ]
    vec->operator()({0, 0}) = -3.04698968f;
    vec->operator()({0, 1}) = -3.57507586f;

    // mat =
    // [ [4,  5 ],
    //   [6,  7 ],
    //   [8,  9 ] ]
    mat->operator()({0, 0}) = 0.5f;
    mat->operator()({0, 1}) = 0.5f;
    mat->operator()({1, 0}) = 0.5f;
    mat->operator()({1, 1}) = 0.5f;

    auto matMul = std::make_shared<MatMul>();
    auto result = matMul->mul(vec, mat); // shape should be (1, 2)

    // Verify the dimensions of the result
    const auto& resultDims = result->get_dimensions();
    ASSERT_EQ(resultDims.size(), 2ul);
    EXPECT_EQ(resultDims[0], 1);
    EXPECT_EQ(resultDims[1], 2);

    // Expected result = vec * mat = [ [40, 46] ]
    EXPECT_FLOAT_EQ(result->operator()({0, 0}), -3.31103277f);
    EXPECT_FLOAT_EQ(result->operator()({0, 1}), -3.31103277f);
}

TEST(MatMulTest, ForwardPassMxM) {
    auto matA = std::make_shared<Tensor>(std::vector<ssize_t>{2, 3}); // shape (2, 3)
    auto matB = std::make_shared<Tensor>(std::vector<ssize_t>{3, 2}); // shape (3, 2)

    // matA =
    // [ [1, 2, 3],
    //   [4, 5, 6] ]
    matA->operator()({0, 0}) = 1.0f;
    matA->operator()({0, 1}) = 2.0f;
    matA->operator()({0, 2}) = 3.0f;
    matA->operator()({1, 0}) = 4.0f;
    matA->operator()({1, 1}) = 5.0f;
    matA->operator()({1, 2}) = 6.0f;

    // matB =
    // [ [7,  8 ],
    //   [9,  10],
    //   [11, 12] ]
    matB->operator()({0, 0}) = 7.0f;
    matB->operator()({0, 1}) = 8.0f;
    matB->operator()({1, 0}) = 9.0f;
    matB->operator()({1, 1}) = 10.0f;
    matB->operator()({2, 0}) = 11.0f;
    matB->operator()({2, 1}) = 12.0f;

    auto matMul = std::make_shared<MatMul>();
    auto result = matMul->mul(matA, matB); // shape should be (2, 2)

    // Verify the dimensions of the result
    const auto& resultDims = result->get_dimensions();
    ASSERT_EQ(resultDims.size(), 2ul);
    EXPECT_EQ(resultDims[0], 2);
    EXPECT_EQ(resultDims[1], 2);

    // Expected result:
    // [ [58,  64],
    //   [139, 154] ]
    EXPECT_FLOAT_EQ(result->operator()({0, 0}), 58.0f);
    EXPECT_FLOAT_EQ(result->operator()({0, 1}), 64.0f);
    EXPECT_FLOAT_EQ(result->operator()({1, 0}), 139.0f);
    EXPECT_FLOAT_EQ(result->operator()({1, 1}), 154.0f);
}
//
// TEST(MatMulTest, BackwardPassVxM) {
//     auto vec = std::make_shared<Tensor>(std::vector<ssize_t>{1, 3}); // shape (1, 3)
//     auto mat = std::make_shared<Tensor>(std::vector<ssize_t>{3, 2}); // shape (3, 2)
//
//     vec->requires_grad = true;
//     mat->requires_grad = true;
//
//     // vec = [ [1, 2, 3] ]
//     vec->operator()({0, 0}) = 1.0f;
//     vec->operator()({0, 1}) = 2.0f;
//     vec->operator()({0, 2}) = 3.0f;
//
//     // mat =
//     // [ [4,  5 ],
//     //   [6,  7 ],
//     //   [8,  9 ] ]
//     mat->operator()({0, 0}) = 4.0f;
//     mat->operator()({0, 1}) = 5.0f;
//     mat->operator()({1, 0}) = 6.0f;
//     mat->operator()({1, 1}) = 7.0f;
//     mat->operator()({2, 0}) = 8.0f;
//     mat->operator()({2, 1}) = 9.0f;
//
//     auto matMul = std::make_shared<MatMul>();
//     auto output = matMul->mul(vec, mat);
//
//     // Backward pass: Use an error tensor to propagate
//     auto error = std::make_shared<Tensor>(std::vector<ssize_t>{1, 2}); // shape (1, 2)
//     error->operator()({0, 0}) = 1.0f;
//     error->operator()({0, 1}) = 2.0f;
//
//     matMul->compute_gradients(error, output);
//
//     // Verify gradients for vec
//     auto vec_grad = vec->gradients;
//     ASSERT_TRUE(vec_grad != nullptr);
//     EXPECT_FLOAT_EQ(vec_grad->operator()({0, 0}), 14.0f); // 1 * 4 + 2 * 6
//     EXPECT_FLOAT_EQ(vec_grad->operator()({0, 1}), 18.0f); // 1 * 5 + 2 * 7
//     EXPECT_FLOAT_EQ(vec_grad->operator()({0, 2}), 22.0f); // 1 * 8 + 2 * 9
//
//     // Verify gradients for mat
//     auto mat_grad = mat->gradients;
//     ASSERT_TRUE(mat_grad != nullptr);
//     EXPECT_FLOAT_EQ(mat_grad->operator()({0, 0}), 1.0f); // vec[0] * error[0]
//     EXPECT_FLOAT_EQ(mat_grad->operator()({0, 1}), 2.0f);
//     EXPECT_FLOAT_EQ(mat_grad->operator()({1, 0}), 2.0f); // vec[1] * error[1]
//     EXPECT_FLOAT_EQ(mat_grad->operator()({1, 1}), 4.0f);
//     EXPECT_FLOAT_EQ(mat_grad->operator()({2, 0}), 3.0f); // vec[2] * error[2]
//     EXPECT_FLOAT_EQ(mat_grad->operator()({2, 1}), 6.0f);
// }

TEST(MatMulTest, InvalidDimensions) {
    auto matA = std::make_shared<Tensor>(std::vector<ssize_t>{2, 3}); // shape (2, 3)
    auto matB = std::make_shared<Tensor>(std::vector<ssize_t>{4, 5}); // shape (4, 5)

    auto matMul = std::make_shared<MatMul>();

    EXPECT_THROW({
        auto result = matMul->mul(matA, matB);
        (void)result; // Prevent unused variable warning
    }, std::runtime_error);
}

//===================================================
//=== CrossEntropy
//===================================================
TEST(CrossEntropyTest, Backwards) {
    auto y_logits = std::make_shared<Tensor>(std::vector<ssize_t>{1, 3}); // Logits for 3 classes
    auto y_actual = std::make_shared<Tensor>(std::vector<ssize_t>{1, 3}); // One-hot encoded labels

    y_logits->operator()({0, 0}) = 2.0f;
    y_logits->operator()({0, 1}) = 1.0f;
    y_logits->operator()({0, 2}) = 0.1f;

    y_actual->operator()({0, 0}) = 1.0f;
    y_actual->operator()({0, 1}) = 0.0f;
    y_actual->operator()({0, 2}) = 0.0f;

    auto ce = std::make_shared<CrossEntropy>();
    auto forward_result_tensor = ce->calculate(y_logits, y_actual);

    forward_result_tensor->backwards();

    auto result = y_logits->gradients;

    EXPECT_NEAR(result->operator()({0, 0}), -0.3409988f, 1e-4);
    EXPECT_NEAR(result->operator()({0, 1}), 0.24243299f, 1e-4);
    EXPECT_NEAR(result->operator()({0, 2}), 0.09856589f, 1e-4);

}

TEST(CrossEntropyTest, ValidInputs) {
    auto y_logits = std::make_shared<Tensor>(std::vector<ssize_t>{1, 3});
    auto y_actual = std::make_shared<Tensor>(std::vector<ssize_t>{1, 3});

    y_logits->operator()({0, 0}) = 2.0f;
    y_logits->operator()({0, 1}) = 1.0f;
    y_logits->operator()({0, 2}) = 0.1f;

    y_actual->operator()({0, 0}) = 1.0f;
    y_actual->operator()({0, 1}) = 0.0f;
    y_actual->operator()({0, 2}) = 0.0f;

    auto ce = std::make_shared<CrossEntropy>();
    auto result_tensor = ce->calculate(y_logits, y_actual);
    auto result = result_tensor->data[0];

    // Verify the resulting loss tensor dimensions and value
    ASSERT_EQ(result_tensor->get_dimensions().size(), 2ul);
    EXPECT_EQ(result_tensor->get_dimensions()[0], 1);
    EXPECT_EQ(result_tensor->get_dimensions()[1], 1);

    // Check the loss value (manually calculated)
    float expected_loss = 0.41702f;
    EXPECT_NEAR(result, expected_loss, 1e-4);
}

//===================================================
//=== ReLU
//===================================================
TEST(ReLUTest, HandlesBasicInput) {
    // Create an input tensor
    auto input = std::make_shared<Tensor>(std::vector<ssize_t>{1, 5});
    input->operator()({0, 0}) = -3.0f;
    input->operator()({0, 1}) = -1.0f;
    input->operator()({0, 2}) = 0.0f;
    input->operator()({0, 3}) = 2.0f;
    input->operator()({0, 4}) = 5.0f;

    // Create an expected output tensor
    auto expected_output = std::make_shared<Tensor>(std::vector<ssize_t>{1, 5});
    expected_output->operator()({0, 0}) = 0.0f;
    expected_output->operator()({0, 1}) = 0.0f;
    expected_output->operator()({0, 2}) = 0.0f;
    expected_output->operator()({0, 3}) = 2.0f;
    expected_output->operator()({0, 4}) = 5.0f;

    // Create a ReLU operator instance
    auto relu_op = std::make_shared<ReLU>();

    // Compute ReLU using the operator
    auto result = relu_op->relu(input);

    // Verify the size of the output tensor
    ASSERT_EQ(result->get_dimensions(), expected_output->get_dimensions());

    // Verify each element
    for (int i = 0; i < result->data.size(); ++i) {
        EXPECT_FLOAT_EQ(result->data[i], expected_output->data[i]);
    }
}

//===================================================
//=== Generel Tests
//===================================================
TEST(OperationsTest, LinearLayer) {

    auto x = std::make_shared<Tensor>(std::vector<ssize_t>{1, 3}); // shape (1,2)
    auto w = std::make_shared<Tensor>(std::vector<ssize_t>{3, 2}); // shape (2,2)
    auto b = std::make_shared<Tensor>(std::vector<ssize_t>{1, 2}); // shape (1,2)

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
    auto matMul = std::make_shared<MatMul>();
    auto intermediate = matMul->mul(x, w); // shape should be (1, 2)


    // | 58 64 |
    // +
    // | 9 10 |
    // =
    // | 67 74 |
    // [ [ 67  74 ] ] dim(1,2)
    auto tenAdd = std::make_shared<TenAdd>();
    auto result = tenAdd->add(intermediate, b); // shape should be (1, 2)

    // Check the resulting dimensions
    const auto& resultDims = result->get_dimensions();
    ASSERT_EQ(resultDims.size(), 2ul);
    EXPECT_EQ(resultDims[0], 1);
    EXPECT_EQ(resultDims[1], 2);

    // Check result data
    EXPECT_FLOAT_EQ(result->operator()({0, 0}), 67.0f);
    EXPECT_FLOAT_EQ(result->operator()({0, 1}), 74.0f);
}

TEST(OperationsTest, LinearNet) {

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
    auto result = result_tensor->data[0];


    // Check the resulting dimensions
    ASSERT_EQ(result_tensor->data.size(), 1);
    EXPECT_EQ(result_tensor->dimensions[0], 1);
    EXPECT_EQ(result_tensor->dimensions[1], 1);

    // Check result data
    float expected_loss = 7.000910282;
    EXPECT_NEAR(result, expected_loss, 1e-4);
}

TEST(OperationsTest, LinearNetBackwards) {

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

    // Verify gradients for `w` (weights)
    auto w_grad = w->gradients;
    EXPECT_NEAR(w_grad->operator()({0, 0}), -0.9991f, 1e-4);
    EXPECT_NEAR(w_grad->operator()({0, 1}), 0.9991f, 1e-4);
    EXPECT_NEAR(w_grad->operator()({1, 0}), -1.9982f, 1e-4);
    EXPECT_NEAR(w_grad->operator()({1, 1}), 1.9982f, 1e-4);
    EXPECT_NEAR(w_grad->operator()({2, 0}), -2.9973f, 1e-4);
    EXPECT_NEAR(w_grad->operator()({2, 1}), 2.9973f, 1e-4);

    // Verify gradients for `b` (bias)
    auto b_grad = b->gradients;
    EXPECT_NEAR(b_grad->operator()({0, 0}), -0.9991f, 1e-4);
    EXPECT_NEAR(b_grad->operator()({0, 1}), 0.9991f, 1e-4);
}
