#include <gtest/gtest.h>
#include "math.hpp"
#include "tensor.hpp"

TEST(SoftmaxTest, HandlesBasicInput) {
    auto input = std::make_shared<Tensor>(std::vector<ssize_t>{1, 3}, 1.0f);
    input->data = {1.0f, 2.0f, 3.0f};

    auto expected_output = std::make_shared<Tensor>(std::vector<ssize_t>{1, 3}, 1.0f);
    expected_output->data = {0.09003057f, 0.24472847f, 0.66524096f};

    // Compute softmax
    auto result = softmax(input);

    // Verify the size of the output
    ASSERT_EQ(result->dimensions, expected_output->dimensions);

    // Verify each element
    for (size_t i = 0; i < result->data.size(); ++i) {
        EXPECT_NEAR(result->data[i], expected_output->data[i], 1e-5f); // Allow for small floating-point errors
    }
}

// Test case for numerical stability
TEST(SubtractTensorTest, BasicInput) {

    auto a = std::make_shared<Tensor>(std::vector<ssize_t>{1, 3}, 1.0f);
    a->data = {1.0f, 2.0f, 3.0f};

    auto b = std::make_shared<Tensor>(std::vector<ssize_t>{1, 3}, 1.0f);
    b->data = {1.0f, 2.0f, 3.0f};

    auto expected_output = std::make_shared<Tensor>(std::vector<ssize_t>{1, 3}, 1.0f);
    expected_output->data = {0.0, 0.0, 0.0};


    // Compute softmax
    auto result = subtract_tensors(a, b);

    // Verify the size of the output
    ASSERT_EQ(result->dimensions, expected_output->dimensions);

    // Verify each element
    for (size_t i = 0; i < result->data.size(); ++i) {
        EXPECT_NEAR(result->data[i], expected_output->data[i], 1e-5f); // Allow for small floating-point errors
    }
}

TEST(MatrixMultiplyTest, VectorTimesMatrix) {
    auto lhs = std::make_shared<Tensor>(std::vector<ssize_t>{1, 3}); // shape (1,3)
    auto rhs = std::make_shared<Tensor>(std::vector<ssize_t>{3, 2}); // shape (3,2)

    // Let lhs = [ [1, 2, 3] ]
    lhs->operator()({0, 0}) = 1.0f;
    lhs->operator()({0, 1}) = 2.0f;
    lhs->operator()({0, 2}) = 3.0f;

    // Let rhs =
    // [ [4,  5 ],
    //   [6,  7 ],
    //   [8,  9 ] ]
    rhs->operator()({0, 0}) = 4.0f;
    rhs->operator()({0, 1}) = 5.0f;
    rhs->operator()({1, 0}) = 6.0f;
    rhs->operator()({1, 1}) = 7.0f;
    rhs->operator()({2, 0}) = 8.0f;
    rhs->operator()({2, 1}) = 9.0f;

    auto result = matrix_multiply(lhs, rhs); // shape should be (1, 2)

    const auto& resultDims = result->get_dimensions();
    ASSERT_EQ(resultDims.size(), 2ul);
    EXPECT_EQ(resultDims[0], 1);
    EXPECT_EQ(resultDims[1], 2);

    // Expected result = lhs * rhs = [ [ (1*4 + 2*6 + 3*8), (1*5 + 2*7 + 3*9) ] ]
    //                           = [ [ (4 + 12 + 24), (5 + 14 + 27) ] ]
    //                           = [ [ 40, 46 ] ]
    EXPECT_FLOAT_EQ(result->operator()({0, 0}), 40.0f);
    EXPECT_FLOAT_EQ(result->operator()({0, 1}), 46.0f);
}

TEST(MatrixMultiplyTest, MatrixTimesMatrix) {
    auto lhs = std::make_shared<Tensor>(std::vector<ssize_t>{2, 3}); // shape (2,3)
    auto rhs = std::make_shared<Tensor>(std::vector<ssize_t>{3, 2}); // shape (3,2)

    // lhs =
    // [ [1, 2, 3],
    //   [4, 5, 6] ]
    lhs->operator()({0, 0}) = 1.0f;
    lhs->operator()({0, 1}) = 2.0f;
    lhs->operator()({0, 2}) = 3.0f;
    lhs->operator()({1, 0}) = 4.0f;
    lhs->operator()({1, 1}) = 5.0f;
    lhs->operator()({1, 2}) = 6.0f;

    // rhs =
    // [ [7,  8 ],
    //   [9,  10],
    //   [11, 12] ]
    rhs->operator()({0, 0}) = 7.0f;
    rhs->operator()({0, 1}) = 8.0f;
    rhs->operator()({1, 0}) = 9.0f;
    rhs->operator()({1, 1}) = 10.0f;
    rhs->operator()({2, 0}) = 11.0f;
    rhs->operator()({2, 1}) = 12.0f;

    auto result = matrix_multiply(lhs, rhs); // shape should be (2, 2)

    const auto& resultDims = result->get_dimensions();
    ASSERT_EQ(resultDims.size(), 2ul);
    EXPECT_EQ(resultDims[0], 2);
    EXPECT_EQ(resultDims[1], 2);

    // Expected = lhs * rhs =
    // [ [ (1*7 + 2*9 + 3*11),   (1*8 + 2*10 + 3*12) ],
    //   [ (4*7 + 5*9 + 6*11),   (4*8 + 5*10 + 6*12) ] ]
    // = [ [ (7 + 18 + 33),      (8 + 20 + 36) ],
    //     [ (28 + 45 + 66),     (32 + 50 + 72) ] ]
    // = [ [ 58, 64 ],
    //     [ 139, 154 ] ]
    EXPECT_FLOAT_EQ(result->operator()({0, 0}), 58.0f);
    EXPECT_FLOAT_EQ(result->operator()({0, 1}), 64.0f);
    EXPECT_FLOAT_EQ(result->operator()({1, 0}), 139.0f);
    EXPECT_FLOAT_EQ(result->operator()({1, 1}), 154.0f);
}

TEST(MatrixMultiplyTest, InvalidDimensions) {
    auto lhs = std::make_shared<Tensor>(std::vector<ssize_t>{2, 3}); // shape (2,3)
    auto rhs = std::make_shared<Tensor>(std::vector<ssize_t>{4, 5}); // shape (4,5)

    // Inner dimensions (3 and 4) do not match
    EXPECT_THROW({
        auto result = matrix_multiply(lhs, rhs);
        (void)result; // Prevent unused variable warning
    }, std::runtime_error);
}

TEST(ReluTest, HandlesBasicInput) {
    // Create an input tensor
    auto input = std::make_shared<Tensor>(std::vector<ssize_t>{1, 5}, 1.0f);
    input->data = {-3.0f, -1.0f, 0.0f, 2.0f, 5.0f};

    // Create an expected output tensor
    auto expected_output = std::make_shared<Tensor>(std::vector<ssize_t>{1, 5}, 1.0f);
    expected_output->data = {0.0f, 0.0f, 0.0f, 2.0f, 5.0f};

    // Compute ReLU
    auto result = math::relu(input);

    // Verify the size of the output
    ASSERT_EQ(result->dimensions, expected_output->dimensions);

    // Verify each element
    for (size_t i = 0; i < result->data.size(); ++i) {
        EXPECT_FLOAT_EQ(result->data[i], expected_output->data[i]); // Exact match for float comparison
    }
}