#include <gtest/gtest.h>
#include "tensor.hpp"

TEST(TensorTest, Constructor) {
    Tensor tensor({2, 3}, 0.0f);

    // Verify dimensions
    EXPECT_EQ(tensor.get_dimensions(), (std::vector<ssize_t>{2, 3}));

    // Verify size
    EXPECT_EQ(tensor.size(), 6);

    // Verify initial values
    for (ssize_t i = 0; i < 2; ++i) {
        for (ssize_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(tensor({i, j}), 0.0f);
        }
    }
}

TEST(TensorTest, ElementAccess) {
    Tensor tensor({2, 2}, 1.0f);

    tensor({0, 0}) = 2.0f;
    tensor({1, 1}) = 3.5f;

    // Verify modified values
    EXPECT_FLOAT_EQ(tensor({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(tensor({1, 1}), 3.5f);

    // Verify unchanged values
    EXPECT_FLOAT_EQ(tensor({0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(tensor({1, 0}), 1.0f);
}

TEST(TensorTest, Fill) {
    Tensor tensor({2, 3}, 0.0f);

    tensor.fill(4.5f);

    // Verify all elements have the new value
    for (ssize_t i = 0; i < 2; ++i) {
        for (ssize_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(tensor({i, j}), 4.5f);
        }
    }
}

TEST(TensorTest, InvalidAccess) {
    Tensor tensor({2, 2}, 0.0f);

    // Access with mismatched indices
    EXPECT_THROW(tensor({0}), std::invalid_argument);
    EXPECT_THROW(tensor({0, 1, 2}), std::invalid_argument);
}

TEST(TensorTest, InternalIndexingConsistency) {
    Tensor tensor({2, 3}, 0.0f);

    tensor({0, 0}) = 1.0f;
    tensor({1, 2}) = 2.5f;

    // Verify flat indexing produces consistent results
    EXPECT_FLOAT_EQ(tensor({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(tensor({1, 2}), 2.5f);
}

TEST(TensorTest, Transpose) {
    // Create a 2x3 tensor with specific values
    Tensor tensor({2, 3}, 0.0f);
    tensor({0, 0}) = 1.0f;
    tensor({0, 1}) = 2.0f;
    tensor({0, 2}) = 3.0f;
    tensor({1, 0}) = 4.0f;
    tensor({1, 1}) = 5.0f;
    tensor({1, 2}) = 6.0f;

    // Perform the transpose
    auto transposed = tensor.transpose();

    // Verify dimensions
    EXPECT_EQ(transposed->get_dimensions(), (std::vector<ssize_t>{3, 2}));

    // Verify transposed values
    EXPECT_FLOAT_EQ((*transposed)({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ((*transposed)({1, 0}), 2.0f);
    EXPECT_FLOAT_EQ((*transposed)({2, 0}), 3.0f);
    EXPECT_FLOAT_EQ((*transposed)({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ((*transposed)({1, 1}), 5.0f);
    EXPECT_FLOAT_EQ((*transposed)({2, 1}), 6.0f);
}

TEST(TensorTest, TransposeInvalidDimensions) {
    // Create a 1D tensor (invalid for transpose)
    Tensor tensor({3}, 0.0f);

    // Attempt to transpose and expect an exception
    EXPECT_THROW(tensor.transpose(), std::invalid_argument);

    // Create a 3D tensor (invalid for transpose)
    Tensor tensor3D({2, 3, 4}, 0.0f);

    // Attempt to transpose and expect an exception
    EXPECT_THROW(tensor3D.transpose(), std::invalid_argument);
}
