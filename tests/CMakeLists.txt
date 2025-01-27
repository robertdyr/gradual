# Enable testing
enable_testing()

# Fetch Google Test
include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)

FetchContent_MakeAvailable(googletest)

# Define a function to streamline test creation
function(add_gradual_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name}
            PRIVATE
            gradual_lib
            GTest::gtest
            GTest::gtest_main
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Add tests
add_gradual_test(test_tensor)
add_gradual_test(test_operations)
add_gradual_test(test_math)
add_gradual_test(test_optimizers)
