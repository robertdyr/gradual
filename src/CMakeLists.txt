# Create the tensor library
add_library(gradual_lib
        tensor.cpp
        operations.cpp
        math.cpp
        optimizers.cpp)

# Include the current directory for headers
target_include_directories(gradual_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

