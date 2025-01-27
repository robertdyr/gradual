# Create the tensor library
add_library(gradual_lib
        tensor.cpp
        operations.cpp
        math.cpp
        optimizers.cpp
        utils.hpp)

# Include the current directory for headers
target_include_directories(gradual_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Create the main executable
add_executable(gradual main.cpp)

# Link the main executable with the tensor library
target_link_libraries(gradual PRIVATE gradual_lib)
