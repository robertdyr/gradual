# Create an executable that uses the library
add_executable(gradual_example main.cpp)

# Link the library we built in ../src
target_link_libraries(gradual_example PRIVATE gradual_lib)
