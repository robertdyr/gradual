cmake_minimum_required(VERSION 3.16)
project(Gradual)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Enable testing at the root level
include(CTest)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
