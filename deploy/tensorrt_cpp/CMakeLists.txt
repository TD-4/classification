cmake_minimum_required(VERSION 3.14)
project(tensorrt_cpp)

set(CMAKE_CXX_STANDARD 11)

# ===================
# compile main
# ===================
add_subdirectory(classifier)
include_directories(classifier)
link_directories(classifier)

add_executable(tensorrt_cpp main.cpp)

target_link_libraries(tensorrt_cpp trt)