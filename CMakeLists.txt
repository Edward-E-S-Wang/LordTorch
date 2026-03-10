cmake_minimum_required(VERSION 3.16)
project(mtorch VERSION 0.1.0 LANGUAGES CXX)

option(MTORCH_BUILD_TESTS "Build tests" ON)
option(MTORCH_BUILD_EXAMPLES "Build examples" ON)
option(MTORCH_BUILD_PYTHON "Build Python bindings" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(mtorch
  src/core/storage.cpp
  src/core/tensor_impl.cpp
  src/core/tensor.cpp
  src/ops/creation.cpp
  src/ops/elementwise.cpp
  src/ops/reduction.cpp
  src/ops/matmul.cpp
  src/ops/activation.cpp
  src/autograd/node.cpp
  src/autograd/engine.cpp
  src/nn/module.cpp
  src/nn/parameter.cpp
  src/nn/linear.cpp
  src/nn/loss.cpp
  src/optim/optimizer.cpp
  src/optim/sgd.cpp
  src/utils/print.cpp
)

target_include_directories(mtorch
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MTORCH_BUILD_EXAMPLES)
  add_executable(example_tensor_basics examples/tensor_basics.cpp)
  target_link_libraries(example_tensor_basics PRIVATE mtorch)

  add_executable(example_linear_regression examples/linear_regression.cpp)
  target_link_libraries(example_linear_regression PRIVATE mtorch)
endif()

if(MTORCH_BUILD_TESTS)
  enable_testing()
  add_executable(test_tensor tests/test_tensor.cpp)
  target_link_libraries(test_tensor PRIVATE mtorch)
  add_test(NAME test_tensor COMMAND test_tensor)

  add_executable(test_reduction tests/test_reduction.cpp)
  target_link_libraries(test_reduction PRIVATE mtorch)
  add_test(NAME test_reduction COMMAND test_reduction)
endif()
