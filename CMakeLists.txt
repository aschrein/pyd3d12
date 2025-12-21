cmake_minimum_required(VERSION 3.10)

project(PyD3D12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

execute_process(
  COMMAND python -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "pybind11_DIR: '${pybind11_DIR}'")

find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(native)