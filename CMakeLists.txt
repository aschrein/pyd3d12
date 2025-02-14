cmake_minimum_required(VERSION 3.10)

project(PyD3D12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(native)