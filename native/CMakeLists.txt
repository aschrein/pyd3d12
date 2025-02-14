cmake_minimum_required(VERSION 3.10)

function(_add_module module_name)
    pybind11_add_module(${module_name} ${ARGN})
    install(TARGETS ${module_name} LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${module_name})
endfunction()

_add_module(native src/lib.cpp)