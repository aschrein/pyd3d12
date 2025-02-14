cmake_minimum_required(VERSION 3.10)

function(_add_module module_name)
    pybind11_add_module(${module_name} ${ARGN})

    target_include_directories(${module_name}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../bin/dxc_2024_07_31/inc
            ${CMAKE_CURRENT_SOURCE_DIR}/../bin/Microsoft.Direct3D.D3D12.1.615.0/build/native/include
        )

    target_link_libraries(${module_name}
            PRIVATE
            dxgi
            d3d12
        )

    install(TARGETS ${module_name} LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${module_name})
endfunction()

_add_module(native src/lib.cpp)