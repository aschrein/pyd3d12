cmake_minimum_required(VERSION 3.10)

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)

function(_add_module module_name)
    pybind11_add_module(${module_name} ${ARGN})

    target_include_directories(${module_name}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../bin/dxc_2024_07_31/inc
            ${CMAKE_CURRENT_SOURCE_DIR}/../bin/Microsoft.Direct3D.D3D12.1.615.0/build/native/include
            ${THIRD_PARTY_DIR}/imgui
            ${THIRD_PARTY_DIR}
        )


    install(TARGETS ${module_name} LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${module_name})
endfunction()


_add_module(native
    src/lib.cpp
    src/d3d12_bindings_0.cpp
)
target_link_libraries(native PRIVATE 
    dxgi
    d3d12
    ${THIRD_PARTY_DIR}/ags_lib/lib/amd_ags_x64.lib
)
add_custom_command(TARGET native POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${THIRD_PARTY_DIR}/ags_lib/lib/amd_ags_x64.dll
    "$<TARGET_FILE_DIR:native>/amd_ags_x64.dll"
)

_add_module(rdoc src/rdoc.cpp)
_add_module(pix src/pix.cpp)

_add_module(imgui
    src/imgui_bindings_0.cpp
    ${THIRD_PARTY_DIR}/imgui/imgui.cpp
    ${THIRD_PARTY_DIR}/imgui/imgui_draw.cpp
    ${THIRD_PARTY_DIR}/imgui/imgui_tables.cpp
    ${THIRD_PARTY_DIR}/imgui/imgui_demo.cpp
    ${THIRD_PARTY_DIR}/imgui/imgui_widgets.cpp
)
target_compile_definitions(imgui PRIVATE IMGUI_ENABLE_DOCKING=1)