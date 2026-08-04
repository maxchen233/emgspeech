cmake_minimum_required(VERSION 3.10)
project(emg_inference CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

add_executable(emg_inference
    src/main.cpp
    src/ops.cpp
    src/emg_model.cpp
)
target_include_directories(emg_inference PRIVATE src)

find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(cpp_forward
        src/pybind_module.cpp
        src/ops.cpp
        src/emg_model.cpp
    )
    target_include_directories(cpp_forward PRIVATE src)
    target_compile_options(cpp_forward PRIVATE -O3 -march=native)
    message(STATUS "pybind11 found - building cpp_forward Python module")
else()
    message(STATUS "pybind11 not found - skipping Python module")
endif()
