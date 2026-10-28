cmake_minimum_required(VERSION 3.10)
project(emg_inference CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

# ── Common source files ──────────────────────────────────────
set(COMMON_SOURCES
    src/ops.cpp
    src/emg_model.cpp
    src/hifigan.cpp
)

# ── Platform-specific BLAS ───────────────────────────────────
if(APPLE)
    find_library(ACCELERATE_LIB Accelerate)
    # Find vecLib headers for cblas.h
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(VECLIB_HEADERS "${SDK_PATH}/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers")
    set(BLAS_INCLUDE_DIRS ${VECLIB_HEADERS})
    set(BLAS_LIBRARIES ${ACCELERATE_LIB})
else()
    set(BLAS_INCLUDE_DIRS "/usr/include/openblas")
    set(BLAS_LIBRARIES openblas)
endif()

# ── Golden test executable (original) ────────────────────────
add_executable(emg_inference
    src/main.cpp
    ${COMMON_SOURCES}
)
target_include_directories(emg_inference PRIVATE src ${BLAS_INCLUDE_DIRS})
target_link_libraries(emg_inference ${BLAS_LIBRARIES})

# ── HiFi-GAN golden test ────────────────────────────────────
add_executable(hifigan_test
    src/hifigan_test.cpp
    ${COMMON_SOURCES}
)
target_include_directories(hifigan_test PRIVATE src ${BLAS_INCLUDE_DIRS})
target_link_libraries(hifigan_test ${BLAS_LIBRARIES})

# ── Full pipeline executable ─────────────────────────────────
add_executable(emg_pipeline
    src/pipeline_main.cpp
    src/streaming_pipeline.cpp
    ${COMMON_SOURCES}
)
target_include_directories(emg_pipeline PRIVATE src ${BLAS_INCLUDE_DIRS})
target_link_libraries(emg_pipeline ${BLAS_LIBRARIES})
target_compile_options(emg_pipeline PRIVATE -O3 -march=native)

# ── pybind11 module (optional) ───────────────────────────────
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(cpp_forward
        src/pybind_module.cpp
        ${COMMON_SOURCES}
    )
    target_include_directories(cpp_forward PRIVATE src ${BLAS_INCLUDE_DIRS})
    target_link_libraries(cpp_forward PRIVATE ${BLAS_LIBRARIES})
    target_compile_options(cpp_forward PRIVATE -O3 -march=native)
    message(STATUS "pybind11 found - building cpp_forward Python module")
else()
    message(STATUS "pybind11 not found - skipping Python module")
endif()
