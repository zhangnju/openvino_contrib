# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.21)
project(OpenVINOROCMGpuPlugin CXX HIP )

# Initialize CMAKE_rocm_ARCHITECTURES when CMAKE_rocm_COMPILER_ID is AMD.
# Raise an error if rocm_ARCHITECTURES is empty.
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()
if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

find_package(HIP QUIET)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
#set(CMAKE_CXX_COMPILER /opt/rocm/bin/amdclang++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_AMD__")
#set(GPU_TARGETS "gfx90a;gfx942")
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
     set(CMAKE_ROCM_STANDARD 14)
endif()

if (rocm_KERNEL_PRINT_LOG)
    add_definitions(-DROCM_KERNEL_PRINT_LOG)
endif()
if (ENABLE_MIOPEN_BACKEND_API)
    add_definitions(-DENABLE_MIOPEN_BACKEND_API)
endif()

find_package(OpenVINODeveloperPackage REQUIRED
             PATHS "${InferenceEngineDeveloperPackage_DIR}")

include(cmake/features.cmake)

set(HAS_REQUIRED REQUIRED)

find_package(HIP REQUIRED)
find_package(ROCM REQUIRED)
find_package(Threads REQUIRED)
find_package(miopen REQUIRED)
find_package(rocblas REQUIRED)
find_package(hiptensor REQUIRED)

find_library(MIOPEN_PATH
             NAMES MIOpen
             ${HAS_REQUIRED}
             HINTS "$ENV{MIOPEN_PATH}" "/opt/rocm/" "$ENV{rocm_PATH}"
             PATH_SUFFIXES /opt/rocm/ lib)

find_library(HIPTENSOR_PATH
             NAMES hiptensor
             ${HAS_REQUIRED}
             HINTS "$ENV{HIPTENSOR_PATH}" "/opt/rocm" "$ENV{rocm_PATH}" 
             PATH_SUFFIXES /opt/rocm/ lib)

find_library(ROCBLAS_PATH
             NAMES rocblas
             ${HAS_REQUIRED}
             HINTS "$ENV{ROCBLAS_PATH}" "/opt/rocm" "$ENV{rocm_PATH}" 
             PATH_SUFFIXES /opt/rocm/ lib)

find_path(HIPTENSOR_INCLUDE_DIR 
            NAMES hiptensor.hpp
            ${HAS_REQUIRED}
            HINTS ${rocm_PATH} /opt/rocm/include
            PATH_SUFFIXES hiptensor include)             

find_path(MIOPEN_INCLUDE_DIR 
            NAMES miopen.h
            ${HAS_REQUIRED}
            HINTS ${rocm_PATH} /opt/rocm/include
            PATH_SUFFIXES miopen include)

find_path(ROCBLAS_INCLUDE_DIR 
            NAMES rocblas.h
            ${HAS_REQUIRED}
            HINTS ${rocm_PATH} /opt/rocm/include
            PATH_SUFFIXES rocblas include)

if(WIN32)
    string(REPLACE "-Zi" "-Z7" CMAKE_rocm_FLAGS_DEBUG "${CMAKE_rocm_FLAGS_DEBUG}")
    message("-- [rocm_gpu] CMAKE_rocm_FLAGS_DEBUG ${CMAKE_rocm_FLAGS_DEBUG}")
endif()

message("-- [rocm_gpu] HIPTENSOR_PATH ${HIPTENSOR_PATH}")
message("-- [rocm_gpu] HIPTENSOR_INCLUDE_DIR ${HIPTENSOR_INCLUDE_DIR}")
message("-- [rocm_gpu] MIOPEN_PATH ${MIOPEN_PATH}")
message("-- [rocm_gpu] MIOPEN_INCLUDE_DIR ${MIOPEN_INCLUDE_DIR}")
message("-- [rocm_gpu] ROCBLAS_PATH ${ROCBLAS_PATH}")
message("-- [rocm_gpu] ROCBLAS_INCLUDE_DIR ${ROCBLAS_INCLUDE_DIR}")

add_library(rocm::MIOpen SHARED IMPORTED)
set_target_properties(rocm::MIOpen PROPERTIES IMPORTED_LOCATION "${MIOPEN_PATH}")
set_target_properties(rocm::MIOpen PROPERTIES IMPORTED_IMPLIB "${MIOPEN_PATH}")
add_library(rocm::rocblas SHARED IMPORTED)
set_target_properties(rocm::rocblas PROPERTIES IMPORTED_LOCATION "${ROCBLAS_PATH}")
set_target_properties(rocm::rocblas PROPERTIES IMPORTED_IMPLIB "${ROCBLAS_PATH}")
add_library(rocm::hiptensor SHARED IMPORTED)
set_target_properties(rocm::hiptensor PROPERTIES IMPORTED_LOCATION "${HIPTENSOR_PATH}")
set_target_properties(rocm::hiptensor PROPERTIES IMPORTED_IMPLIB "${HIPTENSOR_PATH}")

include(FetchContent)

# In this example we are picking a specific tag.
# You can also pick a specific commit, if you need to.
FetchContent_Declare(GSL
                     GIT_REPOSITORY "https://github.com/microsoft/GSL"
                     GIT_TAG "v3.1.0")

FetchContent_MakeAvailable(GSL)

# it contains c++14 requirement, which overrides c++17 setting in CMAKE_CUDA_FLAGS
set_property(TARGET GSL PROPERTY INTERFACE_COMPILE_FEATURES)

FetchContent_Declare(fmt-header-only
                     GIT_REPOSITORY "https://github.com/fmtlib/fmt"
                     GIT_TAG "7.1.3")

FetchContent_MakeAvailable(fmt-header-only)

# it contains c++11 requirement, which overrides c++17 setting in CMAKE_CUDA_FLAGS
set_property(TARGET fmt-header-only PROPERTY INTERFACE_COMPILE_FEATURES)

add_subdirectory(src)


if(ENABLE_TESTS)
    include(CTest)
    enable_testing()

    add_subdirectory(tests/unit)
    if(ENABLE_FUNCTIONAL_TESTS)
        add_subdirectory(tests/functional)
    endif()
endif()

# install

# ATTENTION: uncomment to install component
# ov_cpack(template)
