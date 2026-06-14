# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_rocm_gpu_plugin)
set(OBJ_NAME ${TARGET_NAME}_obj)

set(SOURCE_MASKS *.cpp *.hip)
set(HEADER_MASKS *.hpp *.hpp)
set(CMAKE_C_COMPILER "/opt/rocm/bin/amdclang++")
set(CMAKE_CXX_COMPILER "/opt/rocm/bin/amdclang++")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_AMD__")

file(GLOB_RECURSE
	SOURCES
	CONFIGURE_DEPENDS
	${SOURCE_MASKS}
)
list(FILTER SOURCES EXCLUDE REGEX ".*rocm_create_plugin\\.cpp$")
list(FILTER SOURCES EXCLUDE REGEX ".*rocm_create_extensions\\.cpp$")
list(FILTER SOURCES EXCLUDE REGEX ".*ops/examples/.*$")
file(GLOB_RECURSE
	HEADERS
	${HEADER_MASKS}
)

set_source_files_properties(*.hip *.hpp PROPERTIES LANGUAGE hip)

add_library(${OBJ_NAME} STATIC ${SOURCES})
target_compile_definitions(${OBJ_NAME} PRIVATE IMPLEMENT_INFERENCE_ENGINE_PLUGIN IMPLEMENT_OPENVINO_EXTENSION_API)

# Adds a shared library with plugin
ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "ROCM"
              SOURCES ${HEADERS} rocm_create_plugin.cpp rocm_create_extensions.cpp
              DEFAULT_CONFIG ${PLUGIN_DEFAULT_CONFIG}
              VERSION_DEFINES_FOR rocm_create_plugin.cpp
              ADD_CLANG_FORMAT)

# Enable support of CC for the plugin
ov_mark_target_as_cc(${TARGET_NAME})

set_property(TARGET ${OBJ_NAME} PROPERTY ROCM_ARCHITECTURES ${CMAKE_ROCM_ARCHITECTURES})

ov_target_link_whole_archive(${TARGET_NAME} ${OBJ_NAME})

LINK_DIRECTORIES("/opt/rocm/lib")

target_include_directories(${OBJ_NAME}
                           SYSTEM PUBLIC
                           "/opt/rocm/include"
)
target_include_directories(${OBJ_NAME}
                           PUBLIC
                           "${HIPTENSOR_INCLUDE_DIR}"
                           "${CMAKE_CURRENT_SOURCE_DIR}"
                           "${OpenVINOROCMGpuPlugin_SOURCE_DIR}/include"
                           PRIVATE $<TARGET_PROPERTY:openvino::conditional_compilation,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(${OBJ_NAME}
                      PUBLIC
                      GSL
                      fmt-header-only
                      openvino::runtime
                      openvino::runtime::dev
                      openvino::core::dev
                      PRIVATE
                      hip::host
                      hip::device
                      MIOpen
                      roc::rocblas
                      ${NGRAPH_LIBRARIES}
                      dl
)

# CK WMMA: add include path for CK templates (header-only device code)
if(ENABLE_CK_WMMA)
    target_include_directories(${OBJ_NAME} SYSTEM PUBLIC "/opt/rocm/include")
    # Link CK static library containing pre-instantiated conv instances
    if(TARGET composable_kernel::device_conv_operations)
        target_link_libraries(${OBJ_NAME} PRIVATE composable_kernel::device_conv_operations)
    elseif(EXISTS "/opt/rocm/lib/libdevice_conv_operations.a")
        target_link_libraries(${OBJ_NAME} PRIVATE "/opt/rocm/lib/libdevice_conv_operations.a")
    endif()
    # CK requires C++17 (already set above)
    target_compile_options(${OBJ_NAME} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-std=c++17>)
endif()
#target_compile_features(${OBJ_NAME} PRIVATE cxx_std_11)
# set_target_properties(${OBJ_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})


# ATTENTION: uncomment to register a plugin in the plugins.xml file
if(ENABLE_ROCM)
    ov_register_plugins(MAIN_TARGET ${TARGET_NAME}
                        POSSIBLE_PLUGINS ${TARGET_NAME})
endif()
