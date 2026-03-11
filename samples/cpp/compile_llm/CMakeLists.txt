# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT TARGET openvino::npu_ops)
    message(FATAL_ERROR "Target openvino::npu_ops is not available. Enable/build Intel NPU ops first.")
endif()

ov_add_sample(NAME compile_llm
              SOURCES
                "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              DEPENDENCIES
                format_reader
                ie_samples_utils
                openvino::npu_ops)

target_include_directories(compile_llm PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugins/intel_npu/src/ops/include
)