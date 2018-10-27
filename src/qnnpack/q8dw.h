/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/params.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_Q8DW_FUNCTION(fn_name)                                  \
  void fn_name(                                                         \
    size_t channels,                                                    \
    size_t output_width,                                                \
    const uint8_t** input,                                              \
    const void* weights,                                                \
    uint8_t* output,                                                    \
    size_t input_stride,                                                \
    size_t output_increment,                                            \
    const union qnnp_conv_quantization_params* quantization_params);

DECLARE_Q8DW_FUNCTION(q8dw_ukernel_9c8__neon)
DECLARE_Q8DW_FUNCTION(q8dw_ukernel_9c8__sse2)

#ifdef __cplusplus
} /* extern "C" */
#endif
