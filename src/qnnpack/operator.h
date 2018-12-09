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

#include <qnnpack/requantization.h>


enum qnnp_format {
  qnnp_format_quint8 = 0x02000000,
  qnnp_format_float32 = 0x02020202,
  qnnp_format_float16 = 0x01010101,
};

enum qnnp_ukernel_type {
  qnnp_ukernel_type_none = 0,
  qnnp_ukernel_type_add,
  qnnp_ukernel_type_average_pooling,
  qnnp_ukernel_type_channel_shuffle,
  qnnp_ukernel_type_clamp,
  qnnp_ukernel_type_conv,
  qnnp_ukernel_type_dwconv,
  qnnp_ukernel_type_gemm,
  qnnp_ukernel_type_global_average_pooling,
  qnnp_ukernel_type_lut,
  qnnp_ukernel_type_max_pooling,
  qnnp_ukernel_type_softargmax,
  qnnp_ukernel_type_xzp_gemm,
};

struct qnnp_operator {
  size_t batch_size;
  uint32_t input_padding_top;
  uint32_t input_padding_right;
  uint32_t input_padding_bottom;
  uint32_t input_padding_left;
  uint32_t adjustment_height;
  uint32_t adjustment_width;
  uint32_t kernel_height;
  uint32_t kernel_width;
  uint32_t stride_height;
  uint32_t stride_width;
  uint32_t dilation_height;
  uint32_t dilation_width;
  uint32_t groups;
  size_t group_stride;
  size_t group_channels;
  size_t group_input_channels;
  size_t group_output_channels;
  size_t channels;

  size_t input_height;
  size_t input_width;
  size_t input_pixel_stride;
  const void* input;
  const void** indirection_buffer;
  void* a_sum;

  size_t input2_pixel_stride;
  const void* input2;

  size_t output_height;
  size_t output_width;
  size_t output_pixel_stride;
  void* output;

  void* packed_weights;
  float input_scale;
  float output_scale;
  uint8_t input_zero_point;
  uint8_t kernel_zero_point;
  uint8_t output_zero_point;
  uint8_t output_min;
  uint8_t output_max;

  size_t valid_batch_size;
  size_t last_input_height;
  size_t last_input_width;
  const void* last_input;

  void* zero_buffer;
  void* zero_pointer;
  void* lookup_table;

  union {
    union qnnp_q31_requantization_params requantization_params;
    union qnnp_conv_quantization_params conv_quantization_params;
    union qnnp_add_quantization_params add_quantization_params;
    union qnnp_avgpool_quantization_params avgpool_quantization_params;
    union qnnp_u8_clamping_params u8_clamping_params;
  };
  enum qnnp_ukernel_type ukernel_type;
  enum qnnp_format format;
};

static inline uint32_t qnnp_operator_get_log2_output_element_size(const struct qnnp_operator* convolution) {
  return (uint32_t) (convolution->format & UINT32_C(0xFF));
}

static inline uint32_t qnnp_operator_get_log2_input_element_size(const struct qnnp_operator* convolution) {
  return (uint32_t) ((convolution->format >> 8) & UINT32_C(0xFF));
}

static inline uint32_t qnnp_operator_get_log2_kernel_element_size(const struct qnnp_operator* convolution) {
  return (uint32_t) ((convolution->format >> 16) & UINT32_C(0xFF));
}

static inline uint32_t qnnp_operator_get_log2_bias_element_size(const struct qnnp_operator* convolution) {
  return (uint32_t) ((convolution->format >> 24) & UINT32_C(0xFF));
}
