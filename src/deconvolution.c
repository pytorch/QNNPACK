/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include <cpuinfo.h>
#include <fp16.h>

#include <qnnpack.h>
#include <qnnpack/convolution.h>
#include <qnnpack/requantization.h>
#include <qnnpack/log.h>
#include <qnnpack/math.h>
#include <qnnpack/pack.h>
#include <qnnpack/params.h>

static inline size_t compute_output_dimension(
    size_t input_dimension,
    size_t input_padding_dimension,
    size_t adjustment_dimension,
    size_t kernel_dimension,
    size_t dilation_dimension,
    size_t stride_dimension)
{
  const size_t effective_kernel_dimension = (kernel_dimension - 1) * dilation_dimension + 1;
  return stride_dimension * (input_dimension - 1) + adjustment_dimension + effective_kernel_dimension - input_padding_dimension;
}

enum qnnp_status qnnp_create_deconvolution2d_nhwc_q8(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t adjustment_height,
    uint32_t adjustment_width,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t kernel_zero_point,
    float kernel_scale,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    qnnp_operator_t* deconvolution_out)
{
  qnnp_operator_t deconvolution = NULL;
  enum qnnp_status status = qnnp_status_uninitialized;

  if (!qnnp_params.initialized) {
    qnnp_log_error("qnnp_create_deconvolution2d_nhwc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = qnnp_status_invalid_parameter;

  if (kernel_width == 0 || kernel_height == 0) {
    qnnp_log_error(
      "failed to create deconvolution with %" PRIu32 "x%" PRIu32 " kernel: kernel dimensions must be non-zero",
      kernel_width, kernel_height);
    goto error;
  }

  if (stride_width == 0 || stride_height == 0) {
    qnnp_log_error(
      "failed to create deconvolution with %" PRIu32 "x%" PRIu32 " stride: "
      "stride dimensions must be non-zero",
      stride_width, stride_height);
    goto error;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    qnnp_log_error(
      "failed to create deconvolution with %" PRIu32 "x%" PRIu32 " dilation: "
      "dilation dimensions must be non-zero",
      dilation_width, dilation_height);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    qnnp_log_error(
      "failed to create deconvolution with %.7g input scale: scale must be finite and positive", input_scale);
    goto error;
  }

  if (kernel_scale <= 0.0f || !isnormal(kernel_scale)) {
    qnnp_log_error(
      "failed to create deconvolution with %.7g kernel scale: scale must be finite and positive", kernel_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    qnnp_log_error(
      "failed to create deconvolution with %.7g output scale: scale must be finite and positive", output_scale);
    goto error;
  }

  status = qnnp_status_unsupported_parameter;

  const float deconvolution_scale = input_scale * kernel_scale / output_scale;
  if (deconvolution_scale >= 1.0f) {
    qnnp_log_error(
      "failed to create deconvolution with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
      "deconvolution scale %.7g is greater or equal to 1.0",
      input_scale, kernel_scale, output_scale, deconvolution_scale);
    goto error;
  }

  status = qnnp_status_out_of_memory;

  deconvolution = calloc(1, sizeof(struct qnnp_operator));
  if (deconvolution == NULL) {
    qnnp_log_error("failed to allocate %zu bytes for qnnp_operator structure", sizeof(struct qnnp_operator));
    goto error;
  }

  uint32_t flags = QNNP_CONVOLUTION_FLAG_ZERO;

  const uint32_t nr = qnnp_params.q8conv.nr;
  const uint32_t kr = qnnp_params.q8conv.kr;

  const uint32_t n_stride = (group_output_channels + (nr - 1)) & -nr;
  const uint32_t k_stride = (group_input_channels + (kr - 1)) & -kr;
  deconvolution->packed_kernel = malloc(sizeof(uint8_t) * kernel_height * kernel_width * groups * k_stride * n_stride);
  if (deconvolution->packed_kernel == NULL) {
    qnnp_log_error("failed to allocate %zu bytes for packed kernel data",
      sizeof(uint8_t) * kernel_height * kernel_width * groups * k_stride * n_stride);
    goto error;
  }
  const size_t kernel_size = kernel_height * kernel_width;
  memset(deconvolution->packed_kernel, kernel_zero_point, sizeof(uint8_t) * groups * kernel_size * k_stride * n_stride);

  for (uint32_t group = 0; group < groups; group++) {
    pack_q8deconv_b(
        group_output_channels, kernel_size, group_input_channels,
        nr, kr,
        kernel + group * group_output_channels * kernel_size * group_input_channels,
        deconvolution->packed_kernel + group * kernel_size * n_stride * k_stride);
  }

  deconvolution->bias = malloc(sizeof(int32_t) * groups * n_stride);
  if (deconvolution->bias == NULL) {
    qnnp_log_error("failed to allocate %zu bytes for packed bias data", sizeof(int32_t) * groups * n_stride);
    goto error;
  }
  for (uint32_t group = 0; group < groups; group++) {
    memcpy(
      (int32_t*) deconvolution->bias + group * n_stride,
      bias + group * group_output_channels,
      sizeof(int32_t) * group_output_channels);
  }

  if (flags & QNNP_CONVOLUTION_FLAG_ZERO) {
    const size_t zero_size = sizeof(uint8_t) * k_stride + (group_input_channels >= 8 ? 0 : 8);
    deconvolution->zero = malloc(zero_size);
    if (deconvolution->zero == NULL) {
      qnnp_log_error("failed to allocate %zu bytes for zero padding", zero_size);
      goto error;
    }
    memset(deconvolution->zero, input_zero_point, zero_size);
  }

  deconvolution->input_padding_top = input_padding_top;
  deconvolution->input_padding_right = input_padding_right;
  deconvolution->input_padding_bottom = input_padding_bottom;
  deconvolution->input_padding_left = input_padding_left;
  deconvolution->adjustment_height = adjustment_height;
  deconvolution->adjustment_width = adjustment_width;

  deconvolution->kernel_height = kernel_height;
  deconvolution->kernel_width = kernel_width;
  deconvolution->stride_height = stride_height;
  deconvolution->stride_width = stride_width;
  deconvolution->dilation_height = dilation_height;
  deconvolution->dilation_width = dilation_width;
  deconvolution->groups = groups;
  deconvolution->group_input_channels = group_input_channels;
  deconvolution->group_output_channels = group_output_channels;

  deconvolution->input_zero_point = input_zero_point;
  deconvolution->kernel_zero_point = kernel_zero_point;

  deconvolution->conv_quantization_params =
    qnnp_compute_conv_quantization_params(
      input_zero_point, kernel_zero_point,
      deconvolution_scale, output_zero_point, output_min, output_max);

  deconvolution->format = qnnp_format_quint8;
  deconvolution->flags = flags;

  *deconvolution_out = deconvolution;
  return qnnp_status_success;

error:
  qnnp_delete_operator(deconvolution);
  return status;
}

enum qnnp_status qnnp_setup_deconvolution2d_nhwc_q8(
    qnnp_operator_t deconvolution,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const uint8_t* input,
    size_t input_pixel_stride,
    uint8_t* output,
    size_t output_pixel_stride,
    pthreadpool_t threadpool)
{
  if (!qnnp_params.initialized) {
    qnnp_log_error("qnnp_setup_deconvolution2d_nhwc_q8 failed because QNNPACK is not properly initialized");
    return qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    qnnp_log_error("failed to setup deconvolution with batch size %zu: batch size must be non-zero", batch_size);
    return qnnp_status_invalid_parameter;
  }

  if (input_width == 0 || input_height == 0) {
    qnnp_log_error(
      "failed to setup deconvolution with %zux%zu input: input dimensions must be non-zero",
      input_width,
      input_height);
    return qnnp_status_invalid_parameter;
  }

  deconvolution->batch_size = batch_size;
  deconvolution->input_height = input_height;
  deconvolution->input_width = input_width;
  deconvolution->input = input;
  deconvolution->input_pixel_stride = input_pixel_stride;
  deconvolution->output = output;
  deconvolution->output_pixel_stride = output_pixel_stride;

  const size_t kernel_height = deconvolution->kernel_height;
  const size_t kernel_width = deconvolution->kernel_width;
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t stride_height = deconvolution->stride_height;
  const size_t stride_width = deconvolution->stride_width;
  const size_t output_height = deconvolution->output_height = compute_output_dimension(
    input_height, deconvolution->input_padding_top + deconvolution->input_padding_bottom,
    deconvolution->adjustment_height, kernel_height, deconvolution->dilation_height, stride_height);
  const size_t output_width = deconvolution->output_width = compute_output_dimension(
    input_width, deconvolution->input_padding_left + deconvolution->input_padding_right,
    deconvolution->adjustment_width, kernel_width, deconvolution->dilation_width, stride_width);

  const size_t groups = deconvolution->groups;
  const size_t output_size = output_height * output_width;
  const size_t output_tile_size = qnnp_params.q8conv.mr;
  const size_t tiled_output_size = round_up(output_size, output_tile_size);
  const size_t im2col_buffer_size = sizeof(void*) * batch_size * groups * tiled_output_size * kernel_size;

  const void** im2col_buffer = (const void**) realloc(deconvolution->im2col_buffer, im2col_buffer_size);
  if (im2col_buffer == NULL) {
    qnnp_log_error("failed to allocate %zu bytes for im2col buffer", im2col_buffer_size);
    return qnnp_status_out_of_memory;
  }
  deconvolution->im2col_buffer = im2col_buffer;

  const void* zero = deconvolution->zero;
  if (deconvolution->group_input_channels < 8) {
    zero = (const void*) ((uintptr_t) zero + 8);
  }

  for (size_t group = 0; group < groups; group++) {
    for (size_t image = 0; image < batch_size; image++) {
      for (size_t output_tile_start = 0; output_tile_start < tiled_output_size; output_tile_start += output_tile_size) {
        for (size_t output_tile_offset = 0; output_tile_offset < output_tile_size; output_tile_offset++) {
          const size_t tiled_output_index = output_tile_start + output_tile_offset;
          const size_t output_index = min(tiled_output_index, output_size - 1);
          const size_t output_y = output_index / output_width;
          const size_t output_x = output_index % output_width;
          for (size_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
            const size_t y = output_y + deconvolution->input_padding_top - kernel_y * deconvolution->dilation_height;
            const size_t input_y = y / stride_height;
            for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
              const size_t x = output_x + deconvolution->input_padding_left - kernel_x * deconvolution->dilation_width;
              const size_t input_x = x / stride_width;
              const size_t im2col_index =
                (group * batch_size + image) * tiled_output_size * kernel_size + output_tile_start * kernel_size + (kernel_y * kernel_width + kernel_x) * output_tile_size + output_tile_offset;
              if (input_y * stride_height == y && input_y < input_height && input_x * stride_width == x && input_x < input_width) {
                im2col_buffer[im2col_index] =
                  input + ((image * input_height + input_y) * input_width + input_x) * input_pixel_stride + group * deconvolution->group_input_channels;
              } else {
                im2col_buffer[im2col_index] = zero;
              }
            }
          }
        }
      }
    }
  }

  return qnnp_status_success;
}
