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

#include <cpuinfo.h>
#include <fp16.h>
#include <fxdiv.h>

#include <qnnpack.h>
#include <qnnpack/convolution.h>
#include <qnnpack/requantization.h>
#include <qnnpack/log.h>
#include <qnnpack/math.h>
#include <qnnpack/pack.h>
#include <qnnpack/params.h>
#include <qnnpack/q8gemm.h>


enum qnnp_status qnnp_create_fully_connected_nc_q8(
    size_t input_channels,
    size_t output_channels,
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
    qnnp_operator_t* fully_connected_out)
{
  qnnp_operator_t fully_connected = NULL;
  enum qnnp_status status = qnnp_status_uninitialized;

  if (!qnnp_params.initialized) {
    qnnp_log_error("qnnp_create_fully_connected_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = qnnp_status_unsupported_parameter;

  const float requantization_scale = input_scale * kernel_scale / output_scale;
  if (requantization_scale >= 1.0f) {
    qnnp_log_error(
      "failed to create fully connected operator with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
      "requantization scale %.7g is greater or equal to 1.0",
      input_scale, kernel_scale, output_scale, requantization_scale);
    goto error;
  }

  status = qnnp_status_out_of_memory;

  fully_connected = calloc(1, sizeof(struct qnnp_operator));
  if (fully_connected == NULL) {
    qnnp_log_error("failed to allocate %zu bytes for qnnp_operator structure", sizeof(struct qnnp_operator));
    goto error;
  }

  const uint32_t nr = qnnp_params.q8conv.nr;
  const uint32_t kr = qnnp_params.q8conv.kr;

  const uint32_t n_stride = (output_channels + (nr - 1)) & -nr;
  const uint32_t k_stride = (input_channels + (kr - 1)) & -kr;

  fully_connected->packed_kernel = malloc(sizeof(uint8_t) * k_stride * n_stride);
  if (fully_connected->packed_kernel == NULL) {
    qnnp_log_error("failed to allocate %zu bytes for packed kernel data",
      sizeof(uint8_t) * k_stride * n_stride);
    goto error;
  }
  memset(fully_connected->packed_kernel, kernel_zero_point, sizeof(uint8_t) * k_stride * n_stride);

  pack_q8gemm_b(
      output_channels, input_channels,
      nr, kr,
      kernel,
      fully_connected->packed_kernel);

  fully_connected->bias = malloc(sizeof(int32_t) * n_stride);
  if (fully_connected->bias == NULL) {
    qnnp_log_error("failed to allocate %zu bytes for packed bias data", sizeof(int32_t) * n_stride);
    goto error;
  }
  memcpy(fully_connected->bias, bias, sizeof(uint32_t) * output_channels);

  fully_connected->groups = 1;
  fully_connected->group_input_channels = input_channels;
  fully_connected->group_output_channels = output_channels;

  fully_connected->input_zero_point = input_zero_point;
  fully_connected->kernel_zero_point = kernel_zero_point;

  fully_connected->requantization_params =
    qnnp_compute_requantization_params(
      requantization_scale, output_zero_point, output_min, output_max);

  fully_connected->format = qnnp_format_quint8;
  fully_connected->flags = QNNP_CONVOLUTION_FLAG_GEMM;

  *fully_connected_out = fully_connected;
  return qnnp_status_success;

error:
  qnnp_delete_operator(fully_connected);
  return status;
}

enum qnnp_status qnnp_setup_fully_connected_nc_q8(
    qnnp_operator_t convolution,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride,
    pthreadpool_t threadpool)
{
  if (!qnnp_params.initialized) {
    qnnp_log_error("qnnp_setup_fully_connected_nc_q8 failed because QNNPACK is not properly initialized");
    return qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    qnnp_log_error("failed to setup fully connected operator with batch size %zu: batch size must be non-zero", batch_size);
    return qnnp_status_invalid_parameter;
  }

  convolution->batch_size = 1;
  convolution->input_height = batch_size;
  convolution->input_width = 1;
  convolution->input = input;
  convolution->input_pixel_stride = input_stride;

  convolution->output_height = batch_size;
  convolution->output_width = 1;
  convolution->output = output;
  convolution->output_pixel_stride = output_stride;

  return qnnp_status_success;
}
