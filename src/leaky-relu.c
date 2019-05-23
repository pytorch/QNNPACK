/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <qnnpack.h>
#include <qnnpack/operator.h>
#include <qnnpack/log.h>


enum qnnp_status qnnp_create_leaky_relu_nc_q8(
    size_t channels,
    float negative_slope,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    qnnp_operator_t* leaky_relu_out)
{
  qnnp_operator_t leaky_relu_op = NULL;
  enum qnnp_status status = qnnp_status_uninitialized;

  if (!qnnp_params.initialized) {
    qnnp_log_error("qnnp_create_leaky_relu_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = qnnp_status_invalid_parameter;

  if (channels == 0) {
    qnnp_log_error(
      "failed to create Leaky ReLU operator with %zu channels: number of channels must be non-zero", channels);
    goto error;
  }

  if (negative_slope <= 0.0f || !isnormal(negative_slope)) {
    qnnp_log_error(
      "failed to create Leaky ReLU operator with %.7g negative slope: slope must be finite and positive", negative_slope);
    goto error;
  }

  if (negative_slope > 1.0f) {
    qnnp_log_error(
      "failed to create Leaky ReLU operator with %.7g negative slope: slope must not exceed 1.0", negative_slope);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    qnnp_log_error(
      "failed to create Leaky ReLU operator with %.7g input scale: scale must be finite and positive", input_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    qnnp_log_error(
      "failed to create Leaky ReLU operator with %.7g output scale: scale must be finite and positive", output_scale);
    goto error;
  }

  if (output_min >= output_max) {
    qnnp_log_error(
      "failed to create Leaky ReLU operator with [%" PRIu8 ", %" PRIu8 "] output range: range min must be below range max",
      output_min, output_max);
    goto error;
  }

  status = qnnp_status_unsupported_parameter;

  const float input_output_scale = input_scale / output_scale;
  if (input_output_scale < 0x1.0p-8f || input_output_scale >= 0x1.0p+8f) {
    qnnp_log_error(
      "failed to create Leaky ReLU operator with %.7g input-to-output scale ratio: "
      "scale ratio must be in [2**-8, 2**8) range",
      input_output_scale);
    goto error;
  }

  status = qnnp_status_out_of_memory;

  leaky_relu_op = calloc(1, sizeof(struct qnnp_operator));
  if (leaky_relu_op == NULL) {
    qnnp_log_error("failed to allocate %zu bytes for qnnp_operator structure", sizeof(struct qnnp_operator));
    goto error;
  }

  leaky_relu_op->lookup_table = malloc(256 * sizeof(uint8_t));
  if (leaky_relu_op->lookup_table == NULL) {
    qnnp_log_error("failed to allocate 256 bytes for Leaky ReLU lookup table");
    goto error;
  }

  uint8_t* lookup_table = leaky_relu_op->lookup_table;
  const float scaled_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const float scaled_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (int32_t i = 0; i < 256; i++) {
    const float x = input_output_scale * (float) (i - (int32_t) (uint32_t) input_zero_point);
    float y = x < 0.0f ? x * negative_slope : x;
    if (y < scaled_min_less_zero_point) {
      y = scaled_min_less_zero_point;
    }
    if (y > scaled_max_less_zero_point) {
      y = scaled_max_less_zero_point;
    }
    lookup_table[(uint32_t) i] = (uint8_t) (lrintf(y) + (long) output_zero_point);
  }

  leaky_relu_op->channels = channels;

  leaky_relu_op->ukernel_type = qnnp_ukernel_type_lut;
  leaky_relu_op->format = qnnp_format_quint8;

  *leaky_relu_out = leaky_relu_op;
  return qnnp_status_success;

error:
  qnnp_delete_operator(leaky_relu_op);
  return status;
}

enum qnnp_status qnnp_setup_leaky_relu_nc_q8(
    qnnp_operator_t leaky_relu,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride)
{
  if (!qnnp_params.initialized) {
    qnnp_log_error("qnnp_setup_leaky_relu_nc_q8 failed because QNNPACK is not properly initialized");
    return qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    leaky_relu->batch_size = 0;
    return qnnp_status_success;
  }

  leaky_relu->batch_size = batch_size;
  leaky_relu->input = input;
  leaky_relu->input_pixel_stride = input_stride;
  leaky_relu->output = output;
  leaky_relu->output_pixel_stride = output_stride;

  return qnnp_status_success;
}
