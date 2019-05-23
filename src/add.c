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
#include <qnnpack/requantization.h>
#include <qnnpack/log.h>
#include <qnnpack/params.h>


enum qnnp_status qnnp_create_add_nc_q8(
    size_t channels,
    uint8_t a_zero_point,
    float a_scale,
    uint8_t b_zero_point,
    float b_scale,
    uint8_t sum_zero_point,
    float sum_scale,
    uint8_t sum_min,
    uint8_t sum_max,
    uint32_t flags,
    qnnp_operator_t* add_out)
{
  qnnp_operator_t add_op = NULL;
  enum qnnp_status status = qnnp_status_uninitialized;

  if (!qnnp_params.initialized) {
    qnnp_log_error("qnnp_create_add_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = qnnp_status_invalid_parameter;

  if (channels == 0) {
    qnnp_log_error(
      "failed to create add operator with %zu channels: number of channels must be non-zero", channels);
    goto error;
  }

  if (a_scale <= 0.0f || !isnormal(a_scale)) {
    qnnp_log_error(
      "failed to create add operator with %.7g A scale: scale must be finite and positive", a_scale);
    goto error;
  }

  if (b_scale <= 0.0f || !isnormal(b_scale)) {
    qnnp_log_error(
      "failed to create add operator with %.7g B scale: scale must be finite and positive", b_scale);
    goto error;
  }

  if (sum_scale <= 0.0f || !isnormal(sum_scale)) {
    qnnp_log_error(
      "failed to create add operator with %.7g output scale: scale must be finite and positive", sum_scale);
    goto error;
  }

  if (sum_min >= sum_max) {
    qnnp_log_error(
      "failed to create add operator with [%" PRIu8 ", %" PRIu8 "] output range: range min must be below range max",
      sum_min, sum_max);
    goto error;
  }

  status = qnnp_status_unsupported_parameter;

  const float a_output_scale = a_scale / sum_scale;
  if (a_output_scale < 0x1.0p-14f || a_output_scale >= 0x1.0p+8f) {
    qnnp_log_error(
      "failed to create add operator with %.7g A-to-output scale ratio: scale ratio must be in [2**-14, 2**8) range",
      a_output_scale);
    goto error;
  }

  const float b_output_scale = b_scale / sum_scale;
  if (b_output_scale < 0x1.0p-14f || b_output_scale >= 0x1.0p+8f) {
    qnnp_log_error(
      "failed to create add operator with %.7g A-to-output scale ratio: scale ratio must be in [2**-14, 2**8) range",
      b_output_scale);
    goto error;
  }

  status = qnnp_status_out_of_memory;

  add_op = calloc(1, sizeof(struct qnnp_operator));
  if (add_op == NULL) {
    qnnp_log_error("failed to allocate %zu bytes for qnnp_operator structure", sizeof(struct qnnp_operator));
    goto error;
  }

  add_op->channels = channels;
  add_op->add_quantization_params =
    qnnp_compute_add_quantization_params(
      a_zero_point, b_zero_point, sum_zero_point,
      a_scale / sum_scale, b_scale / sum_scale,
      sum_min, sum_max);

  add_op->ukernel_type = qnnp_ukernel_type_add;
  add_op->format = qnnp_format_quint8;

  *add_out = add_op;
  return qnnp_status_success;

error:
  qnnp_delete_operator(add_op);
  return status;
}

enum qnnp_status qnnp_setup_add_nc_q8(
    qnnp_operator_t add_op,
    size_t batch_size,
    const uint8_t* a,
    size_t a_stride,
    const uint8_t* b,
    size_t b_stride,
    uint8_t* sum,
    size_t sum_stride)
{
  if (!qnnp_params.initialized) {
    qnnp_log_error("qnnp_setup_add_nc_q8 failed because QNNPACK is not properly initialized");
    return qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    add_op->batch_size = 0;
    return qnnp_status_success;
  }

  add_op->batch_size = batch_size;
  add_op->input = a;
  add_op->input_pixel_stride = a_stride;
  add_op->input2 = b;
  add_op->input2_pixel_stride = b_stride;
  add_op->output = sum;
  add_op->output_pixel_stride = sum_stride;

  return qnnp_status_success;
}
