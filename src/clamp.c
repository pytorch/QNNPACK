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


enum qnnp_status qnnp_create_clamp_nc_u8(
    size_t channels,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    qnnp_operator_t* clamp_out)
{
  qnnp_operator_t clamp_op = NULL;
  enum qnnp_status status = qnnp_status_uninitialized;

  if (!qnnp_params.initialized) {
    qnnp_log_error("qnnp_create_clamp_nc_u8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = qnnp_status_invalid_parameter;

  if (channels == 0) {
    qnnp_log_error(
      "failed to create Clamp operator with %zu channels: number of channels must be non-zero", channels);
    goto error;
  }

  if (output_min > output_max) {
    qnnp_log_error(
      "failed to create Clamp operator with [%" PRIu8 ", %" PRIu8 "] output range: range min must be below range max",
      output_min, output_max);
    goto error;
  }

  status = qnnp_status_out_of_memory;

  clamp_op = calloc(1, sizeof(struct qnnp_operator));
  if (clamp_op == NULL) {
    qnnp_log_error("failed to allocate %zu bytes for qnnp_operator structure", sizeof(struct qnnp_operator));
    goto error;
  }

  clamp_op->channels = channels;
  clamp_op->u8_clamping_params = qnnp_compute_u8_clamping_params(output_min, output_max);

  clamp_op->ukernel_type = qnnp_ukernel_type_clamp;
  clamp_op->format = qnnp_format_quint8;

  *clamp_out = clamp_op;
  return qnnp_status_success;

error:
  qnnp_delete_operator(clamp_op);
  return status;
}

enum qnnp_status qnnp_setup_clamp_nc_u8(
    qnnp_operator_t clamp,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride)
{
  if (!qnnp_params.initialized) {
    qnnp_log_error("qnnp_setup_clamp_nc_u8 failed because QNNPACK is not properly initialized");
    return qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    clamp->batch_size = 0;
    return qnnp_status_success;
  }

  clamp->batch_size = batch_size;
  clamp->input = input;
  clamp->input_pixel_stride = input_stride;
  clamp->output = output;
  clamp->output_pixel_stride = output_stride;

  return qnnp_status_success;
}
