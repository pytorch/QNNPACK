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
#include <qnnpack/params.h>


enum qnnp_status qnnp_create_channel_shuffle_nc_x8(
    size_t groups,
    size_t group_channels,
    uint32_t flags,
    qnnp_operator_t* channel_shuffle_out)
{
  qnnp_operator_t channel_shuffle_op = NULL;
  enum qnnp_status status = qnnp_status_uninitialized;

  if (!qnnp_params.initialized) {
    qnnp_log_error("qnnp_create_channel_shuffle_nc_x8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = qnnp_status_invalid_parameter;

  if (groups <= 1) {
    qnnp_log_error(
      "failed to create channel shuffle operator with %zu groups: "
      "at least two groups required", groups);
    goto error;
  }

  if (group_channels == 0) {
    qnnp_log_error(
      "failed to create channel shuffle operator with %zu group channels: "
      "number of group channels must be non-zero", group_channels);
    goto error;
  }

  status = qnnp_status_out_of_memory;

  channel_shuffle_op = calloc(1, sizeof(struct qnnp_operator));
  if (channel_shuffle_op == NULL) {
    qnnp_log_error("failed to allocate %zu bytes for qnnp_operator structure", sizeof(struct qnnp_operator));
    goto error;
  }

  channel_shuffle_op->groups = groups;
  channel_shuffle_op->group_channels = group_channels;

  channel_shuffle_op->ukernel_type = qnnp_ukernel_type_channel_shuffle;
  channel_shuffle_op->format = qnnp_format_quint8;

  *channel_shuffle_out = channel_shuffle_op;
  return qnnp_status_success;

error:
  qnnp_delete_operator(channel_shuffle_op);
  return status;
}

enum qnnp_status qnnp_setup_channel_shuffle_nc_x8(
    qnnp_operator_t channel_shuffle_op,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride)
{
  if (!qnnp_params.initialized) {
    qnnp_log_error("qnnp_setup_channel_shuffle_nc_x8 failed because QNNPACK is not properly initialized");
    return qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    channel_shuffle_op->batch_size = 0;
    return qnnp_status_success;
  }

  channel_shuffle_op->batch_size = batch_size;
  channel_shuffle_op->input = input;
  channel_shuffle_op->input_pixel_stride = input_stride;
  channel_shuffle_op->output = output;
  channel_shuffle_op->output_pixel_stride = output_stride;

  return qnnp_status_success;
}
