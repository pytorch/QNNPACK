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

static inline size_t compute_output_dimension(
    size_t padded_input_dimension,
    size_t kernel_dimension,
    size_t dilation_dimension,
    size_t subsampling_dimension)
{
  const size_t effective_kernel_dimension = (kernel_dimension - 1) * dilation_dimension + 1;
  return (padded_input_dimension - effective_kernel_dimension) / subsampling_dimension + 1;
}

static void q8gemm_compute_row_sum(
    const uint8_t* restrict a,
    size_t batch_size,
    size_t groups,
    size_t m,
    size_t k,
    size_t a_stride,
    const int32_t multiplier,
    int32_t* restrict a_sum,
    size_t a_sum_stride,
    pthreadpool_t threadpool);

enum qnnp_status qnnp_create_convolution2d_nhwc_q8(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t subsampling_height,
    uint32_t subsampling_width,
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
    qnnp_operator_t* convolution_out)
{
  qnnp_operator_t convolution = NULL;
  enum qnnp_status status = qnnp_status_uninitialized;

  if (!qnnp_params.initialized) {
    qnnp_log_error("qnnp_create_convolution2d_nhwc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = qnnp_status_invalid_parameter;

  if (kernel_width == 0 || kernel_height == 0) {
    qnnp_log_error(
      "failed to create convolution with %" PRIu32 "x%" PRIu32 " kernel: kernel dimensions must be non-zero",
      kernel_width, kernel_height);
    goto error;
  }

  if (subsampling_width == 0 || subsampling_height == 0) {
    qnnp_log_error(
      "failed to create convolution with %" PRIu32 "x%" PRIu32 " subsampling: "
      "subsampling dimensions must be non-zero",
      subsampling_width, subsampling_height);
    goto error;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    qnnp_log_error(
      "failed to create convolution with %" PRIu32 "x%" PRIu32 " dilation: "
      "dilation dimensions must be non-zero",
      dilation_width, dilation_height);
    goto error;
  }

  status = qnnp_status_unsupported_parameter;

  if (subsampling_height > kernel_height) {
    qnnp_log_info(
      "inefficiency in convolution with %" PRIu32 "x%" PRIu32 " kernel and %" PRIu32 "x%" PRIu32 " subsampling: "
      "height subsampling is greater than kernel height; subsampling should be performed before the convolution",
      kernel_width, kernel_height, subsampling_width, subsampling_height);
  }

  if (subsampling_width > kernel_width) {
    qnnp_log_info(
      "inefficiency in convolution with %" PRIu32 "x%" PRIu32 " kernel and %" PRIu32 "x%" PRIu32 " subsampling: "
      "width subsampling is greater than kernel width; subsampling should be performed before the convolution",
      kernel_width, kernel_height, subsampling_width, subsampling_height);
  }

  if (input_padding_top >= kernel_height) {
    qnnp_log_info(
      "inefficiency in convolution with %" PRIu32 "x%" PRIu32 " kernel and %" PRIu32 "+%" PRIu32 " height padding: "
      "input top padding is greater or equal to kernel height",
      kernel_width, kernel_height, input_padding_top, input_padding_bottom);
  }

  if (input_padding_bottom >= kernel_height) {
    qnnp_log_info(
      "inefficiency in convolution with %" PRIu32 "x%" PRIu32 " kernel and %" PRIu32 "+%" PRIu32 " height padding: "
      "input bottom padding is greater or equal to kernel height",
      kernel_width, kernel_height, input_padding_top, input_padding_bottom);
  }

  if (input_padding_right >= kernel_width) {
    qnnp_log_info(
      "inefficiency in convolution with %" PRIu32 "x%" PRIu32 " kernel and %" PRIu32 "+%" PRIu32 " width padding: "
      "input right padding is greater or equal to kernel width",
      kernel_width, kernel_height, input_padding_left, input_padding_right);
  }

  if (input_padding_left >= kernel_width) {
    qnnp_log_info(
      "inefficiency in convolution with %" PRIu32 "x%" PRIu32 " kernel and %" PRIu32 "+%" PRIu32 " width padding: "
      "input left padding is greater or equal to kernel width",
      kernel_width, kernel_height, input_padding_left, input_padding_right);
  }

  const float convolution_scale = input_scale * kernel_scale / output_scale;
  if (convolution_scale >= 1.0f) {
    qnnp_log_error(
      "failed to create convolution with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
      "convolution scale %.7g is greater or equal to 1.0",
      input_scale, kernel_scale, output_scale, convolution_scale);
    goto error;
  }

  status = qnnp_status_out_of_memory;

  convolution = calloc(1, sizeof(struct qnnp_operator));
  if (convolution == NULL) {
    qnnp_log_error("failed to allocate %zu bytes for qnnp_operator structure", sizeof(struct qnnp_operator));
    goto error;
  }

  const size_t kernel_size = kernel_height * kernel_width;

  uint32_t flags = 0;
  if (kernel_size == 9 && dilation_width == 1 && group_input_channels == 1 && group_output_channels == 1 && groups > 1) {
    flags |= QNNP_CONVOLUTION_FLAG_DW;
  } else if (kernel_size == 1 && subsampling_height == 1 && subsampling_width == 1) {
    if (group_input_channels >= qnnp_params.q8conv_xzp.kthreshold) {
      flags |= QNNP_CONVOLUTION_FLAG_XZP_GEMM;
    } else {
      flags |= QNNP_CONVOLUTION_FLAG_GEMM;
    }
  }
  if ((input_padding_left | input_padding_top | input_padding_right | input_padding_bottom) != 0) {
    flags |= QNNP_CONVOLUTION_FLAG_ZERO;
  }

  if (flags & QNNP_CONVOLUTION_FLAG_DW) {
    const uint32_t cr = qnnp_params.q8dw9.cr;
    const uint32_t c_stride = (groups + (cr - 1)) & -cr;
    const size_t packed_weights_size = (sizeof(uint8_t) * kernel_size + sizeof(int32_t)) * c_stride;
    convolution->packed_kernel = malloc(packed_weights_size);
    if (convolution->packed_kernel == NULL) {
      qnnp_log_error("failed to allocate %zu bytes for packed kernel data", packed_weights_size);
      goto error;
    }

    pack_q8dw_w(
      kernel_height, kernel_width,
      groups, cr,
      kernel, bias, convolution->packed_kernel);

    if (flags & QNNP_CONVOLUTION_FLAG_ZERO) {
      const size_t zero_size = sizeof(uint8_t) * c_stride + (groups >= 8 ? 0 : 8);
      convolution->zero = malloc(zero_size);
      if (convolution->zero == NULL) {
        qnnp_log_error("failed to allocate %zu bytes for zero padding", zero_size);
        goto error;
      }
      memset(convolution->zero, input_zero_point, zero_size);
    }
  } else {
    uint32_t nr = qnnp_params.q8conv.nr;
    uint32_t kr = qnnp_params.q8conv.kr;

    if (flags & QNNP_CONVOLUTION_FLAG_XZP_GEMM) {
      nr = qnnp_params.q8conv_xzp.nr;
      kr = qnnp_params.q8conv_xzp.kr;
    }

    const uint32_t n_stride = (group_output_channels + (nr - 1)) & -nr;
    const uint32_t k_stride = (group_input_channels + (kr - 1)) & -kr;

    convolution->packed_kernel = malloc(sizeof(uint8_t) * kernel_size * groups * k_stride * n_stride);
    if (convolution->packed_kernel == NULL) {
      qnnp_log_error("failed to allocate %zu bytes for packed kernel data",
        sizeof(uint8_t) * kernel_size * groups * k_stride * n_stride);
      goto error;
    }
    if (flags & QNNP_CONVOLUTION_FLAG_XZP_GEMM) {
      /* The XZP ukernel needs the padding to be 0 */
      memset(convolution->packed_kernel, 0, sizeof(uint8_t) * groups * kernel_size * k_stride * n_stride);
    } else {
      memset(convolution->packed_kernel, kernel_zero_point, sizeof(uint8_t) * groups * kernel_size * k_stride * n_stride);
    }

    if (flags & QNNP_CONVOLUTION_FLAG_GEMM) {
      for (uint32_t group = 0; group < groups; group++) {
        pack_q8gemm_b(
            group_output_channels, group_input_channels,
            nr, kr,
            kernel + group * group_output_channels * group_input_channels,
            convolution->packed_kernel + group * n_stride * k_stride);
      }
    } else if (flags & QNNP_CONVOLUTION_FLAG_XZP_GEMM) {
      for (uint32_t group = 0; group < groups; group++) {
        const uint32_t kc = qnnp_params.q8conv_xzp.kc;
        pack_q8gemm_b_diagonal(
            group_output_channels, group_input_channels,
            nr, kr, kc,
            kernel + group * group_output_channels * group_input_channels,
            convolution->packed_kernel + group * n_stride * k_stride);
      }
    } else {
      for (uint32_t group = 0; group < groups; group++) {
        pack_q8conv_b(
            group_output_channels, kernel_size, group_input_channels,
            nr, kr,
            kernel + group * group_output_channels * kernel_size * group_input_channels,
            convolution->packed_kernel + group * kernel_size * n_stride * k_stride);
      }
    }

    convolution->bias = malloc(sizeof(int32_t) * groups * n_stride);
    if (convolution->bias == NULL) {
      qnnp_log_error("failed to allocate %zu bytes for packed bias data", sizeof(int32_t) * groups * n_stride);
      goto error;
    }
    for (uint32_t group = 0; group < groups; group++) {
      memcpy(
        (int32_t*) convolution->bias + group * n_stride,
        bias + group * group_output_channels,
        sizeof(int32_t) * group_output_channels);
    }

    if (flags & QNNP_CONVOLUTION_FLAG_XZP_GEMM) {
      /* compute row sum for b and fold into bias */
      int32_t* bias_buf = (int32_t*) malloc(sizeof(int32_t) * groups * n_stride);
      const int32_t zero_point_product = group_input_channels * input_zero_point * kernel_zero_point;
      int32_t* bias = (int32_t*) convolution->bias;
      /* kernel: G x OC x kH x kW x IC */
      /* row_sum:G x OC */
      /* swap groups and batch_size */
      q8gemm_compute_row_sum(
        kernel,
        groups,
        1,
        group_output_channels,
        group_input_channels,
        group_input_channels,
        -input_zero_point,
        bias_buf,
        n_stride,
        NULL);
      for (uint32_t group = 0; group < groups; group++) {
        for (uint32_t i = 0; i < group_output_channels; i++) {
          bias[group * n_stride + i] += bias_buf[group * n_stride + i] + zero_point_product;
        }
      }
      free(bias_buf);
    }

    if (flags & QNNP_CONVOLUTION_FLAG_ZERO) {
      const size_t zero_size = sizeof(uint8_t) * k_stride + (group_input_channels >= 8 ? 0 : 8);
      convolution->zero = malloc(zero_size);
      if (convolution->zero == NULL) {
        qnnp_log_error("failed to allocate %zu bytes for zero padding", zero_size);
        goto error;
      }
      memset(convolution->zero, input_zero_point, zero_size);
    }
  }

  convolution->input_padding_top = input_padding_top;
  convolution->input_padding_right = input_padding_right;
  convolution->input_padding_bottom = input_padding_bottom;
  convolution->input_padding_left = input_padding_left;

  convolution->kernel_height = kernel_height;
  convolution->kernel_width = kernel_width;
  convolution->stride_height = subsampling_height;
  convolution->stride_width = subsampling_width;
  convolution->dilation_height = dilation_height;
  convolution->dilation_width = dilation_width;
  convolution->groups = groups;
  convolution->group_input_channels = group_input_channels;
  convolution->group_output_channels = group_output_channels;

  convolution->input_zero_point = input_zero_point;
  convolution->kernel_zero_point = kernel_zero_point;

  convolution->requantization_params =
    qnnp_compute_requantization_params(
      convolution_scale, output_zero_point, output_min, output_max);

  convolution->format = qnnp_format_quint8;
  convolution->flags = flags;

  *convolution_out = convolution;
  return qnnp_status_success;

error:
  qnnp_delete_operator(convolution);
  return status;
}

enum qnnp_status qnnp_setup_convolution2d_nhwc_q8(
    qnnp_operator_t convolution,
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
    qnnp_log_error("qnnp_setup_convolution2d_nhwc_q8 failed because QNNPACK is not properly initialized");
    return qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    qnnp_log_error("failed to setup convolution with batch size %zu: batch size must be non-zero", batch_size);
    return qnnp_status_invalid_parameter;
  }

  if (input_width == 0 || input_height == 0) {
    qnnp_log_error(
      "failed to setup convolution with %zux%zu input: input dimensions must be non-zero",
      input_width,
      input_height);
    return qnnp_status_invalid_parameter;
  }

  convolution->batch_size = batch_size;
  convolution->input_height = input_height;
  convolution->input_width = input_width;
  convolution->input = input;
  convolution->input_pixel_stride = input_pixel_stride;

  convolution->output_height = compute_output_dimension(
      convolution->input_padding_top + input_height + convolution->input_padding_bottom,
      convolution->kernel_height,
      convolution->dilation_height,
      convolution->stride_height);
  convolution->output_width = compute_output_dimension(
      convolution->input_padding_left + input_width + convolution->input_padding_right,
      convolution->kernel_width,
      convolution->dilation_width,
      convolution->stride_width);
  convolution->output = output;
  convolution->output_pixel_stride = output_pixel_stride;

  if (convolution->flags & QNNP_CONVOLUTION_FLAG_GEMM) {
    /* Convolution maps directly to GEMM and doesn't use im2col buffer */
    return qnnp_status_success;
  }

  const size_t groups = convolution->groups;

  if (convolution->flags & QNNP_CONVOLUTION_FLAG_XZP_GEMM) {
    void* a_sum = (void*) realloc(convolution->a_sum, sizeof(int32_t) * batch_size * groups * input_height * input_width);
    if (a_sum == NULL) {
      qnnp_log_error("failed to allocate %zu bytes for a row sum data",
        sizeof(int32_t) * batch_size * groups * input_height * input_width);
      return qnnp_status_out_of_memory;
    }
    convolution->a_sum = a_sum;
    return qnnp_status_success;
  }

  const size_t kernel_height = convolution->kernel_height;
  const size_t kernel_width = convolution->kernel_width;
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t output_height = convolution->output_height;
  const size_t output_width = convolution->output_width;
  if (convolution->flags & QNNP_CONVOLUTION_FLAG_DW) {
    const size_t subsampling_width = convolution->stride_width;
    const size_t im2col_buffer_size = sizeof(void*) * batch_size * output_height *
      (kernel_size + (output_width * subsampling_width - 1) * kernel_height);

    const void** im2col_buffer = (const void**) realloc(convolution->im2col_buffer, im2col_buffer_size);
    if (im2col_buffer == NULL) {
      qnnp_log_error("failed to allocate %zu bytes for im2col buffer", im2col_buffer_size);
      return qnnp_status_out_of_memory;
    }
    convolution->im2col_buffer = im2col_buffer;

    const void* zero = convolution->zero;
    if (groups < 8) {
      zero = (const void*) ((uintptr_t) zero + 8);
    }

    for (size_t group = 0; group < groups; group++) {
      for (size_t image = 0; image < batch_size; image++) {
        for (size_t output_y = 0; output_y < output_height; output_y++) {
          for (size_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
            const size_t input_y =
              output_y * convolution->stride_height + kernel_y * convolution->dilation_height - convolution->input_padding_top;
            if (input_y < input_height) {
              for (size_t output_x = 0; output_x < output_width; output_x++) {
                for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                  const size_t input_x =
                    output_x * subsampling_width + kernel_x * convolution->dilation_width - convolution->input_padding_left;
                  const size_t im2col_index =
                    (image * output_height + output_y) * (kernel_size + (output_width * subsampling_width - 1) * kernel_height) +
                    output_x * subsampling_width * kernel_height + kernel_x * kernel_height + kernel_y;
                  if (input_x < input_width) {
                    im2col_buffer[im2col_index] = input + ((image * input_height + input_y) * input_width + input_x) * input_pixel_stride;
                  } else {
                    im2col_buffer[im2col_index] = zero;
                  }
                }
              }
            } else {
              for (size_t output_x = 0; output_x < output_width; output_x++) {
                for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                  const size_t im2col_index =
                    (image * output_height + output_y) * (kernel_size + (output_width * subsampling_width - 1) * kernel_height) +
                    output_x * subsampling_width * kernel_height + kernel_x * kernel_height + kernel_y;
                  im2col_buffer[im2col_index] = zero;
                }
              }
            }
          }
        }
      }
    }
  } else {
    const size_t output_size = output_height * output_width;
    const size_t output_tile_size = qnnp_params.q8conv.mr;
    const size_t tiled_output_size = round_up(output_size, output_tile_size);
    const size_t im2col_buffer_size = sizeof(void*) * batch_size * groups * tiled_output_size * kernel_size;

    const void** im2col_buffer = (const void**) realloc(convolution->im2col_buffer, im2col_buffer_size);
    if (im2col_buffer == NULL) {
      qnnp_log_error("failed to allocate %zu bytes for im2col buffer", im2col_buffer_size);
      return qnnp_status_out_of_memory;
    }
    convolution->im2col_buffer = im2col_buffer;

    const void* zero = convolution->zero;
    if (convolution->group_input_channels < 8) {
      zero = (const void*) ((uintptr_t) zero + 8);
    }

    const struct fxdiv_divisor_size_t output_width_divisor = fxdiv_init_size_t(output_width);
    for (size_t group = 0; group < groups; group++) {
      for (size_t image = 0; image < batch_size; image++) {
        for (size_t output_tile_start = 0; output_tile_start < tiled_output_size; output_tile_start += output_tile_size) {
          for (size_t output_tile_offset = 0; output_tile_offset < output_tile_size; output_tile_offset++) {
            const size_t tiled_output_index = output_tile_start + output_tile_offset;
            const size_t output_index = min(tiled_output_index, output_size - 1);
            const struct fxdiv_result_size_t output_index_components =
              fxdiv_divide_size_t(output_index, output_width_divisor);
            const size_t output_y = output_index_components.quotient;
            const size_t output_x = output_index_components.remainder;
            for (size_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
              const size_t input_y =
                output_y * convolution->stride_height + kernel_y * convolution->dilation_height - convolution->input_padding_top;
              if (input_y < input_height) {
                for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                  const size_t input_x =
                    output_x * convolution->stride_width + kernel_x * convolution->dilation_width - convolution->input_padding_left;
                  const size_t im2col_index =
                    (group * batch_size + image) * tiled_output_size * kernel_size + output_tile_start * kernel_size + (kernel_y * kernel_width + kernel_x) * output_tile_size + output_tile_offset;
                  if (input_x < input_width) {
                    im2col_buffer[im2col_index] =
                      input + ((image * input_height + input_y) * input_width + input_x) * input_pixel_stride + group * convolution->group_input_channels;
                  } else {
                    im2col_buffer[im2col_index] = zero;
                  }
                }
              } else {
                for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                  const size_t im2col_index =
                    (group * batch_size + image) * tiled_output_size * kernel_size + output_tile_start * kernel_size + (kernel_y * kernel_width + kernel_x) * output_tile_size + output_tile_offset;
                  im2col_buffer[im2col_index] = zero;
                }
              }
            }
          }
        }
      }
    }
  }

  return qnnp_status_success;
}

struct q8gemm_context {
  size_t k;
  size_t k_stride;
  size_t n;
  size_t n_stride;
  const uint8_t* a;
  size_t a_stride;
  const uint8_t* packed_b;
  const int32_t* bias;
  uint8_t* c;
  size_t c_stride;
  uint8_t a_zero_point;
  uint8_t b_zero_point;
  union qnnp_q31_requantization_params requantization_params;
  const q8gemm_ukernel_function ukernel;
};

static void compute_q8gemm(
    const struct q8gemm_context context[restrict static 1],
    size_t group_index,
    size_t pixel_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t group_range /* always 1 */,
    size_t pixel_range,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t k = context->k;
  const size_t k_stride = context->k_stride;
  const size_t n = context->n;
  const size_t n_stride = context->n_stride;
  const uint8_t* restrict a = context->a;
  const size_t a_stride = context->a_stride;
  const uint8_t* restrict packed_b = context->packed_b;
  const int32_t* restrict bias = context->bias;
  uint8_t* restrict c = context->c;
  const size_t c_stride = context->c_stride;
  const uint8_t a_zero_point = context->a_zero_point;
  const uint8_t b_zero_point = context->b_zero_point;

  context->ukernel(
      mr_block_size,
      nr_block_size,
      k,
      a + (pixel_index + mr_block_start) * a_stride + group_index * k,
      a_stride,
      packed_b + nr_block_start * k_stride + group_index * k_stride * n_stride,
      bias + nr_block_start + group_index * n_stride,
      c + (pixel_index + mr_block_start) * c_stride + nr_block_start + group_index * n,
      c_stride,
      a_zero_point,
      b_zero_point,
      &context->requantization_params);
}

struct q8sum_rows_context {
  const uint8_t* a;
  size_t groups;
  size_t m;
  size_t k;
  size_t a_stride;
  const int32_t multiplier;
  int32_t* a_sum;
  size_t a_sum_stride;
  const q8sum_rows_ukernel_function ukernel;
};

static void compute_sum_rows(
    const struct q8sum_rows_context context[restrict static 1],
    size_t group_index,
    size_t batch_index,
    size_t block_start,
    size_t group_range /* always 1 */,
    size_t batch_range /* always 1 */,
    size_t block_size)
{
  const uint8_t* a = context->a;
  const size_t groups = context->groups;
  const size_t m = context->m;
  const size_t k = context->k;
  const size_t a_stride = context->a_stride;
  const int32_t multiplier = context->multiplier;
  int32_t* a_sum = context->a_sum;
  const size_t a_sum_stride = context->a_sum_stride;

  context->ukernel(
      a + batch_index * m * a_stride + group_index * k + block_start * a_stride,
      min(block_size, m - block_start),
      k,
      a_stride,
      multiplier,
      a_sum + batch_index * groups * a_sum_stride + group_index * a_sum_stride + block_start);
}

/*   a: batch_size * m * groups * k   */
/*   a_sum: batch_size * groups * m   */
static void q8gemm_compute_row_sum(
    const uint8_t* restrict a,
    size_t batch_size,
    size_t groups,
    size_t m,
    size_t k,
    size_t a_stride,
    const int32_t multiplier,
    int32_t* restrict a_sum,
    size_t a_sum_stride,
    pthreadpool_t threadpool)
{
  struct q8sum_rows_context context = {
      .a = a,
      .groups = groups,
      .m = m,
      .k = k,
      .a_stride = a_stride,
      .multiplier = multiplier,
      .a_sum = a_sum,
      .a_sum_stride = a_sum_stride,
      .ukernel = qnnp_params.q8sum_rows.sum_rows,
  };

  pthreadpool_compute_3d_tiled(
    threadpool,
    (pthreadpool_function_3d_tiled_t) compute_sum_rows,
    &context,
    groups, batch_size, m,
    1, 1, qnnp_params.q8sum_rows.m);
}

struct q8gemm_xzp_context {
  size_t k;
  size_t k_stride;
  size_t n;
  size_t n_stride;
  const uint8_t* a;
  size_t a_stride;
  const uint8_t* packed_b;
  const int32_t* bias;
  uint8_t* c;
  size_t c_stride;
  const int32_t* a_sum;
  size_t groups;
  size_t batch_size;
  size_t a_sum_stride;
  union qnnp_q31_requantization_params requantization_params;
  const q8gemm_xzp_ukernel_function ukernel;
};

static void compute_q8gemm_xzp(
    const struct q8gemm_xzp_context context[restrict static 1],
    size_t group_index,
    size_t pixel_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t group_range /* always 1 */,
    size_t pixel_range,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t k = context->k;
  const size_t k_stride = context->k_stride;
  const size_t n = context->n;
  const size_t n_stride = context->n_stride;
  const uint8_t* restrict a = context->a;
  const size_t a_stride = context->a_stride;
  const uint8_t* restrict packed_b = context->packed_b;
  const int32_t* restrict bias = context->bias;
  uint8_t* restrict c = context->c;
  const size_t c_stride = context->c_stride;
  const int32_t* a_sum = context->a_sum;
  const size_t groups = context->groups;
  const size_t batch_size = context->batch_size;
  const size_t a_sum_stride = context->a_sum_stride;

  context->ukernel(
      mr_block_size,
      nr_block_size,
      k,
      a + (pixel_index + mr_block_start) * a_stride + group_index * k,
      a_stride,
      packed_b + nr_block_start * k_stride + group_index * k_stride * n_stride,
      bias + nr_block_start + group_index * n_stride,
      c + (pixel_index + mr_block_start) * c_stride + nr_block_start + group_index * n,
      c_stride,
      a_sum + pixel_index * groups + group_index * a_sum_stride + mr_block_start,
      &context->requantization_params);
}

struct q8conv_context {
  size_t bs;
  size_t ks;
  size_t kc;
  size_t kc_stride;
  size_t m;
  size_t m_stride;
  size_t n;
  size_t n_stride;
  const uint8_t** im2col_a;
  const uint8_t* packed_b;
  const int32_t* bias;
  uint8_t* c;
  size_t c_stride;
  uint8_t a_zero_point;
  uint8_t b_zero_point;
  union qnnp_q31_requantization_params requantization_params;
  const q8conv_ukernel_function ukernel;
};

static void compute_q8conv(
    const struct q8conv_context context[restrict static 1],
    size_t group_index,
    size_t image_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t group_range /* always 1 */,
    size_t image_range /* always 1 */,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t bs = context->bs;
  const size_t ks = context->ks;
  const size_t kc = context->kc;
  const size_t kc_stride = context->kc_stride;
  const size_t m = context->m;
  const size_t m_stride = context->m_stride;
  const size_t n = context->n;
  const size_t n_stride = context->n_stride;
  const uint8_t** restrict im2col_a = context->im2col_a;
  const uint8_t* restrict packed_b = context->packed_b;
  const int32_t* restrict bias = context->bias;
  uint8_t* restrict c = context->c;
  const size_t c_stride = context->c_stride;
  const uint8_t a_zero_point = context->a_zero_point;
  const uint8_t b_zero_point = context->b_zero_point;

  context->ukernel(
      mr_block_size,
      nr_block_size,
      kc,
      ks,
      im2col_a + (mr_block_start + (image_index + group_index * bs) * m_stride) * ks,
      packed_b + (nr_block_start + group_index * n_stride) * kc_stride,
      bias + nr_block_start + group_index * n_stride,
      c + (mr_block_start + image_index * m) * c_stride + group_index * n + nr_block_start,
      c_stride,
      a_zero_point,
      b_zero_point,
      &context->requantization_params);
}

struct q8dw_context {
  size_t groups;
  const uint8_t** im2col_buffer;
  size_t im2col_row_stride;
  size_t im2col_col_stride;
  const uint8_t* packed_kernel;
  const int32_t* bias;
  uint8_t* output;
  size_t output_height;
  size_t output_width;
  size_t output_row_stride;
  size_t output_col_increment;
  uint8_t input_zero_point;
  uint8_t kernel_zero_point;
  union qnnp_q31_requantization_params requantization_params;
  const q8dw_ukernel_function ukernel;
};

static void compute_q8dw(
    const struct q8dw_context context[restrict static 1],
    size_t image,
    size_t output_y)
{
  const size_t output_height = context->output_height;

  context->ukernel(
    context->groups,
    context->output_width,
    context->im2col_buffer + (image * output_height + output_y) * context->im2col_row_stride,
    context->packed_kernel,
    context->output + (image * output_height + output_y) * context->output_row_stride,
    context->im2col_col_stride,
    context->output_col_increment,
    context->input_zero_point,
    context->kernel_zero_point,
    &context->requantization_params);
}

enum qnnp_status qnnp_run_operator(qnnp_operator_t op, pthreadpool_t threadpool)
{
  if (op->flags & QNNP_CONVOLUTION_FLAG_DW) {
    const size_t batch_size = op->batch_size;
    const size_t groups = op->groups;
    const size_t kernel_height = op->kernel_height;
    const size_t kernel_width = op->kernel_width;
    const size_t kernel_size = kernel_height * kernel_width;
    const size_t subsampling_width = op->stride_width;
    const size_t output_height = op->output_height;
    const size_t output_width = op->output_width;

    struct q8dw_context q8dw_context = {
        .groups = groups,
        .im2col_buffer = (const uint8_t**) op->im2col_buffer,
        .im2col_row_stride = kernel_size + (output_width * subsampling_width - 1) * kernel_height,
        .im2col_col_stride = kernel_height * subsampling_width * sizeof(void*),
        .packed_kernel = op->packed_kernel,
        .bias = op->bias,
        .output = op->output,
        .output_height = output_height,
        .output_width = output_width,
        .output_row_stride = output_width * op->output_pixel_stride,
        .output_col_increment = (op->output_pixel_stride - groups) * sizeof(uint8_t),
        .input_zero_point = op->input_zero_point,
        .kernel_zero_point = op->kernel_zero_point,
        .requantization_params = op->requantization_params,
        .ukernel = qnnp_params.q8dw9.dw,
    };
    pthreadpool_compute_2d(
        threadpool,
        (pthreadpool_function_2d_t) compute_q8dw,
        &q8dw_context,
        batch_size, output_height);
  } else if (op->flags & QNNP_CONVOLUTION_FLAG_XZP_GEMM) {
    const size_t batch_size = op->batch_size;
    const size_t groups = op->groups;
    const size_t group_input_channels = op->group_input_channels;
    const size_t group_output_channels = op->group_output_channels;
    const uint32_t mr = qnnp_params.q8conv_xzp.mr;
    const uint32_t nr = qnnp_params.q8conv_xzp.nr;
    const uint32_t kr = qnnp_params.q8conv_xzp.kr;
    const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
    const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;

    /* compute input row sum */
    const size_t input_size = op->input_height * op->input_width;
    int32_t* a_sum = (int32_t*) op->a_sum;
    q8gemm_compute_row_sum(
        op->input,
        batch_size,
        groups,
        input_size, /* m */
        group_input_channels, /* k */
        op->input_pixel_stride, /* input stride */
        -op->kernel_zero_point, /* multiplier */
        a_sum,
        input_size,
        threadpool);
    struct q8gemm_xzp_context q8gemm_xzp_context = {
        .k = group_input_channels,
        .k_stride = k_stride,
        .n = group_output_channels,
        .n_stride = n_stride,
        .a = op->input,
        .a_stride = op->input_pixel_stride,
        .packed_b = op->packed_kernel,
        .bias = op->bias,
        .c = op->output,
        .c_stride = op->output_pixel_stride,
        .a_sum = a_sum,
        .groups = op->groups,
        .batch_size = batch_size,
        .a_sum_stride = input_size,
        .requantization_params = op->requantization_params,
        .ukernel = qnnp_params.q8conv_xzp.gemm,
    };
    pthreadpool_compute_4d_tiled(
        threadpool,
        (pthreadpool_function_4d_tiled_t) compute_q8gemm_xzp,
        &q8gemm_xzp_context,
        groups, batch_size * input_size, input_size, group_output_channels,
        1, input_size, mr, nr);
  } else {
    const size_t batch_size = op->batch_size;
    const size_t groups = op->groups;
    const size_t group_input_channels = op->group_input_channels;
    const size_t group_output_channels = op->group_output_channels;
    const uint32_t mr = qnnp_params.q8conv.mr;
    const uint32_t nr = qnnp_params.q8conv.nr;
    const uint32_t kr = qnnp_params.q8conv.kr;
    const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
    const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;

    const size_t output_size = op->output_height * op->output_width;
    if (op->flags & QNNP_CONVOLUTION_FLAG_GEMM) {
      struct q8gemm_context q8gemm_context = {
          .k = group_input_channels,
          .k_stride = k_stride,
          .n = group_output_channels,
          .n_stride = n_stride,
          .a = op->input,
          .a_stride = op->input_pixel_stride,
          .packed_b = op->packed_kernel,
          .bias = op->bias,
          .c = op->output,
          .c_stride = op->output_pixel_stride,
          .a_zero_point = op->input_zero_point,
          .b_zero_point = op->kernel_zero_point,
          .requantization_params = op->requantization_params,
          .ukernel = qnnp_params.q8conv.gemm,
      };

      pthreadpool_compute_4d_tiled(
          threadpool,
          (pthreadpool_function_4d_tiled_t) compute_q8gemm,
          &q8gemm_context,
          groups, batch_size * output_size, output_size, group_output_channels,
          1, output_size, mr, nr);
    } else {
      const size_t kernel_size = op->kernel_height * op->kernel_width;
      const size_t m_stride = round_up(output_size, mr);
      struct q8conv_context q8conv_context = {
          .bs = batch_size,
          .ks = kernel_size,
          .kc = group_input_channels,
          .kc_stride = k_stride * kernel_size,
          .m = output_size,
          .m_stride = m_stride,
          .n = group_output_channels,
          .n_stride = n_stride,
          .im2col_a = (const uint8_t**) op->im2col_buffer,
          .packed_b = op->packed_kernel,
          .bias = (const int32_t*) op->bias,
          .c = op->output,
          .c_stride = op->output_pixel_stride,
          .a_zero_point = op->input_zero_point,
          .b_zero_point = op->kernel_zero_point,
          .requantization_params = op->requantization_params,
          .ukernel = qnnp_params.q8conv.conv,
      };

      pthreadpool_compute_4d_tiled(
          threadpool,
          (pthreadpool_function_4d_tiled_t) compute_q8conv,
          &q8conv_context,
          groups, batch_size, output_size, group_output_channels,
          1, 1, mr, nr);
    }
  }
  return qnnp_status_success;
}

enum qnnp_status qnnp_delete_operator(qnnp_operator_t op)
{
  if (op != NULL) {
    free(op->im2col_buffer);
    free(op->packed_kernel);
    free(op->a_sum);
    free(op->bias);
    free(op->zero);
    free(op);
    return qnnp_status_success;
  }
  return qnnp_status_invalid_parameter;
}
