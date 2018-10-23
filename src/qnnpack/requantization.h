/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

#include <fp16/bitcasts.h>

#include <qnnpack/params.h>
#include <qnnpack/scalar-utils.h>


static inline union qnnp_q31_requantization_params qnnp_compute_scalar_requantization_params(
  float scale,
  uint8_t zero_point,
  uint8_t min,
  uint8_t max)
{
  /* Compute requantization parameters */
  const uint32_t scale_bits = fp32_to_bits(scale);

  /* Multiplier is in [0x40000000, 0x7FFFFF80] range */
  const int32_t multiplier = (int32_t)(((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  /* Shift is in [0, 31] range */
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  union qnnp_q31_requantization_params params;
  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  const uint32_t remainder_threshold = remainder_mask >> 1;
  params.scalar.multiplier = multiplier;
  params.scalar.remainder_mask = (int32_t) remainder_mask;
  params.scalar.remainder_threshold = (int32_t) remainder_threshold;
  params.scalar.shift = (uint32_t) shift;
  params.scalar.min_less_zero_point = (int32_t) (uint32_t) min - (int32_t) (uint32_t) zero_point;
  params.scalar.max_less_zero_point = (int32_t) (uint32_t) max - (int32_t) (uint32_t) zero_point;
  params.scalar.zero_point = (int32_t) (uint32_t) zero_point;
  return params;
}

static inline union qnnp_q31_requantization_params qnnp_compute_requantization_params(
  float scale,
  uint8_t zero_point,
  uint8_t min,
  uint8_t max)
{
  /* Compute requantization parameters */
  const uint32_t scale_bits = fp32_to_bits(scale);

  /* Multiplier is in [0x40000000, 0x7FFFFF80] range */
  const int32_t multiplier = (int32_t)(((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  /* Shift is in [0, 31] range */
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  union qnnp_q31_requantization_params params;
  #if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
    const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
    const uint32_t remainder_threshold = remainder_mask >> 1;
    params.sse2.multiplier[0] = multiplier;
    params.sse2.multiplier[1] = multiplier;
    params.sse2.multiplier[2] = multiplier;
    params.sse2.multiplier[3] = multiplier;
    params.sse2.rounding[0] = UINT64_C(0x40000000);
    params.sse2.rounding[1] = UINT64_C(0x40000000);
    params.sse2.remainder_mask[0] = (int32_t) remainder_mask;
    params.sse2.remainder_mask[1] = (int32_t) remainder_mask;
    params.sse2.remainder_mask[2] = (int32_t) remainder_mask;
    params.sse2.remainder_mask[3] = (int32_t) remainder_mask;
    params.sse2.remainder_threshold[0] = (int32_t) remainder_threshold;
    params.sse2.remainder_threshold[1] = (int32_t) remainder_threshold;
    params.sse2.remainder_threshold[2] = (int32_t) remainder_threshold;
    params.sse2.remainder_threshold[3] = (int32_t) remainder_threshold;
    params.sse2.shift[0] = (uint64_t) (uint32_t) shift;
    params.sse2.shift[1] = (uint64_t) (uint32_t) shift;
    for (uint32_t i = 0; i < 8; i++) {
      params.sse2.zero_point[i] = (int16_t) (uint16_t) zero_point;  
    }
    for (uint32_t i = 0; i < 16; i++) {
      params.sse2.max[i] = max;
      params.sse2.min[i] = min;
    }
  #elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
    params.neon.multiplier = multiplier;
    params.neon.right_shift = -shift;
    params.neon.zero_point = (int16_t) (uint16_t) zero_point;
    params.neon.max = max;
    params.neon.min = min;
  #else
    const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
    const uint32_t remainder_threshold = remainder_mask >> 1;
    params.scalar.multiplier = multiplier;
    params.scalar.remainder_mask = (int32_t) remainder_mask;
    params.scalar.remainder_threshold = (int32_t) remainder_threshold;
    params.scalar.shift = (uint32_t) shift;
    params.scalar.min_less_zero_point = (int32_t) (uint32_t) min - (int32_t) (uint32_t) zero_point;
    params.scalar.max_less_zero_point = (int32_t) (uint32_t) max - (int32_t) (uint32_t) zero_point;
    params.scalar.zero_point = (int32_t) (uint32_t) zero_point;
  #endif
  return params;
}

static inline uint8_t qnnp_q31_requantize(
  int32_t n,
  union qnnp_q31_requantization_params params)
{
  const int64_t product = (int64_t) n * (int64_t) params.scalar.multiplier;
  const int32_t q31product = (int32_t) (uint32_t) ((uint64_t) (product + INT64_C(0x40000000)) >> 31);
  const int32_t remainder = (q31product & params.scalar.remainder_mask) - (int32_t) (n < 0);
  n = asr_s32(q31product, params.scalar.shift) + (int32_t) (remainder > params.scalar.remainder_threshold);
  if (n < params.scalar.min_less_zero_point) {
    n = params.scalar.min_less_zero_point;
  }
  if (n > params.scalar.max_less_zero_point) {
    n = params.scalar.max_less_zero_point;
  }

  return (uint8_t) (n + params.scalar.zero_point);
}
