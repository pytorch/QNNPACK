/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <immintrin.h>

QNNP_INLINE __m128i quantize(const __m128i a, const __m128i zp)
{
#if QNNPACK_RUNTIME_QUANTIZATION
  // Run-time quantization
  return _mm_sub_epi16(a, zp);
#else
  // Design-time quantization (no-op)
  return a;
#endif
}
