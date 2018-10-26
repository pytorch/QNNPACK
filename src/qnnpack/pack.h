/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <qnnpack/math.h>

static inline void pack_q8gemm_b(
    size_t n,
    size_t k,
    uint32_t nr,
    uint32_t kr,
    const uint8_t* b,
    uint8_t* packed_b)
{
  const size_t k_stride = (k + (kr - 1)) & -kr;
  for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
    const size_t nr_block_size = min(n - nr_block_start, nr);
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
      for (size_t kr_block_start = 0; kr_block_start < k; kr_block_start += kr) {
        const size_t kr_block_size = min(k - kr_block_start, kr);
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
          packed_b[nr_block_start * k_stride + kr_block_start * nr + nr_block_offset * kr + kr_block_offset] =
              b[(nr_block_start + nr_block_offset) * k + (kr_block_start + kr_block_offset)];
        }
      }
    }
  }
}

static inline void pack_q8conv_b(
    size_t n,
    size_t ks,
    size_t kc,
    uint32_t nr,
    uint32_t kr,
    const uint8_t* b,
    uint8_t* packed_b)
{
  const size_t kc_stride = (kc + (kr - 1)) & -kr;
  for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
    const size_t nr_block_size = min(n - nr_block_start, nr);
    for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
      const size_t kr_block_size = min(kc - kr_block_start, kr);
      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
            packed_b[(nr_block_start * ks + ki * nr) * kc_stride + kr_block_start * nr + nr_block_offset * kr + kr_block_offset] =
                b[((nr_block_start + nr_block_offset) * ks + ki) * kc + (kr_block_start + kr_block_offset)];
          }
        }
      }
    }
  }
}

static inline void pack_q8deconv_b(
    size_t n,
    size_t ks,
    size_t kc,
    uint32_t nr,
    uint32_t kr,
    const uint8_t* b,
    uint8_t* packed_b)
{
  const size_t kc_stride = (kc + (kr - 1)) & -kr;
  for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
    const size_t nr_block_size = min(n - nr_block_start, nr);
    for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
      const size_t kr_block_size = min(kc - kr_block_start, kr);
      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
            packed_b[(nr_block_start * ks + ki * nr) * kc_stride + kr_block_start * nr + nr_block_offset * kr + kr_block_offset] =
                b[((kr_block_start + kr_block_offset) * ks + ki) * n + (nr_block_start + nr_block_offset)];
          }
        }
      }
    }
  }
}

static inline void pack_q8dw_w(
  size_t h,
  size_t w,
  size_t c,
  size_t cr,
  const uint8_t* k,
  const int32_t* b,
  void* packed_w)
{
  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
      *((int32_t*) packed_w) = b[cr_block_start + cr_block_offset];
      packed_w = (void*) ((uintptr_t) packed_w + sizeof(int32_t));
    }
    packed_w = (void*) ((uintptr_t) packed_w + (cr - cr_block_size) * sizeof(int32_t));
    for (size_t x = 0; x < w; x++) {
      for (size_t y = 0; y < h; y++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *((uint8_t*) packed_w) = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          packed_w = (void*) ((uintptr_t) packed_w + sizeof(uint8_t));
        }
        packed_w = (void*) ((uintptr_t) packed_w + (cr - cr_block_size) * sizeof(uint8_t));
      }
    }
  }
}

static inline void pack_q8gemm_b_diagonal(
    size_t n,
    size_t k,
    uint32_t nr,
    uint32_t kr,
    uint32_t kc,
    const uint8_t* b,
    uint8_t* packed_b)
{
  const size_t k_stride = (k + (kr - 1)) & -kr;
  for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
    /* Pack b first in big chunk of size nr x kc,
     *  within the block of size nr x kc, pack b diagonally in size of kr.
     * kc (power of 2) must be multiples of kr (power of 2) */
    const size_t nr_block_size = min(n - nr_block_start, nr);
    size_t kr_chunk_start = 0;
    for (; kr_chunk_start < k / kc * kc; kr_chunk_start += kc) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
        for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            /* When kc is power of 2, x % kc == x & (kc - 1) */
            packed_b[nr_block_start * k_stride + (kr_chunk_start + kr_block_start) * nr + nr_block_offset * kr + kr_block_offset]
              = b[(nr_block_start + nr_block_offset) * k + kr_chunk_start + ((kr_block_start + nr_block_offset * kr + kr_block_offset) & (kc - 1))];
          }
        }
      }
    }

    /* for the remaining k (< kc), pack it in the same way as pack_q8gemm_b */
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
      for (size_t kr_block_start = kr_chunk_start; kr_block_start < k; kr_block_start += kr) {
        const size_t kr_block_size = min(k - kr_block_start, kr);
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
          packed_b[nr_block_start * k_stride + kr_block_start * nr + nr_block_offset * kr + kr_block_offset] =
              b[(nr_block_start + nr_block_offset) * k + (kr_block_start + kr_block_offset)];
        }
      }
    }
  }
}
