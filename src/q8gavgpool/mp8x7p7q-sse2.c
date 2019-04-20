/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <emmintrin.h>

#include <qnnpack/q8gavgpool.h>


void q8gavgpool_ukernel_mp8x7p7q__sse2(
    size_t m,
    size_t n,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    const union qnnp_avgpool_quantization_params quantization_params[RESTRICT_STATIC 1])
{
  assert(m > 7);
  assert(n >= 8);

  const uint8_t* i0 = input;
  const uint8_t* i1 = i0 + input_stride;
  const uint8_t* i2 = i1 + input_stride;
  const uint8_t* i3 = i2 + input_stride;
  const uint8_t* i4 = i3 + input_stride;
  const uint8_t* i5 = i4 + input_stride;
  const uint8_t* i6 = i5 + input_stride;
  const size_t packed_n = (n + 7) & -8;
  const size_t input_increment = 7 * input_stride - packed_n;
  const __m128i vbias = _mm_load_si128((const __m128i*) &quantization_params->sse2.bias);
  const __m128i vzero = _mm_setzero_si128();

  /* note: goes up to 7 elements over bound */
  int32_t* acc = buffer;
  for (size_t k = 0; k < n; k += 8) {
    const __m128i vi0 = _mm_loadl_epi64((const __m128i*) i0); i0 += 8;
    const __m128i vi1 = _mm_loadl_epi64((const __m128i*) i1); i1 += 8;
    const __m128i vi2 = _mm_loadl_epi64((const __m128i*) i2); i2 += 8;
    const __m128i vi3 = _mm_loadl_epi64((const __m128i*) i3); i3 += 8;
    const __m128i vi4 = _mm_loadl_epi64((const __m128i*) i4); i4 += 8;
    const __m128i vi5 = _mm_loadl_epi64((const __m128i*) i5); i5 += 8;
    const __m128i vi6 = _mm_loadl_epi64((const __m128i*) i6); i6 += 8;

    const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
    const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
    const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
    const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
    const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
    const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
    const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

    __m128i vacc_lo = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vxi0, vzero));
    __m128i vacc_hi = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vxi0, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi1, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi1, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi2, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi2, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi3, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi3, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi4, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi4, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi5, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi5, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi6, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi6, vzero));

    _mm_store_si128((__m128i*) acc, vacc_lo);
    _mm_store_si128((__m128i*) acc + 1, vacc_hi);
    acc += 8;
  }
  for (m -= 7; m > 7; m -= 7) {
    acc = buffer;
    i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);

    /* note: goes up to 7 elements over bound */
    for (size_t k = 0; k < n; k += 8) {
      const __m128i vi0 = _mm_loadl_epi64((const __m128i*) i0); i0 += 8;
      const __m128i vi1 = _mm_loadl_epi64((const __m128i*) i1); i1 += 8;
      const __m128i vi2 = _mm_loadl_epi64((const __m128i*) i2); i2 += 8;
      const __m128i vi3 = _mm_loadl_epi64((const __m128i*) i3); i3 += 8;
      const __m128i vi4 = _mm_loadl_epi64((const __m128i*) i4); i4 += 8;
      const __m128i vi5 = _mm_loadl_epi64((const __m128i*) i5); i5 += 8;
      const __m128i vi6 = _mm_loadl_epi64((const __m128i*) i6); i6 += 8;
      __m128i vacc_lo = _mm_load_si128((const __m128i*) acc);
      __m128i vacc_hi = _mm_load_si128((const __m128i*) acc + 1);

      const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
      const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
      const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
      const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
      const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
      const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
      const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi0, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi0, vzero));
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi1, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi1, vzero));
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi2, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi2, vzero));
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi3, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi3, vzero));
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi4, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi4, vzero));
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi5, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi5, vzero));
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi6, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi6, vzero));

      _mm_store_si128((__m128i*) acc, vacc_lo);
      _mm_store_si128((__m128i*) acc + 1, vacc_hi);
      acc += 8;
    }
  }

  const __m128i vmultiplier = _mm_load_si128((const __m128i*) quantization_params->sse2.multiplier);
  const __m128i vrounding = _mm_load_si128((const __m128i*) quantization_params->sse2.rounding);
  const __m128i vright_shift = _mm_loadl_epi64((const __m128i*) quantization_params->sse2.right_shift);

  i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
  if (m < 2) {
    i1 = zero;
  }
  i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
  if (m <= 2) {
    i2 = zero;
  }
  i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
  if (m < 4) {
    i3 = zero;
  }
  i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
  if (m <= 4) {
    i4 = zero;
  }
  i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
  if (m < 6) {
    i5 = zero;
  }
  i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);
  if (m <= 6) {
    i6 = zero;
  }

  acc = buffer;
  do {
    const __m128i vi0 = _mm_loadl_epi64((const __m128i*) i0); i0 += 8;
    const __m128i vi1 = _mm_loadl_epi64((const __m128i*) i1); i1 += 8;
    const __m128i vi2 = _mm_loadl_epi64((const __m128i*) i2); i2 += 8;
    const __m128i vi3 = _mm_loadl_epi64((const __m128i*) i3); i3 += 8;
    const __m128i vi4 = _mm_loadl_epi64((const __m128i*) i4); i4 += 8;
    const __m128i vi5 = _mm_loadl_epi64((const __m128i*) i5); i5 += 8;
    const __m128i vi6 = _mm_loadl_epi64((const __m128i*) i6); i6 += 8;
    __m128i vacc_lo = _mm_load_si128((const __m128i*) acc);
    __m128i vacc_hi = _mm_load_si128((const __m128i*) acc + 1);
    acc += 8;

    const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
    const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
    const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
    const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
    const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
    const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
    const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi0, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi0, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi1, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi1, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi2, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi2, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi3, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi3, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi4, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi4, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi5, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi5, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi6, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi6, vzero));

    const __m128i vneg_mask_lo = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_lo);
    const __m128i vneg_mask_hi = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_hi);

    const __m128i vabs_lo0123 = _mm_sub_epi32(_mm_xor_si128(vacc_lo, vneg_mask_lo), vneg_mask_lo);
    const __m128i vabs_hi0123 = _mm_sub_epi32(_mm_xor_si128(vacc_hi, vneg_mask_hi), vneg_mask_hi);

    const __m128i vabs_lo1032 = _mm_shuffle_epi32(vabs_lo0123, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i vabs_hi1032 = _mm_shuffle_epi32(vabs_hi0123, _MM_SHUFFLE(2, 3, 0, 1));

    const __m128i vabsmul_lo02 = _mm_mul_epu32(vabs_lo0123, vmultiplier);
    const __m128i vabsmul_hi02 = _mm_mul_epu32(vabs_hi0123, vmultiplier);

    const __m128i vabsmul_lo13 = _mm_mul_epu32(vabs_lo1032, vmultiplier);
    const __m128i vabsmul_hi13 = _mm_mul_epu32(vabs_hi1032, vmultiplier);

    const __m128i vabs_scaled_lo02 = _mm_srl_epi64(_mm_add_epi64(vabsmul_lo02, vrounding), vright_shift);
    const __m128i vabs_scaled_lo13 = _mm_srl_epi64(_mm_add_epi64(vabsmul_lo13, vrounding), vright_shift);
    const __m128i vabs_scaled_hi02 = _mm_srl_epi64(_mm_add_epi64(vabsmul_hi02, vrounding), vright_shift);
    const __m128i vabs_scaled_hi13 = _mm_srl_epi64(_mm_add_epi64(vabsmul_hi13, vrounding), vright_shift);

    const __m128i vabs_scaled_lo0213 = _mm_castps_si128(
        _mm_shuffle_ps(_mm_castsi128_ps(vabs_scaled_lo02), _mm_castsi128_ps(vabs_scaled_lo13), _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i vabs_scaled_hi0213 = _mm_castps_si128(
        _mm_shuffle_ps(_mm_castsi128_ps(vabs_scaled_hi02), _mm_castsi128_ps(vabs_scaled_hi13), _MM_SHUFFLE(2, 0, 2, 0)));

    const __m128i vabs_scaled_lo = _mm_shuffle_epi32(vabs_scaled_lo0213, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i vabs_scaled_hi = _mm_shuffle_epi32(vabs_scaled_hi0213, _MM_SHUFFLE(3, 1, 2, 0));

    const __m128i vscaled_lo = _mm_sub_epi32(_mm_xor_si128(vabs_scaled_lo, vneg_mask_lo), vneg_mask_lo);
    const __m128i vscaled_hi = _mm_sub_epi32(_mm_xor_si128(vabs_scaled_hi, vneg_mask_hi), vneg_mask_hi);

    __m128i vout = _mm_packs_epi32(vscaled_lo, vscaled_hi);
    vout = _mm_adds_epi16(vout, _mm_load_si128((const __m128i*) quantization_params->sse2.output_zero_point));
    vout = _mm_packus_epi16(vout, vout);
    vout = _mm_min_epu8(vout, _mm_load_si128((const __m128i*) quantization_params->sse2.output_max));
    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) quantization_params->sse2.output_min));

    _mm_storel_epi64((__m128i*) output, vout); output += 8;

    n -= 8;
  } while (n >= 8);
  if (n != 0) {
    const size_t address_decrement = 8 - n;
    i0 = (const uint8_t*) ((uintptr_t) i0 - address_decrement);
    i1 = (const uint8_t*) ((uintptr_t) i1 - address_decrement);
    i2 = (const uint8_t*) ((uintptr_t) i2 - address_decrement);
    i3 = (const uint8_t*) ((uintptr_t) i3 - address_decrement);
    i4 = (const uint8_t*) ((uintptr_t) i4 - address_decrement);
    i5 = (const uint8_t*) ((uintptr_t) i5 - address_decrement);
    i6 = (const uint8_t*) ((uintptr_t) i6 - address_decrement);
    const __m128i vi_shift = _mm_cvtsi32_si128(8 * address_decrement);

    const __m128i vi0 = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) i0), vi_shift);
    const __m128i vi1 = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) i1), vi_shift);
    const __m128i vi2 = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) i2), vi_shift);
    const __m128i vi3 = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) i3), vi_shift);
    const __m128i vi4 = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) i4), vi_shift);
    const __m128i vi5 = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) i5), vi_shift);
    const __m128i vi6 = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) i6), vi_shift);
    __m128i vacc_lo = _mm_load_si128((const __m128i*) acc);
    __m128i vacc_hi = _mm_load_si128((const __m128i*) acc + 1);

    const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
    const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
    const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
    const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
    const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
    const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
    const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi0, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi0, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi1, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi1, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi2, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi2, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi3, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi3, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi4, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi4, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi5, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi5, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi6, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi6, vzero));

    const __m128i vneg_mask_lo = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_lo);
    const __m128i vneg_mask_hi = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_hi);

    const __m128i vabs_lo0123 = _mm_sub_epi32(_mm_xor_si128(vacc_lo, vneg_mask_lo), vneg_mask_lo);
    const __m128i vabs_hi0123 = _mm_sub_epi32(_mm_xor_si128(vacc_hi, vneg_mask_hi), vneg_mask_hi);

    const __m128i vabs_lo1032 = _mm_shuffle_epi32(vabs_lo0123, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i vabs_hi1032 = _mm_shuffle_epi32(vabs_hi0123, _MM_SHUFFLE(2, 3, 0, 1));

    const __m128i vabsmul_lo02 = _mm_mul_epu32(vabs_lo0123, vmultiplier);
    const __m128i vabsmul_hi02 = _mm_mul_epu32(vabs_hi0123, vmultiplier);

    const __m128i vabsmul_lo13 = _mm_mul_epu32(vabs_lo1032, vmultiplier);
    const __m128i vabsmul_hi13 = _mm_mul_epu32(vabs_hi1032, vmultiplier);

    const __m128i vabs_scaled_lo02 = _mm_srl_epi64(_mm_add_epi64(vabsmul_lo02, vrounding), vright_shift);
    const __m128i vabs_scaled_lo13 = _mm_srl_epi64(_mm_add_epi64(vabsmul_lo13, vrounding), vright_shift);
    const __m128i vabs_scaled_hi02 = _mm_srl_epi64(_mm_add_epi64(vabsmul_hi02, vrounding), vright_shift);
    const __m128i vabs_scaled_hi13 = _mm_srl_epi64(_mm_add_epi64(vabsmul_hi13, vrounding), vright_shift);

    const __m128i vabs_scaled_lo0213 = _mm_castps_si128(
        _mm_shuffle_ps(_mm_castsi128_ps(vabs_scaled_lo02), _mm_castsi128_ps(vabs_scaled_lo13), _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i vabs_scaled_hi0213 = _mm_castps_si128(
        _mm_shuffle_ps(_mm_castsi128_ps(vabs_scaled_hi02), _mm_castsi128_ps(vabs_scaled_hi13), _MM_SHUFFLE(2, 0, 2, 0)));

    const __m128i vabs_scaled_lo = _mm_shuffle_epi32(vabs_scaled_lo0213, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i vabs_scaled_hi = _mm_shuffle_epi32(vabs_scaled_hi0213, _MM_SHUFFLE(3, 1, 2, 0));

    const __m128i vscaled_lo = _mm_sub_epi32(_mm_xor_si128(vabs_scaled_lo, vneg_mask_lo), vneg_mask_lo);
    const __m128i vscaled_hi = _mm_sub_epi32(_mm_xor_si128(vabs_scaled_hi, vneg_mask_hi), vneg_mask_hi);

    __m128i vout = _mm_packs_epi32(vscaled_lo, vscaled_hi);
    vout = _mm_adds_epi16(vout, _mm_load_si128((const __m128i*) quantization_params->sse2.output_zero_point));
    vout = _mm_packus_epi16(vout, vout);
    vout = _mm_min_epu8(vout, _mm_load_si128((const __m128i*) quantization_params->sse2.output_max));
    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) quantization_params->sse2.output_min));

    if (n & 4) {
      *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout);
      output += 4;
      vout = _mm_srli_epi64(vout, 32);
    }
    if (n & 2) {
      *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout, 0);
      output += 2;
      vout = _mm_srli_epi32(vout, 16);
    }
    if (n & 1) {
      *((uint8_t*) output) = (uint8_t) _mm_cvtsi128_si32(vout);
    }
  }
}
