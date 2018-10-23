/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>

#include <qnnpack/q8gemm.h>


void q8gemm_ukernel_6x4__neon(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* restrict a,
    size_t a_stride,
    const uint8_t* restrict b,
    const int32_t* restrict bias,
    uint8_t* restrict c,
    size_t c_stride,
    const uint8_t a_offset,
    const uint8_t b_offset,
    const union qnnp_q31_requantization_params requantization_params[restrict static 1])
{
  int32x4_t vacc0x0123 = vld1q_s32(bias);
  int32x4_t vacc1x0123 = vacc0x0123;
  int32x4_t vacc2x0123 = vacc0x0123;
  int32x4_t vacc3x0123 = vacc0x0123;
  int32x4_t vacc4x0123 = vacc0x0123;
  int32x4_t vacc5x0123 = vacc0x0123;

  const uint8_t* a0 = a;
  const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
  if (mr < 2) {
    a1 = a0;
  }
  const uint8_t* a2 = (const uint8_t*) ((uintptr_t) a1 + a_stride);
  if (mr <= 2) {
    a2 = a1;
  }
  const uint8_t* a3 = (const uint8_t*) ((uintptr_t) a2 + a_stride);
  if (mr < 4) {
    a3 = a2;
  }
  const uint8_t* a4 = (const uint8_t*) ((uintptr_t) a3 + a_stride);
  if (mr <= 4) {
    a4 = a3;
  };
  const uint8_t* a5 = (const uint8_t*) ((uintptr_t) a4 + a_stride);
  if (mr != 6) {
    a5 = a4;
  }

  const uint8x8_t va_offset = vdup_n_u8(a_offset);
  const uint8x8_t vb_offset = vdup_n_u8(b_offset);
  for (; k >= 8; k -= 8) {
    const int16x8_t va0 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(a0), va_offset)); a0 += 8;
    const int16x8_t va1 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(a1), va_offset)); a1 += 8;
    const int16x8_t va2 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(a2), va_offset)); a2 += 8;
    const int16x8_t va3 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(a3), va_offset)); a3 += 8;
    const int16x8_t va4 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(a4), va_offset)); a4 += 8;
    const int16x8_t va5 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(a5), va_offset)); a5 += 8;

    const int16x8_t vb01 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(b), vb_offset)); b += 8;

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb01), vget_low_s16(va0), 0);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb01), vget_low_s16(va1), 0);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb01), vget_low_s16(va2), 0);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb01), vget_low_s16(va3), 0);
    vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vb01), vget_low_s16(va4), 0);
    vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vb01), vget_low_s16(va5), 0);

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_high_s16(vb01), vget_low_s16(va0), 1);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_high_s16(vb01), vget_low_s16(va1), 1);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_high_s16(vb01), vget_low_s16(va2), 1);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_high_s16(vb01), vget_low_s16(va3), 1);
    vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_high_s16(vb01), vget_low_s16(va4), 1);
    vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_high_s16(vb01), vget_low_s16(va5), 1);

    const int16x8_t vb23 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(b), vb_offset)); b += 8;

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb23), vget_low_s16(va0), 2);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb23), vget_low_s16(va1), 2);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb23), vget_low_s16(va2), 2);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb23), vget_low_s16(va3), 2);
    vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vb23), vget_low_s16(va4), 2);
    vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vb23), vget_low_s16(va5), 2);

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_high_s16(vb23), vget_low_s16(va0), 3);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_high_s16(vb23), vget_low_s16(va1), 3);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_high_s16(vb23), vget_low_s16(va2), 3);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_high_s16(vb23), vget_low_s16(va3), 3);
    vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_high_s16(vb23), vget_low_s16(va4), 3);
    vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_high_s16(vb23), vget_low_s16(va5), 3);

    const int16x8_t vb45 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(b), vb_offset)); b += 8;

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb45), vget_high_s16(va0), 0);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb45), vget_high_s16(va1), 0);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb45), vget_high_s16(va2), 0);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb45), vget_high_s16(va3), 0);
    vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vb45), vget_high_s16(va4), 0);
    vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vb45), vget_high_s16(va5), 0);

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_high_s16(vb45), vget_high_s16(va0), 1);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_high_s16(vb45), vget_high_s16(va1), 1);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_high_s16(vb45), vget_high_s16(va2), 1);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_high_s16(vb45), vget_high_s16(va3), 1);
    vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_high_s16(vb45), vget_high_s16(va4), 1);
    vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_high_s16(vb45), vget_high_s16(va5), 1);

    const int16x8_t vb67 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(b), vb_offset)); b += 8;

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb67), vget_high_s16(va0), 2);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb67), vget_high_s16(va1), 2);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb67), vget_high_s16(va2), 2);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb67), vget_high_s16(va3), 2);
    vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vb67), vget_high_s16(va4), 2);
    vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vb67), vget_high_s16(va5), 2);

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_high_s16(vb67), vget_high_s16(va0), 3);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_high_s16(vb67), vget_high_s16(va1), 3);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_high_s16(vb67), vget_high_s16(va2), 3);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_high_s16(vb67), vget_high_s16(va3), 3);
    vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_high_s16(vb67), vget_high_s16(va4), 3);
    vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_high_s16(vb67), vget_high_s16(va5), 3);
  }
  if (k != 0) {
    const int16x8_t va0 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(a0), va_offset)); a0 += 8;
    const int16x8_t va1 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(a1), va_offset)); a1 += 8;
    const int16x8_t va2 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(a2), va_offset)); a2 += 8;
    const int16x8_t va3 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(a3), va_offset)); a3 += 8;
    const int16x8_t va4 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(a4), va_offset)); a4 += 8;
    const int16x8_t va5 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(a5), va_offset)); a5 += 8;

    const int16x8_t vb0 = vreinterpretq_s16_u16(vsubl_u8(
        vreinterpret_u8_u32(vld1_dup_u32((const uint32_t*) b)), vb_offset)); b += 4;

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb0), vget_low_s16(va0), 0);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb0), vget_low_s16(va1), 0);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb0), vget_low_s16(va2), 0);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb0), vget_low_s16(va3), 0);
    vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vb0), vget_low_s16(va4), 0);
    vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vb0), vget_low_s16(va5), 0);

    if (k >= 2) {
      const int16x8_t vb1 = vreinterpretq_s16_u16(vsubl_u8(
          vreinterpret_u8_u32(vld1_dup_u32((const uint32_t*) b)), vb_offset)); b += 4;

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb1), vget_low_s16(va0), 1);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb1), vget_low_s16(va1), 1);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb1), vget_low_s16(va2), 1);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb1), vget_low_s16(va3), 1);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vb1), vget_low_s16(va4), 1);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vb1), vget_low_s16(va5), 1);

      if (k > 2) {
        const int16x8_t vb2 = vreinterpretq_s16_u16(vsubl_u8(
            vreinterpret_u8_u32(vld1_dup_u32((const uint32_t*) b)), vb_offset)); b += 4;

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb2), vget_low_s16(va0), 2);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb2), vget_low_s16(va1), 2);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb2), vget_low_s16(va2), 2);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb2), vget_low_s16(va3), 2);
        vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vb2), vget_low_s16(va4), 2);
        vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vb2), vget_low_s16(va5), 2);

        if (k >= 4) {
          const int16x8_t vb3 = vreinterpretq_s16_u16(vsubl_u8(
              vreinterpret_u8_u32(vld1_dup_u32((const uint32_t*) b)), vb_offset)); b += 4;

          vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb3), vget_low_s16(va0), 3);
          vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb3), vget_low_s16(va1), 3);
          vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb3), vget_low_s16(va2), 3);
          vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb3), vget_low_s16(va3), 3);
          vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vb3), vget_low_s16(va4), 3);
          vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vb3), vget_low_s16(va5), 3);

          if (k > 4) {
            const int16x8_t vb4 = vreinterpretq_s16_u16(vsubl_u8(
                vreinterpret_u8_u32(vld1_dup_u32((const uint32_t*)b)), vb_offset)); b += 4;

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb4), vget_high_s16(va0), 0);
            vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb4), vget_high_s16(va1), 0);
            vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb4), vget_high_s16(va2), 0);
            vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb4), vget_high_s16(va3), 0);
            vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vb4), vget_high_s16(va4), 0);
            vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vb4), vget_high_s16(va5), 0);

            if (k >= 6) {
              const int16x8_t vb5 = vreinterpretq_s16_u16(vsubl_u8(
                  vreinterpret_u8_u32(vld1_dup_u32((const uint32_t*)b)), vb_offset)); b += 4;

              vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb5), vget_high_s16(va0), 1);
              vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb5), vget_high_s16(va1), 1);
              vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb5), vget_high_s16(va2), 1);
              vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb5), vget_high_s16(va3), 1);
              vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vb5), vget_high_s16(va4), 1);
              vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vb5), vget_high_s16(va5), 1);

              if (k > 6) {
                const int16x8_t vb6 = vreinterpretq_s16_u16(vsubl_u8(
                    vreinterpret_u8_u32(vld1_dup_u32((const uint32_t*)b)), vb_offset));

                vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb6), vget_high_s16(va0), 2);
                vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb6), vget_high_s16(va1), 2);
                vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb6), vget_high_s16(va2), 2);
                vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb6), vget_high_s16(va3), 2);
                vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vb6), vget_high_s16(va4), 2);
                vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vb6), vget_high_s16(va5), 2);
              }
            }
          }
        }
      }
    }
  }

  const int32x4_t vmultiplier = vld1q_dup_s32(&requantization_params->neon.multiplier);
  vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier);
  vacc1x0123 = vqrdmulhq_s32(vacc1x0123, vmultiplier);
  vacc2x0123 = vqrdmulhq_s32(vacc2x0123, vmultiplier);
  vacc3x0123 = vqrdmulhq_s32(vacc3x0123, vmultiplier);
  vacc4x0123 = vqrdmulhq_s32(vacc4x0123, vmultiplier);
  vacc5x0123 = vqrdmulhq_s32(vacc5x0123, vmultiplier);

  const int32x4_t vright_shift = vld1q_dup_s32(&requantization_params->neon.right_shift);
  const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
  vacc0x0123 = vsraq_n_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask), 31);
  vacc1x0123 = vsraq_n_s32(vacc1x0123, vbicq_s32(vacc1x0123, vzero_shift_mask), 31);
  vacc2x0123 = vsraq_n_s32(vacc2x0123, vbicq_s32(vacc2x0123, vzero_shift_mask), 31);
  vacc3x0123 = vsraq_n_s32(vacc3x0123, vbicq_s32(vacc3x0123, vzero_shift_mask), 31);
  vacc4x0123 = vsraq_n_s32(vacc4x0123, vbicq_s32(vacc4x0123, vzero_shift_mask), 31);
  vacc5x0123 = vsraq_n_s32(vacc5x0123, vbicq_s32(vacc5x0123, vzero_shift_mask), 31);
  
  vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift);
  vacc1x0123 = vrshlq_s32(vacc1x0123, vright_shift);
  vacc2x0123 = vrshlq_s32(vacc2x0123, vright_shift);
  vacc3x0123 = vrshlq_s32(vacc3x0123, vright_shift);
  vacc4x0123 = vrshlq_s32(vacc4x0123, vright_shift);
  vacc5x0123 = vrshlq_s32(vacc5x0123, vright_shift);

  const int16x8_t vzero_point = vld1q_dup_s16(&requantization_params->neon.zero_point);
#ifdef __aarch64__
  const int16x8_t vacc01x0123 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc1x0123), vzero_point);
  const int16x8_t vacc23x0123 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc3x0123), vzero_point);
  const int16x8_t vacc45x0123 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc4x0123), vacc5x0123), vzero_point);

  uint8x16_t vout0123x0123 = vqmovun_high_s16(vqmovun_s16(vacc01x0123), vacc23x0123);
  uint8x8_t vout45x0123 = vqmovun_s16(vacc45x0123);
#else
  const int16x8_t vacc01x0123 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc1x0123)), vzero_point);
  const int16x8_t vacc23x0123 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc3x0123)), vzero_point);
  const int16x8_t vacc45x0123 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc4x0123), vqmovn_s32(vacc5x0123)), vzero_point);

  uint8x16_t vout0123x0123 = vcombine_u8(vqmovun_s16(vacc01x0123), vqmovun_s16(vacc23x0123));
  uint8x8_t vout45x0123 = vqmovun_s16(vacc45x0123);
#endif
  const uint8x16_t vmin = vld1q_dup_u8(&requantization_params->neon.min);
  const uint8x16_t vmax = vld1q_dup_u8(&requantization_params->neon.max);

  vout0123x0123 = vmaxq_u8(vout0123x0123, vmin);
  vout45x0123 = vmax_u8(vout45x0123, vget_low_u8(vmin));
  vout0123x0123 = vminq_u8(vout0123x0123, vmax);
  vout45x0123 = vmin_u8(vout45x0123, vget_low_u8(vmax));

  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + c_stride);
  if (mr < 2) {
    c1 = c0;
  }
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + c_stride);
  if (mr <= 2) {
    c2 = c1;
  }
  uint8_t* c3 = (uint8_t*) ((uintptr_t) c2 + c_stride);
  if (mr < 4) {
    c3 = c2;
  }
  uint8_t* c4 = (uint8_t*) ((uintptr_t) c3 + c_stride);
  if (mr <= 4) {
    c4 = c3;
  }
  uint8_t* c5 = (uint8_t*) ((uintptr_t) c4 + c_stride);
  if (mr != 6) {
    c5 = c4;
  }
  if (nr == 4) {
    vst1q_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpretq_u32_u8(vout0123x0123), 0);
    vst1q_lane_u32(__builtin_assume_aligned(c1, 1), vreinterpretq_u32_u8(vout0123x0123), 1);
    vst1q_lane_u32(__builtin_assume_aligned(c2, 1), vreinterpretq_u32_u8(vout0123x0123), 2);
    vst1q_lane_u32(__builtin_assume_aligned(c3, 1), vreinterpretq_u32_u8(vout0123x0123), 3);
    vst1_lane_u32( __builtin_assume_aligned(c4, 1), vreinterpret_u32_u8(vout45x0123), 0);
    vst1_lane_u32(__builtin_assume_aligned(c5, 1), vreinterpret_u32_u8(vout45x0123), 1);
  } else {
    if (nr >= 2) {
      vst1q_lane_u16(__builtin_assume_aligned(c0, 1), vreinterpretq_u16_u8(vout0123x0123), 0); c0 += 2;
      vst1q_lane_u16(__builtin_assume_aligned(c1, 1), vreinterpretq_u16_u8(vout0123x0123), 2); c1 += 2;
      vst1q_lane_u16(__builtin_assume_aligned(c2, 1), vreinterpretq_u16_u8(vout0123x0123), 4); c2 += 2;
      vst1q_lane_u16(__builtin_assume_aligned(c3, 1), vreinterpretq_u16_u8(vout0123x0123), 6); c3 += 2;
      vst1_lane_u16(__builtin_assume_aligned(c4, 1), vreinterpret_u16_u8(vout45x0123), 0); c4 += 2;
      vst1_lane_u16(__builtin_assume_aligned(c5, 1), vreinterpret_u16_u8(vout45x0123), 2); c5 += 2;
      vout0123x0123 = vextq_u8(vout0123x0123, vout0123x0123, 2);
      vout45x0123 = vext_u8(vout45x0123, vout45x0123, 2);
      nr -= 2;
    }
    if (nr != 0) {
      vst1q_lane_u8(__builtin_assume_aligned(c0, 1), vout0123x0123, 0);
      vst1q_lane_u8(__builtin_assume_aligned(c1, 1), vout0123x0123, 4);
      vst1q_lane_u8(__builtin_assume_aligned(c2, 1), vout0123x0123, 8);
      vst1q_lane_u8(__builtin_assume_aligned(c3, 1), vout0123x0123, 12);
      vst1_lane_u8(__builtin_assume_aligned(c4, 1), vout45x0123, 0);
      vst1_lane_u8(__builtin_assume_aligned(c5, 1), vout45x0123, 4);
    }
  }
}
