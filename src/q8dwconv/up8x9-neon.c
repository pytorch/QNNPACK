/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>

#include <qnnpack/q8dwconv.h>
#include <requantization/runtime-neon.h>


void q8dwconv_ukernel_up8x9__neon(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    size_t input_stride,
    size_t output_increment,
    const union qnnp_conv_quantization_params quantization_params[restrict static 1])
{
  const uint8x8_t va_zero_point = vld1_dup_u8((const uint8_t*) &quantization_params->neon.input_zero_point);
  const uint8x8_t vkernel_zero_point = vld1_dup_u8((const uint8_t*) &quantization_params->neon.kernel_zero_point);
  const int32x4_t vmultiplier = vld1q_dup_s32(&quantization_params->neon.multiplier);
  const int32x4_t vright_shift = vld1q_dup_s32(&quantization_params->neon.right_shift);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&quantization_params->neon.output_zero_point);
  const uint8x8_t voutput_min = vld1_dup_u8(&quantization_params->neon.output_min);
  const uint8x8_t voutput_max = vld1_dup_u8(&quantization_params->neon.output_max);

#ifdef __aarch64__
  /* Larger number of registers on AArch64 make it possible to process few pixels at a time */
  if (input_stride == 3 * sizeof(void*)) {
    for (; output_width >= 3; output_width -= 3) {
      const uint8_t* i00 = input[ 0];
      const uint8_t* i10 = input[ 1];
      const uint8_t* i20 = input[ 2];
      const uint8_t* i01 = input[ 3];
      const uint8_t* i11 = input[ 4];
      const uint8_t* i21 = input[ 5];
      const uint8_t* i02 = input[ 6];
      const uint8_t* i12 = input[ 7];
      const uint8_t* i22 = input[ 8];
      const uint8_t* i03 = input[ 9];
      const uint8_t* i13 = input[10];
      const uint8_t* i23 = input[11];
      const uint8_t* i04 = input[12];
      const uint8_t* i14 = input[13];
      const uint8_t* i24 = input[14];

      uint8_t* output0 = output;
      uint8_t* output1 = output0 + channels + output_increment;
      uint8_t* output2 = output1 + channels + output_increment;

      input += 9;

      size_t c = channels;
      const void* w = weights;
      for (; c >= 8; c -= 8) {
        int32x4_t vacc0_lo = vld1q_s32(w); w = (void*) ((uintptr_t) w + sizeof(int32x4_t));
        int32x4_t vacc0_hi = vld1q_s32(w); w = (void*) ((uintptr_t) w + sizeof(int32x4_t));
        int32x4_t vacc1_lo = vacc0_lo;
        int32x4_t vacc2_lo = vacc0_lo;
        int32x4_t vacc1_hi = vacc0_hi;
        int32x4_t vacc2_hi = vacc0_hi;

        const uint8x8_t vk00 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
        const uint8x8_t vi00 = vld1_u8(i00); i00 += 8;
        const uint8x8_t vi01 = vld1_u8(i01); i01 += 8;
        const uint8x8_t vi02 = vld1_u8(i02); i02 += 8;
        const int16x8_t vxk00 = vreinterpretq_s16_u16(vsubl_u8(vk00, vkernel_zero_point));
        const int16x8_t vxi00 = vreinterpretq_s16_u16(sub_zero_point(vi00, va_zero_point));
        const int16x8_t vxi01 = vreinterpretq_s16_u16(sub_zero_point(vi01, va_zero_point));
        const int16x8_t vxi02 = vreinterpretq_s16_u16(sub_zero_point(vi02, va_zero_point));
        vacc0_lo = vmlal_s16(vacc0_lo, vget_low_s16(vxk00), vget_low_s16(vxi00));
        vacc0_hi = vmlal_high_s16(vacc0_hi, vxk00, vxi00);
        vacc1_lo = vmlal_s16(vacc1_lo, vget_low_s16(vxk00), vget_low_s16(vxi01));
        vacc1_hi = vmlal_high_s16(vacc1_hi, vxk00, vxi01);
        vacc2_lo = vmlal_s16(vacc2_lo, vget_low_s16(vxk00), vget_low_s16(vxi02));
        vacc2_hi = vmlal_high_s16(vacc2_hi, vxk00, vxi02);

        const uint8x8_t vk10 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
        const uint8x8_t vi10 = vld1_u8(i10); i10 += 8;
        const uint8x8_t vi11 = vld1_u8(i11); i11 += 8;
        const uint8x8_t vi12 = vld1_u8(i12); i12 += 8;
        const int16x8_t vxk10 = vreinterpretq_s16_u16(vsubl_u8(vk10, vkernel_zero_point));
        const int16x8_t vxi10 = vreinterpretq_s16_u16(sub_zero_point(vi10, va_zero_point));
        const int16x8_t vxi11 = vreinterpretq_s16_u16(sub_zero_point(vi11, va_zero_point));
        const int16x8_t vxi12 = vreinterpretq_s16_u16(sub_zero_point(vi12, va_zero_point));
        vacc0_lo = vmlal_s16(vacc0_lo, vget_low_s16(vxk10), vget_low_s16(vxi10));
        vacc0_hi = vmlal_high_s16(vacc0_hi, vxk10, vxi10);
        vacc1_lo = vmlal_s16(vacc1_lo, vget_low_s16(vxk10), vget_low_s16(vxi11));
        vacc1_hi = vmlal_high_s16(vacc1_hi, vxk10, vxi11);
        vacc2_lo = vmlal_s16(vacc2_lo, vget_low_s16(vxk10), vget_low_s16(vxi12));
        vacc2_hi = vmlal_high_s16(vacc2_hi, vxk10, vxi12);

        const uint8x8_t vk20 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
        const uint8x8_t vi20 = vld1_u8(i20); i20 += 8;
        const uint8x8_t vi21 = vld1_u8(i21); i21 += 8;
        const uint8x8_t vi22 = vld1_u8(i22); i22 += 8;
        const int16x8_t vxk20 = vreinterpretq_s16_u16(vsubl_u8(vk20, vkernel_zero_point));
        const int16x8_t vxi20 = vreinterpretq_s16_u16(sub_zero_point(vi20, va_zero_point));
        const int16x8_t vxi21 = vreinterpretq_s16_u16(sub_zero_point(vi21, va_zero_point));
        const int16x8_t vxi22 = vreinterpretq_s16_u16(sub_zero_point(vi22, va_zero_point));
        vacc0_lo = vmlal_s16(vacc0_lo, vget_low_s16(vxk20), vget_low_s16(vxi20));
        vacc0_hi = vmlal_high_s16(vacc0_hi, vxk20, vxi20);
        vacc1_lo = vmlal_s16(vacc1_lo, vget_low_s16(vxk20), vget_low_s16(vxi21));
        vacc1_hi = vmlal_high_s16(vacc1_hi, vxk20, vxi21);
        vacc2_lo = vmlal_s16(vacc2_lo, vget_low_s16(vxk20), vget_low_s16(vxi22));
        vacc2_hi = vmlal_high_s16(vacc2_hi, vxk20, vxi22);

        const uint8x8_t vk01 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
        const uint8x8_t vi03 = vld1_u8(i03); i03 += 8;
        const int16x8_t vxk01 = vreinterpretq_s16_u16(vsubl_u8(vk01, vkernel_zero_point));
        const int16x8_t vxi03 = vreinterpretq_s16_u16(sub_zero_point(vi03, va_zero_point));
        vacc0_lo = vmlal_s16(vacc0_lo, vget_low_s16(vxk01), vget_low_s16(vxi01));
        vacc0_hi = vmlal_high_s16(vacc0_hi, vxk01, vxi01);
        vacc1_lo = vmlal_s16(vacc1_lo, vget_low_s16(vxk01), vget_low_s16(vxi02));
        vacc1_hi = vmlal_high_s16(vacc1_hi, vxk01, vxi02);
        vacc2_lo = vmlal_s16(vacc2_lo, vget_low_s16(vxk01), vget_low_s16(vxi03));
        vacc2_hi = vmlal_high_s16(vacc2_hi, vxk01, vxi03);

        const uint8x8_t vk11 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
        const uint8x8_t vi13 = vld1_u8(i13); i13 += 8;
        const int16x8_t vxk11 = vreinterpretq_s16_u16(vsubl_u8(vk11, vkernel_zero_point));
        const int16x8_t vxi13 = vreinterpretq_s16_u16(sub_zero_point(vi13, va_zero_point));
        vacc0_lo = vmlal_s16(vacc0_lo, vget_low_s16(vxk11), vget_low_s16(vxi11));
        vacc0_hi = vmlal_high_s16(vacc0_hi, vxk11, vxi11);
        vacc1_lo = vmlal_s16(vacc1_lo, vget_low_s16(vxk11), vget_low_s16(vxi12));
        vacc1_hi = vmlal_high_s16(vacc1_hi, vxk11, vxi12);
        vacc2_lo = vmlal_s16(vacc2_lo, vget_low_s16(vxk11), vget_low_s16(vxi13));
        vacc2_hi = vmlal_high_s16(vacc2_hi, vxk11, vxi13);

        const uint8x8_t vk21 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
        const uint8x8_t vi23 = vld1_u8(i23); i23 += 8;
        const int16x8_t vxk21 = vreinterpretq_s16_u16(vsubl_u8(vk21, vkernel_zero_point));
        const int16x8_t vxi23 = vreinterpretq_s16_u16(sub_zero_point(vi23, va_zero_point));
        vacc0_lo = vmlal_s16(vacc0_lo, vget_low_s16(vxk21), vget_low_s16(vxi21));
        vacc0_hi = vmlal_high_s16(vacc0_hi, vxk21, vxi21);
        vacc1_lo = vmlal_s16(vacc1_lo, vget_low_s16(vxk21), vget_low_s16(vxi22));
        vacc1_hi = vmlal_high_s16(vacc1_hi, vxk21, vxi22);
        vacc2_lo = vmlal_s16(vacc2_lo, vget_low_s16(vxk21), vget_low_s16(vxi23));
        vacc2_hi = vmlal_high_s16(vacc2_hi, vxk21, vxi23);

        const uint8x8_t vk02 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
        const uint8x8_t vi04 = vld1_u8(i04); i04 += 8;
        const int16x8_t vxk02 = vreinterpretq_s16_u16(vsubl_u8(vk02, vkernel_zero_point));
        const int16x8_t vxi04 = vreinterpretq_s16_u16(sub_zero_point(vi04, va_zero_point));
        vacc0_lo = vmlal_s16(vacc0_lo, vget_low_s16(vxk02), vget_low_s16(vxi02));
        vacc0_hi = vmlal_high_s16(vacc0_hi, vxk02, vxi02);
        vacc1_lo = vmlal_s16(vacc1_lo, vget_low_s16(vxk02), vget_low_s16(vxi03));
        vacc1_hi = vmlal_high_s16(vacc1_hi, vxk02, vxi03);
        vacc2_lo = vmlal_s16(vacc2_lo, vget_low_s16(vxk02), vget_low_s16(vxi04));
        vacc2_hi = vmlal_high_s16(vacc2_hi, vxk02, vxi04);

        const uint8x8_t vk12 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
        const uint8x8_t vi14 = vld1_u8(i14); i14 += 8;
        const int16x8_t vxk12 = vreinterpretq_s16_u16(vsubl_u8(vk12, vkernel_zero_point));
        const int16x8_t vxi14 = vreinterpretq_s16_u16(sub_zero_point(vi14, va_zero_point));
        vacc0_lo = vmlal_s16(vacc0_lo, vget_low_s16(vxk12), vget_low_s16(vxi12));
        vacc0_hi = vmlal_high_s16(vacc0_hi, vxk12, vxi12);
        vacc1_lo = vmlal_s16(vacc1_lo, vget_low_s16(vxk12), vget_low_s16(vxi13));
        vacc1_hi = vmlal_high_s16(vacc1_hi, vxk12, vxi13);
        vacc2_lo = vmlal_s16(vacc2_lo, vget_low_s16(vxk12), vget_low_s16(vxi14));
        vacc2_hi = vmlal_high_s16(vacc2_hi, vxk12, vxi14);

        const uint8x8_t vk22 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
        const uint8x8_t vi24 = vld1_u8(i24); i24 += 8;
        const int16x8_t vxk22 = vreinterpretq_s16_u16(vsubl_u8(vk22, vkernel_zero_point));
        const int16x8_t vxi24 = vreinterpretq_s16_u16(sub_zero_point(vi24, va_zero_point));
        vacc0_lo = vmlal_s16(vacc0_lo, vget_low_s16(vxk22), vget_low_s16(vxi22));
        vacc0_hi = vmlal_high_s16(vacc0_hi, vxk22, vxi22);
        vacc1_lo = vmlal_s16(vacc1_lo, vget_low_s16(vxk22), vget_low_s16(vxi23));
        vacc1_hi = vmlal_high_s16(vacc1_hi, vxk22, vxi23);
        vacc2_lo = vmlal_s16(vacc2_lo, vget_low_s16(vxk22), vget_low_s16(vxi24));
        vacc2_hi = vmlal_high_s16(vacc2_hi, vxk22, vxi24);

        vacc0_lo = vqrdmulhq_s32(vacc0_lo, vmultiplier);
        vacc0_hi = vqrdmulhq_s32(vacc0_hi, vmultiplier);
        vacc1_lo = vqrdmulhq_s32(vacc1_lo, vmultiplier);
        vacc1_hi = vqrdmulhq_s32(vacc1_hi, vmultiplier);
        vacc2_lo = vqrdmulhq_s32(vacc2_lo, vmultiplier);
        vacc2_hi = vqrdmulhq_s32(vacc2_hi, vmultiplier);

        const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
        vacc0_lo = vsraq_n_s32(vacc0_lo, vbicq_s32(vacc0_lo, vzero_shift_mask), 31);
        vacc0_hi = vsraq_n_s32(vacc0_hi, vbicq_s32(vacc0_hi, vzero_shift_mask), 31);
        vacc1_lo = vsraq_n_s32(vacc1_lo, vbicq_s32(vacc1_lo, vzero_shift_mask), 31);
        vacc1_hi = vsraq_n_s32(vacc1_hi, vbicq_s32(vacc1_hi, vzero_shift_mask), 31);
        vacc2_lo = vsraq_n_s32(vacc2_lo, vbicq_s32(vacc2_lo, vzero_shift_mask), 31);
        vacc2_hi = vsraq_n_s32(vacc2_hi, vbicq_s32(vacc2_hi, vzero_shift_mask), 31);

        vacc0_lo = vrshlq_s32(vacc0_lo, vright_shift);
        vacc0_hi = vrshlq_s32(vacc0_hi, vright_shift);
        vacc1_lo = vrshlq_s32(vacc1_lo, vright_shift);
        vacc1_hi = vrshlq_s32(vacc1_hi, vright_shift);
        vacc2_lo = vrshlq_s32(vacc2_lo, vright_shift);
        vacc2_hi = vrshlq_s32(vacc2_hi, vright_shift);

        const int16x8_t vacc0 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0_lo), vacc0_hi), voutput_zero_point);
        const int16x8_t vacc1 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1_lo), vacc1_hi), voutput_zero_point);
        const int16x8_t vacc2 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2_lo), vacc2_hi), voutput_zero_point);
        uint8x8_t vout0 = vqmovun_s16(vacc0);
        uint8x8_t vout1 = vqmovun_s16(vacc1);
        uint8x8_t vout2 = vqmovun_s16(vacc2);
        vout0 = vmax_u8(vout0, voutput_min);
        vout1 = vmax_u8(vout1, voutput_min);
        vout2 = vmax_u8(vout2, voutput_min);
        vout0 = vmin_u8(vout0, voutput_max);
        vout1 = vmin_u8(vout1, voutput_max);
        vout2 = vmin_u8(vout2, voutput_max);

        vst1_u8(output0, vout0); output0 += 8;
        vst1_u8(output1, vout1); output1 += 8;
        vst1_u8(output2, vout2); output2 += 8;
      }
      if (c != 0) {
        const size_t c_predecrement = 8 - c;
        const int64x1_t vi_shift = vmov_n_s64(-8 * c_predecrement);
        i00 -= c_predecrement;
        i10 -= c_predecrement;
        i20 -= c_predecrement;
        i01 -= c_predecrement;
        i11 -= c_predecrement;
        i21 -= c_predecrement;
        i02 -= c_predecrement;
        i12 -= c_predecrement;
        i22 -= c_predecrement;
        i03 -= c_predecrement;
        i13 -= c_predecrement;
        i23 -= c_predecrement;
        i04 -= c_predecrement;
        i14 -= c_predecrement;
        i24 -= c_predecrement;

        int32x4_t vacc0_lo = vld1q_s32(w); w = (void*) ((uintptr_t) w + sizeof(int32x4_t));
        int32x4_t vacc0_hi = vld1q_s32(w); w = (void*) ((uintptr_t) w + sizeof(int32x4_t));
        int32x4_t vacc1_lo = vacc0_lo;
        int32x4_t vacc2_lo = vacc0_lo;
        int32x4_t vacc1_hi = vacc0_hi;
        int32x4_t vacc2_hi = vacc0_hi;

        const uint8x8_t vk00 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
        const uint8x8_t vi00 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i00)), vi_shift));
        const uint8x8_t vi01 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i01)), vi_shift));
        const uint8x8_t vi02 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i02)), vi_shift));
        const int16x8_t vxk00 = vreinterpretq_s16_u16(vsubl_u8(vk00, vkernel_zero_point));
        const int16x8_t vxi00 = vreinterpretq_s16_u16(sub_zero_point(vi00, va_zero_point));
        const int16x8_t vxi01 = vreinterpretq_s16_u16(sub_zero_point(vi01, va_zero_point));
        const int16x8_t vxi02 = vreinterpretq_s16_u16(sub_zero_point(vi02, va_zero_point));
        vacc0_lo = vmlal_s16(vacc0_lo, vget_low_s16(vxk00), vget_low_s16(vxi00));
        vacc0_hi = vmlal_high_s16(vacc0_hi, vxk00, vxi00);
        vacc1_lo = vmlal_s16(vacc1_lo, vget_low_s16(vxk00), vget_low_s16(vxi01));
        vacc1_hi = vmlal_high_s16(vacc1_hi, vxk00, vxi01);
        vacc2_lo = vmlal_s16(vacc2_lo, vget_low_s16(vxk00), vget_low_s16(vxi02));
        vacc2_hi = vmlal_high_s16(vacc2_hi, vxk00, vxi02);

        const uint8x8_t vk10 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
        const uint8x8_t vi10 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i10)), vi_shift));
        const uint8x8_t vi11 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i11)), vi_shift));
        const uint8x8_t vi12 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i12)), vi_shift));
        const int16x8_t vxk10 = vreinterpretq_s16_u16(vsubl_u8(vk10, vkernel_zero_point));
        const int16x8_t vxi10 = vreinterpretq_s16_u16(sub_zero_point(vi10, va_zero_point));
        const int16x8_t vxi11 = vreinterpretq_s16_u16(sub_zero_point(vi11, va_zero_point));
        const int16x8_t vxi12 = vreinterpretq_s16_u16(sub_zero_point(vi12, va_zero_point));
        vacc0_lo = vmlal_s16(vacc0_lo, vget_low_s16(vxk10), vget_low_s16(vxi10));
        vacc0_hi = vmlal_high_s16(vacc0_hi, vxk10, vxi10);
        vacc1_lo = vmlal_s16(vacc1_lo, vget_low_s16(vxk10), vget_low_s16(vxi11));
        vacc1_hi = vmlal_high_s16(vacc1_hi, vxk10, vxi11);
        vacc2_lo = vmlal_s16(vacc2_lo, vget_low_s16(vxk10), vget_low_s16(vxi12));
        vacc2_hi = vmlal_high_s16(vacc2_hi, vxk10, vxi12);

        const uint8x8_t vk20 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
        const uint8x8_t vi20 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i20)), vi_shift));
        const uint8x8_t vi21 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i21)), vi_shift));
        const uint8x8_t vi22 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i22)), vi_shift));
        const int16x8_t vxk20 = vreinterpretq_s16_u16(vsubl_u8(vk20, vkernel_zero_point));
        const int16x8_t vxi20 = vreinterpretq_s16_u16(sub_zero_point(vi20, va_zero_point));
        const int16x8_t vxi21 = vreinterpretq_s16_u16(sub_zero_point(vi21, va_zero_point));
        const int16x8_t vxi22 = vreinterpretq_s16_u16(sub_zero_point(vi22, va_zero_point));
        vacc0_lo = vmlal_s16(vacc0_lo, vget_low_s16(vxk20), vget_low_s16(vxi20));
        vacc0_hi = vmlal_high_s16(vacc0_hi, vxk20, vxi20);
        vacc1_lo = vmlal_s16(vacc1_lo, vget_low_s16(vxk20), vget_low_s16(vxi21));
        vacc1_hi = vmlal_high_s16(vacc1_hi, vxk20, vxi21);
        vacc2_lo = vmlal_s16(vacc2_lo, vget_low_s16(vxk20), vget_low_s16(vxi22));
        vacc2_hi = vmlal_high_s16(vacc2_hi, vxk20, vxi22);

        const uint8x8_t vk01 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
        const uint8x8_t vi03 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i03)), vi_shift));
        const int16x8_t vxk01 = vreinterpretq_s16_u16(vsubl_u8(vk01, vkernel_zero_point));
        const int16x8_t vxi03 = vreinterpretq_s16_u16(sub_zero_point(vi03, va_zero_point));
        vacc0_lo = vmlal_s16(vacc0_lo, vget_low_s16(vxk01), vget_low_s16(vxi01));
        vacc0_hi = vmlal_high_s16(vacc0_hi, vxk01, vxi01);
        vacc1_lo = vmlal_s16(vacc1_lo, vget_low_s16(vxk01), vget_low_s16(vxi02));
        vacc1_hi = vmlal_high_s16(vacc1_hi, vxk01, vxi02);
        vacc2_lo = vmlal_s16(vacc2_lo, vget_low_s16(vxk01), vget_low_s16(vxi03));
        vacc2_hi = vmlal_high_s16(vacc2_hi, vxk01, vxi03);

        const uint8x8_t vk11 = vld1_u8(w);  w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
        const uint8x8_t vi13 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i13)), vi_shift));
        const int16x8_t vxk11 = vreinterpretq_s16_u16(vsubl_u8(vk11, vkernel_zero_point));
        const int16x8_t vxi13 = vreinterpretq_s16_u16(sub_zero_point(vi13, va_zero_point));
        vacc0_lo = vmlal_s16(vacc0_lo, vget_low_s16(vxk11), vget_low_s16(vxi11));
        vacc0_hi = vmlal_high_s16(vacc0_hi, vxk11, vxi11);
        vacc1_lo = vmlal_s16(vacc1_lo, vget_low_s16(vxk11), vget_low_s16(vxi12));
        vacc1_hi = vmlal_high_s16(vacc1_hi, vxk11, vxi12);
        vacc2_lo = vmlal_s16(vacc2_lo, vget_low_s16(vxk11), vget_low_s16(vxi13));
        vacc2_hi = vmlal_high_s16(vacc2_hi, vxk11, vxi13);

        const uint8x8_t vk21 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
        const uint8x8_t vi23 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i23)), vi_shift));
        const int16x8_t vxk21 = vreinterpretq_s16_u16(vsubl_u8(vk21, vkernel_zero_point));
        const int16x8_t vxi23 = vreinterpretq_s16_u16(sub_zero_point(vi23, va_zero_point));
        vacc0_lo = vmlal_s16(vacc0_lo, vget_low_s16(vxk21), vget_low_s16(vxi21));
        vacc0_hi = vmlal_high_s16(vacc0_hi, vxk21, vxi21);
        vacc1_lo = vmlal_s16(vacc1_lo, vget_low_s16(vxk21), vget_low_s16(vxi22));
        vacc1_hi = vmlal_high_s16(vacc1_hi, vxk21, vxi22);
        vacc2_lo = vmlal_s16(vacc2_lo, vget_low_s16(vxk21), vget_low_s16(vxi23));
        vacc2_hi = vmlal_high_s16(vacc2_hi, vxk21, vxi23);

        const uint8x8_t vk02 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
        const uint8x8_t vi04 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i04)), vi_shift));
        const int16x8_t vxk02 = vreinterpretq_s16_u16(vsubl_u8(vk02, vkernel_zero_point));
        const int16x8_t vxi04 = vreinterpretq_s16_u16(sub_zero_point(vi04, va_zero_point));
        vacc0_lo = vmlal_s16(vacc0_lo, vget_low_s16(vxk02), vget_low_s16(vxi02));
        vacc0_hi = vmlal_high_s16(vacc0_hi, vxk02, vxi02);
        vacc1_lo = vmlal_s16(vacc1_lo, vget_low_s16(vxk02), vget_low_s16(vxi03));
        vacc1_hi = vmlal_high_s16(vacc1_hi, vxk02, vxi03);
        vacc2_lo = vmlal_s16(vacc2_lo, vget_low_s16(vxk02), vget_low_s16(vxi04));
        vacc2_hi = vmlal_high_s16(vacc2_hi, vxk02, vxi04);

        const uint8x8_t vk12 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
        const uint8x8_t vi14 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i14)), vi_shift));
        const int16x8_t vxk12 = vreinterpretq_s16_u16(vsubl_u8(vk12, vkernel_zero_point));
        const int16x8_t vxi14 = vreinterpretq_s16_u16(sub_zero_point(vi14, va_zero_point));
        vacc0_lo = vmlal_s16(vacc0_lo, vget_low_s16(vxk12), vget_low_s16(vxi12));
        vacc0_hi = vmlal_high_s16(vacc0_hi, vxk12, vxi12);
        vacc1_lo = vmlal_s16(vacc1_lo, vget_low_s16(vxk12), vget_low_s16(vxi13));
        vacc1_hi = vmlal_high_s16(vacc1_hi, vxk12, vxi13);
        vacc2_lo = vmlal_s16(vacc2_lo, vget_low_s16(vxk12), vget_low_s16(vxi14));
        vacc2_hi = vmlal_high_s16(vacc2_hi, vxk12, vxi14);

        const uint8x8_t vk22 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
        const uint8x8_t vi24 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i24)), vi_shift));
        const int16x8_t vxk22 = vreinterpretq_s16_u16(vsubl_u8(vk22, vkernel_zero_point));
        const int16x8_t vxi24 = vreinterpretq_s16_u16(sub_zero_point(vi24, va_zero_point));
        vacc0_lo = vmlal_s16(vacc0_lo, vget_low_s16(vxk22), vget_low_s16(vxi22));
        vacc0_hi = vmlal_high_s16(vacc0_hi, vxk22, vxi22);
        vacc1_lo = vmlal_s16(vacc1_lo, vget_low_s16(vxk22), vget_low_s16(vxi23));
        vacc1_hi = vmlal_high_s16(vacc1_hi, vxk22, vxi23);
        vacc2_lo = vmlal_s16(vacc2_lo, vget_low_s16(vxk22), vget_low_s16(vxi24));
        vacc2_hi = vmlal_high_s16(vacc2_hi, vxk22, vxi24);

        vacc0_lo = vqrdmulhq_s32(vacc0_lo, vmultiplier);
        vacc0_hi = vqrdmulhq_s32(vacc0_hi, vmultiplier);
        vacc1_lo = vqrdmulhq_s32(vacc1_lo, vmultiplier);
        vacc1_hi = vqrdmulhq_s32(vacc1_hi, vmultiplier);
        vacc2_lo = vqrdmulhq_s32(vacc2_lo, vmultiplier);
        vacc2_hi = vqrdmulhq_s32(vacc2_hi, vmultiplier);

        const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
        vacc0_lo = vsraq_n_s32(vacc0_lo, vbicq_s32(vacc0_lo, vzero_shift_mask), 31);
        vacc0_hi = vsraq_n_s32(vacc0_hi, vbicq_s32(vacc0_hi, vzero_shift_mask), 31);
        vacc1_lo = vsraq_n_s32(vacc1_lo, vbicq_s32(vacc1_lo, vzero_shift_mask), 31);
        vacc1_hi = vsraq_n_s32(vacc1_hi, vbicq_s32(vacc1_hi, vzero_shift_mask), 31);
        vacc2_lo = vsraq_n_s32(vacc2_lo, vbicq_s32(vacc2_lo, vzero_shift_mask), 31);
        vacc2_hi = vsraq_n_s32(vacc2_hi, vbicq_s32(vacc2_hi, vzero_shift_mask), 31);

        vacc0_lo = vrshlq_s32(vacc0_lo, vright_shift);
        vacc0_hi = vrshlq_s32(vacc0_hi, vright_shift);
        vacc1_lo = vrshlq_s32(vacc1_lo, vright_shift);
        vacc1_hi = vrshlq_s32(vacc1_hi, vright_shift);
        vacc2_lo = vrshlq_s32(vacc2_lo, vright_shift);
        vacc2_hi = vrshlq_s32(vacc2_hi, vright_shift);

        const int16x8_t vacc0 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0_lo), vacc0_hi), voutput_zero_point);
        const int16x8_t vacc1 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1_lo), vacc1_hi), voutput_zero_point);
        const int16x8_t vacc2 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2_lo), vacc2_hi), voutput_zero_point);
        uint8x8_t vout0 = vqmovun_s16(vacc0);
        uint8x8_t vout1 = vqmovun_s16(vacc1);
        uint8x8_t vout2 = vqmovun_s16(vacc2);
        vout0 = vmax_u8(vout0, voutput_min);
        vout1 = vmax_u8(vout1, voutput_min);
        vout2 = vmax_u8(vout2, voutput_min);
        vout0 = vmin_u8(vout0, voutput_max);
        vout1 = vmin_u8(vout1, voutput_max);
        vout2 = vmin_u8(vout2, voutput_max);

        if (c & 4) {
          vst1_lane_u32(__builtin_assume_aligned(output0, 1), vreinterpret_u32_u8(vout0), 0); output0 += 4;
          vst1_lane_u32(__builtin_assume_aligned(output1, 1), vreinterpret_u32_u8(vout1), 0); output1 += 4;
          vst1_lane_u32(__builtin_assume_aligned(output2, 1), vreinterpret_u32_u8(vout2), 0); output2 += 4;
          vout0 = vext_u8(vout0, vout0, 4);
          vout1 = vext_u8(vout1, vout1, 4);
          vout2 = vext_u8(vout2, vout2, 4);
        }
        if (c & 2) {
          vst1_lane_u16(__builtin_assume_aligned(output0, 1), vreinterpret_u16_u8(vout0), 0); output0 += 2;
          vst1_lane_u16(__builtin_assume_aligned(output1, 1), vreinterpret_u16_u8(vout1), 0); output1 += 2;
          vst1_lane_u16(__builtin_assume_aligned(output2, 1), vreinterpret_u16_u8(vout2), 0); output2 += 2;
          vout0 = vext_u8(vout0, vout0, 2);
          vout1 = vext_u8(vout1, vout1, 2);
          vout2 = vext_u8(vout2, vout2, 2);
        }
        if (c & 1) {
          vst1_lane_u8(__builtin_assume_aligned(output0, 1), vout0, 0); output0++;
          vst1_lane_u8(__builtin_assume_aligned(output1, 1), vout1, 0); output1++;
          vst1_lane_u8(__builtin_assume_aligned(output2, 1), vout2, 0); output2++;
        }
      }

      output = (uint8_t*) ((uintptr_t) output2 + output_increment);
    }
    if (output_width == 0) {
      return;
    }
  }
#endif

  do {
    const uint8_t* i0 = input[0];
    const uint8_t* i1 = input[1];
    const uint8_t* i2 = input[2];
    const uint8_t* i3 = input[3];
    const uint8_t* i4 = input[4];
    const uint8_t* i5 = input[5];
    const uint8_t* i6 = input[6];
    const uint8_t* i7 = input[7];
    const uint8_t* i8 = input[8];

    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 8; c -= 8) {
      int32x4_t vaccX1_lo = vld1q_s32(w); w = (void*) ((uintptr_t) w + sizeof(int32x4_t));
      int32x4_t vaccX1_hi = vld1q_s32(w); w = (void*) ((uintptr_t) w + sizeof(int32x4_t));

      const uint8x8_t vk0 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi0 = vld1_u8(i0); i0 += 8;
      const int16x8_t vxk0 = vreinterpretq_s16_u16(vsubl_u8(vk0, vkernel_zero_point));
      const int16x8_t vxi0 = vreinterpretq_s16_u16(sub_zero_point(vi0, va_zero_point));
      int32x4_t vaccX0_lo = vmull_s16(vget_low_s16(vxk0), vget_low_s16(vxi0));
      int32x4_t vaccX0_hi = vmull_s16(vget_high_s16(vxk0), vget_high_s16(vxi0));

      const uint8x8_t vk1 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi1 = vld1_u8(i1); i1 += 8;
      const int16x8_t vxk1 = vreinterpretq_s16_u16(vsubl_u8(vk1, vkernel_zero_point));
      const int16x8_t vxi1 = vreinterpretq_s16_u16(sub_zero_point(vi1, va_zero_point));
      vaccX1_lo = vmlal_s16(vaccX1_lo, vget_low_s16(vxk1), vget_low_s16(vxi1));
      vaccX1_hi = vmlal_s16(vaccX1_hi, vget_high_s16(vxk1), vget_high_s16(vxi1));

      const uint8x8_t vk2 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi2 = vld1_u8(i2); i2 += 8;
      const int16x8_t vxk2 = vreinterpretq_s16_u16(vsubl_u8(vk2, vkernel_zero_point));
      const int16x8_t vxi2 = vreinterpretq_s16_u16(sub_zero_point(vi2, va_zero_point));
      vaccX0_lo = vmlal_s16(vaccX0_lo, vget_low_s16(vxk2), vget_low_s16(vxi2));
      vaccX0_hi = vmlal_s16(vaccX0_hi, vget_high_s16(vxk2), vget_high_s16(vxi2));

      const uint8x8_t vk3 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi3 = vld1_u8(i3); i3 += 8;
      const int16x8_t vxk3 = vreinterpretq_s16_u16(vsubl_u8(vk3, vkernel_zero_point));
      const int16x8_t vxi3 = vreinterpretq_s16_u16(sub_zero_point(vi3, va_zero_point));
      vaccX1_lo = vmlal_s16(vaccX1_lo, vget_low_s16(vxk3), vget_low_s16(vxi3));
      vaccX1_hi = vmlal_s16(vaccX1_hi, vget_high_s16(vxk3), vget_high_s16(vxi3));

      const uint8x8_t vk4 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi4 = vld1_u8(i4); i4 += 8;
      const int16x8_t vxk4 = vreinterpretq_s16_u16(vsubl_u8(vk4, vkernel_zero_point));
      const int16x8_t vxi4 = vreinterpretq_s16_u16(sub_zero_point(vi4, va_zero_point));
      vaccX0_lo = vmlal_s16(vaccX0_lo, vget_low_s16(vxk4), vget_low_s16(vxi4));
      vaccX0_hi = vmlal_s16(vaccX0_hi, vget_high_s16(vxk4), vget_high_s16(vxi4));

      const uint8x8_t vk5 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi5 = vld1_u8(i5); i5 += 8;
      const int16x8_t vxk5 = vreinterpretq_s16_u16(vsubl_u8(vk5, vkernel_zero_point));
      const int16x8_t vxi5 = vreinterpretq_s16_u16(sub_zero_point(vi5, va_zero_point));
      vaccX1_lo = vmlal_s16(vaccX1_lo, vget_low_s16(vxk5), vget_low_s16(vxi5));
      vaccX1_hi = vmlal_s16(vaccX1_hi, vget_high_s16(vxk5), vget_high_s16(vxi5));

      const uint8x8_t vk6 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi6 = vld1_u8(i6); i6 += 8;
      const int16x8_t vxk6 = vreinterpretq_s16_u16(vsubl_u8(vk6, vkernel_zero_point));
      const int16x8_t vxi6 = vreinterpretq_s16_u16(sub_zero_point(vi6, va_zero_point));
      vaccX0_lo = vmlal_s16(vaccX0_lo, vget_low_s16(vxk6), vget_low_s16(vxi6));
      vaccX0_hi = vmlal_s16(vaccX0_hi, vget_high_s16(vxk6), vget_high_s16(vxi6));

      const uint8x8_t vk7 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi7 = vld1_u8(i7); i7 += 8;
      const int16x8_t vxk7 = vreinterpretq_s16_u16(vsubl_u8(vk7, vkernel_zero_point));
      const int16x8_t vxi7 = vreinterpretq_s16_u16(sub_zero_point(vi7, va_zero_point));
      vaccX1_lo = vmlal_s16(vaccX1_lo, vget_low_s16(vxk7), vget_low_s16(vxi7));
      vaccX1_hi = vmlal_s16(vaccX1_hi, vget_high_s16(vxk7), vget_high_s16(vxi7));

      const uint8x8_t vk8 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi8 = vld1_u8(i8); i8 += 8;
      const int16x8_t vxk8 = vreinterpretq_s16_u16(vsubl_u8(vk8, vkernel_zero_point));
      const int16x8_t vxi8 = vreinterpretq_s16_u16(sub_zero_point(vi8, va_zero_point));
      vaccX0_lo = vmlal_s16(vaccX0_lo, vget_low_s16(vxk8), vget_low_s16(vxi8));
      vaccX0_hi = vmlal_s16(vaccX0_hi, vget_high_s16(vxk8), vget_high_s16(vxi8));

      int32x4_t vacc_lo = vaddq_s32(vaccX0_lo, vaccX1_lo);
      int32x4_t vacc_hi = vaddq_s32(vaccX0_hi, vaccX1_hi);

      vacc_lo = vqrdmulhq_s32(vacc_lo, vmultiplier);
      vacc_hi = vqrdmulhq_s32(vacc_hi, vmultiplier);

      const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
      vacc_lo = vsraq_n_s32(vacc_lo, vbicq_s32(vacc_lo, vzero_shift_mask), 31);
      vacc_hi = vsraq_n_s32(vacc_hi, vbicq_s32(vacc_hi, vzero_shift_mask), 31);

      vacc_lo = vrshlq_s32(vacc_lo, vright_shift);
      vacc_hi = vrshlq_s32(vacc_hi, vright_shift);

#ifdef __aarch64__
      const int16x8_t vacc = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);
#else
      const int16x8_t vacc = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi)), voutput_zero_point);
#endif
      uint8x8_t vout = vqmovun_s16(vacc);
      vout = vmax_u8(vout, voutput_min);
      vout = vmin_u8(vout, voutput_max);

      vst1_u8(output, vout); output += 8;
    }
    if (c != 0) {
      const size_t c_predecrement = 8 - c;
      const int64x1_t vi_shift = vmov_n_s64(-8 * c_predecrement);
      i0 -= c_predecrement;
      i1 -= c_predecrement;
      i2 -= c_predecrement;
      i3 -= c_predecrement;
      i4 -= c_predecrement;
      i5 -= c_predecrement;
      i6 -= c_predecrement;
      i7 -= c_predecrement;
      i8 -= c_predecrement;

      int32x4_t vaccX1_lo = vld1q_s32(w); w = (void*) ((uintptr_t) w + sizeof(int32x4_t));
      int32x4_t vaccX1_hi = vld1q_s32(w); w = (void*) ((uintptr_t) w + sizeof(int32x4_t));

      const uint8x8_t vk0 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi0 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i0)), vi_shift));
      const int16x8_t vxk0 = vreinterpretq_s16_u16(vsubl_u8(vk0, vkernel_zero_point));
      const int16x8_t vxi0 = vreinterpretq_s16_u16(sub_zero_point(vi0, va_zero_point));
      int32x4_t vaccX0_lo = vmull_s16(vget_low_s16(vxk0), vget_low_s16(vxi0));
      int32x4_t vaccX0_hi = vmull_s16(vget_high_s16(vxk0), vget_high_s16(vxi0));

      const uint8x8_t vk1 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi1 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i1)), vi_shift));
      const int16x8_t vxk1 = vreinterpretq_s16_u16(vsubl_u8(vk1, vkernel_zero_point));
      const int16x8_t vxi1 = vreinterpretq_s16_u16(sub_zero_point(vi1, va_zero_point));
      vaccX1_lo = vmlal_s16(vaccX1_lo, vget_low_s16(vxk1), vget_low_s16(vxi1));
      vaccX1_hi = vmlal_s16(vaccX1_hi, vget_high_s16(vxk1), vget_high_s16(vxi1));

      const uint8x8_t vk2 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi2 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i2)), vi_shift));
      const int16x8_t vxk2 = vreinterpretq_s16_u16(vsubl_u8(vk2, vkernel_zero_point));
      const int16x8_t vxi2 = vreinterpretq_s16_u16(sub_zero_point(vi2, va_zero_point));
      vaccX0_lo = vmlal_s16(vaccX0_lo, vget_low_s16(vxk2), vget_low_s16(vxi2));
      vaccX0_hi = vmlal_s16(vaccX0_hi, vget_high_s16(vxk2), vget_high_s16(vxi2));

      const uint8x8_t vk3 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi3 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i3)), vi_shift));
      const int16x8_t vxk3 = vreinterpretq_s16_u16(vsubl_u8(vk3, vkernel_zero_point));
      const int16x8_t vxi3 = vreinterpretq_s16_u16(sub_zero_point(vi3, va_zero_point));
      vaccX1_lo = vmlal_s16(vaccX1_lo, vget_low_s16(vxk3), vget_low_s16(vxi3));
      vaccX1_hi = vmlal_s16(vaccX1_hi, vget_high_s16(vxk3), vget_high_s16(vxi3));

      const uint8x8_t vk4 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi4 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i4)), vi_shift));
      const int16x8_t vxk4 = vreinterpretq_s16_u16(vsubl_u8(vk4, vkernel_zero_point));
      const int16x8_t vxi4 = vreinterpretq_s16_u16(sub_zero_point(vi4, va_zero_point));
      vaccX0_lo = vmlal_s16(vaccX0_lo, vget_low_s16(vxk4), vget_low_s16(vxi4));
      vaccX0_hi = vmlal_s16(vaccX0_hi, vget_high_s16(vxk4), vget_high_s16(vxi4));

      const uint8x8_t vk5 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi5 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i5)), vi_shift));
      const int16x8_t vxk5 = vreinterpretq_s16_u16(vsubl_u8(vk5, vkernel_zero_point));
      const int16x8_t vxi5 = vreinterpretq_s16_u16(sub_zero_point(vi5, va_zero_point));
      vaccX1_lo = vmlal_s16(vaccX1_lo, vget_low_s16(vxk5), vget_low_s16(vxi5));
      vaccX1_hi = vmlal_s16(vaccX1_hi, vget_high_s16(vxk5), vget_high_s16(vxi5));

      const uint8x8_t vk6 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi6 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i6)), vi_shift));
      const int16x8_t vxk6 = vreinterpretq_s16_u16(vsubl_u8(vk6, vkernel_zero_point));
      const int16x8_t vxi6 = vreinterpretq_s16_u16(sub_zero_point(vi6, va_zero_point));
      vaccX0_lo = vmlal_s16(vaccX0_lo, vget_low_s16(vxk6), vget_low_s16(vxi6));
      vaccX0_hi = vmlal_s16(vaccX0_hi, vget_high_s16(vxk6), vget_high_s16(vxi6));

      const uint8x8_t vk7 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi7 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i7)), vi_shift));
      const int16x8_t vxk7 = vreinterpretq_s16_u16(vsubl_u8(vk7, vkernel_zero_point));
      const int16x8_t vxi7 = vreinterpretq_s16_u16(sub_zero_point(vi7, va_zero_point));
      vaccX1_lo = vmlal_s16(vaccX1_lo, vget_low_s16(vxk7), vget_low_s16(vxi7));
      vaccX1_hi = vmlal_s16(vaccX1_hi, vget_high_s16(vxk7), vget_high_s16(vxi7));

      const uint8x8_t vk8 = vld1_u8(w);
      const uint8x8_t vi8 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i8)), vi_shift));
      const int16x8_t vxk8 = vreinterpretq_s16_u16(vsubl_u8(vk8, vkernel_zero_point));
      const int16x8_t vxi8 = vreinterpretq_s16_u16(sub_zero_point(vi8, va_zero_point));
      vaccX0_lo = vmlal_s16(vaccX0_lo, vget_low_s16(vxk8), vget_low_s16(vxi8));
      vaccX0_hi = vmlal_s16(vaccX0_hi, vget_high_s16(vxk8), vget_high_s16(vxi8));

      int32x4_t vacc_lo = vaddq_s32(vaccX0_lo, vaccX1_lo);
      int32x4_t vacc_hi = vaddq_s32(vaccX0_hi, vaccX1_hi);

      vacc_lo = vqrdmulhq_s32(vacc_lo, vmultiplier);
      vacc_hi = vqrdmulhq_s32(vacc_hi, vmultiplier);

      const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
      vacc_lo = vsraq_n_s32(vacc_lo, vbicq_s32(vacc_lo, vzero_shift_mask), 31);
      vacc_hi = vsraq_n_s32(vacc_hi, vbicq_s32(vacc_hi, vzero_shift_mask), 31);

      vacc_lo = vrshlq_s32(vacc_lo, vright_shift);
      vacc_hi = vrshlq_s32(vacc_hi, vright_shift);

#ifdef __aarch64__
      const int16x8_t vacc = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);
#else
      const int16x8_t vacc = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi)), voutput_zero_point);
#endif
      uint8x8_t vout = vqmovun_s16(vacc);
      vout = vmax_u8(vout, voutput_min);
      vout = vmin_u8(vout, voutput_max);

      if (c & 4) {
        vst1_lane_u32(__builtin_assume_aligned(output, 1), vreinterpret_u32_u8(vout), 0); output += 4;
        vout = vext_u8(vout, vout, 4);
      }
      if (c & 2) {
        vst1_lane_u16(__builtin_assume_aligned(output, 1), vreinterpret_u16_u8(vout), 0); output += 2;
        vout = vext_u8(vout, vout, 2);
      }
      if (c & 1) {
        vst1_lane_u8(__builtin_assume_aligned(output, 1), vout, 0); output++;
      }
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
