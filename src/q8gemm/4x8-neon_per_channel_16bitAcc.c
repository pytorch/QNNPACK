/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>

#include <qnnpack/q8gemm.h>
//#include <stdio.h>

// Uncomment one of the following:
// #define OPPORTUNISTIC_ACC_0
//#define OPPORTUNISTIC_ACC_1
// #define OPPORTUNISTIC_ACC_2
//#define FIXED_ACC


// #ifdef OPPORTUNISTIC_ACC_0
// // This version does "opportunistic" accumulation to 16bit: accumulate without further check just until
// // requantization. Works only for 32bit AARCH
// void q8gemm_ukernel_4x8__neon_per_channel_16bitAcc(
//     size_t mr,
//     size_t nr,
//     size_t k,
//     const uint8_t* restrict a,
//     size_t a_stride,
//     const void* restrict w,
//     uint8_t* restrict c,
//     size_t c_stride,
//     const union qnnp_conv_quantization_params quantization_params[restrict static 1],
//     size_t kernel_quantization_params_offset)
// {
//   // Read the bias packed in the first 8 (32-bit) words
//   int32x4_t vacc0123_bias = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 16);
//   int32x4_t vacc4567_bias = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 16);
//
//   // init accumulators with zeros
//   // const int32_t z = 0;
//   // int32x4_t vacc0x0123 = vld1q_dup_s32(&z);
//   // int32x4_t vacc0x4567 = vld1q_dup_s32(&z);
//   // int32x4_t vacc1x0123 = vacc0x0123;
//   // int32x4_t vacc1x4567 = vacc0x4567;
//   // int32x4_t vacc2x0123 = vacc0x0123;
//   // int32x4_t vacc2x4567 = vacc0x4567;
//   // int32x4_t vacc3x0123 = vacc0x0123;
//   // int32x4_t vacc3x4567 = vacc0x4567;
//
//   const size_t kc = k;
//   const int16_t z16 = 0;
//   int16x8_t vacc0x01234567 = vld1q_dup_s16(&z16);
//   int16x8_t vacc1x01234567 = vacc0x01234567;
//   int16x8_t vacc2x01234567 = vacc0x01234567;
//   int16x8_t vacc3x01234567 = vacc0x01234567;
//
//   const uint8_t* a0 = a;
//   const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
//   if (mr < 2) {
//     a1 = a0;
//   }
//   const uint8_t* a2 = (const uint8_t*) ((uintptr_t) a1 + a_stride);
//   if (mr <= 2) {
//     a2 = a1;
//   }
//   const uint8_t* a3 = (const uint8_t*) ((uintptr_t) a2 + a_stride);
//   if (mr != 4) {
//     a3 = a2;
//   }
//
//   int32_t aa, ww, acc, res;
//
//   const uint8x8_t vb_zero_point = vld1_u8((const uint8_t*) &quantization_params->neon.kernel_zero_point_v[kernel_quantization_params_offset]);
//   for (; k >= 8; k -= 8) {
//     const uint8x8_t va0 = vld1_u8(a0); a0 += 8;
//     const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0));
//     const uint8x8_t va1 = vld1_u8(a1); a1 += 8;
//     const int16x8_t vxa1 = vreinterpretq_s16_u16(vmovl_u8(va1));
//     const uint8x8_t va2 = vld1_u8(a2); a2 += 8;
//     const int16x8_t vxa2 = vreinterpretq_s16_u16(vmovl_u8(va2));
//     const uint8x8_t va3 = vld1_u8(a3); a3 += 8;
//     const int16x8_t vxa3 = vreinterpretq_s16_u16(vmovl_u8(va3));
//
//     const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));
//
//     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
//     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
//     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
//     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
//     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
//     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
//     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
//     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
//
//     aa = vgetq_lane_s16(vxa0, 0);
//     for (size_t i = 0; i<8; ++i) {
//         acc = vgetq_lane_s16(vacc0x01234567, i);
//         ww = vgetq_lane_s16(vxb01234567c0, i);
//         res = acc + (aa * ww);
//         if (res > 32767 || res < -32768) {
//           // printf("acc overflow: %ld , kc == %z , k == %z \n", res, kc, k);
//           printf("acc overflow: %ld  acc = %ld  aa = %ld  ww = %ld   kc = %ld  \n", res, acc, aa, ww, kc);
//         }
//     }
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c0, vget_low_s16(vxa0), 0);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c0, vget_low_s16(vxa1), 0);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c0, vget_low_s16(vxa2), 0);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c0, vget_low_s16(vxa3), 0);
//
//     // int16_t a16 = vget_lane_s16(vget_low_s16(vacc0x01234567), 0);
//     // if (a16 > 28000) {
//     //   printf("acc overflow: %i (kc: %z , k = %u)\n", a16, kc, k);
//     // }
//
//     // int16_t a16 = vget_lane_s16(vget_low_s16(vacc0x01234567), 0);
//     // int32_t a32 = (int32_t)a16;
//     // int32_t b32 = vgetq_lane_s32(vacc0x0123, 0);
//     // if ((a32 - b32)*(a32 - b32) > 0.02) {
//     //   printf("accumulator overflow: %u , %u \n", a32, b32);
//     // }
//
//     const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));
//
//     aa = vgetq_lane_s16(vxa0, 1);
//     for (size_t i = 0; i<8; ++i) {
//         acc = vgetq_lane_s16(vacc0x01234567, i);
//         ww = vgetq_lane_s16(vxb01234567c1, i);
//         res = acc + (aa * ww);
//         if (res > 32767 || res < -32768) {
//           // printf("acc overflow: %ld , kc == %z , k == %z \n", res, kc, k);
//           printf("acc overflow: %ld  acc = %ld  aa = %ld  ww = %ld   kc = %ld  \n", res, acc, aa, ww, kc);
//         }
//     }
//
//     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
//     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
//     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
//     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
//     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
//     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
//     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
//     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c1, vget_low_s16(vxa0), 1);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c1, vget_low_s16(vxa1), 1);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c1, vget_low_s16(vxa2), 1);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c1, vget_low_s16(vxa3), 1);
//
//     const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));
//
//     aa = vgetq_lane_s16(vxa0, 2);
//     for (size_t i = 0; i<8; ++i) {
//         acc = vgetq_lane_s16(vacc0x01234567, i);
//         ww = vgetq_lane_s16(vxb01234567c2, i);
//         res = acc + (aa * ww);
//         if (res > 32767 || res < -32768) {
//           // printf("acc overflow: %ld , kc == %z , k == %z \n", res, kc, k);
//           printf("acc overflow: %ld  acc = %ld  aa = %ld  ww = %ld   kc = %ld  \n", res, acc, aa, ww, kc);
//         }
//     }
//
//     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
//     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
//     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
//     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
//     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
//     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
//     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
//     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c2, vget_low_s16(vxa0), 2);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c2, vget_low_s16(vxa1), 2);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c2, vget_low_s16(vxa2), 2);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c2, vget_low_s16(vxa3), 2);
//
//     const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));
//
//     aa = vgetq_lane_s16(vxa0, 3);
//     for (size_t i = 0; i<8; ++i) {
//         acc = vgetq_lane_s16(vacc0x01234567, i);
//         ww = vgetq_lane_s16(vxb01234567c3, i);
//         res = acc + (aa * ww);
//         if (res > 32767 || res < -32768) {
//           // printf("acc overflow: %ld , kc == %z , k == %z \n", res, kc, k);
//           printf("acc overflow: %ld  acc = %ld  aa = %ld  ww = %ld   kc = %ld  \n", res, acc, aa, ww, kc);
//         }
//     }
//
//     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
//     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
//     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
//     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
//     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
//     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
//     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
//     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c3, vget_low_s16(vxa0), 3);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c3, vget_low_s16(vxa1), 3);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c3, vget_low_s16(vxa2), 3);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c3, vget_low_s16(vxa3), 3);
//
//     const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));
//
//     aa = vgetq_lane_s16(vxa0, 4);
//     for (size_t i = 0; i<8; ++i) {
//         acc = vgetq_lane_s16(vacc0x01234567, i);
//         ww = vgetq_lane_s16(vxb01234567c4, i);
//         res = acc + (aa * ww);
//         if (res > 32767 || res < -32768) {
//           // printf("acc overflow: %ld , kc == %z , k == %z \n", res, kc, k);
//           printf("acc overflow: %ld  acc = %ld  aa = %ld  ww = %ld   kc = %ld  \n", res, acc, aa, ww, kc);
//         }
//     }
//
//     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
//     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
//     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
//     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
//     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
//     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
//     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
//     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c4, vget_high_s16(vxa0), 0);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c4, vget_high_s16(vxa1), 0);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c4, vget_high_s16(vxa2), 0);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c4, vget_high_s16(vxa3), 0);
//
//     const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));
//
//     aa = vgetq_lane_s16(vxa0, 5);
//     for (size_t i = 0; i<8; ++i) {
//         acc = vgetq_lane_s16(vacc0x01234567, i);
//         ww = vgetq_lane_s16(vxb01234567c5, i);
//         res = acc + (aa * ww);
//         if (res > 32767 || res < -32768) {
//           // printf("acc overflow: %ld , kc == %z , k == %z \n", res, kc, k);
//           printf("acc overflow: %ld  acc = %ld  aa = %ld  ww = %ld   kc = %ld  \n", res, acc, aa, ww, kc);
//         }
//     }
//
//     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
//     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
//     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
//     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
//     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
//     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
//     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
//     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c5, vget_high_s16(vxa0), 1);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c5, vget_high_s16(vxa1), 1);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c5, vget_high_s16(vxa2), 1);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c5, vget_high_s16(vxa3), 1);
//
//     const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));
//
//     aa = vgetq_lane_s16(vxa0, 6);
//     for (size_t i = 0; i<8; ++i) {
//         acc = vgetq_lane_s16(vacc0x01234567, i);
//         ww = vgetq_lane_s16(vxb01234567c6, i);
//         res = acc + (aa * ww);
//         if (res > 32767 || res < -32768) {
//           // printf("acc overflow: %ld , kc == %z , k == %z \n", res, kc, k);
//           printf("acc overflow: %ld  acc = %ld  aa = %ld  ww = %ld   kc = %ld  \n", res, acc, aa, ww, kc);
//         }
//     }
//
//     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
//     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
//     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
//     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
//     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
//     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
//     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
//     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c6, vget_high_s16(vxa0), 2);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c6, vget_high_s16(vxa1), 2);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c6, vget_high_s16(vxa2), 2);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c6, vget_high_s16(vxa3), 2);
//
//     const uint8x8_t vb01234567c7 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c7 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c7, vb_zero_point));
//
//     aa = vgetq_lane_s16(vxa0, 7);
//     for (size_t i = 0; i<8; ++i) {
//         acc = vgetq_lane_s16(vacc0x01234567, i);
//         ww = vgetq_lane_s16(vxb01234567c7, i);
//         res = acc + (aa * ww);
//         if (res > 32767 || res < -32768) {
//           // printf("acc overflow: %ld , kc == %z , k == %z \n", res, kc, k);
//           printf("acc overflow: %ld  acc = %ld  aa = %ld  ww = %ld   kc = %ld  \n", res, acc, aa, ww, kc);
//         }
//     }
//
//     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
//     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
//     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
//     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
//     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
//     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
//     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
//     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c7, vget_high_s16(vxa0), 3);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c7, vget_high_s16(vxa1), 3);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c7, vget_high_s16(vxa2), 3);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c7, vget_high_s16(vxa3), 3);
//   }
//   if (k != 0) {
//     const size_t a_predecrement = 8 - k;
//     const int64x1_t va_shift = vmov_n_s64(-8 * a_predecrement);
//     const uint8x8_t va0 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a0 - a_predecrement)), va_shift));
//     const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0));
//     const uint8x8_t va1 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a1 - a_predecrement)), va_shift));
//     const int16x8_t vxa1 = vreinterpretq_s16_u16(vmovl_u8(va1));
//     const uint8x8_t va2 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a2 - a_predecrement)), va_shift));
//     const int16x8_t vxa2 = vreinterpretq_s16_u16(vmovl_u8(va2));
//     const uint8x8_t va3 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a3 - a_predecrement)), va_shift));
//     const int16x8_t vxa3 = vreinterpretq_s16_u16(vmovl_u8(va3));
//
//     const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));
//
//     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
//     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
//     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
//     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
//     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
//     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
//     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
//     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c0, vget_low_s16(vxa0), 0);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c0, vget_low_s16(vxa1), 0);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c0, vget_low_s16(vxa2), 0);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c0, vget_low_s16(vxa3), 0);
//
//     if (k >= 2) {
//       const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//       const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));
//
//       // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
//       // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
//       // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
//       // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
//       // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
//       // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
//       // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
//       // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
//
//       vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c1, vget_low_s16(vxa0), 1);
//       vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c1, vget_low_s16(vxa1), 1);
//       vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c1, vget_low_s16(vxa2), 1);
//       vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c1, vget_low_s16(vxa3), 1);
//
//       if (k >= 3) {
//         const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//         const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));
//
//         // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
//         // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
//         // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
//         // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
//         // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
//         // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
//         // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
//         // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
//
//         vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c2, vget_low_s16(vxa0), 2);
//         vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c2, vget_low_s16(vxa1), 2);
//         vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c2, vget_low_s16(vxa2), 2);
//         vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c2, vget_low_s16(vxa3), 2);
//
//         if (k >= 4) {
//           const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//           const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));
//
//           // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
//           // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
//           // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
//           // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
//           // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
//           // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
//           // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
//           // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
//
//           vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c3, vget_low_s16(vxa0), 3);
//           vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c3, vget_low_s16(vxa1), 3);
//           vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c3, vget_low_s16(vxa2), 3);
//           vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c3, vget_low_s16(vxa3), 3);
//
//           if (k >= 5) {
//             const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//             const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));
//
//             // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
//             // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
//             // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
//             // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
//             // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
//             // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
//             // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
//             // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
//
//             vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c4, vget_high_s16(vxa0), 0);
//             vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c4, vget_high_s16(vxa1), 0);
//             vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c4, vget_high_s16(vxa2), 0);
//             vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c4, vget_high_s16(vxa3), 0);
//
//             if (k >= 6) {
//               const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//               const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));
//
//               // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
//               // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
//               // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
//               // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
//               // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
//               // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
//               // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
//               // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
//
//               vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c5, vget_high_s16(vxa0), 1);
//               vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c5, vget_high_s16(vxa1), 1);
//               vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c5, vget_high_s16(vxa2), 1);
//               vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c5, vget_high_s16(vxa3), 1);
//
//               if (k >= 7) {
//                 const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//                 const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));
//
//                 // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
//                 // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
//                 // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
//                 // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
//                 // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
//                 // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
//                 // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
//                 // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
//
//                 vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c6, vget_high_s16(vxa0), 2);
//                 vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c6, vget_high_s16(vxa1), 2);
//                 vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c6, vget_high_s16(vxa2), 2);
//                 vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c6, vget_high_s16(vxa3), 2);
//               }
//             }
//           }
//         }
//       }
//     }
//   }
//
//   // printf("accumulators:  %i , %i , %i , %i , %i , %i , %i , %i\n",
//   // vgetq_lane_s16(vacc0x01234567, 0), vgetq_lane_s16(vacc0x01234567, 1), vgetq_lane_s16(vacc0x01234567, 2), vgetq_lane_s16(vacc0x01234567, 3),
//   // vgetq_lane_s16(vacc0x01234567, 4), vgetq_lane_s16(vacc0x01234567, 5), vgetq_lane_s16(vacc0x01234567, 6), vgetq_lane_s16(vacc0x01234567, 7));
//
//
//   int32x4_t vacc0x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc0x01234567)), vacc0123_bias);
//   int32x4_t vacc0x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc0x01234567)), vacc4567_bias);
//   int32x4_t vacc1x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc1x01234567)), vacc0123_bias);
//   int32x4_t vacc1x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc1x01234567)), vacc4567_bias);
//   int32x4_t vacc2x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc2x01234567)), vacc0123_bias);
//   int32x4_t vacc2x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc2x01234567)), vacc4567_bias);
//   int32x4_t vacc3x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc3x01234567)), vacc0123_bias);
//   int32x4_t vacc3x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc3x01234567)), vacc4567_bias);
//
//
//   // vacc0x0123 = vaddq_s32(vacc0x0123, vacc0123_bias);
//   // vacc0x4567 = vaddq_s32(vacc0x4567, vacc4567_bias);
//   // vacc1x0123 = vaddq_s32(vacc1x0123, vacc0123_bias);
//   // vacc1x4567 = vaddq_s32(vacc1x4567, vacc4567_bias);
//   // vacc2x0123 = vaddq_s32(vacc2x0123, vacc0123_bias);
//   // vacc2x4567 = vaddq_s32(vacc2x4567, vacc4567_bias);
//   // vacc3x0123 = vaddq_s32(vacc3x0123, vacc0123_bias);
//   // vacc3x4567 = vaddq_s32(vacc3x4567, vacc4567_bias);
//
//   // printf("accumulators:  %i , %i , %i , %i , %i , %i , %i , %i\n",
//   // vgetq_lane_s32(vacc0x0123, 0), vgetq_lane_s32(vacc0x0123, 1), vgetq_lane_s32(vacc0x0123, 2), vgetq_lane_s32(vacc0x0123, 3),
//   // vgetq_lane_s32(vacc0x4567, 0), vgetq_lane_s32(vacc0x4567, 1), vgetq_lane_s32(vacc0x4567, 2), vgetq_lane_s32(vacc0x4567, 3));
//
//   // printf("accumulators:  %i , %i , %i , %i , %i , %i , %i , %i\n",
//   // vgetq_lane_s32(vacc0123_bias, 0), vgetq_lane_s32(vacc0123_bias, 1), vgetq_lane_s32(vacc0123_bias, 2), vgetq_lane_s32(vacc0123_bias, 3),
//   // vgetq_lane_s32(vacc4567_bias, 0), vgetq_lane_s32(vacc4567_bias, 1), vgetq_lane_s32(vacc4567_bias, 2), vgetq_lane_s32(vacc4567_bias, 3));
//
//
//
//   const int32x4_t vmultiplier0x0123 = vld1q_s32(&quantization_params->neon.multiplier_v[kernel_quantization_params_offset]);
//   const int32x4_t vmultiplier0x4567 = vld1q_s32(&quantization_params->neon.multiplier_v[kernel_quantization_params_offset + 4]);
//   vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier0x0123);
//   vacc0x4567 = vqrdmulhq_s32(vacc0x4567, vmultiplier0x4567);
//   vacc1x0123 = vqrdmulhq_s32(vacc1x0123, vmultiplier0x0123);
//   vacc1x4567 = vqrdmulhq_s32(vacc1x4567, vmultiplier0x4567);
//   vacc2x0123 = vqrdmulhq_s32(vacc2x0123, vmultiplier0x0123);
//   vacc2x4567 = vqrdmulhq_s32(vacc2x4567, vmultiplier0x4567);
//   vacc3x0123 = vqrdmulhq_s32(vacc3x0123, vmultiplier0x0123);
//   vacc3x4567 = vqrdmulhq_s32(vacc3x4567, vmultiplier0x4567);
//
//   const int32x4_t vright_shift_0x0123 = vld1q_s32(&quantization_params->neon.right_shift_v[kernel_quantization_params_offset]);
//   const int32x4_t vright_shift_0x4567 = vld1q_s32(&quantization_params->neon.right_shift_v[kernel_quantization_params_offset + 4]);
//   const int32x4_t vzero_shift_mask_0x0123 = vreinterpretq_s32_u32(vceqq_s32(vright_shift_0x0123, vmovq_n_s32(0)));
//   const int32x4_t vzero_shift_mask_0x4567 = vreinterpretq_s32_u32(vceqq_s32(vright_shift_0x4567, vmovq_n_s32(0)));
//   vacc0x0123 = vsraq_n_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask_0x0123), 31);
//   vacc0x4567 = vsraq_n_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask_0x4567), 31);
//   vacc1x0123 = vsraq_n_s32(vacc1x0123, vbicq_s32(vacc1x0123, vzero_shift_mask_0x0123), 31);
//   vacc1x4567 = vsraq_n_s32(vacc1x4567, vbicq_s32(vacc1x4567, vzero_shift_mask_0x4567), 31);
//   vacc2x0123 = vsraq_n_s32(vacc2x0123, vbicq_s32(vacc2x0123, vzero_shift_mask_0x0123), 31);
//   vacc2x4567 = vsraq_n_s32(vacc2x4567, vbicq_s32(vacc2x4567, vzero_shift_mask_0x4567), 31);
//   vacc3x0123 = vsraq_n_s32(vacc3x0123, vbicq_s32(vacc3x0123, vzero_shift_mask_0x0123), 31);
//   vacc3x4567 = vsraq_n_s32(vacc3x4567, vbicq_s32(vacc3x4567, vzero_shift_mask_0x4567), 31);
//
//   vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift_0x0123);
//   vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift_0x4567);
//   vacc1x0123 = vrshlq_s32(vacc1x0123, vright_shift_0x0123);
//   vacc1x4567 = vrshlq_s32(vacc1x4567, vright_shift_0x4567);
//   vacc2x0123 = vrshlq_s32(vacc2x0123, vright_shift_0x0123);
//   vacc2x4567 = vrshlq_s32(vacc2x4567, vright_shift_0x4567);
//   vacc3x0123 = vrshlq_s32(vacc3x0123, vright_shift_0x0123);
//   vacc3x4567 = vrshlq_s32(vacc3x4567, vright_shift_0x4567);
//
//   const int16x8_t voutput_zero_point = vld1q_dup_s16(&quantization_params->neon.output_zero_point);
// // #ifdef __aarch64__
// //   const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
// //   const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
// //   const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
// //   const int16x8_t vacc3x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);
// //
// //   uint8x16_t vout0x01234567_1x01234567 = vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc1x01234567);
// //   uint8x16_t vout2x01234567_3x01234567 = vqmovun_high_s16(vqmovun_s16(vacc2x01234567), vacc3x01234567);
// // #else
//   const int16x8_t vacc0x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
//   const int16x8_t vacc1x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
//   const int16x8_t vacc2x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
//   const int16x8_t vacc3x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);
//
//   uint8x16_t vout0x01234567_1x01234567 = vcombine_u8(vqmovun_s16(vacc0x01234567_f), vqmovun_s16(vacc1x01234567_f));
//   uint8x16_t vout2x01234567_3x01234567 = vcombine_u8(vqmovun_s16(vacc2x01234567_f), vqmovun_s16(vacc3x01234567_f));
// // #endif
//   const uint8x16_t voutput_min = vld1q_dup_u8(&quantization_params->neon.output_min);
//   const uint8x16_t voutput_max = vld1q_dup_u8(&quantization_params->neon.output_max);
//
//   vout0x01234567_1x01234567 = vmaxq_u8(vout0x01234567_1x01234567, voutput_min);
//   vout2x01234567_3x01234567 = vmaxq_u8(vout2x01234567_3x01234567, voutput_min);
//   vout0x01234567_1x01234567 = vminq_u8(vout0x01234567_1x01234567, voutput_max);
//   vout2x01234567_3x01234567 = vminq_u8(vout2x01234567_3x01234567, voutput_max);
//
//   uint8_t* c0 = c;
//   uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + c_stride);
//   if (mr < 2) {
//     c1 = c0;
//   }
//   uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + c_stride);
//   if (mr <= 2) {
//     c2 = c1;
//   }
//   uint8_t* c3 = (uint8_t*) ((uintptr_t) c2 + c_stride);
//   if (mr != 4) {
//     c3 = c2;
//   }
//   if (nr == 8) {
//     vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567));
//     vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567));
//     vst1_u8(c2, vget_low_u8(vout2x01234567_3x01234567));
//     vst1_u8(c3, vget_high_u8(vout2x01234567_3x01234567));
//   } else {
//     if (nr >= 4) {
//       vst1q_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpretq_u32_u8(vout0x01234567_1x01234567), 0); c0 += 4;
//       vst1q_lane_u32(__builtin_assume_aligned(c1, 1), vreinterpretq_u32_u8(vout0x01234567_1x01234567), 2); c1 += 4;
//       vst1q_lane_u32(__builtin_assume_aligned(c2, 1), vreinterpretq_u32_u8(vout2x01234567_3x01234567), 0); c2 += 4;
//       vst1q_lane_u32(__builtin_assume_aligned(c3, 1), vreinterpretq_u32_u8(vout2x01234567_3x01234567), 2); c3 += 4;
//       vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
//       vout2x01234567_3x01234567 = vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
//       nr -= 4;
//     }
//     if (nr >= 2) {
//       vst1q_lane_u16(__builtin_assume_aligned(c0, 1), vreinterpretq_u16_u8(vout0x01234567_1x01234567), 0); c0 += 2;
//       vst1q_lane_u16(__builtin_assume_aligned(c1, 1), vreinterpretq_u16_u8(vout0x01234567_1x01234567), 4); c1 += 2;
//       vst1q_lane_u16(__builtin_assume_aligned(c2, 1), vreinterpretq_u16_u8(vout2x01234567_3x01234567), 0); c2 += 2;
//       vst1q_lane_u16(__builtin_assume_aligned(c3, 1), vreinterpretq_u16_u8(vout2x01234567_3x01234567), 4); c3 += 2;
//       vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
//       vout2x01234567_3x01234567 = vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
//       nr -= 2;
//     }
//     if (nr != 0) {
//       vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
//       vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
//       vst1q_lane_u8(c2, vout2x01234567_3x01234567, 0);
//       vst1q_lane_u8(c3, vout2x01234567_3x01234567, 8);
//     }
//   }
// }
// #elif defined OPPORTUNISTIC_ACC_1
//
// // This version does "opportunistic" accumulation to 16bit: accumulate without further check just until
// // requantization. Works only for 32bit AARCH.
// // This does not promote {a, w} values to 16bit prior to multiplication.
// // However this loads a single value at a time and increase the pointer --> x8 adds per loop instead of 1.
// void q8gemm_ukernel_4x8__neon_per_channel_16bitAcc(
//     size_t mr,
//     size_t nr,
//     size_t k,
//     const uint8_t* restrict a,
//     size_t a_stride,
//     const void* restrict w,
//     uint8_t* restrict c,
//     size_t c_stride,
//     const union qnnp_conv_quantization_params quantization_params[restrict static 1],
//     size_t kernel_quantization_params_offset)
// {
//   // Read the bias packed in the first 8 (32-bit) words
//   int32x4_t vacc0123_bias = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 16);
//   int32x4_t vacc4567_bias = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 16);
//
//
//   const size_t kc = k;
//   const int16_t z16 = 0;
//   int16x8_t vacc0x01234567 = vld1q_dup_s16(&z16);
//   int16x8_t vacc1x01234567 = vacc0x01234567;
//   int16x8_t vacc2x01234567 = vacc0x01234567;
//   int16x8_t vacc3x01234567 = vacc0x01234567;
//
//   const uint8_t* a0 = a;
//   const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
//   if (mr < 2) {
//     a1 = a0;
//   }
//   const uint8_t* a2 = (const uint8_t*) ((uintptr_t) a1 + a_stride);
//   if (mr <= 2) {
//     a2 = a1;
//   }
//   const uint8_t* a3 = (const uint8_t*) ((uintptr_t) a2 + a_stride);
//   if (mr != 4) {
//     a3 = a2;
//   }
//
//   const uint8x8_t vb_zero_point_u = vld1_u8((const uint8_t*) &quantization_params->neon.kernel_zero_point_v[kernel_quantization_params_offset]);
//   const int8x8_t vb_zero_point = vreinterpret_s8_u8(vb_zero_point_u);
//   for (; k >= 8; k -= 8) {
//     // const uint8x8_t va0 = vld1_u8(a0); a0 += 8;
//     // const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0));
//     // // const int8x8_t vxa0 = vreinterpretq_s8_u8(va0);
//     // const uint8x8_t va1 = vld1_u8(a1); a1 += 8;
//     // const int16x8_t vxa1 = vreinterpretq_s16_u16(vmovl_u8(va1));
//     // // const int8x8_t vxa1 = vreinterpretq_s8_u8(va1);
//     // const uint8x8_t va2 = vld1_u8(a2); a2 += 8;
//     // const int16x8_t vxa2 = vreinterpretq_s16_u16(vmovl_u8(va2));
//     // // const int8x8_t vxa2 = vreinterpretq_s8_u8(va2);
//     // const uint8x8_t va3 = vld1_u8(a3); a3 += 8;
//     // const int16x8_t vxa3 = vreinterpretq_s16_u16(vmovl_u8(va3));
//     // // const int8x8_t vxa3 = vreinterpretq_s8_u8(va3);
//
//     const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     // const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));
//     const int8x8_t vxb01234567c0 = vqsub_s8(vreinterpret_s8_u8(vb01234567c0), vb_zero_point);
//
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c0, vreinterpret_s8_u8(vld1_dup_u8(a0))); ++a0;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c0, vreinterpret_s8_u8(vld1_dup_u8(a1))); ++a1;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c0, vreinterpret_s8_u8(vld1_dup_u8(a2))); ++a2;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c0, vreinterpret_s8_u8(vld1_dup_u8(a3))); ++a3;
//
//     const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     // const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));
//     const int8x8_t vxb01234567c1 = vqsub_s8(vreinterpret_s8_u8(vb01234567c1), vb_zero_point);
//
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c1, vreinterpret_s8_u8(vld1_dup_u8(a0))); ++a0;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c1, vreinterpret_s8_u8(vld1_dup_u8(a1))); ++a1;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c1, vreinterpret_s8_u8(vld1_dup_u8(a2))); ++a2;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c1, vreinterpret_s8_u8(vld1_dup_u8(a3))); ++a3;
//
//     const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     // const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));
//     const int8x8_t vxb01234567c2 = vqsub_s8(vreinterpret_s8_u8(vb01234567c2), vb_zero_point);
//
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c2, vreinterpret_s8_u8(vld1_dup_u8(a0))); ++a0;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c2, vreinterpret_s8_u8(vld1_dup_u8(a1))); ++a1;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c2, vreinterpret_s8_u8(vld1_dup_u8(a2))); ++a2;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c2, vreinterpret_s8_u8(vld1_dup_u8(a3))); ++a3;
//
//     const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     // const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));
//     const int8x8_t vxb01234567c3 = vqsub_s8(vreinterpret_s8_u8(vb01234567c3), vb_zero_point);
//
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c3, vreinterpret_s8_u8(vld1_dup_u8(a0))); ++a0;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c3, vreinterpret_s8_u8(vld1_dup_u8(a1))); ++a1;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c3, vreinterpret_s8_u8(vld1_dup_u8(a2))); ++a2;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c3, vreinterpret_s8_u8(vld1_dup_u8(a3))); ++a3;
//
//
//     const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     // const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));
//     const int8x8_t vxb01234567c4 = vqsub_s8(vreinterpret_s8_u8(vb01234567c4), vb_zero_point);
//
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c4, vreinterpret_s8_u8(vld1_dup_u8(a0))); ++a0;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c4, vreinterpret_s8_u8(vld1_dup_u8(a1))); ++a1;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c4, vreinterpret_s8_u8(vld1_dup_u8(a2))); ++a2;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c4, vreinterpret_s8_u8(vld1_dup_u8(a3))); ++a3;
//
//     const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     // const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));
//     const int8x8_t vxb01234567c5 = vqsub_s8(vreinterpret_s8_u8(vb01234567c5), vb_zero_point);
//
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c5, vreinterpret_s8_u8(vld1_dup_u8(a0))); ++a0;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c5, vreinterpret_s8_u8(vld1_dup_u8(a1))); ++a1;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c5, vreinterpret_s8_u8(vld1_dup_u8(a2))); ++a2;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c5, vreinterpret_s8_u8(vld1_dup_u8(a3))); ++a3;
//
//     const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     // const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));
//     const int8x8_t vxb01234567c6 = vqsub_s8(vreinterpret_s8_u8(vb01234567c6), vb_zero_point);
//
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c6, vreinterpret_s8_u8(vld1_dup_u8(a0))); ++a0;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c6, vreinterpret_s8_u8(vld1_dup_u8(a1))); ++a1;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c6, vreinterpret_s8_u8(vld1_dup_u8(a2))); ++a2;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c6, vreinterpret_s8_u8(vld1_dup_u8(a3))); ++a3;
//
//     const uint8x8_t vb01234567c7 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     // const int16x8_t vxb01234567c7 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c7, vb_zero_point));
//     const int8x8_t vxb01234567c7 = vqsub_s8(vreinterpret_s8_u8(vb01234567c7), vb_zero_point);
//
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c7, vreinterpret_s8_u8(vld1_dup_u8(a0))); ++a0;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c7, vreinterpret_s8_u8(vld1_dup_u8(a1))); ++a1;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c7, vreinterpret_s8_u8(vld1_dup_u8(a2))); ++a2;
//     vacc0x01234567 = vmlal_s8(vacc0x01234567, vxb01234567c7, vreinterpret_s8_u8(vld1_dup_u8(a3))); ++a3;
//
//   }
//   // if (k != 0) {
//   //   const size_t a_predecrement = 8 - k;
//   //   const int64x1_t va_shift = vmov_n_s64(-8 * a_predecrement);
//   //   const uint8x8_t va0 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a0 - a_predecrement)), va_shift));
//   //   const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0));
//   //   const uint8x8_t va1 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a1 - a_predecrement)), va_shift));
//   //   const int16x8_t vxa1 = vreinterpretq_s16_u16(vmovl_u8(va1));
//   //   const uint8x8_t va2 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a2 - a_predecrement)), va_shift));
//   //   const int16x8_t vxa2 = vreinterpretq_s16_u16(vmovl_u8(va2));
//   //   const uint8x8_t va3 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a3 - a_predecrement)), va_shift));
//   //   const int16x8_t vxa3 = vreinterpretq_s16_u16(vmovl_u8(va3));
//   //
//   //   const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//   //   const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));
//   //
//   //   // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
//   //   // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
//   //   // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
//   //   // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
//   //   // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
//   //   // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
//   //   // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
//   //   // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
//   //
//   //   vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c0, vget_low_s16(vxa0), 0);
//   //   vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c0, vget_low_s16(vxa1), 0);
//   //   vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c0, vget_low_s16(vxa2), 0);
//   //   vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c0, vget_low_s16(vxa3), 0);
//   //
//   //   if (k >= 2) {
//   //     const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//   //     const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));
//   //
//   //     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
//   //     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
//   //     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
//   //     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
//   //     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
//   //     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
//   //     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
//   //     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
//   //
//   //     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c1, vget_low_s16(vxa0), 1);
//   //     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c1, vget_low_s16(vxa1), 1);
//   //     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c1, vget_low_s16(vxa2), 1);
//   //     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c1, vget_low_s16(vxa3), 1);
//   //
//   //     if (k >= 3) {
//   //       const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//   //       const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));
//   //
//   //       // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
//   //       // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
//   //       // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
//   //       // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
//   //       // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
//   //       // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
//   //       // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
//   //       // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
//   //
//   //       vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c2, vget_low_s16(vxa0), 2);
//   //       vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c2, vget_low_s16(vxa1), 2);
//   //       vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c2, vget_low_s16(vxa2), 2);
//   //       vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c2, vget_low_s16(vxa3), 2);
//   //
//   //       if (k >= 4) {
//   //         const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//   //         const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));
//   //
//   //         // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
//   //         // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
//   //         // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
//   //         // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
//   //         // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
//   //         // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
//   //         // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
//   //         // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
//   //
//   //         vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c3, vget_low_s16(vxa0), 3);
//   //         vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c3, vget_low_s16(vxa1), 3);
//   //         vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c3, vget_low_s16(vxa2), 3);
//   //         vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c3, vget_low_s16(vxa3), 3);
//   //
//   //         if (k >= 5) {
//   //           const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//   //           const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));
//   //
//   //           // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
//   //           // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
//   //           // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
//   //           // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
//   //           // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
//   //           // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
//   //           // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
//   //           // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
//   //
//   //           vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c4, vget_high_s16(vxa0), 0);
//   //           vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c4, vget_high_s16(vxa1), 0);
//   //           vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c4, vget_high_s16(vxa2), 0);
//   //           vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c4, vget_high_s16(vxa3), 0);
//   //
//   //           if (k >= 6) {
//   //             const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//   //             const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));
//   //
//   //             // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
//   //             // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
//   //             // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
//   //             // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
//   //             // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
//   //             // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
//   //             // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
//   //             // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
//   //
//   //             vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c5, vget_high_s16(vxa0), 1);
//   //             vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c5, vget_high_s16(vxa1), 1);
//   //             vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c5, vget_high_s16(vxa2), 1);
//   //             vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c5, vget_high_s16(vxa3), 1);
//   //
//   //             if (k >= 7) {
//   //               const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//   //               const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));
//   //
//   //               // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
//   //               // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
//   //               // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
//   //               // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
//   //               // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
//   //               // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
//   //               // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
//   //               // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
//   //
//   //               vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c6, vget_high_s16(vxa0), 2);
//   //               vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c6, vget_high_s16(vxa1), 2);
//   //               vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c6, vget_high_s16(vxa2), 2);
//   //               vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c6, vget_high_s16(vxa3), 2);
//   //             }
//   //           }
//   //         }
//   //       }
//   //     }
//   //   }
//   // }
//
//   // printf("accumulators:  %i , %i , %i , %i , %i , %i , %i , %i\n",
//   // vgetq_lane_s16(vacc0x01234567, 0), vgetq_lane_s16(vacc0x01234567, 1), vgetq_lane_s16(vacc0x01234567, 2), vgetq_lane_s16(vacc0x01234567, 3),
//   // vgetq_lane_s16(vacc0x01234567, 4), vgetq_lane_s16(vacc0x01234567, 5), vgetq_lane_s16(vacc0x01234567, 6), vgetq_lane_s16(vacc0x01234567, 7));
//
//
//   int32x4_t vacc0x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc0x01234567)), vacc0123_bias);
//   int32x4_t vacc0x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc0x01234567)), vacc4567_bias);
//   int32x4_t vacc1x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc1x01234567)), vacc0123_bias);
//   int32x4_t vacc1x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc1x01234567)), vacc4567_bias);
//   int32x4_t vacc2x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc2x01234567)), vacc0123_bias);
//   int32x4_t vacc2x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc2x01234567)), vacc4567_bias);
//   int32x4_t vacc3x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc3x01234567)), vacc0123_bias);
//   int32x4_t vacc3x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc3x01234567)), vacc4567_bias);
//
//
//   // vacc0x0123 = vaddq_s32(vacc0x0123, vacc0123_bias);
//   // vacc0x4567 = vaddq_s32(vacc0x4567, vacc4567_bias);
//   // vacc1x0123 = vaddq_s32(vacc1x0123, vacc0123_bias);
//   // vacc1x4567 = vaddq_s32(vacc1x4567, vacc4567_bias);
//   // vacc2x0123 = vaddq_s32(vacc2x0123, vacc0123_bias);
//   // vacc2x4567 = vaddq_s32(vacc2x4567, vacc4567_bias);
//   // vacc3x0123 = vaddq_s32(vacc3x0123, vacc0123_bias);
//   // vacc3x4567 = vaddq_s32(vacc3x4567, vacc4567_bias);
//
//   // printf("accumulators:  %i , %i , %i , %i , %i , %i , %i , %i\n",
//   // vgetq_lane_s32(vacc0x0123, 0), vgetq_lane_s32(vacc0x0123, 1), vgetq_lane_s32(vacc0x0123, 2), vgetq_lane_s32(vacc0x0123, 3),
//   // vgetq_lane_s32(vacc0x4567, 0), vgetq_lane_s32(vacc0x4567, 1), vgetq_lane_s32(vacc0x4567, 2), vgetq_lane_s32(vacc0x4567, 3));
//
//   // printf("accumulators:  %i , %i , %i , %i , %i , %i , %i , %i\n",
//   // vgetq_lane_s32(vacc0123_bias, 0), vgetq_lane_s32(vacc0123_bias, 1), vgetq_lane_s32(vacc0123_bias, 2), vgetq_lane_s32(vacc0123_bias, 3),
//   // vgetq_lane_s32(vacc4567_bias, 0), vgetq_lane_s32(vacc4567_bias, 1), vgetq_lane_s32(vacc4567_bias, 2), vgetq_lane_s32(vacc4567_bias, 3));
//
//
//
//   const int32x4_t vmultiplier0x0123 = vld1q_s32(&quantization_params->neon.multiplier_v[kernel_quantization_params_offset]);
//   const int32x4_t vmultiplier0x4567 = vld1q_s32(&quantization_params->neon.multiplier_v[kernel_quantization_params_offset + 4]);
//   vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier0x0123);
//   vacc0x4567 = vqrdmulhq_s32(vacc0x4567, vmultiplier0x4567);
//   vacc1x0123 = vqrdmulhq_s32(vacc1x0123, vmultiplier0x0123);
//   vacc1x4567 = vqrdmulhq_s32(vacc1x4567, vmultiplier0x4567);
//   vacc2x0123 = vqrdmulhq_s32(vacc2x0123, vmultiplier0x0123);
//   vacc2x4567 = vqrdmulhq_s32(vacc2x4567, vmultiplier0x4567);
//   vacc3x0123 = vqrdmulhq_s32(vacc3x0123, vmultiplier0x0123);
//   vacc3x4567 = vqrdmulhq_s32(vacc3x4567, vmultiplier0x4567);
//
//   const int32x4_t vright_shift_0x0123 = vld1q_s32(&quantization_params->neon.right_shift_v[kernel_quantization_params_offset]);
//   const int32x4_t vright_shift_0x4567 = vld1q_s32(&quantization_params->neon.right_shift_v[kernel_quantization_params_offset + 4]);
//   const int32x4_t vzero_shift_mask_0x0123 = vreinterpretq_s32_u32(vceqq_s32(vright_shift_0x0123, vmovq_n_s32(0)));
//   const int32x4_t vzero_shift_mask_0x4567 = vreinterpretq_s32_u32(vceqq_s32(vright_shift_0x4567, vmovq_n_s32(0)));
//   vacc0x0123 = vsraq_n_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask_0x0123), 31);
//   vacc0x4567 = vsraq_n_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask_0x4567), 31);
//   vacc1x0123 = vsraq_n_s32(vacc1x0123, vbicq_s32(vacc1x0123, vzero_shift_mask_0x0123), 31);
//   vacc1x4567 = vsraq_n_s32(vacc1x4567, vbicq_s32(vacc1x4567, vzero_shift_mask_0x4567), 31);
//   vacc2x0123 = vsraq_n_s32(vacc2x0123, vbicq_s32(vacc2x0123, vzero_shift_mask_0x0123), 31);
//   vacc2x4567 = vsraq_n_s32(vacc2x4567, vbicq_s32(vacc2x4567, vzero_shift_mask_0x4567), 31);
//   vacc3x0123 = vsraq_n_s32(vacc3x0123, vbicq_s32(vacc3x0123, vzero_shift_mask_0x0123), 31);
//   vacc3x4567 = vsraq_n_s32(vacc3x4567, vbicq_s32(vacc3x4567, vzero_shift_mask_0x4567), 31);
//
//   vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift_0x0123);
//   vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift_0x4567);
//   vacc1x0123 = vrshlq_s32(vacc1x0123, vright_shift_0x0123);
//   vacc1x4567 = vrshlq_s32(vacc1x4567, vright_shift_0x4567);
//   vacc2x0123 = vrshlq_s32(vacc2x0123, vright_shift_0x0123);
//   vacc2x4567 = vrshlq_s32(vacc2x4567, vright_shift_0x4567);
//   vacc3x0123 = vrshlq_s32(vacc3x0123, vright_shift_0x0123);
//   vacc3x4567 = vrshlq_s32(vacc3x4567, vright_shift_0x4567);
//
//   const int16x8_t voutput_zero_point = vld1q_dup_s16(&quantization_params->neon.output_zero_point);
// // #ifdef __aarch64__
// //   const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
// //   const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
// //   const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
// //   const int16x8_t vacc3x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);
// //
// //   uint8x16_t vout0x01234567_1x01234567 = vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc1x01234567);
// //   uint8x16_t vout2x01234567_3x01234567 = vqmovun_high_s16(vqmovun_s16(vacc2x01234567), vacc3x01234567);
// // #else
//   const int16x8_t vacc0x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
//   const int16x8_t vacc1x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
//   const int16x8_t vacc2x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
//   const int16x8_t vacc3x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);
//
//   uint8x16_t vout0x01234567_1x01234567 = vcombine_u8(vqmovun_s16(vacc0x01234567_f), vqmovun_s16(vacc1x01234567_f));
//   uint8x16_t vout2x01234567_3x01234567 = vcombine_u8(vqmovun_s16(vacc2x01234567_f), vqmovun_s16(vacc3x01234567_f));
// // #endif
//   const uint8x16_t voutput_min = vld1q_dup_u8(&quantization_params->neon.output_min);
//   const uint8x16_t voutput_max = vld1q_dup_u8(&quantization_params->neon.output_max);
//
//   vout0x01234567_1x01234567 = vmaxq_u8(vout0x01234567_1x01234567, voutput_min);
//   vout2x01234567_3x01234567 = vmaxq_u8(vout2x01234567_3x01234567, voutput_min);
//   vout0x01234567_1x01234567 = vminq_u8(vout0x01234567_1x01234567, voutput_max);
//   vout2x01234567_3x01234567 = vminq_u8(vout2x01234567_3x01234567, voutput_max);
//
//   uint8_t* c0 = c;
//   uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + c_stride);
//   if (mr < 2) {
//     c1 = c0;
//   }
//   uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + c_stride);
//   if (mr <= 2) {
//     c2 = c1;
//   }
//   uint8_t* c3 = (uint8_t*) ((uintptr_t) c2 + c_stride);
//   if (mr != 4) {
//     c3 = c2;
//   }
//   if (nr == 8) {
//     vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567));
//     vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567));
//     vst1_u8(c2, vget_low_u8(vout2x01234567_3x01234567));
//     vst1_u8(c3, vget_high_u8(vout2x01234567_3x01234567));
//   } else {
//     if (nr >= 4) {
//       vst1q_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpretq_u32_u8(vout0x01234567_1x01234567), 0); c0 += 4;
//       vst1q_lane_u32(__builtin_assume_aligned(c1, 1), vreinterpretq_u32_u8(vout0x01234567_1x01234567), 2); c1 += 4;
//       vst1q_lane_u32(__builtin_assume_aligned(c2, 1), vreinterpretq_u32_u8(vout2x01234567_3x01234567), 0); c2 += 4;
//       vst1q_lane_u32(__builtin_assume_aligned(c3, 1), vreinterpretq_u32_u8(vout2x01234567_3x01234567), 2); c3 += 4;
//       vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
//       vout2x01234567_3x01234567 = vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
//       nr -= 4;
//     }
//     if (nr >= 2) {
//       vst1q_lane_u16(__builtin_assume_aligned(c0, 1), vreinterpretq_u16_u8(vout0x01234567_1x01234567), 0); c0 += 2;
//       vst1q_lane_u16(__builtin_assume_aligned(c1, 1), vreinterpretq_u16_u8(vout0x01234567_1x01234567), 4); c1 += 2;
//       vst1q_lane_u16(__builtin_assume_aligned(c2, 1), vreinterpretq_u16_u8(vout2x01234567_3x01234567), 0); c2 += 2;
//       vst1q_lane_u16(__builtin_assume_aligned(c3, 1), vreinterpretq_u16_u8(vout2x01234567_3x01234567), 4); c3 += 2;
//       vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
//       vout2x01234567_3x01234567 = vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
//       nr -= 2;
//     }
//     if (nr != 0) {
//       vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
//       vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
//       vst1q_lane_u8(c2, vout2x01234567_3x01234567, 0);
//       vst1q_lane_u8(c3, vout2x01234567_3x01234567, 8);
//     }
//   }
// }
// #elif defined OPPORTUNISTIC_ACC_2
// This version does "opportunistic" accumulation to 16bit: accumulate without further check just until
// requantization. Works only for 32bit AARCH
void q8gemm_ukernel_4x8__neon_per_channel_16bitAcc(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t c_stride,
    const union qnnp_conv_quantization_params quantization_params[restrict static 1],
    size_t kernel_quantization_params_offset)
{
  // Read the bias packed in the first 8 (32-bit) words
  int32x4_t vacc0123_bias = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 16);
  int32x4_t vacc4567_bias = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 16);

  // const int16_t z16 = 0;
  int16x8_t vacc0x01234567 = veorq_s16(vacc0x01234567, vacc0x01234567);//vld1q_dup_s16(&z16);
  int16x8_t vacc1x01234567 = vacc0x01234567;
  int16x8_t vacc2x01234567 = vacc0x01234567;
  int16x8_t vacc3x01234567 = vacc0x01234567;

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
  if (mr != 4) {
    a3 = a2;
  }

  const uint8x8_t vb_zero_point = vld1_u8((const uint8_t*) &quantization_params->neon.kernel_zero_point_v[kernel_quantization_params_offset]);
  for (; k >= 8; k -= 8) {
    const uint8x8_t va0 = vld1_u8(a0); a0 += 8;
    const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0));
    const uint8x8_t va1 = vld1_u8(a1); a1 += 8;
    const int16x8_t vxa1 = vreinterpretq_s16_u16(vmovl_u8(va1));
    const uint8x8_t va2 = vld1_u8(a2); a2 += 8;
    const int16x8_t vxa2 = vreinterpretq_s16_u16(vmovl_u8(va2));
    const uint8x8_t va3 = vld1_u8(a3); a3 += 8;
    const int16x8_t vxa3 = vreinterpretq_s16_u16(vmovl_u8(va3));

    const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
    const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));

    vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c0, vget_low_s16(vxa0), 0);
    vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c0, vget_low_s16(vxa1), 0);
    vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c0, vget_low_s16(vxa2), 0);
    vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c0, vget_low_s16(vxa3), 0);

    const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
    const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));

    vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c1, vget_low_s16(vxa0), 1);
    vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c1, vget_low_s16(vxa1), 1);
    vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c1, vget_low_s16(vxa2), 1);
    vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c1, vget_low_s16(vxa3), 1);

    const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
    const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));

    vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c2, vget_low_s16(vxa0), 2);
    vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c2, vget_low_s16(vxa1), 2);
    vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c2, vget_low_s16(vxa2), 2);
    vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c2, vget_low_s16(vxa3), 2);

    const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
    const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));

    vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c3, vget_low_s16(vxa0), 3);
    vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c3, vget_low_s16(vxa1), 3);
    vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c3, vget_low_s16(vxa2), 3);
    vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c3, vget_low_s16(vxa3), 3);

    const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
    const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));

    vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c4, vget_high_s16(vxa0), 0);
    vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c4, vget_high_s16(vxa1), 0);
    vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c4, vget_high_s16(vxa2), 0);
    vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c4, vget_high_s16(vxa3), 0);

    const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
    const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));

    vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c5, vget_high_s16(vxa0), 1);
    vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c5, vget_high_s16(vxa1), 1);
    vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c5, vget_high_s16(vxa2), 1);
    vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c5, vget_high_s16(vxa3), 1);

    const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
    const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));

    vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c6, vget_high_s16(vxa0), 2);
    vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c6, vget_high_s16(vxa1), 2);
    vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c6, vget_high_s16(vxa2), 2);
    vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c6, vget_high_s16(vxa3), 2);

    const uint8x8_t vb01234567c7 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
    const int16x8_t vxb01234567c7 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c7, vb_zero_point));

    vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c7, vget_high_s16(vxa0), 3);
    vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c7, vget_high_s16(vxa1), 3);
    vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c7, vget_high_s16(vxa2), 3);
    vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c7, vget_high_s16(vxa3), 3);
  }
  if (k != 0) {
    const size_t a_predecrement = 8 - k;
    const int64x1_t va_shift = vmov_n_s64(-8 * a_predecrement);
    const uint8x8_t va0 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a0 - a_predecrement)), va_shift));
    const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0));
    const uint8x8_t va1 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a1 - a_predecrement)), va_shift));
    const int16x8_t vxa1 = vreinterpretq_s16_u16(vmovl_u8(va1));
    const uint8x8_t va2 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a2 - a_predecrement)), va_shift));
    const int16x8_t vxa2 = vreinterpretq_s16_u16(vmovl_u8(va2));
    const uint8x8_t va3 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a3 - a_predecrement)), va_shift));
    const int16x8_t vxa3 = vreinterpretq_s16_u16(vmovl_u8(va3));

    const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
    const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));

    vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c0, vget_low_s16(vxa0), 0);
    vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c0, vget_low_s16(vxa1), 0);
    vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c0, vget_low_s16(vxa2), 0);
    vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c0, vget_low_s16(vxa3), 0);

    if (k >= 2) {
      const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
      const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));

      vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c1, vget_low_s16(vxa0), 1);
      vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c1, vget_low_s16(vxa1), 1);
      vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c1, vget_low_s16(vxa2), 1);
      vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c1, vget_low_s16(vxa3), 1);

      if (k >= 3) {
        const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
        const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));

        vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c2, vget_low_s16(vxa0), 2);
        vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c2, vget_low_s16(vxa1), 2);
        vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c2, vget_low_s16(vxa2), 2);
        vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c2, vget_low_s16(vxa3), 2);

        if (k >= 4) {
          const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
          const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));

          vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c3, vget_low_s16(vxa0), 3);
          vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c3, vget_low_s16(vxa1), 3);
          vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c3, vget_low_s16(vxa2), 3);
          vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c3, vget_low_s16(vxa3), 3);

          if (k >= 5) {
            const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
            const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));

            vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c4, vget_high_s16(vxa0), 0);
            vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c4, vget_high_s16(vxa1), 0);
            vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c4, vget_high_s16(vxa2), 0);
            vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c4, vget_high_s16(vxa3), 0);

            if (k >= 6) {
              const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
              const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));

              vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c5, vget_high_s16(vxa0), 1);
              vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c5, vget_high_s16(vxa1), 1);
              vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c5, vget_high_s16(vxa2), 1);
              vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c5, vget_high_s16(vxa3), 1);

              if (k >= 7) {
                const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
                const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));

                vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c6, vget_high_s16(vxa0), 2);
                vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c6, vget_high_s16(vxa1), 2);
                vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c6, vget_high_s16(vxa2), 2);
                vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c6, vget_high_s16(vxa3), 2);
              }
            }
          }
        }
      }
    }
  }

  int32x4_t vacc0x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc0x01234567)), vacc0123_bias);
  int32x4_t vacc0x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc0x01234567)), vacc4567_bias);
  int32x4_t vacc1x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc1x01234567)), vacc0123_bias);
  int32x4_t vacc1x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc1x01234567)), vacc4567_bias);
  int32x4_t vacc2x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc2x01234567)), vacc0123_bias);
  int32x4_t vacc2x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc2x01234567)), vacc4567_bias);
  int32x4_t vacc3x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc3x01234567)), vacc0123_bias);
  int32x4_t vacc3x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc3x01234567)), vacc4567_bias);


  const int32x4_t vmultiplier0x0123 = vld1q_s32(&quantization_params->neon.multiplier_v[kernel_quantization_params_offset]);
  const int32x4_t vmultiplier0x4567 = vld1q_s32(&quantization_params->neon.multiplier_v[kernel_quantization_params_offset + 4]);
  vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier0x0123);
  vacc0x4567 = vqrdmulhq_s32(vacc0x4567, vmultiplier0x4567);
  vacc1x0123 = vqrdmulhq_s32(vacc1x0123, vmultiplier0x0123);
  vacc1x4567 = vqrdmulhq_s32(vacc1x4567, vmultiplier0x4567);
  vacc2x0123 = vqrdmulhq_s32(vacc2x0123, vmultiplier0x0123);
  vacc2x4567 = vqrdmulhq_s32(vacc2x4567, vmultiplier0x4567);
  vacc3x0123 = vqrdmulhq_s32(vacc3x0123, vmultiplier0x0123);
  vacc3x4567 = vqrdmulhq_s32(vacc3x4567, vmultiplier0x4567);

  const int32x4_t vright_shift_0x0123 = vld1q_s32(&quantization_params->neon.right_shift_v[kernel_quantization_params_offset]);
  const int32x4_t vright_shift_0x4567 = vld1q_s32(&quantization_params->neon.right_shift_v[kernel_quantization_params_offset + 4]);
  const int32x4_t vzero_shift_mask_0x0123 = vreinterpretq_s32_u32(vceqq_s32(vright_shift_0x0123, vmovq_n_s32(0)));
  const int32x4_t vzero_shift_mask_0x4567 = vreinterpretq_s32_u32(vceqq_s32(vright_shift_0x4567, vmovq_n_s32(0)));
  vacc0x0123 = vsraq_n_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask_0x0123), 31);
  vacc0x4567 = vsraq_n_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask_0x4567), 31);
  vacc1x0123 = vsraq_n_s32(vacc1x0123, vbicq_s32(vacc1x0123, vzero_shift_mask_0x0123), 31);
  vacc1x4567 = vsraq_n_s32(vacc1x4567, vbicq_s32(vacc1x4567, vzero_shift_mask_0x4567), 31);
  vacc2x0123 = vsraq_n_s32(vacc2x0123, vbicq_s32(vacc2x0123, vzero_shift_mask_0x0123), 31);
  vacc2x4567 = vsraq_n_s32(vacc2x4567, vbicq_s32(vacc2x4567, vzero_shift_mask_0x4567), 31);
  vacc3x0123 = vsraq_n_s32(vacc3x0123, vbicq_s32(vacc3x0123, vzero_shift_mask_0x0123), 31);
  vacc3x4567 = vsraq_n_s32(vacc3x4567, vbicq_s32(vacc3x4567, vzero_shift_mask_0x4567), 31);

  vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift_0x0123);
  vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift_0x4567);
  vacc1x0123 = vrshlq_s32(vacc1x0123, vright_shift_0x0123);
  vacc1x4567 = vrshlq_s32(vacc1x4567, vright_shift_0x4567);
  vacc2x0123 = vrshlq_s32(vacc2x0123, vright_shift_0x0123);
  vacc2x4567 = vrshlq_s32(vacc2x4567, vright_shift_0x4567);
  vacc3x0123 = vrshlq_s32(vacc3x0123, vright_shift_0x0123);
  vacc3x4567 = vrshlq_s32(vacc3x4567, vright_shift_0x4567);

  const int16x8_t voutput_zero_point = vld1q_dup_s16(&quantization_params->neon.output_zero_point);
#ifdef __aarch64__
  const int16x8_t vacc0x01234567_f = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
  const int16x8_t vacc1x01234567_f = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
  const int16x8_t vacc2x01234567_f = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
  const int16x8_t vacc3x01234567_f = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);

  uint8x16_t vout0x01234567_1x01234567 = vqmovun_high_s16(vqmovun_s16(vacc0x01234567_f), vacc1x01234567_f);
  uint8x16_t vout2x01234567_3x01234567 = vqmovun_high_s16(vqmovun_s16(vacc2x01234567_f), vacc3x01234567_f);
#else
  const int16x8_t vacc0x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
  const int16x8_t vacc1x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
  const int16x8_t vacc2x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
  const int16x8_t vacc3x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);

  uint8x16_t vout0x01234567_1x01234567 = vcombine_u8(vqmovun_s16(vacc0x01234567_f), vqmovun_s16(vacc1x01234567_f));
  uint8x16_t vout2x01234567_3x01234567 = vcombine_u8(vqmovun_s16(vacc2x01234567_f), vqmovun_s16(vacc3x01234567_f));
#endif
  const uint8x16_t voutput_min = vld1q_dup_u8(&quantization_params->neon.output_min);
  const uint8x16_t voutput_max = vld1q_dup_u8(&quantization_params->neon.output_max);

  vout0x01234567_1x01234567 = vmaxq_u8(vout0x01234567_1x01234567, voutput_min);
  vout2x01234567_3x01234567 = vmaxq_u8(vout2x01234567_3x01234567, voutput_min);
  vout0x01234567_1x01234567 = vminq_u8(vout0x01234567_1x01234567, voutput_max);
  vout2x01234567_3x01234567 = vminq_u8(vout2x01234567_3x01234567, voutput_max);

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
  if (mr != 4) {
    c3 = c2;
  }
  if (nr == 8) {
    vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567));
    vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567));
    vst1_u8(c2, vget_low_u8(vout2x01234567_3x01234567));
    vst1_u8(c3, vget_high_u8(vout2x01234567_3x01234567));
  } else {
    if (nr >= 4) {
      vst1q_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpretq_u32_u8(vout0x01234567_1x01234567), 0); c0 += 4;
      vst1q_lane_u32(__builtin_assume_aligned(c1, 1), vreinterpretq_u32_u8(vout0x01234567_1x01234567), 2); c1 += 4;
      vst1q_lane_u32(__builtin_assume_aligned(c2, 1), vreinterpretq_u32_u8(vout2x01234567_3x01234567), 0); c2 += 4;
      vst1q_lane_u32(__builtin_assume_aligned(c3, 1), vreinterpretq_u32_u8(vout2x01234567_3x01234567), 2); c3 += 4;
      vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      vout2x01234567_3x01234567 = vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
      nr -= 4;
    }
    if (nr >= 2) {
      vst1q_lane_u16(__builtin_assume_aligned(c0, 1), vreinterpretq_u16_u8(vout0x01234567_1x01234567), 0); c0 += 2;
      vst1q_lane_u16(__builtin_assume_aligned(c1, 1), vreinterpretq_u16_u8(vout0x01234567_1x01234567), 4); c1 += 2;
      vst1q_lane_u16(__builtin_assume_aligned(c2, 1), vreinterpretq_u16_u8(vout2x01234567_3x01234567), 0); c2 += 2;
      vst1q_lane_u16(__builtin_assume_aligned(c3, 1), vreinterpretq_u16_u8(vout2x01234567_3x01234567), 4); c3 += 2;
      vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      vout2x01234567_3x01234567 = vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
      nr -= 2;
    }
    if (nr != 0) {
      vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
      vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
      vst1q_lane_u8(c2, vout2x01234567_3x01234567, 0);
      vst1q_lane_u8(c3, vout2x01234567_3x01234567, 8);
    }
  }
}
// #elif defined FIXED_ACC
// // This version does "opportunistic" accumulation to 16bit: accumulate without further check just until
// // requantization. Works only for 32bit AARCH
// void q8gemm_ukernel_4x8__neon_per_channel_16bitAcc(
//     size_t mr,
//     size_t nr,
//     size_t k,
//     const uint8_t* restrict a,
//     size_t a_stride,
//     const void* restrict w,
//     uint8_t* restrict c,
//     size_t c_stride,
//     const union qnnp_conv_quantization_params quantization_params[restrict static 1],
//     size_t kernel_quantization_params_offset)
// {
//   // Read the bias packed in the first 8 (32-bit) words
//   int32x4_t vacc0123_bias = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 16);
//   int32x4_t vacc4567_bias = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 16);
//
//   // init accumulators with zeros
//   // const int32_t z = 0;
//   // int32x4_t vacc0x0123 = vld1q_dup_s32(&z);
//   // int32x4_t vacc0x4567 = vld1q_dup_s32(&z);
//   // int32x4_t vacc1x0123 = vacc0x0123;
//   // int32x4_t vacc1x4567 = vacc0x4567;
//   // int32x4_t vacc2x0123 = vacc0x0123;
//   // int32x4_t vacc2x4567 = vacc0x4567;
//   // int32x4_t vacc3x0123 = vacc0x0123;
//   // int32x4_t vacc3x4567 = vacc0x4567;
//
//   const size_t kc = k;
//   const int16_t z16 = 0;
//   int16x8_t vacc0x01234567 = vld1q_dup_s16(&z16);
//   int16x8_t vacc1x01234567 = vacc0x01234567;
//   int16x8_t vacc2x01234567 = vacc0x01234567;
//   int16x8_t vacc3x01234567 = vacc0x01234567;
//
//   const uint8_t* a0 = a;
//   const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
//   if (mr < 2) {
//     a1 = a0;
//   }
//   const uint8_t* a2 = (const uint8_t*) ((uintptr_t) a1 + a_stride);
//   if (mr <= 2) {
//     a2 = a1;
//   }
//   const uint8_t* a3 = (const uint8_t*) ((uintptr_t) a2 + a_stride);
//   if (mr != 4) {
//     a3 = a2;
//   }
//
//   int32_t aa, ww, acc, res;
//
//   const uint8x8_t vb_zero_point = vld1_u8((const uint8_t*) &quantization_params->neon.kernel_zero_point_v[kernel_quantization_params_offset]);
//   for (; k >= 8; k -= 8) {
//     const uint8x8_t va0 = vld1_u8(a0); a0 += 8;
//     const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0));
//     const uint8x8_t va1 = vld1_u8(a1); a1 += 8;
//     const int16x8_t vxa1 = vreinterpretq_s16_u16(vmovl_u8(va1));
//     const uint8x8_t va2 = vld1_u8(a2); a2 += 8;
//     const int16x8_t vxa2 = vreinterpretq_s16_u16(vmovl_u8(va2));
//     const uint8x8_t va3 = vld1_u8(a3); a3 += 8;
//     const int16x8_t vxa3 = vreinterpretq_s16_u16(vmovl_u8(va3));
//
//     const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));
//
//     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
//     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
//     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
//     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
//     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
//     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
//     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
//     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
//
//     aa = vgetq_lane_s16(vxa0, 0);
//     for (size_t i = 0; i<8; ++i) {
//         acc = vgetq_lane_s16(vacc0x01234567, i);
//         ww = vgetq_lane_s16(vxb01234567c0, i);
//         res = acc + (aa * ww);
//         if (res > 32767 || res < -32768) {
//           // printf("acc overflow: %ld , kc == %z , k == %z \n", res, kc, k);
//           printf("acc overflow: %ld  acc = %ld  aa = %ld  ww = %ld   kc = %ld  \n", res, acc, aa, ww, kc);
//         }
//     }
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c0, vget_low_s16(vxa0), 0);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c0, vget_low_s16(vxa1), 0);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c0, vget_low_s16(vxa2), 0);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c0, vget_low_s16(vxa3), 0);
//
//     // int16_t a16 = vget_lane_s16(vget_low_s16(vacc0x01234567), 0);
//     // if (a16 > 28000) {
//     //   printf("acc overflow: %i (kc: %z , k = %u)\n", a16, kc, k);
//     // }
//
//     // int16_t a16 = vget_lane_s16(vget_low_s16(vacc0x01234567), 0);
//     // int32_t a32 = (int32_t)a16;
//     // int32_t b32 = vgetq_lane_s32(vacc0x0123, 0);
//     // if ((a32 - b32)*(a32 - b32) > 0.02) {
//     //   printf("accumulator overflow: %u , %u \n", a32, b32);
//     // }
//
//     const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));
//
//     aa = vgetq_lane_s16(vxa0, 1);
//     for (size_t i = 0; i<8; ++i) {
//         acc = vgetq_lane_s16(vacc0x01234567, i);
//         ww = vgetq_lane_s16(vxb01234567c1, i);
//         res = acc + (aa * ww);
//         if (res > 32767 || res < -32768) {
//           // printf("acc overflow: %ld , kc == %z , k == %z \n", res, kc, k);
//           printf("acc overflow: %ld  acc = %ld  aa = %ld  ww = %ld   kc = %ld  \n", res, acc, aa, ww, kc);
//         }
//     }
//
//     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
//     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
//     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
//     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
//     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
//     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
//     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
//     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c1, vget_low_s16(vxa0), 1);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c1, vget_low_s16(vxa1), 1);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c1, vget_low_s16(vxa2), 1);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c1, vget_low_s16(vxa3), 1);
//
//     const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));
//
//     aa = vgetq_lane_s16(vxa0, 2);
//     for (size_t i = 0; i<8; ++i) {
//         acc = vgetq_lane_s16(vacc0x01234567, i);
//         ww = vgetq_lane_s16(vxb01234567c2, i);
//         res = acc + (aa * ww);
//         if (res > 32767 || res < -32768) {
//           // printf("acc overflow: %ld , kc == %z , k == %z \n", res, kc, k);
//           printf("acc overflow: %ld  acc = %ld  aa = %ld  ww = %ld   kc = %ld  \n", res, acc, aa, ww, kc);
//         }
//     }
//
//     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
//     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
//     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
//     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
//     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
//     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
//     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
//     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c2, vget_low_s16(vxa0), 2);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c2, vget_low_s16(vxa1), 2);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c2, vget_low_s16(vxa2), 2);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c2, vget_low_s16(vxa3), 2);
//
//     const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));
//
//     aa = vgetq_lane_s16(vxa0, 3);
//     for (size_t i = 0; i<8; ++i) {
//         acc = vgetq_lane_s16(vacc0x01234567, i);
//         ww = vgetq_lane_s16(vxb01234567c3, i);
//         res = acc + (aa * ww);
//         if (res > 32767 || res < -32768) {
//           // printf("acc overflow: %ld , kc == %z , k == %z \n", res, kc, k);
//           printf("acc overflow: %ld  acc = %ld  aa = %ld  ww = %ld   kc = %ld  \n", res, acc, aa, ww, kc);
//         }
//     }
//
//     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
//     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
//     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
//     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
//     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
//     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
//     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
//     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c3, vget_low_s16(vxa0), 3);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c3, vget_low_s16(vxa1), 3);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c3, vget_low_s16(vxa2), 3);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c3, vget_low_s16(vxa3), 3);
//
//     const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));
//
//     aa = vgetq_lane_s16(vxa0, 4);
//     for (size_t i = 0; i<8; ++i) {
//         acc = vgetq_lane_s16(vacc0x01234567, i);
//         ww = vgetq_lane_s16(vxb01234567c4, i);
//         res = acc + (aa * ww);
//         if (res > 32767 || res < -32768) {
//           // printf("acc overflow: %ld , kc == %z , k == %z \n", res, kc, k);
//           printf("acc overflow: %ld  acc = %ld  aa = %ld  ww = %ld   kc = %ld  \n", res, acc, aa, ww, kc);
//         }
//     }
//
//     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
//     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
//     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
//     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
//     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
//     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
//     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
//     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c4, vget_high_s16(vxa0), 0);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c4, vget_high_s16(vxa1), 0);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c4, vget_high_s16(vxa2), 0);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c4, vget_high_s16(vxa3), 0);
//
//     const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));
//
//     aa = vgetq_lane_s16(vxa0, 5);
//     for (size_t i = 0; i<8; ++i) {
//         acc = vgetq_lane_s16(vacc0x01234567, i);
//         ww = vgetq_lane_s16(vxb01234567c5, i);
//         res = acc + (aa * ww);
//         if (res > 32767 || res < -32768) {
//           // printf("acc overflow: %ld , kc == %z , k == %z \n", res, kc, k);
//           printf("acc overflow: %ld  acc = %ld  aa = %ld  ww = %ld   kc = %ld  \n", res, acc, aa, ww, kc);
//         }
//     }
//
//     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
//     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
//     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
//     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
//     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
//     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
//     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
//     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c5, vget_high_s16(vxa0), 1);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c5, vget_high_s16(vxa1), 1);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c5, vget_high_s16(vxa2), 1);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c5, vget_high_s16(vxa3), 1);
//
//     const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));
//
//     aa = vgetq_lane_s16(vxa0, 6);
//     for (size_t i = 0; i<8; ++i) {
//         acc = vgetq_lane_s16(vacc0x01234567, i);
//         ww = vgetq_lane_s16(vxb01234567c6, i);
//         res = acc + (aa * ww);
//         if (res > 32767 || res < -32768) {
//           // printf("acc overflow: %ld , kc == %z , k == %z \n", res, kc, k);
//           printf("acc overflow: %ld  acc = %ld  aa = %ld  ww = %ld   kc = %ld  \n", res, acc, aa, ww, kc);
//         }
//     }
//
//     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
//     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
//     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
//     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
//     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
//     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
//     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
//     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c6, vget_high_s16(vxa0), 2);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c6, vget_high_s16(vxa1), 2);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c6, vget_high_s16(vxa2), 2);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c6, vget_high_s16(vxa3), 2);
//
//     const uint8x8_t vb01234567c7 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c7 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c7, vb_zero_point));
//
//     aa = vgetq_lane_s16(vxa0, 7);
//     for (size_t i = 0; i<8; ++i) {
//         acc = vgetq_lane_s16(vacc0x01234567, i);
//         ww = vgetq_lane_s16(vxb01234567c7, i);
//         res = acc + (aa * ww);
//         if (res > 32767 || res < -32768) {
//           // printf("acc overflow: %ld , kc == %z , k == %z \n", res, kc, k);
//           printf("acc overflow: %ld  acc = %ld  aa = %ld  ww = %ld   kc = %ld  \n", res, acc, aa, ww, kc);
//         }
//     }
//
//     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
//     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
//     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
//     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
//     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
//     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
//     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
//     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c7, vget_high_s16(vxa0), 3);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c7, vget_high_s16(vxa1), 3);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c7, vget_high_s16(vxa2), 3);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c7, vget_high_s16(vxa3), 3);
//   }
//   if (k != 0) {
//     const size_t a_predecrement = 8 - k;
//     const int64x1_t va_shift = vmov_n_s64(-8 * a_predecrement);
//     const uint8x8_t va0 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a0 - a_predecrement)), va_shift));
//     const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0));
//     const uint8x8_t va1 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a1 - a_predecrement)), va_shift));
//     const int16x8_t vxa1 = vreinterpretq_s16_u16(vmovl_u8(va1));
//     const uint8x8_t va2 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a2 - a_predecrement)), va_shift));
//     const int16x8_t vxa2 = vreinterpretq_s16_u16(vmovl_u8(va2));
//     const uint8x8_t va3 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a3 - a_predecrement)), va_shift));
//     const int16x8_t vxa3 = vreinterpretq_s16_u16(vmovl_u8(va3));
//
//     const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));
//
//     // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
//     // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
//     // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
//     // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
//     // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
//     // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
//     // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
//     // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c0, vget_low_s16(vxa0), 0);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c0, vget_low_s16(vxa1), 0);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c0, vget_low_s16(vxa2), 0);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c0, vget_low_s16(vxa3), 0);
//
//     if (k >= 2) {
//       const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//       const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));
//
//       // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
//       // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
//       // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
//       // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
//       // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
//       // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
//       // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
//       // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
//
//       vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c1, vget_low_s16(vxa0), 1);
//       vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c1, vget_low_s16(vxa1), 1);
//       vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c1, vget_low_s16(vxa2), 1);
//       vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c1, vget_low_s16(vxa3), 1);
//
//       if (k >= 3) {
//         const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//         const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));
//
//         // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
//         // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
//         // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
//         // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
//         // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
//         // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
//         // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
//         // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
//
//         vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c2, vget_low_s16(vxa0), 2);
//         vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c2, vget_low_s16(vxa1), 2);
//         vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c2, vget_low_s16(vxa2), 2);
//         vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c2, vget_low_s16(vxa3), 2);
//
//         if (k >= 4) {
//           const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//           const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));
//
//           // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
//           // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
//           // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
//           // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
//           // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
//           // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
//           // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
//           // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
//
//           vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c3, vget_low_s16(vxa0), 3);
//           vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c3, vget_low_s16(vxa1), 3);
//           vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c3, vget_low_s16(vxa2), 3);
//           vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c3, vget_low_s16(vxa3), 3);
//
//           if (k >= 5) {
//             const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//             const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));
//
//             // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
//             // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
//             // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
//             // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
//             // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
//             // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
//             // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
//             // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
//
//             vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c4, vget_high_s16(vxa0), 0);
//             vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c4, vget_high_s16(vxa1), 0);
//             vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c4, vget_high_s16(vxa2), 0);
//             vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c4, vget_high_s16(vxa3), 0);
//
//             if (k >= 6) {
//               const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//               const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));
//
//               // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
//               // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
//               // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
//               // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
//               // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
//               // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
//               // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
//               // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
//
//               vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c5, vget_high_s16(vxa0), 1);
//               vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c5, vget_high_s16(vxa1), 1);
//               vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c5, vget_high_s16(vxa2), 1);
//               vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c5, vget_high_s16(vxa3), 1);
//
//               if (k >= 7) {
//                 const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//                 const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));
//
//                 // vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
//                 // vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
//                 // vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
//                 // vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
//                 // vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
//                 // vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
//                 // vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
//                 // vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
//
//                 vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c6, vget_high_s16(vxa0), 2);
//                 vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c6, vget_high_s16(vxa1), 2);
//                 vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c6, vget_high_s16(vxa2), 2);
//                 vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c6, vget_high_s16(vxa3), 2);
//               }
//             }
//           }
//         }
//       }
//     }
//   }
//
//   // printf("accumulators:  %i , %i , %i , %i , %i , %i , %i , %i\n",
//   // vgetq_lane_s16(vacc0x01234567, 0), vgetq_lane_s16(vacc0x01234567, 1), vgetq_lane_s16(vacc0x01234567, 2), vgetq_lane_s16(vacc0x01234567, 3),
//   // vgetq_lane_s16(vacc0x01234567, 4), vgetq_lane_s16(vacc0x01234567, 5), vgetq_lane_s16(vacc0x01234567, 6), vgetq_lane_s16(vacc0x01234567, 7));
//
//
//   int32x4_t vacc0x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc0x01234567)), vacc0123_bias);
//   int32x4_t vacc0x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc0x01234567)), vacc4567_bias);
//   int32x4_t vacc1x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc1x01234567)), vacc0123_bias);
//   int32x4_t vacc1x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc1x01234567)), vacc4567_bias);
//   int32x4_t vacc2x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc2x01234567)), vacc0123_bias);
//   int32x4_t vacc2x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc2x01234567)), vacc4567_bias);
//   int32x4_t vacc3x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc3x01234567)), vacc0123_bias);
//   int32x4_t vacc3x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc3x01234567)), vacc4567_bias);
//
//
//   // vacc0x0123 = vaddq_s32(vacc0x0123, vacc0123_bias);
//   // vacc0x4567 = vaddq_s32(vacc0x4567, vacc4567_bias);
//   // vacc1x0123 = vaddq_s32(vacc1x0123, vacc0123_bias);
//   // vacc1x4567 = vaddq_s32(vacc1x4567, vacc4567_bias);
//   // vacc2x0123 = vaddq_s32(vacc2x0123, vacc0123_bias);
//   // vacc2x4567 = vaddq_s32(vacc2x4567, vacc4567_bias);
//   // vacc3x0123 = vaddq_s32(vacc3x0123, vacc0123_bias);
//   // vacc3x4567 = vaddq_s32(vacc3x4567, vacc4567_bias);
//
//   // printf("accumulators:  %i , %i , %i , %i , %i , %i , %i , %i\n",
//   // vgetq_lane_s32(vacc0x0123, 0), vgetq_lane_s32(vacc0x0123, 1), vgetq_lane_s32(vacc0x0123, 2), vgetq_lane_s32(vacc0x0123, 3),
//   // vgetq_lane_s32(vacc0x4567, 0), vgetq_lane_s32(vacc0x4567, 1), vgetq_lane_s32(vacc0x4567, 2), vgetq_lane_s32(vacc0x4567, 3));
//
//   // printf("accumulators:  %i , %i , %i , %i , %i , %i , %i , %i\n",
//   // vgetq_lane_s32(vacc0123_bias, 0), vgetq_lane_s32(vacc0123_bias, 1), vgetq_lane_s32(vacc0123_bias, 2), vgetq_lane_s32(vacc0123_bias, 3),
//   // vgetq_lane_s32(vacc4567_bias, 0), vgetq_lane_s32(vacc4567_bias, 1), vgetq_lane_s32(vacc4567_bias, 2), vgetq_lane_s32(vacc4567_bias, 3));
//
//
//
//   const int32x4_t vmultiplier0x0123 = vld1q_s32(&quantization_params->neon.multiplier_v[kernel_quantization_params_offset]);
//   const int32x4_t vmultiplier0x4567 = vld1q_s32(&quantization_params->neon.multiplier_v[kernel_quantization_params_offset + 4]);
//   vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier0x0123);
//   vacc0x4567 = vqrdmulhq_s32(vacc0x4567, vmultiplier0x4567);
//   vacc1x0123 = vqrdmulhq_s32(vacc1x0123, vmultiplier0x0123);
//   vacc1x4567 = vqrdmulhq_s32(vacc1x4567, vmultiplier0x4567);
//   vacc2x0123 = vqrdmulhq_s32(vacc2x0123, vmultiplier0x0123);
//   vacc2x4567 = vqrdmulhq_s32(vacc2x4567, vmultiplier0x4567);
//   vacc3x0123 = vqrdmulhq_s32(vacc3x0123, vmultiplier0x0123);
//   vacc3x4567 = vqrdmulhq_s32(vacc3x4567, vmultiplier0x4567);
//
//   const int32x4_t vright_shift_0x0123 = vld1q_s32(&quantization_params->neon.right_shift_v[kernel_quantization_params_offset]);
//   const int32x4_t vright_shift_0x4567 = vld1q_s32(&quantization_params->neon.right_shift_v[kernel_quantization_params_offset + 4]);
//   const int32x4_t vzero_shift_mask_0x0123 = vreinterpretq_s32_u32(vceqq_s32(vright_shift_0x0123, vmovq_n_s32(0)));
//   const int32x4_t vzero_shift_mask_0x4567 = vreinterpretq_s32_u32(vceqq_s32(vright_shift_0x4567, vmovq_n_s32(0)));
//   vacc0x0123 = vsraq_n_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask_0x0123), 31);
//   vacc0x4567 = vsraq_n_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask_0x4567), 31);
//   vacc1x0123 = vsraq_n_s32(vacc1x0123, vbicq_s32(vacc1x0123, vzero_shift_mask_0x0123), 31);
//   vacc1x4567 = vsraq_n_s32(vacc1x4567, vbicq_s32(vacc1x4567, vzero_shift_mask_0x4567), 31);
//   vacc2x0123 = vsraq_n_s32(vacc2x0123, vbicq_s32(vacc2x0123, vzero_shift_mask_0x0123), 31);
//   vacc2x4567 = vsraq_n_s32(vacc2x4567, vbicq_s32(vacc2x4567, vzero_shift_mask_0x4567), 31);
//   vacc3x0123 = vsraq_n_s32(vacc3x0123, vbicq_s32(vacc3x0123, vzero_shift_mask_0x0123), 31);
//   vacc3x4567 = vsraq_n_s32(vacc3x4567, vbicq_s32(vacc3x4567, vzero_shift_mask_0x4567), 31);
//
//   vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift_0x0123);
//   vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift_0x4567);
//   vacc1x0123 = vrshlq_s32(vacc1x0123, vright_shift_0x0123);
//   vacc1x4567 = vrshlq_s32(vacc1x4567, vright_shift_0x4567);
//   vacc2x0123 = vrshlq_s32(vacc2x0123, vright_shift_0x0123);
//   vacc2x4567 = vrshlq_s32(vacc2x4567, vright_shift_0x4567);
//   vacc3x0123 = vrshlq_s32(vacc3x0123, vright_shift_0x0123);
//   vacc3x4567 = vrshlq_s32(vacc3x4567, vright_shift_0x4567);
//
//   const int16x8_t voutput_zero_point = vld1q_dup_s16(&quantization_params->neon.output_zero_point);
// // #ifdef __aarch64__
// //   const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
// //   const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
// //   const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
// //   const int16x8_t vacc3x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);
// //
// //   uint8x16_t vout0x01234567_1x01234567 = vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc1x01234567);
// //   uint8x16_t vout2x01234567_3x01234567 = vqmovun_high_s16(vqmovun_s16(vacc2x01234567), vacc3x01234567);
// // #else
//   const int16x8_t vacc0x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
//   const int16x8_t vacc1x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
//   const int16x8_t vacc2x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
//   const int16x8_t vacc3x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);
//
//   uint8x16_t vout0x01234567_1x01234567 = vcombine_u8(vqmovun_s16(vacc0x01234567_f), vqmovun_s16(vacc1x01234567_f));
//   uint8x16_t vout2x01234567_3x01234567 = vcombine_u8(vqmovun_s16(vacc2x01234567_f), vqmovun_s16(vacc3x01234567_f));
// // #endif
//   const uint8x16_t voutput_min = vld1q_dup_u8(&quantization_params->neon.output_min);
//   const uint8x16_t voutput_max = vld1q_dup_u8(&quantization_params->neon.output_max);
//
//   vout0x01234567_1x01234567 = vmaxq_u8(vout0x01234567_1x01234567, voutput_min);
//   vout2x01234567_3x01234567 = vmaxq_u8(vout2x01234567_3x01234567, voutput_min);
//   vout0x01234567_1x01234567 = vminq_u8(vout0x01234567_1x01234567, voutput_max);
//   vout2x01234567_3x01234567 = vminq_u8(vout2x01234567_3x01234567, voutput_max);
//
//   uint8_t* c0 = c;
//   uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + c_stride);
//   if (mr < 2) {
//     c1 = c0;
//   }
//   uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + c_stride);
//   if (mr <= 2) {
//     c2 = c1;
//   }
//   uint8_t* c3 = (uint8_t*) ((uintptr_t) c2 + c_stride);
//   if (mr != 4) {
//     c3 = c2;
//   }
//   if (nr == 8) {
//     vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567));
//     vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567));
//     vst1_u8(c2, vget_low_u8(vout2x01234567_3x01234567));
//     vst1_u8(c3, vget_high_u8(vout2x01234567_3x01234567));
//   } else {
//     if (nr >= 4) {
//       vst1q_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpretq_u32_u8(vout0x01234567_1x01234567), 0); c0 += 4;
//       vst1q_lane_u32(__builtin_assume_aligned(c1, 1), vreinterpretq_u32_u8(vout0x01234567_1x01234567), 2); c1 += 4;
//       vst1q_lane_u32(__builtin_assume_aligned(c2, 1), vreinterpretq_u32_u8(vout2x01234567_3x01234567), 0); c2 += 4;
//       vst1q_lane_u32(__builtin_assume_aligned(c3, 1), vreinterpretq_u32_u8(vout2x01234567_3x01234567), 2); c3 += 4;
//       vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
//       vout2x01234567_3x01234567 = vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
//       nr -= 4;
//     }
//     if (nr >= 2) {
//       vst1q_lane_u16(__builtin_assume_aligned(c0, 1), vreinterpretq_u16_u8(vout0x01234567_1x01234567), 0); c0 += 2;
//       vst1q_lane_u16(__builtin_assume_aligned(c1, 1), vreinterpretq_u16_u8(vout0x01234567_1x01234567), 4); c1 += 2;
//       vst1q_lane_u16(__builtin_assume_aligned(c2, 1), vreinterpretq_u16_u8(vout2x01234567_3x01234567), 0); c2 += 2;
//       vst1q_lane_u16(__builtin_assume_aligned(c3, 1), vreinterpretq_u16_u8(vout2x01234567_3x01234567), 4); c3 += 2;
//       vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
//       vout2x01234567_3x01234567 = vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
//       nr -= 2;
//     }
//     if (nr != 0) {
//       vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
//       vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
//       vst1q_lane_u8(c2, vout2x01234567_3x01234567, 0);
//       vst1q_lane_u8(c3, vout2x01234567_3x01234567, 8);
//     }
//   }
// }
// #endif

// void q8gemm_ukernel_2x8__neon_per_channel_16bitAcc(
//     size_t mr,
//     size_t nr,
//     size_t k,
//     const uint8_t* restrict a,
//     size_t a_stride,
//     const void* restrict w,
//     uint8_t* restrict c,
//     size_t c_stride,
//     const union qnnp_conv_quantization_params quantization_params[restrict static 1],
//     size_t kernel_quantization_params_offset)
// {
//   // Read the bias packed in the first 8 (32-bit) words
//   int32x4_t vacc0123_bias = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 16);
//   int32x4_t vacc4567_bias = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 16);
//
//   const int16_t z16 = 0;
//   int16x8_t vacc0x01234567 = vld1q_dup_s16(&z16);
//   int16x8_t vacc1x01234567 = vacc0x01234567;
//
//   const uint8_t* a0 = a;
//   const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
//   if (mr < 2) {
//     a1 = a0;
//   }
//
//   const uint8x8_t vb_zero_point = vld1_u8((const uint8_t*) &quantization_params->neon.kernel_zero_point_v[kernel_quantization_params_offset]);
//   for (; k >= 8; k -= 8) {
//     const uint8x8_t va0 = vld1_u8(a0); a0 += 8;
//     const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0));
//     const uint8x8_t va1 = vld1_u8(a1); a1 += 8;
//     const int16x8_t vxa1 = vreinterpretq_s16_u16(vmovl_u8(va1));
//
//     const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c0, vget_low_s16(vxa0), 0);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c0, vget_low_s16(vxa1), 0);
//
//     const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c1, vget_low_s16(vxa0), 1);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c1, vget_low_s16(vxa1), 1);
//
//     const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c2, vget_low_s16(vxa0), 2);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c2, vget_low_s16(vxa1), 2);
//
//     const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c3, vget_low_s16(vxa0), 3);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c3, vget_low_s16(vxa1), 3);
//
//     const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c4, vget_high_s16(vxa0), 0);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c4, vget_high_s16(vxa1), 0);
//
//     const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c5, vget_high_s16(vxa0), 1);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c5, vget_high_s16(vxa1), 1);
//
//     const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c6, vget_high_s16(vxa0), 2);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c6, vget_high_s16(vxa1), 2);
//
//     const uint8x8_t vb01234567c7 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c7 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c7, vb_zero_point));
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c7, vget_high_s16(vxa0), 3);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c7, vget_high_s16(vxa1), 3);
//   }
//   if (k != 0) {
//     const size_t a_predecrement = 8 - k;
//     const int64x1_t va_shift = vmov_n_s64(-8 * a_predecrement);
//     const uint8x8_t va0 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a0 - a_predecrement)), va_shift));
//     const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0));
//     const uint8x8_t va1 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a1 - a_predecrement)), va_shift));
//     const int16x8_t vxa1 = vreinterpretq_s16_u16(vmovl_u8(va1));
//
//     const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c0, vget_low_s16(vxa0), 0);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c0, vget_low_s16(vxa1), 0);
//
//     if (k >= 2) {
//       const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//       const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));
//
//       vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c1, vget_low_s16(vxa0), 1);
//       vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c1, vget_low_s16(vxa1), 1);
//
//       if (k >= 3) {
//         const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//         const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));
//
//         vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c2, vget_low_s16(vxa0), 2);
//         vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c2, vget_low_s16(vxa1), 2);
//
//         if (k >= 4) {
//           const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//           const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));
//
//           vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c3, vget_low_s16(vxa0), 3);
//           vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c3, vget_low_s16(vxa1), 3);
//
//           if (k >= 5) {
//             const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//             const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));
//
//             vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c4, vget_high_s16(vxa0), 0);
//             vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c4, vget_high_s16(vxa1), 0);
//
//             if (k >= 6) {
//               const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//               const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));
//
//               vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c5, vget_high_s16(vxa0), 1);
//               vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c5, vget_high_s16(vxa1), 1);
//
//               if (k >= 7) {
//                 const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//                 const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));
//
//                 vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c6, vget_high_s16(vxa0), 2);
//                 vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c6, vget_high_s16(vxa1), 2);
//               }
//             }
//           }
//         }
//       }
//     }
//   }
//
//   int32x4_t vacc0x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc0x01234567)), vacc0123_bias);
//   int32x4_t vacc0x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc0x01234567)), vacc4567_bias);
//   int32x4_t vacc1x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc1x01234567)), vacc0123_bias);
//   int32x4_t vacc1x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc1x01234567)), vacc4567_bias);
//
//
//   const int32x4_t vmultiplier0x0123 = vld1q_s32(&quantization_params->neon.multiplier_v[kernel_quantization_params_offset]);
//   const int32x4_t vmultiplier0x4567 = vld1q_s32(&quantization_params->neon.multiplier_v[kernel_quantization_params_offset + 4]);
//   vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier0x0123);
//   vacc0x4567 = vqrdmulhq_s32(vacc0x4567, vmultiplier0x4567);
//   vacc1x0123 = vqrdmulhq_s32(vacc1x0123, vmultiplier0x0123);
//   vacc1x4567 = vqrdmulhq_s32(vacc1x4567, vmultiplier0x4567);
//
//   const int32x4_t vright_shift_0x0123 = vld1q_s32(&quantization_params->neon.right_shift_v[kernel_quantization_params_offset]);
//   const int32x4_t vright_shift_0x4567 = vld1q_s32(&quantization_params->neon.right_shift_v[kernel_quantization_params_offset + 4]);
//   const int32x4_t vzero_shift_mask_0x0123 = vreinterpretq_s32_u32(vceqq_s32(vright_shift_0x0123, vmovq_n_s32(0)));
//   const int32x4_t vzero_shift_mask_0x4567 = vreinterpretq_s32_u32(vceqq_s32(vright_shift_0x4567, vmovq_n_s32(0)));
//   vacc0x0123 = vsraq_n_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask_0x0123), 31);
//   vacc0x4567 = vsraq_n_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask_0x4567), 31);
//   vacc1x0123 = vsraq_n_s32(vacc1x0123, vbicq_s32(vacc1x0123, vzero_shift_mask_0x0123), 31);
//   vacc1x4567 = vsraq_n_s32(vacc1x4567, vbicq_s32(vacc1x4567, vzero_shift_mask_0x4567), 31);
//
//   vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift_0x0123);
//   vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift_0x4567);
//   vacc1x0123 = vrshlq_s32(vacc1x0123, vright_shift_0x0123);
//   vacc1x4567 = vrshlq_s32(vacc1x4567, vright_shift_0x4567);
//
//   const int16x8_t voutput_zero_point = vld1q_dup_s16(&quantization_params->neon.output_zero_point);
// #ifdef __aarch64__
//   const int16x8_t vacc0x01234567_f = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
//   const int16x8_t vacc1x01234567_f = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
//
//   uint8x16_t vout0x01234567_1x01234567 = vqmovun_high_s16(vqmovun_s16(vacc0x01234567_f), vacc1x01234567_f);
// #else
//   const int16x8_t vacc0x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
//   const int16x8_t vacc1x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
//
//   uint8x16_t vout0x01234567_1x01234567 = vcombine_u8(vqmovun_s16(vacc0x01234567_f), vqmovun_s16(vacc1x01234567_f));
// #endif
//   const uint8x16_t voutput_min = vld1q_dup_u8(&quantization_params->neon.output_min);
//   const uint8x16_t voutput_max = vld1q_dup_u8(&quantization_params->neon.output_max);
//
//   vout0x01234567_1x01234567 = vmaxq_u8(vout0x01234567_1x01234567, voutput_min);
//   vout0x01234567_1x01234567 = vminq_u8(vout0x01234567_1x01234567, voutput_max);
//
//
//   uint8_t* c0 = c;
//   uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + c_stride);
//   if (mr < 2) {
//     c1 = c0;
//   }
//   if (nr == 8) {
//     vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567));
//     vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567));
//   } else {
//     if (nr >= 4) {
//       vst1q_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpretq_u32_u8(vout0x01234567_1x01234567), 0); c0 += 4;
//       vst1q_lane_u32(__builtin_assume_aligned(c1, 1), vreinterpretq_u32_u8(vout0x01234567_1x01234567), 2); c1 += 4;
//       vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
//       nr -= 4;
//     }
//     if (nr >= 2) {
//       vst1q_lane_u16(__builtin_assume_aligned(c0, 1), vreinterpretq_u16_u8(vout0x01234567_1x01234567), 0); c0 += 2;
//       vst1q_lane_u16(__builtin_assume_aligned(c1, 1), vreinterpretq_u16_u8(vout0x01234567_1x01234567), 4); c1 += 2;
//       vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
//       nr -= 2;
//     }
//     if (nr != 0) {
//       vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
//       vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
//     }
//   }
// }
//
// void q8gemm_ukernel_4x4__neon_per_channel_16bitAcc(
//     size_t mr,
//     size_t nr,
//     size_t k,
//     const uint8_t* restrict a,
//     size_t a_stride,
//     const void* restrict w,
//     uint8_t* restrict c,
//     size_t c_stride,
//     const union qnnp_conv_quantization_params quantization_params[restrict static 1],
//     size_t kernel_quantization_params_offset)
// {
//   // Read the bias packed in the first (32-bit) words
//   int32x4_t vacc0123_bias = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 16);
//
//   int16x4_t vacc0x0123 = veor_s16(vacc0x0123, vacc0x0123);
//   int16x4_t vacc1x0123 = vacc0x0123;
//   int16x4_t vacc2x0123 = vacc0x0123;
//   int16x4_t vacc3x0123 = vacc0x0123;
//
//   int16x8_t vacc01x01230123 = veor_s16(vacc01x01230123, vacc01x01230123);
//
//   const uint8_t* a0 = a;
//   const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
//   if (mr < 2) {
//     a1 = a0;
//   }
//   const uint8_t* a2 = (const uint8_t*) ((uintptr_t) a1 + a_stride);
//   if (mr <= 2) {
//     a2 = a1;
//   }
//   const uint8_t* a3 = (const uint8_t*) ((uintptr_t) a2 + a_stride);
//   if (mr != 4) {
//     a3 = a2;
//   }
//
//   const uint8x8_t vb_zero_point = vld1_u8((const uint8_t*) &quantization_params->neon.kernel_zero_point_v[kernel_quantization_params_offset]);
//   for (; k >= 8; k -= 8) {
//     const uint8x8_t va0 = vld1_u8(a0); a0 += 8;
//     const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0));
//     const uint8x8_t va1 = vld1_u8(a1); a1 += 8;
//     const int16x8_t vxa1 = vreinterpretq_s16_u16(vmovl_u8(va1));
//     const uint8x8_t va2 = vld1_u8(a2); a2 += 8;
//     const int16x8_t vxa2 = vreinterpretq_s16_u16(vmovl_u8(va2));
//     const uint8x8_t va3 = vld1_u8(a3); a3 += 8;
//     const int16x8_t vxa3 = vreinterpretq_s16_u16(vmovl_u8(va3));
//
//     const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c0, vget_low_s16(vxa0), 0);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c0, vget_low_s16(vxa1), 0);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c0, vget_low_s16(vxa2), 0);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c0, vget_low_s16(vxa3), 0);
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c0, vget_low_s16(vxa0), 0);
//
//     const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c1, vget_low_s16(vxa0), 1);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c1, vget_low_s16(vxa1), 1);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c1, vget_low_s16(vxa2), 1);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c1, vget_low_s16(vxa3), 1);
//
//     const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c2, vget_low_s16(vxa0), 2);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c2, vget_low_s16(vxa1), 2);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c2, vget_low_s16(vxa2), 2);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c2, vget_low_s16(vxa3), 2);
//
//     const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c3, vget_low_s16(vxa0), 3);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c3, vget_low_s16(vxa1), 3);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c3, vget_low_s16(vxa2), 3);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c3, vget_low_s16(vxa3), 3);
//
//     const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c4, vget_high_s16(vxa0), 0);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c4, vget_high_s16(vxa1), 0);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c4, vget_high_s16(vxa2), 0);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c4, vget_high_s16(vxa3), 0);
//
//     const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c5, vget_high_s16(vxa0), 1);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c5, vget_high_s16(vxa1), 1);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c5, vget_high_s16(vxa2), 1);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c5, vget_high_s16(vxa3), 1);
//
//     const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c6, vget_high_s16(vxa0), 2);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c6, vget_high_s16(vxa1), 2);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c6, vget_high_s16(vxa2), 2);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c6, vget_high_s16(vxa3), 2);
//
//     const uint8x8_t vb01234567c7 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c7 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c7, vb_zero_point));
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c7, vget_high_s16(vxa0), 3);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c7, vget_high_s16(vxa1), 3);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c7, vget_high_s16(vxa2), 3);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c7, vget_high_s16(vxa3), 3);
//   }
//   if (k != 0) {
//     const size_t a_predecrement = 8 - k;
//     const int64x1_t va_shift = vmov_n_s64(-8 * a_predecrement);
//     const uint8x8_t va0 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a0 - a_predecrement)), va_shift));
//     const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0));
//     const uint8x8_t va1 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a1 - a_predecrement)), va_shift));
//     const int16x8_t vxa1 = vreinterpretq_s16_u16(vmovl_u8(va1));
//     const uint8x8_t va2 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a2 - a_predecrement)), va_shift));
//     const int16x8_t vxa2 = vreinterpretq_s16_u16(vmovl_u8(va2));
//     const uint8x8_t va3 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a3 - a_predecrement)), va_shift));
//     const int16x8_t vxa3 = vreinterpretq_s16_u16(vmovl_u8(va3));
//
//     const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//     const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));
//
//     vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c0, vget_low_s16(vxa0), 0);
//     vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c0, vget_low_s16(vxa1), 0);
//     vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c0, vget_low_s16(vxa2), 0);
//     vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c0, vget_low_s16(vxa3), 0);
//
//     if (k >= 2) {
//       const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//       const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));
//
//       vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c1, vget_low_s16(vxa0), 1);
//       vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c1, vget_low_s16(vxa1), 1);
//       vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c1, vget_low_s16(vxa2), 1);
//       vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c1, vget_low_s16(vxa3), 1);
//
//       if (k >= 3) {
//         const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//         const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));
//
//         vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c2, vget_low_s16(vxa0), 2);
//         vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c2, vget_low_s16(vxa1), 2);
//         vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c2, vget_low_s16(vxa2), 2);
//         vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c2, vget_low_s16(vxa3), 2);
//
//         if (k >= 4) {
//           const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//           const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));
//
//           vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c3, vget_low_s16(vxa0), 3);
//           vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c3, vget_low_s16(vxa1), 3);
//           vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c3, vget_low_s16(vxa2), 3);
//           vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c3, vget_low_s16(vxa3), 3);
//
//           if (k >= 5) {
//             const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//             const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));
//
//             vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c4, vget_high_s16(vxa0), 0);
//             vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c4, vget_high_s16(vxa1), 0);
//             vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c4, vget_high_s16(vxa2), 0);
//             vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c4, vget_high_s16(vxa3), 0);
//
//             if (k >= 6) {
//               const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//               const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));
//
//               vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c5, vget_high_s16(vxa0), 1);
//               vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c5, vget_high_s16(vxa1), 1);
//               vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c5, vget_high_s16(vxa2), 1);
//               vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c5, vget_high_s16(vxa3), 1);
//
//               if (k >= 7) {
//                 const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
//                 const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));
//
//                 vacc0x01234567 = vmlaq_lane_s16(vacc0x01234567, vxb01234567c6, vget_high_s16(vxa0), 2);
//                 vacc1x01234567 = vmlaq_lane_s16(vacc1x01234567, vxb01234567c6, vget_high_s16(vxa1), 2);
//                 vacc2x01234567 = vmlaq_lane_s16(vacc2x01234567, vxb01234567c6, vget_high_s16(vxa2), 2);
//                 vacc3x01234567 = vmlaq_lane_s16(vacc3x01234567, vxb01234567c6, vget_high_s16(vxa3), 2);
//               }
//             }
//           }
//         }
//       }
//     }
//   }
//
//   int32x4_t vacc0x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc0x01234567)), vacc0123_bias);
//   int32x4_t vacc0x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc0x01234567)), vacc4567_bias);
//   int32x4_t vacc1x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc1x01234567)), vacc0123_bias);
//   int32x4_t vacc1x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc1x01234567)), vacc4567_bias);
//   int32x4_t vacc2x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc2x01234567)), vacc0123_bias);
//   int32x4_t vacc2x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc2x01234567)), vacc4567_bias);
//   int32x4_t vacc3x0123 = vaddq_s32(vmovl_s16(vget_low_s16(vacc3x01234567)), vacc0123_bias);
//   int32x4_t vacc3x4567 = vaddq_s32(vmovl_s16(vget_high_s16(vacc3x01234567)), vacc4567_bias);
//
//
//   const int32x4_t vmultiplier0x0123 = vld1q_s32(&quantization_params->neon.multiplier_v[kernel_quantization_params_offset]);
//   const int32x4_t vmultiplier0x4567 = vld1q_s32(&quantization_params->neon.multiplier_v[kernel_quantization_params_offset + 4]);
//   vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier0x0123);
//   vacc0x4567 = vqrdmulhq_s32(vacc0x4567, vmultiplier0x4567);
//   vacc1x0123 = vqrdmulhq_s32(vacc1x0123, vmultiplier0x0123);
//   vacc1x4567 = vqrdmulhq_s32(vacc1x4567, vmultiplier0x4567);
//   vacc2x0123 = vqrdmulhq_s32(vacc2x0123, vmultiplier0x0123);
//   vacc2x4567 = vqrdmulhq_s32(vacc2x4567, vmultiplier0x4567);
//   vacc3x0123 = vqrdmulhq_s32(vacc3x0123, vmultiplier0x0123);
//   vacc3x4567 = vqrdmulhq_s32(vacc3x4567, vmultiplier0x4567);
//
//   const int32x4_t vright_shift_0x0123 = vld1q_s32(&quantization_params->neon.right_shift_v[kernel_quantization_params_offset]);
//   const int32x4_t vright_shift_0x4567 = vld1q_s32(&quantization_params->neon.right_shift_v[kernel_quantization_params_offset + 4]);
//   const int32x4_t vzero_shift_mask_0x0123 = vreinterpretq_s32_u32(vceqq_s32(vright_shift_0x0123, vmovq_n_s32(0)));
//   const int32x4_t vzero_shift_mask_0x4567 = vreinterpretq_s32_u32(vceqq_s32(vright_shift_0x4567, vmovq_n_s32(0)));
//   vacc0x0123 = vsraq_n_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask_0x0123), 31);
//   vacc0x4567 = vsraq_n_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask_0x4567), 31);
//   vacc1x0123 = vsraq_n_s32(vacc1x0123, vbicq_s32(vacc1x0123, vzero_shift_mask_0x0123), 31);
//   vacc1x4567 = vsraq_n_s32(vacc1x4567, vbicq_s32(vacc1x4567, vzero_shift_mask_0x4567), 31);
//   vacc2x0123 = vsraq_n_s32(vacc2x0123, vbicq_s32(vacc2x0123, vzero_shift_mask_0x0123), 31);
//   vacc2x4567 = vsraq_n_s32(vacc2x4567, vbicq_s32(vacc2x4567, vzero_shift_mask_0x4567), 31);
//   vacc3x0123 = vsraq_n_s32(vacc3x0123, vbicq_s32(vacc3x0123, vzero_shift_mask_0x0123), 31);
//   vacc3x4567 = vsraq_n_s32(vacc3x4567, vbicq_s32(vacc3x4567, vzero_shift_mask_0x4567), 31);
//
//   vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift_0x0123);
//   vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift_0x4567);
//   vacc1x0123 = vrshlq_s32(vacc1x0123, vright_shift_0x0123);
//   vacc1x4567 = vrshlq_s32(vacc1x4567, vright_shift_0x4567);
//   vacc2x0123 = vrshlq_s32(vacc2x0123, vright_shift_0x0123);
//   vacc2x4567 = vrshlq_s32(vacc2x4567, vright_shift_0x4567);
//   vacc3x0123 = vrshlq_s32(vacc3x0123, vright_shift_0x0123);
//   vacc3x4567 = vrshlq_s32(vacc3x4567, vright_shift_0x4567);
//
//   const int16x8_t voutput_zero_point = vld1q_dup_s16(&quantization_params->neon.output_zero_point);
// #ifdef __aarch64__
//   const int16x8_t vacc0x01234567_f = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
//   const int16x8_t vacc1x01234567_f = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
//   const int16x8_t vacc2x01234567_f = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
//   const int16x8_t vacc3x01234567_f = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);
//
//   uint8x16_t vout0x01234567_1x01234567 = vqmovun_high_s16(vqmovun_s16(vacc0x01234567_f), vacc1x01234567_f);
//   uint8x16_t vout2x01234567_3x01234567 = vqmovun_high_s16(vqmovun_s16(vacc2x01234567_f), vacc3x01234567_f);
// #else
//   const int16x8_t vacc0x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
//   const int16x8_t vacc1x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
//   const int16x8_t vacc2x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
//   const int16x8_t vacc3x01234567_f = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);
//
//   uint8x16_t vout0x01234567_1x01234567 = vcombine_u8(vqmovun_s16(vacc0x01234567_f), vqmovun_s16(vacc1x01234567_f));
//   uint8x16_t vout2x01234567_3x01234567 = vcombine_u8(vqmovun_s16(vacc2x01234567_f), vqmovun_s16(vacc3x01234567_f));
// #endif
//   const uint8x16_t voutput_min = vld1q_dup_u8(&quantization_params->neon.output_min);
//   const uint8x16_t voutput_max = vld1q_dup_u8(&quantization_params->neon.output_max);
//
//   vout0x01234567_1x01234567 = vmaxq_u8(vout0x01234567_1x01234567, voutput_min);
//   vout2x01234567_3x01234567 = vmaxq_u8(vout2x01234567_3x01234567, voutput_min);
//   vout0x01234567_1x01234567 = vminq_u8(vout0x01234567_1x01234567, voutput_max);
//   vout2x01234567_3x01234567 = vminq_u8(vout2x01234567_3x01234567, voutput_max);
//
//   uint8_t* c0 = c;
//   uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + c_stride);
//   if (mr < 2) {
//     c1 = c0;
//   }
//   uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + c_stride);
//   if (mr <= 2) {
//     c2 = c1;
//   }
//   uint8_t* c3 = (uint8_t*) ((uintptr_t) c2 + c_stride);
//   if (mr != 4) {
//     c3 = c2;
//   }
//   if (nr == 8) {
//     vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567));
//     vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567));
//     vst1_u8(c2, vget_low_u8(vout2x01234567_3x01234567));
//     vst1_u8(c3, vget_high_u8(vout2x01234567_3x01234567));
//   } else {
//     if (nr >= 4) {
//       vst1q_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpretq_u32_u8(vout0x01234567_1x01234567), 0); c0 += 4;
//       vst1q_lane_u32(__builtin_assume_aligned(c1, 1), vreinterpretq_u32_u8(vout0x01234567_1x01234567), 2); c1 += 4;
//       vst1q_lane_u32(__builtin_assume_aligned(c2, 1), vreinterpretq_u32_u8(vout2x01234567_3x01234567), 0); c2 += 4;
//       vst1q_lane_u32(__builtin_assume_aligned(c3, 1), vreinterpretq_u32_u8(vout2x01234567_3x01234567), 2); c3 += 4;
//       vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
//       vout2x01234567_3x01234567 = vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
//       nr -= 4;
//     }
//     if (nr >= 2) {
//       vst1q_lane_u16(__builtin_assume_aligned(c0, 1), vreinterpretq_u16_u8(vout0x01234567_1x01234567), 0); c0 += 2;
//       vst1q_lane_u16(__builtin_assume_aligned(c1, 1), vreinterpretq_u16_u8(vout0x01234567_1x01234567), 4); c1 += 2;
//       vst1q_lane_u16(__builtin_assume_aligned(c2, 1), vreinterpretq_u16_u8(vout2x01234567_3x01234567), 0); c2 += 2;
//       vst1q_lane_u16(__builtin_assume_aligned(c3, 1), vreinterpretq_u16_u8(vout2x01234567_3x01234567), 4); c3 += 2;
//       vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
//       vout2x01234567_3x01234567 = vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
//       nr -= 2;
//     }
//     if (nr != 0) {
//       vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
//       vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
//       vst1q_lane_u8(c2, vout2x01234567_3x01234567, 0);
//       vst1q_lane_u8(c3, vout2x01234567_3x01234567, 8);
//     }
//   }
// }
