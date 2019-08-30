/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <qnnpack/assembly.h>

.syntax unified

# void q8gemm_ukernel_4x8__aarch32_neon_per_channel_16bitAcc(
#     size_t mr,
#     size_t nr,
#     size_t k,
#     const uint8_t*restrict a,
#     size_t a_stride,
#     const void*restrict w,
#     uint8_t*restrict c,
#     size_t c_stride,
#     const union qnnp_conv_quantization_params quantization_params[restrict static 1],
#     size_t kernel_quantization_params_offset)
BEGIN_FUNCTION q8gemm_ukernel_4x8__aarch32_neon_per_channel_16bitAcc
	.arm
#ifndef __APPLE__
	.arch armv7-a
	.fpu neon
#endif
	# Load w
	# - ip = w
	LDR ip, [sp, 4]
	PUSH {r4-r8}

	VPUSH {d8-d15}

	# Load quantization params
	# - r7 = quantization_params
	LDR r7, [sp, 100]

	# load kernel_quantization_params_offset
	LDR r5, [sp, 104]

  # Load bias0123, bias4567 and push to stack
	# VLDM ip!, {d16-d19}
  VLDM ip!, {d28-d31}

	# Load a_stride
	# - r6 = a_stride
	LDR r6, [sp, 84]
	CMP r0, 2

  # vacc0x01234567: {d16-d17} AKA q8 is the first accumulators row, set them to zero
  VEOR q8, q8, q8
  VEOR q9, q9, q9
  VEOR q10, q10, q10
  VEOR q11, q11, q11

	# a1 = a0 + a_stride
	ADD r4, r3, r6

	# Load b_zero_point from kernel_zero_point_v:
	# - d15 = b_zero_point
	MOV r8, 16
	ADD r8, r7, r8
	LDR r8, [r8, r5]
	VLD1.8 {d15}, [r8]
	MOVLO r4, r3

	# r8 - kernel_quantization_params_offset
	MOV r8, r5

	# ADD r7, r7, 4
	ADD r5, r4, r6

	# q9 := vacc1x01234567
	# VMOV.I16 q9, q8

  MOVLS r5, r4

  # q10 := vacc2x01234567
	# VMOV.I16 q10, q8

  ADD r6, r5, r6
  CMP r0, 4

  # q11 := vacc3x01234567
	# VMOV.I16 q11, q8

	MOVNE r6, r5
	SUBS r2, r2, 8

	BLO 1f

	.p2align 5
0:
	# Load a0
	# - d1 = a0
	VLD1.8 {d1}, [r3]!

	# Load a1
	# - d3 = a1
	VLD1.8 {d3}, [r4]!

	# Load b0-b7 (channel 0)
	# - d9 = b0-b7
	VLD1.8 {d9}, [ip:64]!

	# Load a2
	# - d5 = a2
	VLD1.8 {d5}, [r5]!

	# q0 = va0 = a0
	VMOVL.U8 q0, d1

	# Load a3
	# - d7 = a3
	VLD1.8 {d7}, [r6]!

	# q1 = va1 = a1
	VMOVL.U8 q1, d3

	# q4 = b0:7 - b_zero_point
  # - q4 - vxb01234567c
	VSUBL.U8 q4, d9, d15

	# q2 = va2 = a2
	VMOVL.U8 q2, d5
	# q3 = va3 = a3
	VMOVL.U8 q3, d7

	### Channel 0 ###

	# Load b0-b7 (channel 1)
	# - d11 = b0-b7
	VLD1.8 {d11}, [ip:64]!

	# vacc0x01234567 += vxb01234567c0 * va0[0]
	VMLA.I16 q8, q4, d0[0]

	# vacc1x01234567 += vxb01234567 * va1[0]
	VMLA.I16 q9, q4, d2[0]

	# vacc1x01234567 += vxb01234567 * va2[0]
	VMLA.I16 q10, q4, d4[0]

	# q5 = b0:7 - b_zero_point
	# - d10 = vb0123 (channel 1)
	# - d11 = vb4567 (channel 1)
	VSUBL.U8 q5, d11, d15

	# vacc1x01234567 += vxb01234567 * va3[0]
	VMLA.I16 q11, q4, d6[0]

	### Channel 1 ###

	# Load b0-b7 (channel 2)
	# - d9 = b0-b7
	VLD1.8 {d9}, [ip:64]!

	# vacc0x01234567 += vxb01234567c0 * va0[1]
	VMLA.I16 q8, q5, d0[1]

	# vacc1x01234567 += vxb01234567 * va1[1]
	VMLA.I16 q9, q5, d2[1]

	# vacc1x01234567 += vxb01234567 * va2[1]
	VMLA.I16 q10, q5, d4[1]

	# q4 = b0:7 - b_zero_point
	# - d8 = vb0123 (channel 2)
	# - d9 = vb4567 (channel 2)
	VSUBL.U8 q4, d9, d15

	# vacc1x01234567 += vxb01234567 * va3[1]
	VMLA.I16 q11, q5, d6[1]

	### Channel 2 ###

	# Load b0-b7 (channel 3)
	# - d11 = b0-b7
	VLD1.8 {d11}, [ip:64]!

	# vacc0x01234567 += vxb01234567c0 * va0[2]
	VMLA.I16 q8, q4, d0[2]

	# vacc1x01234567 += vxb01234567 * va1[2]
	VMLA.I16 q9, q4, d2[2]

	# vacc2x01234567 += vxb01234567 * va2[2]
	VMLA.I16 q10, q4, d4[2]

	# q5 = b0:7 - b_zero_point
	# - d10 = vb0123 (channel 3)
	# - d11 = vb4567 (channel 3)
	VSUBL.U8 q5, d11, d15

  # vacc3x01234567 += vxb01234567 * va3[2]
	VMLA.I16 q11, q4, d6[2]

	### Channel 3 ###

	# Load b0-b7 (channel 4)
	# - d9 = b0-b7
	VLD1.8 {d9}, [ip:64]!

	# vacc0x01234567 += vxb01234567c0 * va0[3]
	VMLA.I16 q8, q5, d0[3]

	# vacc1x01234567 += vxb01234567 * va1[3]
	VMLA.I16 q9, q5, d2[3]

	# vacc1x01234567 += vxb01234567 * va2[3]
	VMLA.I16 q10, q5, d4[3]

	# q5 = b0:7 - b_zero_point
	# - d10 = vb0123 (channel 4)
	# - d11 = vb4567 (channel 4)
	VSUBL.U8 q4, d9, d15

  # vacc3x01234567 += vxb01234567 * va3[3]
	VMLA.I16 q11, q5, d6[3]

	### Channel 4 ###

	# Load b0-b7 (channel 5)
	# - d11 = b0-b7
	VLD1.8 {d11}, [ip:64]!

  # vacc0x01234567 += vxb01234567c0 * va0[4]
  VMLA.I16 q8, q4, d1[0]

  # vacc1x01234567 += vxb01234567 * va1[4]
  VMLA.I16 q9, q4, d3[0]

  # vacc2x01234567 += vxb01234567 * va2[4]
  VMLA.I16 q10, q4, d5[0]

	# q4 = b0:7 - b_zero_point
	# - d8 = vb0123 (channel 5)
	# - d9 = vb4567 (channel 5)
	VSUBL.U8 q5, d11, d15

  # vacc3x01234567 += vxb01234567 * va3[4]
  VMLA.I16 q11, q4, d7[0]

	### Channel 5 ###

	# Load b0-b7 (channel 6)
	# - d9 = b0-b7
	VLD1.8 {d9}, [ip:64]!

  # vacc0x01234567 += vxb01234567c0 * va0[5]
  VMLA.I16 q8, q5, d1[1]

  # vacc1x01234567 += vxb01234567 * va1[5]
  VMLA.I16 q9, q5, d3[1]

  # vacc2x01234567 += vxb01234567 * va2[5]
  VMLA.I16 q10, q5, d5[1]

	# q4 = b0:7 - b_zero_point
	# - d8 = vb0123 (channel 6)
	# - d9 = vb4567 (channel 6)
	VSUBL.U8 q4, d9, d15

  # vacc3x01234567 += vxb01234567 * va3[5]
  VMLA.I16 q11, q5, d7[1]

	### Channel 6 ###

	# Load b0-b7 (channel 7)
	# - d11 = b0-b7
	VLD1.8 {d11}, [ip:64]!

  # vacc0x01234567 += vxb01234567c0 * va0[6]
  VMLA.I16 q8, q4, d1[2]

  # vacc1x01234567 += vxb01234567 * va1[6]
  VMLA.I16 q9, q4, d3[2]

  # vacc2x01234567 += vxb01234567 * va2[6]
  VMLA.I16 q10, q4, d5[2]

	# q5 = b0:7 - b_zero_point
	# - d10 = vb0123 (channel 7)
	# - d11 = vb4567 (channel 7)
	VSUBL.U8 q5, d11, d15

  # vacc3x01234567 += vxb01234567 * va3[6]
  VMLA.I16 q11, q4, d7[2]

	### Channel 7 ###
	SUBS r2, r2, 8

  # vacc0x01234567 += vxb01234567c0 * va0[7]
  VMLA.I16 q8, q5, d1[3]

  # vacc1x01234567 += vxb01234567 * va1[7]
  VMLA.I16 q9, q5, d3[3]

  # vacc2x01234567 += vxb01234567 * va2[7]
  VMLA.I16 q10, q5, d5[3]

  # vacc3x01234567 += vxb01234567 * va3[7]
  VMLA.I16 q11, q5, d7[3]

	BHS 0b

1:
	CMP r2, -8
	BEQ 2f

	# Adjust a0, a1, a2, a3
	ADD r3, r2
	ADD r4, r2
	ADD r5, r2
	ADD r6, r2

	# a_shift = 8 * k - 64
	LSL r2, r2, 3
	VDUP.32 d13, r2

	# Load a0
	# - d1 = a0
	VLD1.8 {d1}, [r3]

	# Load a1
	# - d3 = a1
	VLD1.8 {d3}, [r4]

	# Load b0-b7 (channel 0)
	# - d9 = b0-b7
	VLD1.8 {d9}, [ip:64]!

	# Load a2
	# - d5 = a2
	VLD1.8 {d5}, [r5]

	# q0 = va0 = a0
	VSHL.U64 d1, d1, d13
	VMOVL.U8 q0, d1

	# Load a3
	# - d7 = a3
	VLD1.8 {d7}, [r6]

	# q1 = va1 = a1
	VSHL.U64 d3, d3, d13
	VMOVL.U8 q1, d3

	# q4 = b0:7 - b_zero_point
	# - d8 = vb0123 (channel 0)
	# - d9 = vb4567 (channel 0)
	VSUBL.U8 q4, d9, d15

	# q2 = va2 = a2
	VSHL.U64 d5, d5, d13
	VMOVL.U8 q2, d5
	# q3 = va3 = a3
	VSHL.U64 d7, d7, d13
	VMOVL.U8 q3, d7

	### Channel 0 ###

  # vacc0x01234567 += vxb01234567c0 * va0[0]
  VMLA.I16 q8, q4, d0[0]

  # vacc1x01234567 += vxb01234567 * va1[0]
  VMLA.I16 q9, q4, d2[0]

  # vacc2x01234567 += vxb01234567 * va2[0]
  VMLA.I16 q10, q4, d4[0]

  # vacc3x01234567 += vxb01234567 * va3[0]
  VMLA.I16 q11, q4, d6[0]

	CMP r2, -48
	BLO 2f

	### Channel 1 ###

	# Load b0-b7 (channel 1)
	# - d11 = b0-b7
	VLD1.8 {d11}, [ip:64]!

	# q5 = b0:7 - b_zero_point
	# - d10 = vb0123 (channel 1)
	# - d11 = vb4567 (channel 1)
	VSUBL.U8 q5, d11, d15

  # vacc0x01234567 += vxb01234567c * va0[1]
  VMLA.I16 q8, q5, d0[1]

  # vacc1x01234567 += vxb01234567 * va1[1]
  VMLA.I16 q9, q5, d2[1]

  # vacc2x01234567 += vxb01234567 * va2[1]
  VMLA.I16 q10, q5, d4[1]

  # vacc3x01234567 += vxb01234567 * va3[1]
  VMLA.I16 q11, q5, d6[1]

	### Channel 2 ###
	BLS 2f

	# Load b0-b7 (channel 2)
	# - d9 = b0-b7
	VLD1.8 {d9}, [ip:64]!

	# q4 = b0:7 - b_zero_point
	# - d8 = vb0123 (channel 2)
	# - d9 = vb4567 (channel 2)
	VSUBL.U8 q4, d9, d15

  # vacc0x01234567 += vxb01234567c * va0[2]
  VMLA.I16 q8, q4, d0[2]

  # vacc1x01234567 += vxb01234567 * va1[2]
  VMLA.I16 q9, q4, d2[2]

  # vacc2x01234567 += vxb01234567 * va2[2]
  VMLA.I16 q10, q4, d4[2]

  # vacc3x01234567 += vxb01234567 * va3[2]
  VMLA.I16 q11, q4, d6[2]

	### Channel 3 ###
	CMP r2, -32
	BLO 2f

	# Load b0-b7 (channel 3)
	# - d9 = b0-b7
	VLD1.8 {d11}, [ip:64]!

	# q4 = b0:7 - b_zero_point
	# - d8 = vb0123 (channel 3)
	# - d9 = vb4567 (channel 3)
	VSUBL.U8 q5, d11, d15

  # vacc0x01234567 += vxb01234567c * va0[3]
  VMLA.I16 q8, q5, d0[3]

  # vacc1x01234567 += vxb01234567 * va1[3]
  VMLA.I16 q9, q5, d2[3]

  # vacc2x01234567 += vxb01234567 * va2[3]
  VMLA.I16 q10, q5, d4[3]

  # vacc3x01234567 += vxb01234567 * va3[3]
  VMLA.I16 q11, q5, d6[3]

	### Channel 4 ###
	BLS 2f

	# Load b0-b7 (channel 4)
	# - d11 = b0-b7
	VLD1.8 {d9}, [ip:64]!

	# q5 = b0:7 - b_zero_point
	# - d10 = vb0123 (channel 4)
	# - d11 = vb4567 (channel 4)
	VSUBL.U8 q4, d9, d15

  # vacc0x01234567 += vxb01234567c0 * va0[4]
  VMLA.I16 q8, q4, d1[0]

  # vacc1x01234567 += vxb01234567 * va1[4]
  VMLA.I16 q9, q4, d3[0]

  # vacc2x01234567 += vxb01234567 * va2[4]
  VMLA.I16 q10, q4, d5[0]

  # vacc3x01234567 += vxb01234567 * va3[4]
  VMLA.I16 q11, q4, d7[0]

	### Channel 5 ###
	CMP r2, -16
	BLO 2f

	# Load b0-b7 (channel 5)
	# - d13 = b0-b7
	VLD1.8 {d11}, [ip:64]!

	# q5 = b0:7 - b_zero_point
	# - d10 = vb0123 (channel 5)
	# - d11 = vb4567 (channel 5)
	VSUBL.U8 q5, d11, d15

  # vacc0x01234567 += vxb01234567c0 * va0[5]
  VMLA.I16 q8, q5, d1[1]

  # vacc1x01234567 += vxb01234567 * va1[5]
  VMLA.I16 q9, q5, d3[1]

  # vacc2x01234567 += vxb01234567 * va2[5]
  VMLA.I16 q10, q5, d5[1]

  # vacc3x01234567 += vxb01234567 * va3[5]
  VMLA.I16 q11, q5, d7[1]

	### Channel 6 ###
	BLS 2f

	# Load b0-b7 (channel 6)
	# - d9 = b0-b7
	VLD1.8 {d9}, [ip:64]

	# q4 = b0:7 - b_zero_point
	# - d8 = vb0123 (channel 6)
	# - d9 = vb4567 (channel 6)
	VSUBL.U8 q4, d9, d15

  # vacc0x01234567 += vxb01234567c0 * va0[6]
  VMLA.I16 q8, q4, d1[2]

  # vacc1x01234567 += vxb01234567 * va1[6]
  VMLA.I16 q9, q4, d3[2]

  # vacc2x01234567 += vxb01234567 * va2[6]
  VMLA.I16 q10, q4, d5[2]

  # vacc3x01234567 += vxb01234567 * va3[6]
  VMLA.I16 q11, q4, d7[2]

	.p2align 4
2:

  # Move bias
  VMOV.I32 q5, q15
  VMOV.I32 q4, q14

  # Load multiplier:
  ADD r7, 20
  LDR r8, [r7, r8]
  LDR r2, [sp, 104]

  # Split int16 accumulators to int32 accumulators, then add bias
  # q15 = q11[high] == d23
  VMOVL.S16 q15, d23
  # q14 = q11[low] == d22
  VMOVL.S16 q14, d22

  # q13 = q10[high] == d21
  VMOVL.S16 q13, d21
  # q12 = q10[low] == d20
  VMOVL.S16 q12, d20

  # q11 = q9[high] == d19
  VMOVL.S16 q11, d19
  # q10 = q9[low] == d18
  VMOVL.S16 q10, d18

  # q9 = q8[high] == d17
  VMOVL.S16 q9, d17
  # q8 = q8[low] == d16
  VMOVL.S16 q8, d16

  # - q6 = vmultiplier ( vmultiplier0x0123 )
  VLD1.32 {d12, d13}, [r8]!

  VADD.I32  q15, q15, q5
  VADD.I32  q14, q14, q4

  VADD.I32  q13, q13, q5
  VADD.I32  q12, q12, q4

  VADD.I32  q11, q11, q5
  VADD.I32  q10, q10, q4

  VADD.I32  q9, q9, q5
  VADD.I32  q8, q8, q4


	VQRDMULH.S32 q14, q14, q6
  VQRDMULH.S32 q12, q12, q6
  VQRDMULH.S32 q10, q10, q6
	VQRDMULH.S32  q8, q8, q6

  # - q6 = vmultiplier ( vmultiplier0x4567 )
	VLD1.32 {d12, d13}, [r8]

	# Load right_shift
	# - q4 = d8:d9   = vright_shift_0x0123
	ADD r7, 4
	LDR r8, [r7, r2]
	VLD1.32 {d8, d9}, [r8]!
	SUB r7, 12

  # Compute vzero_shift_mask
	# - q5 = vzero_shift_mask_0x0123
	VCEQ.S32 q5, q4, 0

	VBIC q0,  q8, q5
	VBIC q1,  q10, q5
	VBIC q2,  q12, q5
	VBIC q3,  q14, q5

  VQRDMULH.S32  q9, q9, q6
  VQRDMULH.S32 q11, q11, q6
  VQRDMULH.S32 q13, q13, q6
  VQRDMULH.S32 q15, q15, q6

	VSRA.S32  q8, q0, 31
	VSRA.S32  q10, q1, 31
	VSRA.S32  q12, q2, 31
	VSRA.S32  q14, q3, 31

	VRSHL.S32  q8,  q8, q4
	VRSHL.S32  q10,  q10, q4
	VRSHL.S32  q12,  q12, q4
	VRSHL.S32  q14,  q14, q4

	# - q4 = d8:d9 = vright_shift_0x4567
	VLD1.32 {d8, d9}, [r8]

	# Compute vzero_shift_mask
	# - q5 = vzero_shift_mask_0x4567
	VCEQ.S32 q5, q4, 0

	VBIC q0, q9, q5
	VBIC q1, q11, q5
	VBIC q2, q13, q5
	VBIC q3, q15, q5

	VSRA.S32 q9, q0, 31
	VSRA.S32 q11, q1, 31
	VSRA.S32 q13, q2, 31
	VSRA.S32 q15, q3, 31

	VRSHL.S32  q9,  q9, q4
	VRSHL.S32 q11, q11, q4
	VRSHL.S32 q13, q13, q4
	VRSHL.S32 q15, q15, q4

	# Load output_zero_point
	# - q7 = d14:d15 = voutput_zero_point
	VLD1.16 {d14[], d15[]}, [r7]!

	# Load max:
	# - q5 = d10:d11 = vmax
	VLD1.8 {d10[], d11[]}, [r7]!

	# Load c, c_stride:
	# - r2 = c
	# - r2 = c_stride
	LDRD r2, r3, [sp, 92]

	VQMOVN.S32 d16,  q8
	VQMOVN.S32 d17,  q9
	VQMOVN.S32 d18, q10
	VQMOVN.S32 d19, q11
	VQMOVN.S32 d20, q12
	VQMOVN.S32 d21, q13
	VQMOVN.S32 d22, q14
	VQMOVN.S32 d23, q15

	# Load min:
	# - q4 = q8:q9 = vmin
	VLD1.8 {d8[], d9[]}, [r7]!
	ADD r4, r2, r3

	VQADD.S16  q8,  q8, q7
	VQADD.S16  q9,  q9, q7
	CMP r0, 2
	VQADD.S16 q10, q10, q7
	VQADD.S16 q11, q11, q7
	MOVLO r4, r2

	VQMOVUN.S16 d16,  q8
	VQMOVUN.S16 d17,  q9
	ADD r5, r4, r3
	VQMOVUN.S16 d18, q10
	VQMOVUN.S16 d19, q11
	MOVLS r5, r4

	VMIN.U8 q8, q8, q5
	CMP r0, 4
	VMIN.U8 q9, q9, q5
	ADD r3, r5, r3

	VMAX.U8 q8, q8, q4
	MOVNE r3, r5
	CMP r1, 8
	VMAX.U8 q9, q9, q4

	BNE 4f

	VST1.8 {d16}, [r2]
	VST1.8 {d17}, [r4]
	VST1.8 {d18}, [r5]
	VST1.8 {d19}, [r3]

	VPOP {d8-d15}
	POP {r4-r8}
	BX lr

	.p2align 3
4:
	CMP r1, 4
	BLO 5f

	VST1.32 {d16[0]}, [r2]!
	VST1.32 {d17[0]}, [r4]!
	VST1.32 {d18[0]}, [r5]!
	VST1.32 {d19[0]}, [r3]!

	SUB r1, 4
	VEXT.8 q8, q8, q8, 4
	VEXT.8 q9, q9, q9, 4

5:
	CMP r1, 2
	BLO 6f

	VST1.16 {d16[0]}, [r2]!
	VST1.16 {d17[0]}, [r4]!
	VST1.16 {d18[0]}, [r5]!
	VST1.16 {d19[0]}, [r3]!

	SUB r1, 2
	VEXT.8 q8, q8, q8, 2
	VEXT.8 q9, q9, q9, 2

6:
	TEQ r1, 0
	BEQ 7f

	VST1.8 {d16[0]}, [r2]
	VST1.8 {d17[0]}, [r4]
	VST1.8 {d18[0]}, [r5]
	VST1.8 {d19[0]}, [r3]

7:
	VPOP {d8-d15}
	POP {r4-r8}
	BX lr
END_FUNCTION q8gemm_ukernel_4x8__aarch32_neon_per_channel_16bitAcc

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
