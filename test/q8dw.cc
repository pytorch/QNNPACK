/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <cpuinfo.h>
#include <depthwise-microkernel-tester.h>
#include <qnnpack/q8dw.h>


#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  TEST(Q8DW_25c8_NEON, single_output_channels_eq_8) {
    DepthwiseMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .test(q8dw_ukernel_9c8__neon);
  }
  TEST(Q8DW_25c8_NEON, multi_output_channels_eq_8_with_subsampling) {
    DepthwiseMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .subsampling(2)
      .cr(8)
      .channels(8)
      .width(5)
      .test(q8dw_ukernel_9c8__neon);
  }

  TEST(Q8DW_25c8_NEON, multi_output_channels_eq_8_with_input_stride) {
    DepthwiseMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(5)
      .inputStride(17)
      .test(q8dw_ukernel_9c8__neon);
  }

  TEST(Q8DW_25c8_NEON, multi_output_channels_eq_8_with_output_stride) {
    DepthwiseMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(5)
      .outputStride(19)
      .test(q8dw_ukernel_9c8__neon);
  }

  TEST(Q8DW_25c8_NEON, single_output_channels_eq_8_with_qmin) {
    DepthwiseMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .qmin(8)
      .test(q8dw_ukernel_9c8__neon);
  }

  TEST(Q8DW_25c8_NEON, single_output_channels_eq_8_with_qmax) {
    DepthwiseMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .qmax(218)
      .test(q8dw_ukernel_9c8__neon);
  }

  TEST(Q8DW_25c8_NEON, multip_output_channels_eq_8) {
    DepthwiseMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(3)
      .test(q8dw_ukernel_9c8__neon);
  }

  TEST(Q8DW_25c8_NEON, single_output_channels_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DepthwiseMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(q8dw_ukernel_9c8__neon);
    }
  }

  TEST(Q8DW_25c8_NEON, multi_output_channels_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DepthwiseMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(q8dw_ukernel_9c8__neon);
    }
  }

  TEST(Q8DW_25c8_NEON, multi_output_channels_div_8_with_output_stride) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DepthwiseMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(171)
        .test(q8dw_ukernel_9c8__neon);
    }
  }

  TEST(Q8DW_25c8_NEON, single_output_channels_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(q8dw_ukernel_9c8__neon);
    }
  }

  TEST(Q8DW_25c8_NEON, single_output_channels_gt_8_with_qmin) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmin(128)
        .test(q8dw_ukernel_9c8__neon);
    }
  }

  TEST(Q8DW_25c8_NEON, single_output_channels_gt_8_with_qmax) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmax(128)
        .test(q8dw_ukernel_9c8__neon);
    }
  }

  TEST(Q8DW_25c8_NEON, multi_output_channels_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(q8dw_ukernel_9c8__neon);
    }
  }

  TEST(Q8DW_25c8_NEON, multi_output_channels_gt_8_with_output_stride) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(17)
        .test(q8dw_ukernel_9c8__neon);
    }
  }

  TEST(Q8DW_9c8_NEON, single_output_channels_eq_8) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .test(q8dw_ukernel_9c8__neon);
  }

  TEST(Q8DW_9c8_NEON, single_output_channels_eq_8_with_qmin) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmin(128)
      .test(q8dw_ukernel_9c8__neon);
  }

  TEST(Q8DW_9c8_NEON, single_output_channels_eq_8_with_qmax) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmax(128)
      .test(q8dw_ukernel_9c8__neon);
  }

  TEST(Q8DW_9c8_NEON, multi_output_channels_eq_8) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .test(q8dw_ukernel_9c8__neon);
  }

  TEST(Q8DW_9c8_NEON, multi_output_channels_eq_8_with_subsampling) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .subsampling(2)
      .cr(8)
      .channels(8)
      .width(5)
      .test(q8dw_ukernel_9c8__neon);
  }

  TEST(Q8DW_9c8_NEON, multi_output_channels_eq_8_with_input_stride) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .inputStride(17)
      .test(q8dw_ukernel_9c8__neon);
  }

  TEST(Q8DW_9c8_NEON, multi_output_channels_eq_8_with_output_stride) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .outputStride(19)
      .test(q8dw_ukernel_9c8__neon);
  }

  TEST(Q8DW_9c8_NEON, single_output_channels_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(q8dw_ukernel_9c8__neon);
    }
  }

  TEST(Q8DW_9c8_NEON, multi_output_channels_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(q8dw_ukernel_9c8__neon);
    }
  }

  TEST(Q8DW_9c8_NEON, multi_output_channels_div_8_with_output_stride) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(171)
        .test(q8dw_ukernel_9c8__neon);
    }
  }

  TEST(Q8DW_9c8_NEON, single_output_channels_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(q8dw_ukernel_9c8__neon);
    }
  }

  TEST(Q8DW_9c8_NEON, single_output_channels_gt_8_with_qmin) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmin(128)
        .test(q8dw_ukernel_9c8__neon);
    }
  }

  TEST(Q8DW_9c8_NEON, single_output_channels_gt_8_with_qmax) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmax(128)
        .test(q8dw_ukernel_9c8__neon);
    }
  }

  TEST(Q8DW_9c8_NEON, multi_output_channels_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(q8dw_ukernel_9c8__neon);
    }
  }

  TEST(Q8DW_9c8_NEON, multi_output_channels_gt_8_with_output_stride) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(17)
        .test(q8dw_ukernel_9c8__neon);
    }
  }
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */

#if CPUINFO_ARCH_ARM
  TEST(Q8DW_9c8_AARCH32_NEON, single_output_channels_eq_8) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .test(q8dw_ukernel_9c8__aarch32_neon);
  }

  TEST(Q8DW_9c8_AARCH32_NEON, single_output_channels_eq_8_with_qmin) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmin(128)
      .test(q8dw_ukernel_9c8__aarch32_neon);
  }

  TEST(Q8DW_9c8_AARCH32_NEON, single_output_channels_eq_8_with_qmax) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmax(128)
      .test(q8dw_ukernel_9c8__aarch32_neon);
  }

  TEST(Q8DW_9c8_AARCH32_NEON, multi_output_channels_eq_8) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .test(q8dw_ukernel_9c8__aarch32_neon);
  }

  TEST(Q8DW_9c8_AARCH32_NEON, multi_output_channels_eq_8_with_subsampling) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .subsampling(2)
      .cr(8)
      .channels(8)
      .width(5)
      .test(q8dw_ukernel_9c8__aarch32_neon);
  }

  TEST(Q8DW_9c8_AARCH32_NEON, multi_output_channels_eq_8_with_input_stride) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .inputStride(17)
      .test(q8dw_ukernel_9c8__aarch32_neon);
  }

  TEST(Q8DW_9c8_AARCH32_NEON, multi_output_channels_eq_8_with_output_stride) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .outputStride(19)
      .test(q8dw_ukernel_9c8__aarch32_neon);
  }

  TEST(Q8DW_9c8_AARCH32_NEON, single_output_channels_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(q8dw_ukernel_9c8__aarch32_neon);
    }
  }

  TEST(Q8DW_9c8_AARCH32_NEON, multi_output_channels_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(q8dw_ukernel_9c8__aarch32_neon);
    }
  }

  TEST(Q8DW_9c8_AARCH32_NEON, multi_output_channels_div_8_with_output_stride) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(171)
        .test(q8dw_ukernel_9c8__aarch32_neon);
    }
  }

  TEST(Q8DW_9c8_AARCH32_NEON, single_output_channels_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(q8dw_ukernel_9c8__aarch32_neon);
    }
  }

  TEST(Q8DW_9c8_AARCH32_NEON, single_output_channels_gt_8_with_qmin) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmin(128)
        .test(q8dw_ukernel_9c8__aarch32_neon);
    }
  }

  TEST(Q8DW_9c8_AARCH32_NEON, single_output_channels_gt_8_with_qmax) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmax(128)
        .test(q8dw_ukernel_9c8__aarch32_neon);
    }
  }

  TEST(Q8DW_9c8_AARCH32_NEON, multi_output_channels_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(q8dw_ukernel_9c8__aarch32_neon);
    }
  }

  TEST(Q8DW_9c8_AARCH32_NEON, multi_output_channels_gt_8_with_output_stride) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(17)
        .test(q8dw_ukernel_9c8__aarch32_neon);
    }
  }
#endif /* CPUINFO_ARCH_ARM */

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  TEST(Q8DW_9c8_SSE2, single_output_channels_eq_8) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .test(q8dw_ukernel_9c8__sse2);
  }

  TEST(Q8DW_9c8_SSE2, single_output_channels_eq_8_with_qmin) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmin(128)
      .test(q8dw_ukernel_9c8__sse2);
  }

  TEST(Q8DW_9c8_SSE2, single_output_channels_eq_8_with_qmax) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmax(128)
      .test(q8dw_ukernel_9c8__sse2);
  }

  TEST(Q8DW_9c8_SSE2, multi_output_channels_eq_8) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .test(q8dw_ukernel_9c8__sse2);
  }

  TEST(Q8DW_9c8_SSE2, multi_output_channels_eq_8_with_subsampling) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .subsampling(2)
      .cr(8)
      .channels(8)
      .width(5)
      .test(q8dw_ukernel_9c8__sse2);
  }

  TEST(Q8DW_9c8_SSE2, multi_output_channels_eq_8_with_input_stride) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .inputStride(17)
      .test(q8dw_ukernel_9c8__sse2);
  }

  TEST(Q8DW_9c8_SSE2, multi_output_channels_eq_8_with_output_stride) {
    DepthwiseMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .outputStride(19)
      .test(q8dw_ukernel_9c8__sse2);
  }

  TEST(Q8DW_9c8_SSE2, single_output_channels_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(q8dw_ukernel_9c8__sse2);
    }
  }

  TEST(Q8DW_9c8_SSE2, multi_output_channels_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(q8dw_ukernel_9c8__sse2);
    }
  }

  TEST(Q8DW_9c8_SSE2, multi_output_channels_div_8_with_output_stride) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(171)
        .test(q8dw_ukernel_9c8__sse2);
    }
  }

  TEST(Q8DW_9c8_SSE2, single_output_channels_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(q8dw_ukernel_9c8__sse2);
    }
  }

  TEST(Q8DW_9c8_SSE2, single_output_channels_gt_8_with_qmin) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmin(128)
        .test(q8dw_ukernel_9c8__sse2);
    }
  }

  TEST(Q8DW_9c8_SSE2, single_output_channels_gt_8_with_qmax) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmax(128)
        .test(q8dw_ukernel_9c8__sse2);
    }
  }

  TEST(Q8DW_9c8_SSE2, multi_output_channels_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(q8dw_ukernel_9c8__sse2);
    }
  }

  TEST(Q8DW_9c8_SSE2, multi_output_channels_gt_8_with_output_stride) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DepthwiseMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(17)
        .test(q8dw_ukernel_9c8__sse2);
    }
  }
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */
