/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cpuinfo.h>
#include <gemm-tester.h>
#include <qnnpack/q8gemm.h>

// clang-format off

#if CPUINFO_ARCH_ARM
  TEST(Q8CONV_4x8_AARCH32_NEON, k_eq_8) {
    GemmTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aStride(37)
      .testMicroKernel(q8conv_ukernel_4x8__aarch32_neon);
  }

  TEST(Q8CONV_4x8_AARCH32_NEON, k_eq_8_strided_c) {
    GemmTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aStride(37)
      .cStride(17)
      .testMicroKernel(q8conv_ukernel_4x8__aarch32_neon);
  }

  TEST(Q8CONV_4x8_AARCH32_NEON, k_eq_8_qmin128) {
    GemmTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmin(128)
      .testMicroKernel(q8conv_ukernel_4x8__aarch32_neon);
  }

  TEST(Q8CONV_4x8_AARCH32_NEON, k_eq_8_qmax128) {
    GemmTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmax(128)
      .testMicroKernel(q8conv_ukernel_4x8__aarch32_neon);
  }

  TEST(Q8CONV_4x8_AARCH32_NEON, k_eq_8_azp_only) {
    GemmTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(255)
      .bZeroPoint(0)
      .testMicroKernel(q8conv_ukernel_4x8__aarch32_neon);
  }

  TEST(Q8CONV_4x8_AARCH32_NEON, k_eq_8_bzp_only) {
    GemmTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(255)
      .testMicroKernel(q8conv_ukernel_4x8__aarch32_neon);
  }

  TEST(Q8CONV_4x8_AARCH32_NEON, k_gt_8) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(37)
        .testMicroKernel(q8conv_ukernel_4x8__aarch32_neon);
    }
  }

  TEST(Q8CONV_4x8_AARCH32_NEON, k_gt_8_strided_c) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(37)
        .cStride(17)
        .testMicroKernel(q8conv_ukernel_4x8__aarch32_neon);
    }
  }

  TEST(Q8CONV_4x8_AARCH32_NEON, k_gt_8_azp_only) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(37)
        .aZeroPoint(255)
        .bZeroPoint(0)
        .testMicroKernel(q8conv_ukernel_4x8__aarch32_neon);
    }
  }

  TEST(Q8CONV_4x8_AARCH32_NEON, k_gt_8_bzp_only) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(37)
        .aZeroPoint(0)
        .bZeroPoint(255)
        .testMicroKernel(q8conv_ukernel_4x8__aarch32_neon);
    }
  }

  TEST(Q8CONV_4x8_AARCH32_NEON, k_gt_8_subtile) {
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .aStride(37)
            .iterations(3)
            .testMicroKernel(q8conv_ukernel_4x8__aarch32_neon);
        }
      }
    }
  }

  TEST(Q8CONV_4x8_AARCH32_NEON, k_div_8) {
    for (size_t k = 16; k < 128; k += 8) {
      GemmTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(171)
        .testMicroKernel(q8conv_ukernel_4x8__aarch32_neon);
    }
  }

  TEST(Q8CONV_4x8_AARCH32_NEON, k_div_8_strided_c) {
    for (size_t k = 16; k < 128; k += 8) {
      GemmTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(171)
        .cStride(17)
        .testMicroKernel(q8conv_ukernel_4x8__aarch32_neon);
    }
  }

  TEST(Q8CONV_4x8_AARCH32_NEON, k_div_8_subtile) {
    for (size_t k = 16; k < 128; k += 24) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .aStride(171)
            .iterations(3)
            .testMicroKernel(q8conv_ukernel_4x8__aarch32_neon);
        }
      }
    }
  }
#endif

#if CPUINFO_ARCH_ARM64
  TEST(Q8CONV_8x8_AARCH64_NEON, k_eq_8) {
    GemmTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aStride(37)
      .testMicroKernel(q8conv_ukernel_8x8__aarch64_neon);
  }

  TEST(Q8CONV_8x8_AARCH64_NEON, k_eq_8_strided_c) {
    GemmTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aStride(37)
      .cStride(17)
      .testMicroKernel(q8conv_ukernel_8x8__aarch64_neon);
  }

  TEST(Q8CONV_8x8_AARCH64_NEON, k_eq_8_qmin128) {
    GemmTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .qmin(128)
      .testMicroKernel(q8conv_ukernel_8x8__aarch64_neon);
  }

  TEST(Q8CONV_8x8_AARCH64_NEON, k_eq_8_qmax128) {
    GemmTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .qmax(128)
      .testMicroKernel(q8conv_ukernel_8x8__aarch64_neon);
  }

  TEST(Q8CONV_8x8_AARCH64_NEON, k_eq_8_azp_only) {
    GemmTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aZeroPoint(255)
      .bZeroPoint(0)
      .testMicroKernel(q8conv_ukernel_8x8__aarch64_neon);
  }

  TEST(Q8CONV_8x8_AARCH64_NEON, k_eq_8_bzp_only) {
    GemmTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(255)
      .testMicroKernel(q8conv_ukernel_8x8__aarch64_neon);
  }

  TEST(Q8CONV_8x8_AARCH64_NEON, k_gt_8) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(37)
        .testMicroKernel(q8conv_ukernel_8x8__aarch64_neon);
    }
  }

  TEST(Q8CONV_8x8_AARCH64_NEON, k_gt_8_strided_c) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(37)
        .cStride(17)
        .testMicroKernel(q8conv_ukernel_8x8__aarch64_neon);
    }
  }

  TEST(Q8CONV_8x8_AARCH64_NEON, k_gt_8_azp_only) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(37)
        .aZeroPoint(255)
        .bZeroPoint(0)
        .testMicroKernel(q8conv_ukernel_8x8__aarch64_neon);
    }
  }

  TEST(Q8CONV_8x8_AARCH64_NEON, k_gt_8_bzp_only) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(37)
        .aZeroPoint(0)
        .bZeroPoint(255)
        .testMicroKernel(q8conv_ukernel_8x8__aarch64_neon);
    }
  }

  TEST(Q8CONV_8x8_AARCH64_NEON, k_gt_8_subtile) {
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 8; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmTester()
            .mr(8)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .aStride(37)
            .iterations(3)
            .testMicroKernel(q8conv_ukernel_8x8__aarch64_neon);
        }
      }
    }
  }

  TEST(Q8CONV_8x8_AARCH64_NEON, k_div_8) {
    for (size_t k = 16; k < 128; k += 8) {
      GemmTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(171)
        .testMicroKernel(q8conv_ukernel_8x8__aarch64_neon);
    }
  }

  TEST(Q8CONV_8x8_AARCH64_NEON, k_div_8_strided_c) {
    for (size_t k = 16; k < 128; k += 8) {
      GemmTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(171)
        .cStride(17)
        .testMicroKernel(q8conv_ukernel_8x8__aarch64_neon);
    }
  }

  TEST(Q8CONV_8x8_AARCH64_NEON, k_div_8_subtile) {
    for (size_t k = 16; k < 128; k += 24) {
      for (uint32_t m = 1; m <= 8; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmTester()
            .mr(8)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .aStride(171)
            .iterations(3)
            .testMicroKernel(q8conv_ukernel_8x8__aarch64_neon);
        }
      }
    }
  }
#endif

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  TEST(Q8CONV_4x8_NEON, k_eq_8) {
    GemmTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aStride(37)
      .testMicroKernel(q8conv_ukernel_4x8__neon);
  }

  TEST(Q8CONV_4x8_NEON, k_eq_8_strided_c) {
    GemmTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aStride(37)
      .cStride(17)
      .testMicroKernel(q8conv_ukernel_4x8__neon);
  }

  TEST(Q8CONV_4x8_NEON, k_eq_8_qmin128) {
    GemmTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmin(128)
      .testMicroKernel(q8conv_ukernel_4x8__neon);
  }

  TEST(Q8CONV_4x8_NEON, k_eq_8_qmax128) {
    GemmTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmax(128)
      .testMicroKernel(q8conv_ukernel_4x8__neon);
  }

  TEST(Q8CONV_4x8_NEON, k_eq_8_azp_only) {
    GemmTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(255)
      .bZeroPoint(0)
      .testMicroKernel(q8conv_ukernel_4x8__neon);
  }

  TEST(Q8CONV_4x8_NEON, k_eq_8_bzp_only) {
    GemmTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(255)
      .testMicroKernel(q8conv_ukernel_4x8__neon);
  }

  TEST(Q8CONV_4x8_NEON, k_gt_8) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(37)
        .testMicroKernel(q8conv_ukernel_4x8__neon);
    }
  }

  TEST(Q8CONV_4x8_NEON, k_gt_8_strided_c) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(37)
        .cStride(17)
        .testMicroKernel(q8conv_ukernel_4x8__neon);
    }
  }

  TEST(Q8CONV_4x8_NEON, k_gt_8_azp_only) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(37)
        .aZeroPoint(255)
        .bZeroPoint(0)
        .testMicroKernel(q8conv_ukernel_4x8__neon);
    }
  }

  TEST(Q8CONV_4x8_NEON, k_gt_8_bzp_only) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(37)
        .aZeroPoint(0)
        .bZeroPoint(255)
        .testMicroKernel(q8conv_ukernel_4x8__neon);
    }
  }

  TEST(Q8CONV_4x8_NEON, k_gt_8_subtile) {
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .aStride(37)
            .iterations(3)
            .testMicroKernel(q8conv_ukernel_4x8__neon);
        }
      }
    }
  }

  TEST(Q8CONV_4x8_NEON, k_div_8) {
    for (size_t k = 16; k < 128; k += 8) {
      GemmTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(171)
        .testMicroKernel(q8conv_ukernel_4x8__neon);
    }
  }

  TEST(Q8CONV_4x8_NEON, k_div_8_strided_c) {
    for (size_t k = 16; k < 128; k += 8) {
      GemmTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(171)
        .cStride(17)
        .testMicroKernel(q8conv_ukernel_4x8__neon);
    }
  }

  TEST(Q8CONV_4x8_NEON, k_div_8_subtile) {
    for (size_t k = 16; k < 128; k += 24) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .aStride(171)
            .iterations(3)
            .testMicroKernel(q8conv_ukernel_4x8__neon);
        }
      }
    }
  }

  TEST(Q8CONV_8x8_NEON, k_eq_8) {
    GemmTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aStride(37)
      .testMicroKernel(q8conv_ukernel_8x8__neon);
  }

  TEST(Q8CONV_8x8_NEON, k_eq_8_strided_c) {
    GemmTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aStride(37)
      .cStride(17)
      .testMicroKernel(q8conv_ukernel_8x8__neon);
  }

  TEST(Q8CONV_8x8_NEON, k_eq_8_qmin128) {
    GemmTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .qmin(128)
      .testMicroKernel(q8conv_ukernel_8x8__neon);
  }

  TEST(Q8CONV_8x8_NEON, k_eq_8_qmax128) {
    GemmTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .qmax(128)
      .testMicroKernel(q8conv_ukernel_8x8__neon);
  }

  TEST(Q8CONV_8x8_NEON, k_eq_8_azp_only) {
    GemmTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aZeroPoint(255)
      .bZeroPoint(0)
      .testMicroKernel(q8conv_ukernel_8x8__neon);
  }

  TEST(Q8CONV_8x8_NEON, k_eq_8_bzp_only) {
    GemmTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(255)
      .testMicroKernel(q8conv_ukernel_8x8__neon);
  }

  TEST(Q8CONV_8x8_NEON, k_gt_8) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(37)
        .testMicroKernel(q8conv_ukernel_8x8__neon);
    }
  }

  TEST(Q8CONV_8x8_NEON, k_gt_8_strided_c) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(37)
        .cStride(17)
        .testMicroKernel(q8conv_ukernel_8x8__neon);
    }
  }

  TEST(Q8CONV_8x8_NEON, k_gt_8_azp_only) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(37)
        .aZeroPoint(255)
        .bZeroPoint(0)
        .testMicroKernel(q8conv_ukernel_8x8__neon);
    }
  }

  TEST(Q8CONV_8x8_NEON, k_gt_8_bzp_only) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(37)
        .aZeroPoint(0)
        .bZeroPoint(255)
        .testMicroKernel(q8conv_ukernel_8x8__neon);
    }
  }

  TEST(Q8CONV_8x8_NEON, k_gt_8_subtile) {
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 8; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmTester()
            .mr(8)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .aStride(37)
            .iterations(3)
            .testMicroKernel(q8conv_ukernel_8x8__neon);
        }
      }
    }
  }

  TEST(Q8CONV_8x8_NEON, k_div_8) {
    for (size_t k = 16; k < 128; k += 8) {
      GemmTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(171)
        .testMicroKernel(q8conv_ukernel_8x8__neon);
    }
  }

  TEST(Q8CONV_8x8_NEON, k_div_8_strided_c) {
    for (size_t k = 16; k < 128; k += 8) {
      GemmTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(171)
        .cStride(17)
        .testMicroKernel(q8conv_ukernel_8x8__neon);
    }
  }

  TEST(Q8CONV_8x8_NEON, k_div_8_subtile) {
    for (size_t k = 16; k < 128; k += 24) {
      for (uint32_t m = 1; m <= 8; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmTester()
            .mr(8)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .aStride(171)
            .iterations(3)
            .testMicroKernel(q8conv_ukernel_8x8__neon);
        }
      }
    }
  }
#endif

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  TEST(Q8CONV_4x4c2_SSE2, k_eq_8) {
    GemmTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(8)
      .aStride(37)
      .testMicroKernel(q8conv_ukernel_4x4c2__sse2);
  }

  TEST(Q8CONV_4x4c2_SSE2, k_eq_8_strided_c) {
    GemmTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(8)
      .aStride(37)
      .cStride(17)
      .testMicroKernel(q8conv_ukernel_4x4c2__sse2);
  }

  TEST(Q8CONV_4x4c2_SSE2, k_eq_8_qmin128) {
    GemmTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(8)
      .qmin(128)
      .testMicroKernel(q8conv_ukernel_4x4c2__sse2);
  }

  TEST(Q8CONV_4x4c2_SSE2, k_eq_8_qmax128) {
    GemmTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(8)
      .qmax(128)
      .testMicroKernel(q8conv_ukernel_4x4c2__sse2);
  }

  TEST(Q8CONV_4x4c2_SSE2, k_eq_8_azp_only) {
    GemmTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(8)
      .aZeroPoint(255)
      .bZeroPoint(0)
      .testMicroKernel(q8conv_ukernel_4x4c2__sse2);
  }

  TEST(Q8CONV_4x4c2_SSE2, k_eq_8_bzp_only) {
    GemmTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(255)
      .testMicroKernel(q8conv_ukernel_4x4c2__sse2);
  }

  TEST(Q8CONV_4x4c2_SSE2, k_gt_8) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(4)
        .nr(4)
        .np(4)
        .kr(2)
        .m(4)
        .n(4)
        .k(k)
        .aStride(37)
        .testMicroKernel(q8conv_ukernel_4x4c2__sse2);
    }
  }

  TEST(Q8CONV_4x4c2_SSE2, k_gt_8_strided_c) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(4)
        .nr(4)
        .np(4)
        .kr(2)
        .m(4)
        .n(4)
        .k(k)
        .aStride(37)
        .cStride(17)
        .testMicroKernel(q8conv_ukernel_4x4c2__sse2);
    }
  }

  TEST(Q8CONV_4x4c2_SSE2, k_gt_8_azp_only) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(4)
        .nr(4)
        .np(4)
        .kr(2)
        .m(4)
        .n(4)
        .k(k)
        .aStride(37)
        .aZeroPoint(255)
        .bZeroPoint(0)
        .testMicroKernel(q8conv_ukernel_4x4c2__sse2);
    }
  }

  TEST(Q8CONV_4x4c2_SSE2, k_gt_8_bzp_only) {
    for (size_t k = 9; k < 16; k++) {
      GemmTester()
        .mr(4)
        .nr(4)
        .np(4)
        .kr(2)
        .m(4)
        .n(4)
        .k(k)
        .aStride(37)
        .aZeroPoint(0)
        .bZeroPoint(255)
        .testMicroKernel(q8conv_ukernel_4x4c2__sse2);
    }
  }

  TEST(Q8CONV_4x4c2_SSE2, k_gt_8_subtile) {
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmTester()
            .mr(4)
            .nr(4)
            .np(4)
            .kr(2)
            .m(m)
            .n(n)
            .k(k)
            .aStride(37)
            .iterations(3)
            .testMicroKernel(q8conv_ukernel_4x4c2__sse2);
        }
      }
    }
  }

  TEST(Q8CONV_4x4c2_SSE2, k_div_8) {
    for (size_t k = 16; k < 128; k += 8) {
      GemmTester()
        .mr(4)
        .nr(4)
        .np(4)
        .kr(2)
        .m(4)
        .n(4)
        .k(k)
        .aStride(171)
        .testMicroKernel(q8conv_ukernel_4x4c2__sse2);
    }
  }

  TEST(Q8CONV_4x4c2_SSE2, k_div_8_strided_c) {
    for (size_t k = 16; k < 128; k += 8) {
      GemmTester()
        .mr(4)
        .nr(4)
        .np(4)
        .kr(2)
        .m(4)
        .n(4)
        .k(k)
        .aStride(171)
        .cStride(17)
        .testMicroKernel(q8conv_ukernel_4x4c2__sse2);
    }
  }

  TEST(Q8CONV_4x4c2_SSE2, k_div_8_subtile) {
    for (size_t k = 16; k < 128; k += 24) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmTester()
            .mr(4)
            .nr(4)
            .np(4)
            .kr(2)
            .m(m)
            .n(n)
            .k(k)
            .aStride(171)
            .iterations(3)
            .testMicroKernel(q8conv_ukernel_4x4c2__sse2);
        }
      }
    }
  }
#endif
