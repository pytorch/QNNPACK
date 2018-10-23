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
#include <qnnpack/sgemm.h>

// clang-format off

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(SGEMM_5x8_NEON, k_eq_2) {
  GemmTester()
    .mr(5)
    .nr(8)
    .np(8)
    .kr(1)
    .m(5)
    .n(8)
    .k(2)
    .testMicroKernel(sgemm_ukernel_5x8__neon);
}

TEST(SGEMM_5x8_NEON, k_eq_2_strided_a) {
  GemmTester()
    .mr(5)
    .nr(8)
    .np(8)
    .kr(1)
    .m(5)
    .n(8)
    .k(2)
    .aStride(37)
    .testMicroKernel(sgemm_ukernel_5x8__neon);
}

TEST(SGEMM_5x8_NEON, k_eq_2_strided_c) {
  GemmTester()
    .mr(5)
    .nr(8)
    .np(8)
    .kr(1)
    .m(5)
    .n(8)
    .k(2)
    .cStride(17)
    .testMicroKernel(sgemm_ukernel_5x8__neon);
}

TEST(SGEMM_5x8_NEON, k_eq_8_rmin128) {
  GemmTester()
    .mr(4)
    .nr(8)
    .np(8)
    .kr(1)
    .m(4)
    .n(8)
    .k(8)
    .qmin(128)
    .testMicroKernel(sgemm_ukernel_5x8__neon);
}

TEST(SGEMM_5x8_NEON, k_eq_8_qmax128) {
  GemmTester()
    .mr(4)
    .nr(8)
    .np(8)
    .kr(1)
    .m(4)
    .n(8)
    .k(8)
    .qmax(128)
    .testMicroKernel(sgemm_ukernel_5x8__neon);
}

TEST(SGEMM_5x8_NEON, k_gt_2) {
  for (size_t k = 3; k < 16; k++) {
    GemmTester()
      .mr(5)
      .nr(8)
      .np(8)
      .kr(1)
      .m(5)
      .n(8)
      .k(k)
      .testMicroKernel(sgemm_ukernel_5x8__neon);
  }
}

TEST(SGEMM_5x8_NEON, k_gt_2_strided_a) {
  for (size_t k = 3; k < 16; k++) {
    GemmTester()
      .mr(5)
      .nr(8)
      .np(8)
      .kr(1)
      .m(5)
      .n(8)
      .k(k)
      .aStride(37)
      .testMicroKernel(sgemm_ukernel_5x8__neon);
  }
}

TEST(SGEMM_5x8_NEON, k_gt_2_strided_c) {
  for (size_t k = 3; k < 16; k++) {
    GemmTester()
      .mr(5)
      .nr(8)
      .np(8)
      .kr(1)
      .m(5)
      .n(8)
      .k(k)
      .cStride(17)
      .testMicroKernel(sgemm_ukernel_5x8__neon);
  }
}

TEST(SGEMM_5x8_NEON, k_gt_2_subtile) {
  for (size_t k = 3; k < 16; k++) {
    for (uint32_t m = 1; m <= 5; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmTester()
          .mr(5)
          .nr(8)
          .np(8)
          .kr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(3)
          .testMicroKernel(sgemm_ukernel_5x8__neon);
      }
    }
  }
}

TEST(SGEMM_5x8_NEON, k_div_2) {
  for (size_t k = 2; k < 32; k += 2) {
    GemmTester()
      .mr(5)
      .nr(8)
      .np(8)
      .kr(1)
      .m(5)
      .n(8)
      .k(k)
      .testMicroKernel(sgemm_ukernel_5x8__neon);
  }
}

TEST(SGEMM_5x8_NEON, k_div_2_strided_a) {
  for (size_t k = 2; k < 32; k += 2) {
    GemmTester()
      .mr(5)
      .nr(8)
      .np(8)
      .kr(1)
      .m(5)
      .n(8)
      .k(k)
      .aStride(171)
      .testMicroKernel(sgemm_ukernel_5x8__neon);
  }
}

TEST(SGEMM_5x8_NEON, k_div_2_strided_c) {
  for (size_t k = 2; k < 32; k += 2) {
    GemmTester()
      .mr(5)
      .nr(8)
      .np(8)
      .kr(1)
      .m(5)
      .n(8)
      .k(k)
      .cStride(17)
      .testMicroKernel(sgemm_ukernel_5x8__neon);
  }
}

TEST(SGEMM_5x8_NEON, k_div_2_subtile) {
  for (size_t k = 2; k < 32; k += 6) {
    for (uint32_t m = 1; m <= 5; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmTester()
          .mr(5)
          .nr(8)
          .np(8)
          .kr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(3)
          .testMicroKernel(sgemm_ukernel_5x8__neon);
      }
    }
  }
}

TEST(SGEMM_6x8_NEON, k_eq_2) {
  GemmTester()
    .mr(6)
    .nr(8)
    .np(8)
    .kr(1)
    .m(6)
    .n(8)
    .k(2)
    .testMicroKernel(sgemm_ukernel_6x8__neon);
}

TEST(SGEMM_6x8_NEON, k_eq_2_strided_a) {
  GemmTester()
    .mr(6)
    .nr(8)
    .np(8)
    .kr(1)
    .m(6)
    .n(8)
    .k(2)
    .aStride(37)
    .testMicroKernel(sgemm_ukernel_6x8__neon);
}

TEST(SGEMM_6x8_NEON, k_eq_2_strided_c) {
  GemmTester()
    .mr(6)
    .nr(8)
    .np(8)
    .kr(1)
    .m(6)
    .n(8)
    .k(2)
    .cStride(17)
    .testMicroKernel(sgemm_ukernel_6x8__neon);
}

TEST(SGEMM_6x8_NEON, k_eq_8_qmin128) {
  GemmTester()
    .mr(6)
    .nr(8)
    .np(8)
    .kr(1)
    .m(6)
    .n(8)
    .k(8)
    .qmin(128)
    .testMicroKernel(sgemm_ukernel_6x8__neon);
}

TEST(SGEMM_6x8_NEON, k_eq_8_qmax128) {
  GemmTester()
    .mr(6)
    .nr(8)
    .np(8)
    .kr(1)
    .m(6)
    .n(8)
    .k(8)
    .qmax(128)
    .testMicroKernel(sgemm_ukernel_6x8__neon);
}

TEST(SGEMM_6x8_NEON, k_gt_2) {
  for (size_t k = 3; k < 16; k++) {
    GemmTester()
      .mr(6)
      .nr(8)
      .np(8)
      .kr(1)
      .m(6)
      .n(8)
      .k(k)
      .testMicroKernel(sgemm_ukernel_6x8__neon);
  }
}

TEST(SGEMM_6x8_NEON, k_gt_2_strided_a) {
  for (size_t k = 3; k < 16; k++) {
    GemmTester()
      .mr(6)
      .nr(8)
      .np(8)
      .kr(1)
      .m(6)
      .n(8)
      .k(k)
      .aStride(37)
      .testMicroKernel(sgemm_ukernel_6x8__neon);
  }
}

TEST(SGEMM_6x8_NEON, k_gt_2_strided_c) {
  for (size_t k = 3; k < 16; k++) {
    GemmTester()
      .mr(6)
      .nr(8)
      .np(8)
      .kr(1)
      .m(6)
      .n(8)
      .k(k)
      .cStride(17)
      .testMicroKernel(sgemm_ukernel_6x8__neon);
  }
}

TEST(SGEMM_6x8_NEON, k_gt_2_subtile) {
  for (size_t k = 3; k < 16; k++) {
    for (uint32_t m = 1; m <= 6; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmTester()
          .mr(6)
          .nr(8)
          .np(8)
          .kr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(3)
          .testMicroKernel(sgemm_ukernel_6x8__neon);
      }
    }
  }
}

TEST(SGEMM_6x8_NEON, k_div_2) {
  for (size_t k = 2; k < 32; k += 2) {
    GemmTester()
      .mr(6)
      .nr(8)
      .np(8)
      .kr(1)
      .m(6)
      .n(8)
      .k(k)
      .testMicroKernel(sgemm_ukernel_6x8__neon);
  }
}

TEST(SGEMM_6x8_NEON, k_div_2_strided_a) {
  for (size_t k = 2; k < 32; k += 2) {
    GemmTester()
      .mr(6)
      .nr(8)
      .np(8)
      .kr(1)
      .m(6)
      .n(8)
      .k(k)
      .aStride(171)
      .testMicroKernel(sgemm_ukernel_6x8__neon);
  }
}

TEST(SGEMM_6x8_NEON, k_div_2_strided_c) {
  for (size_t k = 2; k < 32; k += 2) {
    GemmTester()
      .mr(6)
      .nr(8)
      .np(8)
      .kr(1)
      .m(6)
      .n(8)
      .k(k)
      .cStride(17)
      .testMicroKernel(sgemm_ukernel_6x8__neon);
  }
}

TEST(SGEMM_6x8_NEON, k_div_2_subtile) {
  for (size_t k = 2; k < 32; k += 6) {
    for (uint32_t m = 1; m <= 6; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmTester()
          .mr(6)
          .nr(8)
          .np(8)
          .kr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(3)
          .testMicroKernel(sgemm_ukernel_6x8__neon);
      }
    }
  }
}
#endif

TEST(SGEMM_6x8_PSIMD, k_eq_2) {
  GemmTester()
    .mr(6)
    .nr(8)
    .np(8)
    .kr(1)
    .m(6)
    .n(8)
    .k(2)
    .testMicroKernel(sgemm_ukernel_6x8__psimd);
}

TEST(SGEMM_6x8_PSIMD, k_eq_2_strided_a) {
  GemmTester()
    .mr(6)
    .nr(8)
    .np(8)
    .kr(1)
    .m(6)
    .n(8)
    .k(2)
    .aStride(37)
    .testMicroKernel(sgemm_ukernel_6x8__psimd);
}

TEST(SGEMM_6x8_PSIMD, k_eq_2_strided_c) {
  GemmTester()
    .mr(6)
    .nr(8)
    .np(8)
    .kr(1)
    .m(6)
    .n(8)
    .k(2)
    .cStride(17)
    .testMicroKernel(sgemm_ukernel_6x8__psimd);
}

TEST(SGEMM_6x8_PSIMD, k_eq_8_qmin128) {
  GemmTester()
    .mr(6)
    .nr(8)
    .np(8)
    .kr(1)
    .m(6)
    .n(8)
    .k(8)
    .qmin(128)
    .testMicroKernel(sgemm_ukernel_6x8__psimd);
}

TEST(SGEMM_6x8_PSIMD, k_eq_8_qmax128) {
  GemmTester()
    .mr(6)
    .nr(8)
    .np(8)
    .kr(1)
    .m(6)
    .n(8)
    .k(8)
    .qmax(128)
    .testMicroKernel(sgemm_ukernel_6x8__psimd);
}

TEST(SGEMM_6x8_PSIMD, k_gt_2) {
  for (size_t k = 3; k < 16; k++) {
    GemmTester()
      .mr(6)
      .nr(8)
      .np(8)
      .kr(1)
      .m(6)
      .n(8)
      .k(k)
      .testMicroKernel(sgemm_ukernel_6x8__psimd);
  }
}

TEST(SGEMM_6x8_PSIMD, k_gt_2_strided_a) {
  for (size_t k = 3; k < 16; k++) {
    GemmTester()
      .mr(6)
      .nr(8)
      .np(8)
      .kr(1)
      .m(6)
      .n(8)
      .k(k)
      .aStride(37)
      .testMicroKernel(sgemm_ukernel_6x8__psimd);
  }
}

TEST(SGEMM_6x8_PSIMD, k_gt_2_strided_c) {
  for (size_t k = 3; k < 16; k++) {
    GemmTester()
      .mr(6)
      .nr(8)
      .np(8)
      .kr(1)
      .m(6)
      .n(8)
      .k(k)
      .cStride(17)
      .testMicroKernel(sgemm_ukernel_6x8__psimd);
  }
}

TEST(SGEMM_6x8_PSIMD, k_gt_2_subtile) {
  for (size_t k = 3; k < 16; k++) {
    for (uint32_t m = 1; m <= 6; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmTester()
          .mr(6)
          .nr(8)
          .np(8)
          .kr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(3)
          .testMicroKernel(sgemm_ukernel_6x8__psimd);
      }
    }
  }
}

TEST(SGEMM_6x8_PSIMD, k_div_2) {
  for (size_t k = 2; k < 32; k += 2) {
    GemmTester()
      .mr(6)
      .nr(8)
      .np(8)
      .kr(1)
      .m(6)
      .n(8)
      .k(k)
      .testMicroKernel(sgemm_ukernel_6x8__psimd);
  }
}

TEST(SGEMM_6x8_PSIMD, k_div_2_strided_a) {
  for (size_t k = 2; k < 32; k += 2) {
    GemmTester()
      .mr(6)
      .nr(8)
      .np(8)
      .kr(1)
      .m(6)
      .n(8)
      .k(k)
      .aStride(171)
      .testMicroKernel(sgemm_ukernel_6x8__psimd);
  }
}

TEST(SGEMM_6x8_PSIMD, k_div_2_strided_c) {
  for (size_t k = 2; k < 32; k += 2) {
    GemmTester()
      .mr(6)
      .nr(8)
      .np(8)
      .kr(1)
      .m(6)
      .n(8)
      .k(k)
      .cStride(17)
      .testMicroKernel(sgemm_ukernel_6x8__psimd);
  }
}

TEST(SGEMM_6x8_PSIMD, k_div_2_subtile) {
  for (size_t k = 2; k < 32; k += 6) {
    for (uint32_t m = 1; m <= 6; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmTester()
          .mr(6)
          .nr(8)
          .np(8)
          .kr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(3)
          .testMicroKernel(sgemm_ukernel_6x8__psimd);
      }
    }
  }
}
