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
#include <qnnpack/hgemm.h>

// clang-format off

#if CPUINFO_ARCH_ARM
  TEST(HGEMM_8x8_AARCH32_NEONFP16ARITH, k_eq_4) {
    GemmTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(4)
      .testMicroKernel(hgemm_ukernel_8x8__aarch32_neonfp16arith);
  }

  TEST(HGEMM_8x8_AARCH32_NEONFP16ARITH, k_eq_4_strided_a) {
    GemmTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(4)
      .aStride(37)
      .testMicroKernel(hgemm_ukernel_8x8__aarch32_neonfp16arith);
  }

  TEST(HGEMM_8x8_AARCH32_NEONFP16ARITH, k_eq_4_strided_c) {
    GemmTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(4)
      .cStride(17)
      .testMicroKernel(hgemm_ukernel_8x8__aarch32_neonfp16arith);
  }

  TEST(HGEMM_8x8_AARCH32_NEONFP16ARITH, k_eq_4_qmin128) {
    GemmTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(4)
      .qmin(128)
      .testMicroKernel(hgemm_ukernel_8x8__aarch32_neonfp16arith);
  }

  TEST(HGEMM_8x8_AARCH32_NEONFP16ARITH, k_eq_4_qmax128) {
    GemmTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(4)
      .qmax(128)
      .testMicroKernel(hgemm_ukernel_8x8__aarch32_neonfp16arith);
  }

  TEST(HGEMM_8x8_AARCH32_NEONFP16ARITH, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .testMicroKernel(hgemm_ukernel_8x8__aarch32_neonfp16arith);
    }
  }

  TEST(HGEMM_8x8_AARCH32_NEONFP16ARITH, k_gt_4_strided_a) {
    for (size_t k = 5; k < 8; k++) {
      GemmTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(37)
        .testMicroKernel(hgemm_ukernel_8x8__aarch32_neonfp16arith);
    }
  }

  TEST(HGEMM_8x8_AARCH32_NEONFP16ARITH, k_gt_4_strided_c) {
    for (size_t k = 5; k < 8; k++) {
      GemmTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .cStride(17)
        .testMicroKernel(hgemm_ukernel_8x8__aarch32_neonfp16arith);
    }
  }

  TEST(HGEMM_8x8_AARCH32_NEONFP16ARITH, k_gt_4_subtile) {
    for (size_t k = 5; k < 8; k++) {
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
            .iterations(3)
            .testMicroKernel(hgemm_ukernel_8x8__aarch32_neonfp16arith);
        }
      }
    }
  }

  TEST(HGEMM_8x8_AARCH32_NEONFP16ARITH, k_div_4) {
    for (size_t k = 8; k < 64; k += 4) {
      GemmTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .testMicroKernel(hgemm_ukernel_8x8__aarch32_neonfp16arith);
    }
  }

  TEST(HGEMM_8x8_AARCH32_NEONFP16ARITH, k_div_4_strided_a) {
    for (size_t k = 8; k < 64; k += 4) {
      GemmTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(171)
        .testMicroKernel(hgemm_ukernel_8x8__aarch32_neonfp16arith);
    }
  }

  TEST(HGEMM_8x8_AARCH32_NEONFP16ARITH, k_div_4_strided_c) {
    for (size_t k = 8; k < 64; k += 4) {
      GemmTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .cStride(17)
        .testMicroKernel(hgemm_ukernel_8x8__aarch32_neonfp16arith);
    }
  }

  TEST(HGEMM_8x8_AARCH32_NEONFP16ARITH, k_div_4_subtile) {
    for (size_t k = 8; k < 64; k += 12) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 8; n <= 8; n++) {
          GemmTester()
            .mr(8)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .testMicroKernel(hgemm_ukernel_8x8__aarch32_neonfp16arith);
        }
      }
    }
  }
#endif
