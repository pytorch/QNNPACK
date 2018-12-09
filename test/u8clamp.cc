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
#include <clamp-microkernel-tester.h>
#include <qnnpack/u8clamp.h>


#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
TEST(U8CLAMP__SSE2, n_eq_8) {
  ClampMicrokernelTester()
    .n(8)
    .test(u8clamp_ukernel__sse2);
}

TEST(U8CLAMP__SSE2, n_div_8) {
  for (size_t n = 8; n < 512; n += 8) {
    ClampMicrokernelTester()
      .n(n)
      .test(u8clamp_ukernel__sse2);
  }
}

TEST(U8CLAMP__SSE2, n_gt_8) {
  for (size_t n = 9; n < 16; n++) {
    ClampMicrokernelTester()
      .n(n)
      .test(u8clamp_ukernel__sse2);
  }
}

TEST(U8CLAMP__SSE2, n_lt_8) {
  for (size_t n = 1; n < 8; n++) {
    ClampMicrokernelTester()
      .n(n)
      .test(u8clamp_ukernel__sse2);
  }
}

TEST(U8CLAMP__SSE2, inplace) {
  for (size_t n = 1; n < 128; n += 5) {
    ClampMicrokernelTester()
      .iterations(1)
      .n(n)
      .inplace(true)
      .test(u8clamp_ukernel__sse2);
  }
}

TEST(U8CLAMP__SSE2, qmin) {
  for (size_t n = 1; n < 128; n += 11) {
    ClampMicrokernelTester()
      .iterations(1)
      .n(n)
      .qmin(128)
      .test(u8clamp_ukernel__sse2);
  }
}

TEST(U8CLAMP__SSE2, qmax) {
  for (size_t n = 1; n < 128; n += 11) {
    ClampMicrokernelTester()
      .iterations(1)
      .n(n)
      .qmax(128)
      .test(u8clamp_ukernel__sse2);
  }
}
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(U8CLAMP__NEON, n_eq_8) {
  ClampMicrokernelTester()
    .n(8)
    .test(u8clamp_ukernel__neon);
}

TEST(U8CLAMP__NEON, n_div_8) {
  for (size_t n = 8; n < 512; n += 8) {
    ClampMicrokernelTester()
      .n(n)
      .test(u8clamp_ukernel__neon);
  }
}

TEST(U8CLAMP__NEON, n_gt_8) {
  for (size_t n = 9; n < 16; n++) {
    ClampMicrokernelTester()
      .n(n)
      .test(u8clamp_ukernel__neon);
  }
}

TEST(U8CLAMP__NEON, n_lt_8) {
  for (size_t n = 1; n < 8; n++) {
    ClampMicrokernelTester()
      .n(n)
      .test(u8clamp_ukernel__neon);
  }
}

TEST(U8CLAMP__NEON, inplace) {
  for (size_t n = 1; n < 128; n += 5) {
    ClampMicrokernelTester()
      .iterations(1)
      .n(n)
      .inplace(true)
      .test(u8clamp_ukernel__neon);
  }
}

TEST(U8CLAMP__NEON, qmin) {
  for (size_t n = 1; n < 128; n += 11) {
    ClampMicrokernelTester()
      .iterations(1)
      .n(n)
      .qmin(128)
      .test(u8clamp_ukernel__neon);
  }
}

TEST(U8CLAMP__NEON, qmax) {
  for (size_t n = 1; n < 128; n += 11) {
    ClampMicrokernelTester()
      .iterations(1)
      .n(n)
      .qmax(128)
      .test(u8clamp_ukernel__neon);
  }
}
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */
