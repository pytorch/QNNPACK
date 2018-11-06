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
#include <uvadd-microkernel-tester.h>
#include <qnnpack/q8add.h>


#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
TEST(Q8UVADD__SSE2, n_eq_8) {
  UVAddMicrokernelTester()
    .n(8)
    .test(q8uvadd_ukernel__sse2);
}

TEST(Q8UVADD__SSE2, n_div_8) {
  for (size_t n = 8; n < 128; n += 24) {
    UVAddMicrokernelTester()
      .n(n)
      .test(q8uvadd_ukernel__sse2);
  }
}

TEST(Q8UVADD__SSE2, n_gt_8) {
  for (size_t n = 9; n < 16; n++) {
    UVAddMicrokernelTester()
      .n(n)
      .test(q8uvadd_ukernel__sse2);
  }
}

TEST(Q8UVADD__SSE2, n_lt_8) {
  for (size_t n = 1; n < 8; n++) {
    UVAddMicrokernelTester()
      .n(n)
      .test(q8uvadd_ukernel__sse2);
  }
}

TEST(Q8UVADD__SSE2, inplace_a) {
  for (size_t n = 1; n < 128; n += 11) {
    UVAddMicrokernelTester()
      .iterations(1)
      .n(n)
      .inplaceA(true)
      .test(q8uvadd_ukernel__sse2);
  }
}

TEST(Q8UVADD__SSE2, inplace_b) {
  for (size_t n = 1; n < 128; n += 11) {
    UVAddMicrokernelTester()
      .iterations(1)
      .n(n)
      .inplaceB(true)
      .test(q8uvadd_ukernel__sse2);
  }
}

TEST(Q8UVADD__SSE2, inplace_a_and_b) {
  for (size_t n = 1; n < 128; n += 11) {
    UVAddMicrokernelTester()
      .iterations(1)
      .n(n)
      .inplaceA(true)
      .inplaceB(true)
      .test(q8uvadd_ukernel__sse2);
  }
}

TEST(Q8UVADD__SSE2, a_scale) {
  for (size_t n = 1; n < 128; n += 11) {
    for (float aScale = 1.0e-2; aScale < 1.0e+2; aScale *= 1.7f) {
      UVAddMicrokernelTester()
        .iterations(1)
        .n(n)
        .aScale(aScale)
        .test(q8uvadd_ukernel__sse2);
    }
  }
}

TEST(Q8UVADD__SSE2, b_scale) {
  for (size_t n = 1; n < 128; n += 11) {
    for (float bScale = 1.0e-2; bScale < 1.0e+2; bScale *= 1.7f) {
      UVAddMicrokernelTester()
        .iterations(1)
        .n(n)
        .bScale(bScale)
        .test(q8uvadd_ukernel__sse2);
    }
  }
}

TEST(Q8UVADD__SSE2, y_scale) {
  for (size_t n = 1; n < 128; n += 11) {
    for (float yScale = 1.0e-2; yScale < 1.0e+2; yScale *= 1.7f) {
      UVAddMicrokernelTester()
        .iterations(1)
        .n(n)
        .yScale(yScale)
        .test(q8uvadd_ukernel__sse2);
    }
  }
}

TEST(Q8UVADD__SSE2, a_zero_point) {
  for (size_t n = 1; n < 128; n += 11) {
    for (int32_t aZeroPoint = 0; aZeroPoint <= 255; aZeroPoint += 51) {
      UVAddMicrokernelTester()
        .iterations(1)
        .n(n)
        .aZeroPoint(uint8_t(aZeroPoint))
        .test(q8uvadd_ukernel__sse2);
    }
  }
}

TEST(Q8UVADD__SSE2, b_zero_point) {
  for (size_t n = 1; n < 128; n += 11) {
    for (int32_t bZeroPoint = 0; bZeroPoint <= 255; bZeroPoint += 51) {
      UVAddMicrokernelTester()
        .iterations(1)
        .n(n)
        .bZeroPoint(uint8_t(bZeroPoint))
        .test(q8uvadd_ukernel__sse2);
    }
  }
}

TEST(Q8UVADD__SSE2, y_zero_point) {
  for (size_t n = 1; n < 128; n += 11) {
    for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
      UVAddMicrokernelTester()
        .iterations(1)
        .n(n)
        .yZeroPoint(uint8_t(yZeroPoint))
        .test(q8uvadd_ukernel__sse2);
    }
  }
}

TEST(Q8UVADD__SSE2, qmin) {
  for (size_t n = 1; n < 128; n += 11) {
    UVAddMicrokernelTester()
      .iterations(1)
      .n(n)
      .qmin(128)
      .test(q8uvadd_ukernel__sse2);
  }
}

TEST(Q8UVADD__SSE2, qmax) {
  for (size_t n = 1; n < 128; n += 11) {
    UVAddMicrokernelTester()
      .iterations(1)
      .n(n)
      .qmax(128)
      .test(q8uvadd_ukernel__sse2);
  }
}
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(Q8UVADD__NEON, n_eq_8) {
  UVAddMicrokernelTester()
    .n(8)
    .test(q8uvadd_ukernel__neon);
}

TEST(Q8UVADD__NEON, n_div_8) {
  for (size_t n = 8; n < 128; n += 24) {
    UVAddMicrokernelTester()
      .n(n)
      .test(q8uvadd_ukernel__neon);
  }
}

TEST(Q8UVADD__NEON, n_gt_8) {
  for (size_t n = 9; n < 16; n++) {
    UVAddMicrokernelTester()
      .n(n)
      .test(q8uvadd_ukernel__neon);
  }
}

TEST(Q8UVADD__NEON, n_lt_8) {
  for (size_t n = 1; n < 8; n++) {
    UVAddMicrokernelTester()
      .n(n)
      .test(q8uvadd_ukernel__neon);
  }
}

TEST(Q8UVADD__NEON, inplace_a) {
  for (size_t n = 1; n < 128; n += 11) {
    UVAddMicrokernelTester()
      .iterations(1)
      .n(n)
      .inplaceA(true)
      .test(q8uvadd_ukernel__neon);
  }
}

TEST(Q8UVADD__NEON, inplace_b) {
  for (size_t n = 1; n < 128; n += 11) {
    UVAddMicrokernelTester()
      .iterations(1)
      .n(n)
      .inplaceB(true)
      .test(q8uvadd_ukernel__neon);
  }
}

TEST(Q8UVADD__NEON, inplace_a_and_b) {
  for (size_t n = 1; n < 128; n += 11) {
    UVAddMicrokernelTester()
      .iterations(1)
      .n(n)
      .inplaceA(true)
      .inplaceB(true)
      .test(q8uvadd_ukernel__neon);
  }
}

TEST(Q8UVADD__NEON, a_scale) {
  for (size_t n = 1; n < 128; n += 11) {
    for (float aScale = 1.0e-2; aScale < 1.0e+2; aScale *= 1.7f) {
      UVAddMicrokernelTester()
        .iterations(1)
        .n(n)
        .aScale(aScale)
        .test(q8uvadd_ukernel__neon);
    }
  }
}

TEST(Q8UVADD__NEON, b_scale) {
  for (size_t n = 1; n < 128; n += 11) {
    for (float bScale = 1.0e-2; bScale < 1.0e+2; bScale *= 1.7f) {
      UVAddMicrokernelTester()
        .iterations(1)
        .n(n)
        .bScale(bScale)
        .test(q8uvadd_ukernel__neon);
    }
  }
}

TEST(Q8UVADD__NEON, y_scale) {
  for (size_t n = 1; n < 128; n += 11) {
    for (float yScale = 1.0e-2; yScale < 1.0e+2; yScale *= 1.7f) {
      UVAddMicrokernelTester()
        .iterations(1)
        .n(n)
        .yScale(yScale)
        .test(q8uvadd_ukernel__neon);
    }
  }
}

TEST(Q8UVADD__NEON, a_zero_point) {
  for (size_t n = 1; n < 128; n += 11) {
    for (int32_t aZeroPoint = 0; aZeroPoint <= 255; aZeroPoint += 51) {
      UVAddMicrokernelTester()
        .iterations(1)
        .n(n)
        .aZeroPoint(uint8_t(aZeroPoint))
        .test(q8uvadd_ukernel__neon);
    }
  }
}

TEST(Q8UVADD__NEON, b_zero_point) {
  for (size_t n = 1; n < 128; n += 11) {
    for (int32_t bZeroPoint = 0; bZeroPoint <= 255; bZeroPoint += 51) {
      UVAddMicrokernelTester()
        .iterations(1)
        .n(n)
        .bZeroPoint(uint8_t(bZeroPoint))
        .test(q8uvadd_ukernel__neon);
    }
  }
}

TEST(Q8UVADD__NEON, y_zero_point) {
  for (size_t n = 1; n < 128; n += 11) {
    for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
      UVAddMicrokernelTester()
        .iterations(1)
        .n(n)
        .yZeroPoint(uint8_t(yZeroPoint))
        .test(q8uvadd_ukernel__neon);
    }
  }
}

TEST(Q8UVADD__NEON, qmin) {
  for (size_t n = 1; n < 128; n += 11) {
    UVAddMicrokernelTester()
      .iterations(1)
      .n(n)
      .qmin(128)
      .test(q8uvadd_ukernel__neon);
  }
}

TEST(Q8UVADD__NEON, qmax) {
  for (size_t n = 1; n < 128; n += 11) {
    UVAddMicrokernelTester()
      .iterations(1)
      .n(n)
      .qmax(128)
      .test(q8uvadd_ukernel__neon);
  }
}
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */
