#pragma once

#include <cpuinfo.h>


#define TEST_REQUIRES_X86_SSE2 \
  do { \
    if (!cpuinfo_initialize() || !cpuinfo_has_x86_sse2()) { \
      return; \
    } \
  } while (0)

#define TEST_REQUIRES_ARM_NEON \
  do { \
    if (!cpuinfo_initialize() || !cpuinfo_has_arm_neon()) { \
      return; \
    } \
  } while (0)

#define TEST_REQUIRES_ARM_NEON_FP16_ARITH \
  do { \
    if (!cpuinfo_initialize() || !cpuinfo_has_arm_neon_fp16_arith()) { \
      return; \
    } \
  } while (0)
