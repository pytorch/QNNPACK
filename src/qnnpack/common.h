/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once


#if defined(__GNUC__)
  #if defined(__clang__) || (__GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ >= 5)
    #define QNNP_UNREACHABLE do { __builtin_unreachable(); } while (0)
  #else
    #define QNNP_UNREACHABLE do { __builtin_trap(); } while (0)
  #endif
#elif defined(_MSC_VER)
  #define QNNP_UNREACHABLE __assume(0)
#else
  #define QNNP_UNREACHABLE do { } while (0)
#endif

#if defined(_MSC_VER)
#define QNNP_ALIGN(alignment) __declspec(align(alignment))
#else
#define QNNP_ALIGN(alignment) __attribute__((__aligned__(alignment)))
#endif

#define QNNP_COUNT_OF(array) (sizeof(array) / sizeof(0[array]))

#if defined(__GNUC__)
  #define QNNP_LIKELY(condition) (__builtin_expect(!!(condition), 1))
  #define QNNP_UNLIKELY(condition) (__builtin_expect(!!(condition), 0))
#else
  #define QNNP_LIKELY(condition) (!!(condition))
  #define QNNP_UNLIKELY(condition) (!!(condition))
#endif

#if defined(__GNUC__)
  #define QNNP_INLINE inline __attribute__((__always_inline__))
#else
  #define QNNP_INLINE inline
#endif

#ifndef QNNP_INTERNAL
  #if defined(__ELF__)
    #define QNNP_INTERNAL __attribute__((__visibility__("internal")))
  #elif defined(__MACH__)
    #define QNNP_INTERNAL __attribute__((__visibility__("hidden")))
  #else
    #define QNNP_INTERNAL
  #endif
#endif

#ifndef QNNP_PRIVATE
  #if defined(__ELF__)
    #define QNNP_PRIVATE __attribute__((__visibility__("hidden")))
  #elif defined(__MACH__)
    #define QNNP_PRIVATE __attribute__((__visibility__("hidden")))
  #else
    #define QNNP_PRIVATE
  #endif
#endif

#if defined(_MSC_VER)
#define RESTRICT_STATIC
#define restrict
#else
#define RESTRICT_STATIC restrict static
#endif

#if defined(_MSC_VER)
#define __builtin_prefetch
#endif
