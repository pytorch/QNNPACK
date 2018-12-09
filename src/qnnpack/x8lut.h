/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/params.h>
#include <qnnpack/common.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_X8LUT_UKERNEL_FUNCTION(fn_name) \
  QNNP_INTERNAL void fn_name(                   \
      size_t n,                                 \
      const uint8_t* x,                         \
      const uint8_t* t,                         \
      uint8_t* y);

DECLARE_X8LUT_UKERNEL_FUNCTION(x8lut_ukernel__scalar)

#ifdef __cplusplus
} /* extern "C" */
#endif
