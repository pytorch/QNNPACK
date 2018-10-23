/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <inttypes.h>

#include <clog.h>

#ifndef QNNP_LOG_LEVEL
#define QNNP_LOG_LEVEL CLOG_DEBUG
#endif

CLOG_DEFINE_LOG_DEBUG(qnnp_log_debug, "QNNPACK", QNNP_LOG_LEVEL);
CLOG_DEFINE_LOG_INFO(qnnp_log_info, "QNNPACK", QNNP_LOG_LEVEL);
CLOG_DEFINE_LOG_WARNING(qnnp_log_warning, "QNNPACK", QNNP_LOG_LEVEL);
CLOG_DEFINE_LOG_ERROR(qnnp_log_error, "QNNPACK", QNNP_LOG_LEVEL);
CLOG_DEFINE_LOG_FATAL(qnnp_log_fatal, "QNNPACK", QNNP_LOG_LEVEL);
