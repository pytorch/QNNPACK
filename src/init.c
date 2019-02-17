/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _MSC_VER
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <cpuinfo.h>
#include <qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/params.h>
#include <qnnpack/q8avgpool.h>
#include <qnnpack/q8conv.h>
#include <qnnpack/q8dwconv.h>
#include <qnnpack/q8gavgpool.h>
#include <qnnpack/q8gemm.h>
#include <qnnpack/q8vadd.h>
#include <qnnpack/u8clamp.h>
#include <qnnpack/u8lut32norm.h>
#include <qnnpack/u8maxpool.h>
#include <qnnpack/u8rmax.h>
#include <qnnpack/x8lut.h>
#include <qnnpack/x8zip.h>

#ifdef _MSC_VER
static INIT_ONCE init_guard;
BOOL CALLBACK init_win(PINIT_ONCE InitOnce, PVOID Parameter, PVOID *lpContex);
#else
static pthread_once_t init_guard = PTHREAD_ONCE_INIT;
#endif

struct qnnp_parameters qnnp_params = {
  .initialized = false
};

static void init(void) {
#if CPUINFO_ARCH_ARM
  if (!cpuinfo_has_arm_neon()) {
    qnnp_log_error("QNNPACK initialization failed: NEON is not supported");
    return;
  }
  qnnp_params.q8conv = (struct q8conv_parameters) {
      .gemm = q8gemm_ukernel_4x8__aarch32_neon,
      .conv = q8conv_ukernel_4x8__aarch32_neon,
      .mr = 4,
      .nr = 8,
      .kr = 1,
  };
  qnnp_params.q8conv_xzp = (struct q8conv_xzp_parameters) {
      .gemm = q8gemm_xzp_ukernel_4x8c2__aarch32_neon,
      .mr = 4,
      .nr = 8,
      .kr = 2,
      .kc = 8,
      .kthreshold = SIZE_MAX,
  };
  /* setup xzp threshold based on measurements */
  switch (cpuinfo_get_core(0)->uarch) {
    case cpuinfo_uarch_cortex_a72:
      qnnp_params.q8conv_xzp.kthreshold = 64;
      break;
    case cpuinfo_uarch_cortex_a73:
      qnnp_params.q8conv_xzp.kthreshold = 256;
      break;
    case cpuinfo_uarch_cortex_a75:
      qnnp_params.q8conv_xzp.kthreshold = 32;
      break;
    case cpuinfo_uarch_cortex_a76:
      qnnp_params.q8conv_xzp.kthreshold = 16;
      break;
    default:
      break;
  }
  qnnp_params.q8dw9 = (struct q8dwconv_up_parameters) {
      .updw = q8dwconv_ukernel_up8x9__aarch32_neon,
      .cr = 8,
  };
  qnnp_params.q8dw25 = (struct q8dwconv_mp_parameters) {
      .mpdw = q8dwconv_ukernel_mp8x25__neon,
      .cr = 8,
  };
  qnnp_params.q8sum_rows = (struct q8sum_rows_parameters) {
      .sum_rows = q8sumrows_ukernel_4x__neon,
      .m = 4,
  };
  qnnp_params.q8vadd = q8vadd_ukernel__neon;
  qnnp_params.q8gavgpool = (struct q8gavgpool_parameters) {
      .ltnr = q8gavgpool_ukernel_up8xm__neon,
      .genr_lemr = q8gavgpool_ukernel_up8x7__neon,
      .genr_gtmr = q8gavgpool_ukernel_mp8x7p7q__neon,
      .mr = 7,
      .nr = 8,
  };
  qnnp_params.q8avgpool = (struct q8avgpool_parameters) {
      .ltkr = q8avgpool_ukernel_up8xm__neon,
      .gekr_lemr = q8avgpool_ukernel_up8x9__neon,
      .gekr_gtmr = q8avgpool_ukernel_mp8x9p8q__neon,
      .mr = 9,
      .qr = 8,
      .kr = 8,
  };
  qnnp_params.u8maxpool = (struct u8maxpool_parameters) {
      .ltkr = u8maxpool_ukernel_sub16__neon,
      .gekr = u8maxpool_ukernel_16x9p8q__neon,
      .mr = 9,
      .qr = 8,
      .kr = 16,
  };
  qnnp_params.x8zip = (struct x8zip_parameters) {
      .x2 = qnnp_x8zip_x2__neon,
      .x3 = qnnp_x8zip_x3__neon,
      .x4 = qnnp_x8zip_x4__neon,
      .xm = qnnp_x8zip_xm__neon,
  };
  qnnp_params.u8clamp = u8clamp_ukernel__neon;
  qnnp_params.u8rmax = u8rmax_ukernel__neon;
  qnnp_params.u8lut32norm = u8lut32norm_ukernel__scalar;
  qnnp_params.x8lut = x8lut_ukernel__scalar;
#elif CPUINFO_ARCH_ARM64
  qnnp_params.q8conv = (struct q8conv_parameters) {
      .gemm = q8gemm_ukernel_8x8__aarch64_neon,
      .conv = q8conv_ukernel_8x8__aarch64_neon,
      .mr = 8,
      .nr = 8,
      .kr = 1,
  };
  qnnp_params.q8conv_xzp = (struct q8conv_xzp_parameters) {
      .kthreshold = SIZE_MAX,
  };
  qnnp_params.q8dw9 = (struct q8dwconv_up_parameters) {
      .updw = q8dwconv_ukernel_up8x9__neon,
      .cr = 8,
  };
  qnnp_params.q8dw25 = (struct q8dwconv_mp_parameters) {
      .mpdw = q8dwconv_ukernel_mp8x25__neon,
      .cr = 8,
  };
  qnnp_params.q8vadd = q8vadd_ukernel__neon;
  qnnp_params.q8gavgpool = (struct q8gavgpool_parameters) {
      .ltnr = q8gavgpool_ukernel_up8xm__neon,
      .genr_lemr = q8gavgpool_ukernel_up8x7__neon,
      .genr_gtmr = q8gavgpool_ukernel_mp8x7p7q__neon,
      .mr = 7,
      .nr = 8,
  };
  qnnp_params.q8avgpool = (struct q8avgpool_parameters) {
      .ltkr = q8avgpool_ukernel_up8xm__neon,
      .gekr_lemr = q8avgpool_ukernel_up8x9__neon,
      .gekr_gtmr = q8avgpool_ukernel_mp8x9p8q__neon,
      .mr = 9,
      .qr = 8,
      .kr = 8,
  };
  qnnp_params.u8maxpool = (struct u8maxpool_parameters) {
      .ltkr = u8maxpool_ukernel_sub16__neon,
      .gekr = u8maxpool_ukernel_16x9p8q__neon,
      .mr = 9,
      .qr = 8,
      .kr = 16,
  };
  qnnp_params.x8zip = (struct x8zip_parameters) {
      .x2 = qnnp_x8zip_x2__neon,
      .x3 = qnnp_x8zip_x3__neon,
      .x4 = qnnp_x8zip_x4__neon,
      .xm = qnnp_x8zip_xm__neon,
  };
  qnnp_params.u8clamp = u8clamp_ukernel__neon;
  qnnp_params.u8rmax = u8rmax_ukernel__neon;
  qnnp_params.u8lut32norm = u8lut32norm_ukernel__scalar;
  qnnp_params.x8lut = x8lut_ukernel__scalar;
#elif CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  if (!cpuinfo_has_x86_sse2()) {
    qnnp_log_error("QNNPACK initialization failed: SSE2 is not supported");
    return;
  }
  qnnp_params.q8conv = (struct q8conv_parameters){
      .gemm = q8gemm_ukernel_4x4c2__sse2,
      .conv = q8conv_ukernel_4x4c2__sse2,
      .mr = 4,
      .nr = 4,
      .kr = 2,
  };
  qnnp_params.q8conv_xzp = (struct q8conv_xzp_parameters) {
      .kthreshold = SIZE_MAX,
  };
  qnnp_params.q8dw9 = (struct q8dwconv_up_parameters) {
      .updw = q8dwconv_ukernel_up8x9__sse2,
      .cr = 8,
  };
  qnnp_params.q8dw25 = (struct q8dwconv_mp_parameters) {
      .mpdw = q8dwconv_ukernel_mp8x25__sse2,
      .cr = 8,
  };
  qnnp_params.q8vadd = q8vadd_ukernel__sse2;
  qnnp_params.q8gavgpool = (struct q8gavgpool_parameters) {
      .ltnr = q8gavgpool_ukernel_up8xm__sse2,
      .genr_lemr = q8gavgpool_ukernel_up8x7__sse2,
      .genr_gtmr = q8gavgpool_ukernel_mp8x7p7q__sse2,
      .mr = 7,
      .nr = 8,
  };
  qnnp_params.q8avgpool = (struct q8avgpool_parameters) {
      .ltkr = q8avgpool_ukernel_up8xm__sse2,
      .gekr_lemr = q8avgpool_ukernel_up8x9__sse2,
      .gekr_gtmr = q8avgpool_ukernel_mp8x9p8q__sse2,
      .mr = 9,
      .qr = 8,
      .kr = 8,
  };
  qnnp_params.u8maxpool = (struct u8maxpool_parameters) {
      .ltkr = u8maxpool_ukernel_sub16__sse2,
      .gekr = u8maxpool_ukernel_16x9p8q__sse2,
      .mr = 9,
      .qr = 8,
      .kr = 16,
  };
  qnnp_params.x8zip = (struct x8zip_parameters) {
      .x2 = qnnp_x8zip_x2__sse2,
      .x3 = qnnp_x8zip_x3__sse2,
      .x4 = qnnp_x8zip_x4__sse2,
      .xm = qnnp_x8zip_xm__sse2,
  };
  qnnp_params.u8clamp = u8clamp_ukernel__sse2;
  qnnp_params.u8rmax = u8rmax_ukernel__sse2;
  qnnp_params.u8lut32norm = u8lut32norm_ukernel__scalar;
  qnnp_params.x8lut = x8lut_ukernel__scalar;
#else
  #error "Unsupported architecture"
#endif
  qnnp_params.initialized = true;
}

enum qnnp_status qnnp_initialize(void) {
  if (!cpuinfo_initialize()) {
    return qnnp_status_out_of_memory;
  }
#ifdef _MSC_VER
  InitOnceExecuteOnce(&init_guard, init_win, NULL, NULL);
#else
  pthread_once(&init_guard, &init);
#endif
  if (qnnp_params.initialized) {
    return qnnp_status_success;
  } else {
    return qnnp_status_unsupported_hardware;
  }
}

enum qnnp_status qnnp_deinitialize(void) {
  cpuinfo_deinitialize();
  return qnnp_status_success;
}

#ifdef _MSC_VER
BOOL CALLBACK init_win(PINIT_ONCE InitOnce, PVOID Parameter, PVOID *lpContex) {
    init();
    return TRUE;
}
#endif
