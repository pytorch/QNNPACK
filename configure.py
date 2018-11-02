#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import confu
from confu import arm, x86
parser = confu.standard_parser()


def main(args):
    options = parser.parse_args(args)
    build = confu.Build.from_options(options)

    build.export_cpath("include", ["q8gemm.h"])

    with build.options(source_dir="src",
            deps=[build.deps.cpuinfo, build.deps.clog, build.deps.psimd, build.deps.fxdiv, build.deps.pthreadpool, build.deps.FP16],
            extra_include_dirs="src"):

        requantization_objects = [
            build.cc("requantization/precise-scalar.c"),
            build.cc("requantization/fp32-scalar.c"),
            build.cc("requantization/q31-scalar.c"),
            build.cc("requantization/gemmlowp-scalar.c"),
        ]
        with build.options(isa=arm.neon if build.target.is_arm else None):
            requantization_objects += [
                build.cc("requantization/precise-psimd.c"),
                build.cc("requantization/fp32-psimd.c"),
            ]
        if build.target.is_x86 or build.target.is_x86_64:
            with build.options(isa=x86.sse2):
                requantization_objects += [
                    build.cc("requantization/precise-sse2.c"),
                    build.cc("requantization/fp32-sse2.c"),
                    build.cc("requantization/q31-sse2.c"),
                    build.cc("requantization/gemmlowp-sse2.c"),
                ]
            with build.options(isa=x86.ssse3):
                requantization_objects += [
                    build.cc("requantization/precise-ssse3.c"),
                    build.cc("requantization/q31-ssse3.c"),
                    build.cc("requantization/gemmlowp-ssse3.c"),
                ]
            with build.options(isa=x86.sse4_1):
                requantization_objects += [
                    build.cc("requantization/precise-sse4.c"),
                    build.cc("requantization/q31-sse4.c"),
                    build.cc("requantization/gemmlowp-sse4.c"),
                ]
        if build.target.is_arm or build.target.is_arm64:
            with build.options(isa=arm.neon if build.target.is_arm else None):
                requantization_objects += [
                    build.cc("requantization/precise-neon.c"),
                    build.cc("requantization/fp32-neon.c"),
                    build.cc("requantization/q31-neon.c"),
                    build.cc("requantization/gemmlowp-neon.c"),
                ]

        qnnpack_objects = [
            build.cc("init.c"),
            build.cc("convolution.c"),
            build.cc("deconvolution.c"),
            build.cc("fully-connected.c"),
        ]

        with build.options(isa=arm.neon if build.target.is_arm else None):
            qnnpack_objects += [
                build.cc("sgemm/6x8-psimd.c"),
            ]

        with build.options(isa=arm.neon if build.target.is_arm else None):
            if build.target.is_arm or build.target.is_arm64:
                qnnpack_objects += [
                    build.cc("q8gemm/4x8-neon.c"),
                    build.cc("q8gemm/4x-sumrows-neon.c"),
                    build.cc("q8gemm/4x8c2-xzp-neon.c"),
                    build.cc("q8gemm/8x8-neon.c"),
                    build.cc("q8gemm/6x4-neon.c"),
                    build.cc("q8conv/4x8-neon.c"),
                    build.cc("q8conv/8x8-neon.c"),
                    build.cc("q8dw/9c8-neon.c"),
                    build.cc("q8dw/25c8-neon.c"),
                    build.cc("sgemm/5x8-neon.c"),
                    build.cc("sgemm/6x8-neon.c"),
                ]
            if build.target.is_arm:
                qnnpack_objects += [
                    build.cc("q8gemm/4x8-aarch32-neon.S"),
                    build.cc("q8gemm/4x8c2-xzp-aarch32-neon.S"),
                    build.cc("q8conv/4x8-aarch32-neon.S"),
                    build.cc("hgemm/8x8-aarch32-neonfp16arith.S"),
                ]
            if build.target.is_arm64:
                qnnpack_objects += [
                    build.cc("q8gemm/8x8-aarch64-neon.S"),
                    build.cc("q8conv/8x8-aarch64-neon.S"),
                ]
            if build.target.is_x86 or build.target.is_x86_64:
                with build.options(isa=x86.sse2):
                    qnnpack_objects += [
                        build.cc("q8gemm/2x4c8-sse2.c"),
                        build.cc("q8gemm/4x4c2-sse2.c"),
                        build.cc("q8conv/4x4c2-sse2.c"),
                        build.cc("q8dw/9c8-sse2.c"),
                    ]
            build.static_library("qnnpack", qnnpack_objects)


    with build.options(source_dir="test",
            deps={
                (build, build.deps.cpuinfo, build.deps.clog, build.deps.pthreadpool, build.deps.FP16, build.deps.googletest): any,
                "log": build.target.is_android},
            extra_include_dirs=["src", "test"]):

        build.unittest("q8gemm-test", build.cxx("q8gemm.cc"))
        build.unittest("q8conv-test", build.cxx("q8conv.cc"))
        build.unittest("q8dw-test", build.cxx("q8dw.cc"))
        build.unittest("hgemm-test", build.cxx("hgemm.cc"))
        build.unittest("sgemm-test", build.cxx("sgemm.cc"))
        build.unittest("convolution-test", build.cxx("convolution.cc"))
        build.unittest("deconvolution-test", build.cxx("deconvolution.cc"))
        build.unittest("fully-connected-test", build.cxx("fully-connected.cc"))
        build.unittest("requantization-test", [build.cxx("requantization.cc")] + requantization_objects)

    benchmark_isa = None
    if build.target.is_arm:
        benchmark_isa = arm.neon
    elif build.target.is_x86:
        benchmark_isa = x86.sse4_1
    with build.options(source_dir="bench",
            deps={
                (build, build.deps.cpuinfo, build.deps.clog, build.deps.pthreadpool, build.deps.FP16, build.deps.googlebenchmark): any,
                "log": build.target.is_android},
            isa=benchmark_isa,
            extra_include_dirs="src"):

        build.benchmark("convolution-bench", build.cxx("convolution.cc"))
        build.benchmark("q8gemm-bench", build.cxx("q8gemm.cc"))
        build.benchmark("hgemm-bench", build.cxx("hgemm.cc"))
        build.benchmark("sgemm-bench", build.cxx("sgemm.cc"))
        build.benchmark("requantization-bench", [build.cxx("requantization.cc")] + requantization_objects)

    return build

if __name__ == "__main__":
    import sys
    main(sys.argv[1:]).generate()
