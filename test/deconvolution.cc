/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "deconvolution-tester.h"


TEST(DECONVOLUTION, 1x1) {
  DeconvolutionTester()
    .inputSize(27, 29)
    .kernelSize(1, 1)
    .groupInputChannels(23)
    .groupOutputChannels(19)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, 1x1_with_qmin) {
  DeconvolutionTester()
    .inputSize(27, 29)
    .kernelSize(1, 1)
    .groupInputChannels(23)
    .groupOutputChannels(19)
    .qmin(128)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, 1x1_with_qmax) {
  DeconvolutionTester()
    .inputSize(27, 29)
    .kernelSize(1, 1)
    .groupInputChannels(23)
    .groupOutputChannels(19)
    .qmax(128)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, 1x1_with_input_stride) {
  DeconvolutionTester()
    .inputSize(27, 29)
    .kernelSize(1, 1)
    .inputPixelStride(28)
    .groupInputChannels(23)
    .groupOutputChannels(19)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, 1x1_with_output_stride) {
  DeconvolutionTester()
    .inputSize(27, 29)
    .kernelSize(1, 1)
    .outputPixelStride(29)
    .groupInputChannels(23)
    .groupOutputChannels(19)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, 1x1_with_batch) {
  DeconvolutionTester()
    .inputSize(13, 14)
    .kernelSize(1, 1)
    .batchSize(3)
    .groupInputChannels(23)
    .groupOutputChannels(19)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, grouped_1x1) {
  DeconvolutionTester()
    .inputSize(24, 25)
    .kernelSize(1, 1)
    .groups(2)
    .groupInputChannels(17)
    .groupOutputChannels(19)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, 1x3) {
  DeconvolutionTester()
    .inputSize(20, 19)
    .paddingWidth(1)
    .kernelSize(1, 3)
    .groupInputChannels(17)
    .groupOutputChannels(15)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, grouped_1x3) {
  DeconvolutionTester()
    .inputSize(20, 19)
    .paddingWidth(1)
    .kernelSize(1, 3)
    .groups(2)
    .groupInputChannels(17)
    .groupOutputChannels(15)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, 3x1) {
  DeconvolutionTester()
    .inputSize(19, 20)
    .paddingHeight(1)
    .kernelSize(3, 1)
    .groupInputChannels(17)
    .groupOutputChannels(15)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, grouped_3x1) {
  DeconvolutionTester()
    .inputSize(19, 20)
    .paddingHeight(1)
    .kernelSize(3, 1)
    .groups(2)
    .groupInputChannels(17)
    .groupOutputChannels(15)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, 3x3) {
  DeconvolutionTester()
    .inputSize(13, 12)
    .padding(1)
    .kernelSize(3, 3)
    .groupInputChannels(15)
    .groupOutputChannels(17)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, 3x3_with_input_stride) {
  DeconvolutionTester()
    .inputSize(13, 12)
    .padding(1)
    .kernelSize(3, 3)
    .inputPixelStride(22)
    .groupInputChannels(15)
    .groupOutputChannels(17)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, 3x3_with_output_stride) {
  DeconvolutionTester()
    .inputSize(13, 12)
    .padding(1)
    .kernelSize(3, 3)
    .outputPixelStride(23)
    .groupInputChannels(15)
    .groupOutputChannels(17)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, 3x3_with_batch) {
  DeconvolutionTester()
    .inputSize(10, 9)
    .padding(1)
    .kernelSize(3, 3)
    .batchSize(3)
    .groupInputChannels(15)
    .groupOutputChannels(17)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, grouped_3x3) {
  DeconvolutionTester()
    .inputSize(10, 11)
    .padding(1)
    .kernelSize(3, 3)
    .groups(2)
    .groupInputChannels(14)
    .groupOutputChannels(13)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, 3x3s2) {
  DeconvolutionTester()
    .inputSize(19, 21)
    .padding(1)
    .kernelSize(3, 3)
    .stride(2)
    .groupInputChannels(27)
    .groupOutputChannels(19)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, 3x3s1x2) {
  DeconvolutionTester()
    .inputSize(13, 13)
    .padding(1)
    .kernelSize(3, 3)
    .stride(1, 2)
    .groupInputChannels(27)
    .groupOutputChannels(19)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, 3x3s2x1) {
  DeconvolutionTester()
    .inputSize(13, 13)
    .padding(1)
    .kernelSize(3, 3)
    .stride(2, 1)
    .groupInputChannels(27)
    .groupOutputChannels(19)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, 3x3d2) {
  DeconvolutionTester()
    .inputSize(13, 14)
    .padding(2)
    .kernelSize(3, 3)
    .dilation(2)
    .groupInputChannels(27)
    .groupOutputChannels(19)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, 3x3d1x2) {
  DeconvolutionTester()
    .inputSize(14, 15)
    .padding(1, 2)
    .kernelSize(3, 3)
    .dilation(1, 2)
    .groupInputChannels(27)
    .groupOutputChannels(19)
    .iterations(3)
    .test();
}

TEST(DECONVOLUTION, 3x3d2x1) {
  DeconvolutionTester()
    .inputSize(15, 14)
    .padding(2, 1)
    .kernelSize(3, 3)
    .dilation(2, 1)
    .groupInputChannels(27)
    .groupOutputChannels(19)
    .iterations(3)
    .test();
}
