/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "fully-connected-tester.h"

#include <qnnpack/params.h>


TEST(FULLY_CONNECTED, unit_batch) {
  FullyConnectedTester()
    .batchSize(1)
    .inputChannels(23)
    .outputChannels(19)
    .iterations(3)
    .test();
}

TEST(FULLY_CONNECTED, unit_batch_with_qmin) {
  FullyConnectedTester()
    .batchSize(1)
    .inputChannels(23)
    .outputChannels(19)
    .qmin(128)
    .iterations(3)
    .test();
}

TEST(FULLY_CONNECTED, unit_batch_with_qmax) {
  FullyConnectedTester()
    .batchSize(1)
    .inputChannels(23)
    .outputChannels(19)
    .qmax(128)
    .iterations(3)
    .test();
}

TEST(FULLY_CONNECTED, unit_batch_with_input_stride) {
  FullyConnectedTester()
    .batchSize(1)
    .inputChannels(23)
    .inputStride(28)
    .outputChannels(19)
    .iterations(3)
    .test();
}

TEST(FULLY_CONNECTED, unit_batch_with_output_stride) {
  FullyConnectedTester()
    .batchSize(1)
    .inputChannels(23)
    .outputChannels(19)
    .outputStride(29)
    .iterations(3)
    .test();
}

TEST(FULLY_CONNECTED, small_batch) {
  FullyConnectedTester()
    .batchSize(12)
    .inputChannels(23)
    .outputChannels(19)
    .iterations(3)
    .test();
}

TEST(FULLY_CONNECTED, small_batch_with_qmin) {
  FullyConnectedTester()
    .batchSize(12)
    .inputChannels(23)
    .outputChannels(19)
    .qmin(128)
    .iterations(3)
    .test();
}

TEST(FULLY_CONNECTED, small_batch_with_qmax) {
  FullyConnectedTester()
    .batchSize(12)
    .inputChannels(23)
    .outputChannels(19)
    .qmax(128)
    .iterations(3)
    .test();
}

TEST(FULLY_CONNECTED, small_batch_with_input_stride) {
  FullyConnectedTester()
    .batchSize(12)
    .inputChannels(23)
    .inputStride(28)
    .outputChannels(19)
    .iterations(3)
    .test();
}

TEST(FULLY_CONNECTED, small_batch_with_output_stride) {
  FullyConnectedTester()
    .batchSize(12)
    .inputChannels(23)
    .outputChannels(19)
    .outputStride(29)
    .iterations(3)
    .test();
}
