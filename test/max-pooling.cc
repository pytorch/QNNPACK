/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "max-pooling-operator-tester.h"

#include <qnnpack/params.h>


TEST(MAX_POOLING_OP, zero_batch) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  MaxPoolingOperatorTester()
    .batchSize(0)
    .inputHeight(2)
    .inputWidth(6)
    .poolingHeight(1)
    .poolingWidth(8)
    .channels(8)
    .testU8();
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_1xM_pool) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = 2; poolSize <= qnnp_params.u8maxpool.mr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_1xM_pool_with_padding) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = 3; poolSize <= qnnp_params.u8maxpool.mr; poolSize++) {
      for (size_t paddingLeft = 0; paddingLeft <= 1; paddingLeft++) {
        for (size_t paddingRight = 0; paddingRight <= 1; paddingRight++) {
          MaxPoolingOperatorTester()
            .batchSize(1)
            .inputHeight(2)
            .inputWidth(poolSize + 2)
            .paddingLeft(paddingLeft)
            .paddingRight(paddingRight)
            .poolingHeight(1)
            .poolingWidth(poolSize)
            .channels(channels)
            .testU8();
        }
      }
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_1xM_pool_with_stride) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = 2; poolSize <= qnnp_params.u8maxpool.mr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(poolSize + 4)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .strideWidth(2)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_1xM_pool_with_dilation) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = 2; poolSize <= qnnp_params.u8maxpool.mr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(2 * poolSize + 1)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .dilationWidth(2)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_Mx1_pool) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = 2; poolSize <= qnnp_params.u8maxpool.mr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_Mx1_pool_with_padding) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = 2; poolSize <= qnnp_params.u8maxpool.mr; poolSize++) {
      for (size_t paddingTop = 0; paddingTop <= 1; paddingTop++) {
        for (size_t paddingBottom = 0; paddingBottom <= 1; paddingBottom++) {
          MaxPoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .paddingTop(paddingTop)
            .paddingBottom(paddingBottom)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .testU8();
        }
      }
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_Mx1_pool_with_stride) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = 2; poolSize <= qnnp_params.u8maxpool.mr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(poolSize + 3)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .strideHeight(2)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_Mx1_pool_with_dilation) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = 2; poolSize <= qnnp_params.u8maxpool.mr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2 * poolSize)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .dilationHeight(2)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_pool_with_input_stride) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = 2; poolSize <= qnnp_params.u8maxpool.mr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .inputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .inputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_pool_with_output_stride) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = 2; poolSize <= qnnp_params.u8maxpool.mr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .outputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .outputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_pool_with_qmin) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = 2; poolSize <= qnnp_params.u8maxpool.mr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .qmin(192)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .qmin(192)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_pool_with_qmax) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = 2; poolSize <= qnnp_params.u8maxpool.mr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .qmax(192)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .qmax(192)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_1xM_pool) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = qnnp_params.u8maxpool.mr; poolSize <= qnnp_params.u8maxpool.mr + qnnp_params.u8maxpool.qr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_1xM_pool_with_padding) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = 3; poolSize <= qnnp_params.u8maxpool.mr; poolSize++) {
      for (size_t paddingLeft = 0; paddingLeft <= 1; paddingLeft++) {
        for (size_t paddingRight = 0; paddingRight <= 1; paddingRight++) {
          MaxPoolingOperatorTester()
            .batchSize(1)
            .inputHeight(2)
            .inputWidth(poolSize + 2)
            .paddingLeft(paddingLeft)
            .paddingRight(paddingRight)
            .poolingHeight(1)
            .poolingWidth(poolSize)
            .channels(channels)
            .testU8();
        }
      }
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_1xM_pool_with_stride) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = qnnp_params.u8maxpool.mr; poolSize <= qnnp_params.u8maxpool.mr + qnnp_params.u8maxpool.qr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(poolSize + 4)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .strideWidth(2)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_1xM_pool_with_dilation) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = qnnp_params.u8maxpool.mr; poolSize <= qnnp_params.u8maxpool.mr + qnnp_params.u8maxpool.qr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(2 * poolSize + 1)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .dilationWidth(2)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_Mx1_pool) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = qnnp_params.u8maxpool.mr; poolSize <= qnnp_params.u8maxpool.mr + qnnp_params.u8maxpool.qr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_Mx1_pool_with_padding) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = qnnp_params.u8maxpool.mr; poolSize <= qnnp_params.u8maxpool.mr + qnnp_params.u8maxpool.qr; poolSize++) {
      for (size_t paddingTop = 0; paddingTop <= 1; paddingTop++) {
        for (size_t paddingBottom = 0; paddingBottom <= 1; paddingBottom++) {
          MaxPoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .paddingTop(paddingTop)
            .paddingBottom(paddingBottom)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .testU8();
        }
      }
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_Mx1_pool_with_stride) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = qnnp_params.u8maxpool.mr; poolSize <= qnnp_params.u8maxpool.mr + qnnp_params.u8maxpool.qr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(poolSize + 3)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .strideHeight(2)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_Mx1_pool_with_dilation) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = qnnp_params.u8maxpool.mr; poolSize <= qnnp_params.u8maxpool.mr + qnnp_params.u8maxpool.qr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2 * poolSize)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .dilationHeight(2)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_pool_with_input_stride) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = qnnp_params.u8maxpool.mr; poolSize <= qnnp_params.u8maxpool.mr + qnnp_params.u8maxpool.qr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .inputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .inputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_pool_with_output_stride) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = qnnp_params.u8maxpool.mr; poolSize <= qnnp_params.u8maxpool.mr + qnnp_params.u8maxpool.qr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .outputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .outputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_pool_with_qmin) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = qnnp_params.u8maxpool.mr; poolSize <= qnnp_params.u8maxpool.mr + qnnp_params.u8maxpool.qr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .qmin(192)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .qmin(192)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_pool_with_qmax) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = qnnp_params.u8maxpool.mr; poolSize <= qnnp_params.u8maxpool.mr + qnnp_params.u8maxpool.qr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .qmax(192)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .qmax(192)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_1xM_pool) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = 1; channels < qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * qnnp_params.u8maxpool.kr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_1xM_pool_with_padding) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = 1; channels < qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = 3; poolSize <= qnnp_params.u8maxpool.mr; poolSize++) {
      for (size_t paddingLeft = 0; paddingLeft <= 1; paddingLeft++) {
        for (size_t paddingRight = 0; paddingRight <= 1; paddingRight++) {
          MaxPoolingOperatorTester()
            .batchSize(1)
            .inputHeight(2)
            .inputWidth(poolSize + 2)
            .paddingLeft(paddingLeft)
            .paddingRight(paddingRight)
            .poolingHeight(1)
            .poolingWidth(poolSize)
            .channels(channels)
            .testU8();
        }
      }
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_1xM_pool_with_stride) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = 1; channels < qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * qnnp_params.u8maxpool.kr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(poolSize + 4)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .strideWidth(2)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_1xM_pool_with_dilation) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = 1; channels < qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * qnnp_params.u8maxpool.kr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(2 * poolSize + 1)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .dilationWidth(2)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_Mx1_pool) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = 1; channels < qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * qnnp_params.u8maxpool.kr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_Mx1_pool_with_padding) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = 1; channels < qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * qnnp_params.u8maxpool.kr; poolSize++) {
      for (size_t paddingTop = 0; paddingTop <= 1; paddingTop++) {
        for (size_t paddingBottom = 0; paddingBottom <= 1; paddingBottom++) {
          MaxPoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .paddingTop(paddingTop)
            .paddingBottom(paddingBottom)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .testU8();
        }
      }
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_Mx1_pool_with_stride) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = 1; channels < qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * qnnp_params.u8maxpool.kr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(poolSize + 3)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .strideHeight(2)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_Mx1_pool_with_dilation) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = 1; channels < qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * qnnp_params.u8maxpool.kr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2 * poolSize)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .dilationHeight(2)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_with_input_stride) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = 1; channels < qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * qnnp_params.u8maxpool.kr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .inputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .inputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_with_output_stride) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = 1; channels < qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * qnnp_params.u8maxpool.kr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .outputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .outputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_with_qmin) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = 1; channels < qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * qnnp_params.u8maxpool.kr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .qmin(192)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .qmin(192)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_with_qmax) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = 1; channels < qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * qnnp_params.u8maxpool.kr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .qmax(192)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(1)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .qmax(192)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, small_batch_many_channels_small_pool) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = 2; poolSize <= qnnp_params.u8maxpool.mr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(3)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(3)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, small_batch_many_channels_small_pool_with_input_stride) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = 2; poolSize <= qnnp_params.u8maxpool.mr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(3)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .inputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(3)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .inputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, small_batch_many_channels_small_pool_with_output_stride) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 3) {
    for (size_t poolSize = 2; poolSize <= qnnp_params.u8maxpool.mr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(3)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .outputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(3)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .outputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, small_batch_many_channels_large_pool) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = qnnp_params.u8maxpool.mr + 1; poolSize <= qnnp_params.u8maxpool.mr + qnnp_params.u8maxpool.qr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(3)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(3)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, small_batch_many_channels_large_pool_with_input_stride) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 5) {
    for (size_t poolSize = qnnp_params.u8maxpool.mr + 1; poolSize <= qnnp_params.u8maxpool.mr + qnnp_params.u8maxpool.qr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(3)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .inputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(3)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .inputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, small_batch_many_channels_large_pool_with_output_stride) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = qnnp_params.u8maxpool.kr; channels <= 3 * qnnp_params.u8maxpool.kr; channels += 5) {
    for (size_t poolSize = qnnp_params.u8maxpool.mr + 1; poolSize <= qnnp_params.u8maxpool.mr + qnnp_params.u8maxpool.qr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(3)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .outputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(3)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .outputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, small_batch_few_channels) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = 1; channels < qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * qnnp_params.u8maxpool.kr; poolSize++) {
      MaxPoolingOperatorTester()
        .batchSize(3)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(3)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, small_batch_few_channels_with_input_stride) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = 1; channels < qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * qnnp_params.u8maxpool.kr; poolSize += 3) {
      MaxPoolingOperatorTester()
        .batchSize(3)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .inputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(3)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .inputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, small_batch_few_channels_with_output_stride) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  for (size_t channels = 1; channels < qnnp_params.u8maxpool.kr; channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * qnnp_params.u8maxpool.kr; poolSize += 3) {
      MaxPoolingOperatorTester()
        .batchSize(3)
        .inputHeight(poolSize + 1)
        .inputWidth(3)
        .poolingHeight(poolSize)
        .poolingWidth(1)
        .channels(channels)
        .outputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
      MaxPoolingOperatorTester()
        .batchSize(3)
        .inputHeight(2)
        .inputWidth(poolSize + 2)
        .poolingHeight(1)
        .poolingWidth(poolSize)
        .channels(channels)
        .outputPixelStride(5 * qnnp_params.u8maxpool.kr)
        .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, setup_increasing_batch) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  MaxPoolingOperatorTester()
    .batchSize(3)
    .nextBatchSize(5)
    .inputHeight(8)
    .inputWidth(8)
    .poolingHeight(5)
    .poolingWidth(3)
    .channels(24)
    .testSetupU8();
}

TEST(MAX_POOLING_OP, setup_decreasing_batch) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  MaxPoolingOperatorTester()
    .batchSize(5)
    .nextBatchSize(3)
    .inputHeight(8)
    .inputWidth(8)
    .poolingHeight(5)
    .poolingWidth(3)
    .channels(24)
    .testSetupU8();
}

TEST(MAX_POOLING_OP, setup_changing_height) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  MaxPoolingOperatorTester()
    .batchSize(3)
    .inputHeight(8)
    .inputWidth(8)
    .nextInputHeight(9)
    .poolingHeight(5)
    .poolingWidth(3)
    .channels(24)
    .testSetupU8();
  MaxPoolingOperatorTester()
    .batchSize(3)
    .inputHeight(8)
    .inputWidth(8)
    .nextInputHeight(7)
    .poolingHeight(5)
    .poolingWidth(3)
    .channels(24)
    .testSetupU8();
}

TEST(MAX_POOLING_OP, setup_changing_width) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  MaxPoolingOperatorTester()
    .batchSize(3)
    .inputHeight(8)
    .inputWidth(8)
    .nextInputWidth(9)
    .poolingHeight(5)
    .poolingWidth(3)
    .channels(24)
    .testSetupU8();
  MaxPoolingOperatorTester()
    .batchSize(3)
    .inputHeight(8)
    .inputWidth(8)
    .nextInputWidth(7)
    .poolingHeight(5)
    .poolingWidth(3)
    .channels(24)
    .testSetupU8();
}

TEST(MAX_POOLING_OP, setup_swap_height_and_width) {
  ASSERT_EQ(qnnp_status_success, qnnp_initialize());
  MaxPoolingOperatorTester()
    .batchSize(3)
    .inputHeight(9)
    .inputWidth(8)
    .nextInputHeight(8)
    .nextInputWidth(9)
    .poolingHeight(5)
    .poolingWidth(3)
    .channels(24)
    .testSetupU8();
}
