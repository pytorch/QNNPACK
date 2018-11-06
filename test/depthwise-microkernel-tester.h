/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <qnnpack/AlignedAllocator.h>
#include <qnnpack/pack.h>
#include <qnnpack/params.h>
#include <qnnpack/scalar-utils.h>
#include <qnnpack/requantization.h>
#include <qnnpack/q8dw.h>

class DepthwiseMicrokernelTester {
 public:
  inline DepthwiseMicrokernelTester& width(uint32_t width) {
    assert(width >= 1);
    this->width_ = width;
    return *this;
  }

  inline uint32_t width() const {
    return this->width_;
  }

  inline DepthwiseMicrokernelTester& subsampling(uint32_t subsampling) {
    assert(subsampling >= 1);
    this->subsampling_ = subsampling;
    return *this;
  }

  inline uint32_t subsampling() const {
    return this->subsampling_;
  }

  inline DepthwiseMicrokernelTester& channels(uint32_t channels) {
    assert(channels >= 1);
    this->channels_ = channels;
    return *this;
  }

  inline uint32_t channels() const {
    return this->channels_;
  }

  inline DepthwiseMicrokernelTester& cr(uint32_t cr) {
    assert(cr != 0);
    assert((cr & (cr - 1)) == 0);
    this->cr_ = cr;
    return *this;
  }

  inline uint32_t cr() const {
    return this->cr_;
  }

  inline uint32_t packedChannels() const {
    return (channels() | (cr() - 1)) + 1;
  }

  inline DepthwiseMicrokernelTester& kernelHeight(uint32_t kernelHeight) {
    assert(kernelHeight != 0);
    this->kernelHeight_ = kernelHeight;
    return *this;
  }

  inline uint32_t kernelHeight() const {
    return this->kernelHeight_;
  }

  inline DepthwiseMicrokernelTester& kernelWidth(uint32_t kernelWidth) {
    assert(kernelWidth != 0);
    this->kernelWidth_ = kernelWidth;
    return *this;
  }

  inline uint32_t kernelWidth() const {
    return this->kernelWidth_;
  }

  inline uint32_t kernelSize() const {
    return kernelHeight() * kernelWidth();
  }

  inline DepthwiseMicrokernelTester& inputStride(uint32_t inputStride) {
    assert(inputStride != 0);
    this->inputStride_ = inputStride;
    return *this;
  }

  inline uint32_t inputStride() const {
    if (this->inputStride_ == 0) {
      return channels();
    } else {
      assert(this->inputStride_ >= channels());
      return this->inputStride_;
    }
  }

  inline DepthwiseMicrokernelTester& outputStride(uint32_t outputStride) {
    assert(outputStride != 0);
    this->outputStride_ = outputStride;
    return *this;
  }

  inline uint32_t outputStride() const {
    if (this->outputStride_ == 0) {
      return channels();
    } else {
      assert(this->outputStride_ >= channels());
      return this->outputStride_;
    }
  }

  inline DepthwiseMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline DepthwiseMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline DepthwiseMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void test(q8dw_ukernel_function q8dw) const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> input((kernelSize() + (width() * subsampling() - 1) * kernelHeight() - 1) * inputStride() + channels() + 8);
    std::vector<uint8_t> kernel(channels() * kernelSize());
    std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> packedWeights((kernelSize() + sizeof(int32_t) / sizeof(uint8_t)) * packedChannels());
    std::vector<int32_t> bias(packedChannels());
    std::vector<int32_t> accumulators(width() * channels());
    auto channel_stride = (channels() + (cr() - 1)) & -cr();
    std::vector<int32_t> outacc32(width() * channel_stride);
    std::vector<uint8_t> output((width() - 1) * outputStride() + channels());
    std::vector<const uint8_t*> indirectInput(kernelSize() + (width() * subsampling() - 1) * kernelHeight());

    const uint8_t* inputPtr = input.data() + 8;
    const uint8_t inputZeroPoint = 127;
    const uint8_t kernelZeroPoint = 127;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::generate(kernel.begin(), kernel.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(accumulators.begin(), accumulators.end(), 0);
      std::fill(outacc32.begin(), outacc32.end(), 0);

      ASSERT_NE(*std::max_element(input.cbegin(), input.cend()), *std::min_element(input.cbegin(), input.cend()));
      ASSERT_NE(*std::max_element(kernel.cbegin(), kernel.cend()), *std::min_element(kernel.cbegin(), kernel.cend()));

      std::fill(packedWeights.begin(), packedWeights.end(), 0xA5);

      if (kernelSize() == 25) {
        // special path for 5x5 now
        pack_q8dw_w_dilation(
          kernelHeight(), kernelWidth(), channels(), cr(),
          0, kernelHeight(), 0, 2,
          kernel.data(), bias.data(), packedWeights.data(), true);
        pack_q8dw_w_dilation(
          kernelHeight(), kernelWidth(), channels(), cr(),
          0, kernelHeight(), 2, 4,
          kernel.data(), bias.data(), packedWeights.data() + (10 + sizeof(int32_t) / sizeof(uint8_t)) * channel_stride, false);
        pack_q8dw_w_dilation(
          kernelHeight(), kernelWidth(), channels(), cr(),
          0, kernelHeight(), 4, 5,
          kernel.data(), bias.data(), packedWeights.data() + (20 + sizeof(int32_t) / sizeof(uint8_t)) * channel_stride, false);
      } else {
        pack_q8dw_w(
          kernelHeight(), kernelWidth(), channels(), cr(),
          kernel.data(), bias.data(), packedWeights.data());
      }
      for (size_t i = 0; i < kernelSize() + (width() * subsampling() - 1) * kernelHeight(); i++) {
        indirectInput[i] = inputPtr + i * inputStride();
      }
      std::shuffle(indirectInput.begin(), indirectInput.end(), rng);

      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          int32_t acc = bias[c];
          for (size_t kx = 0; kx < kernelWidth(); kx++) {
            for (size_t ky = 0; ky < kernelHeight(); ky++) {
              acc +=
                (int32_t(indirectInput[(x * subsampling() + kx) * kernelHeight() + ky][c]) - int32_t(inputZeroPoint)) *
                (int32_t(kernel[(c * kernelHeight() + ky) * kernelWidth() + kx]) - int32_t(kernelZeroPoint));
            }
          }
          accumulators[x * channels() + c] = acc;
        }
      }
      const int32_t accumulatorsMin = *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulatorsMax = *std::max_element(accumulators.cbegin(), accumulators.cend());
      const uint32_t accumulatorsRange = uint32_t(accumulatorsMax) - uint32_t(accumulatorsMin);
      ASSERT_NE(0, accumulatorsRange);

      const double outputScale = accumulatorsRange >= 256 ? double(accumulatorsRange) / 255.0 : 1.00001;
      const uint8_t outputZeroPoint = uint8_t(std::max(std::min(
        lrint(127.5 - 0.5 * double(accumulatorsMin + accumulatorsMax) / outputScale),
        long(std::numeric_limits<uint8_t>::max())), long(std::numeric_limits<uint8_t>::min())));

      const float requantizationScale = 1.0f / float(outputScale);
      const union qnnp_conv_quantization_params quantizationParams =
        qnnp_compute_conv_quantization_params(
          inputZeroPoint, kernelZeroPoint,
          requantizationScale, outputZeroPoint, qmin(), qmax());
      const union qnnp_q31_requantization_params scalarRequantizationParams =
        qnnp_compute_scalar_requantization_params(
          requantizationScale, outputZeroPoint, qmin(), qmax());

      if (kernelHeight() == 3 && kernelWidth() == 3) {
        q8dw(
          channels(), width(),
          indirectInput.data(), packedWeights.data(), output.data(),
          kernelHeight() * subsampling() * sizeof(void*),
          (outputStride() - channels()) * sizeof(uint8_t),
          &quantizationParams);
      } else {
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
        q8dw_ukernel_25c8__neon(
          channels(), width(),
          indirectInput.data(), packedWeights.data(), outacc32.data(), output.data(),
          kernelHeight() * subsampling() * sizeof(void*),
          (outputStride() - channels()) * sizeof(uint8_t),
          &quantizationParams);
#else
        q8dw(
          channels(), width(),
          indirectInput.data(), packedWeights.data(), output.data(),
          kernelHeight() * subsampling() * sizeof(void*),
          (outputStride() - channels()) * sizeof(uint8_t),
          &quantizationParams);
#endif
      }

      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          const double scaledAccumulator = accumulators[x * channels() + c] / outputScale;
          const double clampedAccumulator = std::max(std::min(scaledAccumulator,
            double(qmax()) - double(outputZeroPoint)),
            double(qmin()) - double(outputZeroPoint));
          ASSERT_NEAR(
            clampedAccumulator,
            (int32_t(output[x * outputStride() + c]) - outputZeroPoint),
            0.6) << "x = " << x << ", channel = " << c;
        }
      }
    }
  }

 private:
  uint32_t channels_{1};
  uint32_t cr_{1};
  uint32_t width_{1};
  uint32_t subsampling_{1};
  uint32_t kernelHeight_{1};
  uint32_t kernelWidth_{1};
  uint32_t inputStride_{0};
  uint32_t outputStride_{0};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{3};
};
