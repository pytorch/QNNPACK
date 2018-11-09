/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <iostream>

#include <qnnpack.h>

#include <benchmark/benchmark.h>


class Q8Convolution : public benchmark::Fixture {
 public:
  virtual void SetUp(const benchmark::State& state) override
  {
    batchSize_ = state.range(0);
    inputHeight_ = state.range(1);
    inputWidth_ = state.range(2);
    kernelHeight_ = state.range(3);
    kernelWidth_ = state.range(4);
    subsampling_ = state.range(5);
    dilation_ = state.range(6);
    groups_ = state.range(7);
    groupInputChannels_ = state.range(8);
    groupOutputChannels_ = state.range(9);

    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    input_.resize(batchSize() * inputHeight() * inputWidth() * inputPixelStride());
    std::generate(input_.begin(), input_.end(), std::ref(u8rng));
    kernel_.resize(groups() * groupOutputChannels() * kernelHeight() * kernelWidth() * groupInputChannels());
    std::generate(kernel_.begin(), kernel_.end(), std::ref(u8rng));
    bias_.resize(groups() * groupOutputChannels());
    std::generate(bias_.begin(), bias_.end(), std::ref(s32rng));
    output_.resize(batchSize() * outputHeight() * outputWidth() * outputPixelStride());

    qnnp_status status = qnnp_initialize();
    assert(status == qnnp_status_success);

    status = qnnp_create_convolution2d_nhwc_q8(
      paddingTop(), paddingRight(), paddingBottom(), paddingLeft(),
      kernelHeight(), kernelWidth(),
      /* subsampling */ subsampling(), subsampling(),
      /* dilation */ dilation(), dilation(),
      groups(), groupInputChannels(), groupOutputChannels(),
      127, 0.5f,
      127, 0.5f,
      kernel(), bias(),
      127, 0.5f, 0, 255,
      &convolutionObject_);
    assert(status == qnnp_status_success);

    status = qnnp_setup_convolution2d_nhwc_q8(
      convolutionObject_,
      batchSize(), inputHeight(), inputWidth(),
      input(), inputPixelStride(),
      output(), outputPixelStride(),
      nullptr /* thread pool */);
    assert(status == qnnp_status_success);
  }

  virtual void TearDown(benchmark::State& state) override
  {
    qnnp_delete_operator(convolutionObject_);
    convolutionObject_ = nullptr;

    state.SetItemsProcessed(
      uint64_t(state.iterations()) * 2 *
        batchSize() * outputHeight() * outputWidth() *
        groups() * groupInputChannels() * groupOutputChannels() *
        kernelHeight() * kernelWidth());
    input_.clear();
    kernel_.clear();
    bias_.clear();
    output_.clear();
  }

  inline const uint8_t* input() const {
    return input_.data();
  }

  inline const uint8_t* kernel() const {
    return kernel_.data();
  }

  inline const int32_t* bias() const {
    return bias_.data();
  }

  inline uint8_t* output() {
    return output_.data();
  }

  inline size_t batchSize() const {
    return batchSize_;
  }

  inline size_t inputHeight() const {
    return inputHeight_;
  }

  inline size_t inputWidth() const {
    return inputWidth_;
  }

  inline uint32_t kernelHeight() const {
    return kernelHeight_;
  }

  inline uint32_t kernelWidth() const {
    return kernelWidth_;
  }

  inline uint32_t effectiveKernelHeight() const {
    return (kernelHeight() - 1) * dilation() + 1;
  }

  inline uint32_t effectiveKernelWidth() const {
    return (kernelWidth() - 1) * dilation() + 1;
  }

  inline uint32_t subsampling() const {
    return subsampling_;
  }

  inline uint32_t dilation() const {
    return dilation_;
  }

  inline uint32_t paddingLeft() const {
    return effectiveKernelWidth() / 2;
  }

  inline uint32_t paddingRight() const {
    return effectiveKernelWidth() - 1 - paddingLeft();
  }

  inline uint32_t paddingTop() const {
    return effectiveKernelHeight() / 2;
  }

  inline uint32_t paddingBottom() const {
    return effectiveKernelHeight() - 1 - paddingTop();
  }

  inline uint32_t outputHeight() const {
    return (paddingTop() + inputHeight() + paddingBottom() - effectiveKernelHeight()) / subsampling() + 1;
  }

  inline uint32_t outputWidth() const {
    return (paddingLeft() + inputWidth() + paddingRight() - effectiveKernelWidth()) / subsampling() + 1;
  }

  inline uint32_t groups() const {
    return groups_;
  }

  inline uint32_t groupInputChannels() const {
    return groupInputChannels_;
  }

  inline uint32_t groupOutputChannels() const {
    return groupOutputChannels_;
  }

  inline qnnp_operator_t convolutionObject() const {
    return convolutionObject_;
  }

  inline size_t inputPixelStride() const {
    return groups() * groupInputChannels();
  }

  inline size_t outputPixelStride() const {
    return groups() * groupOutputChannels();
  }

 private:
  qnnp_operator_t convolutionObject_;
  std::vector<uint8_t> input_;
  std::vector<uint8_t> kernel_;
  std::vector<int32_t> bias_;
  std::vector<uint8_t> output_;
  size_t batchSize_{1};
  size_t inputHeight_{1};
  size_t inputWidth_{1};
  uint32_t kernelHeight_{1};
  uint32_t kernelWidth_{1};
  uint32_t subsampling_{1};
  uint32_t dilation_{1};
  uint32_t groups_{1};
  uint32_t groupInputChannels_{1};
  uint32_t groupOutputChannels_{1};
};

/* ShuffleNet v1 with 1 group */
static void ShuffleNetV1G1(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 224, 224,  3,  3, 2, 1, 1,    3,   24});
  b->Args({1,  56,  56,  1,  1, 1, 1, 1,   24,   30});
  b->Args({1,  28,  28,  1,  1, 1, 1, 1,   30,  120});
  b->Args({1,  28,  28,  1,  1, 1, 1, 1,  144,   36});
  b->Args({1,  28,  28,  1,  1, 1, 1, 1,   36,  144});
  b->Args({1,  14,  14,  1,  1, 1, 1, 1,   36,  144});
  b->Args({1,  14,  14,  1,  1, 1, 1, 1,  288,   72});
  b->Args({1,  14,  14,  1,  1, 1, 1, 1,   72,  288});
  b->Args({1,   7,   7,  1,  1, 1, 1, 1,   72,  288});
  b->Args({1,   7,   7,  1,  1, 1, 1, 1,  576,  144});
  b->Args({1,   7,   7,  1,  1, 1, 1, 1,  144,  576});
}

/* ShuffleNet v1 with 2 groups */
static void ShuffleNetV1G2(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 224, 224,  3,  3, 2, 1, 1,    3,   24});
  b->Args({1,  56,  56,  1,  1, 1, 1, 2,   12,   22});
  b->Args({1,  28,  28,  1,  1, 1, 1, 2,   22,   88});
  b->Args({1,  28,  28,  1,  1, 1, 1, 2,  100,   25});
  b->Args({1,  28,  28,  1,  1, 1, 1, 2,   25,  100});
  b->Args({1,  14,  14,  1,  1, 1, 1, 2,   25,  100});
  b->Args({1,  14,  14,  1,  1, 1, 1, 2,  200,   50});
  b->Args({1,  14,  14,  1,  1, 1, 1, 2,   50,  200});
  b->Args({1,   7,   7,  1,  1, 1, 1, 2,   50,  200});
  b->Args({1,   7,   7,  1,  1, 1, 1, 2,  400,  100});
  b->Args({1,   7,   7,  1,  1, 1, 1, 2,  100,  400});
}

/* ShuffleNet v1 with 3 groups */
static void ShuffleNetV1G3(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 224, 224,  3,  3, 2, 1, 1,    3,   24});
  b->Args({1,  56,  56,  1,  1, 1, 1, 3,    8,   18});
  b->Args({1,  28,  28,  1,  1, 1, 1, 3,   18,   72});
  b->Args({1,  28,  28,  1,  1, 1, 1, 3,   80,   20});
  b->Args({1,  28,  28,  1,  1, 1, 1, 3,   20,   80});
  b->Args({1,  14,  14,  1,  1, 1, 1, 3,   20,   80});
  b->Args({1,  14,  14,  1,  1, 1, 1, 3,  160,   40});
  b->Args({1,  14,  14,  1,  1, 1, 1, 3,   40,  160});
  b->Args({1,   7,   7,  1,  1, 1, 1, 3,   40,  160});
  b->Args({1,   7,   7,  1,  1, 1, 1, 3,  320,   80});
  b->Args({1,   7,   7,  1,  1, 1, 1, 3,   80,  320});
}

/* ShuffleNet v1 with 4 groups */
static void ShuffleNetV1G4(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 224, 224,  3,  3, 2, 1, 1,    3,   24});
  b->Args({1,  56,  56,  1,  1, 1, 1, 4,    6,   15});
  b->Args({1,  28,  28,  1,  1, 1, 1, 4,   15,   62});
  b->Args({1,  28,  28,  1,  1, 1, 1, 4,   68,   17});
  b->Args({1,  28,  28,  1,  1, 1, 1, 4,   17,   68});
  b->Args({1,  14,  14,  1,  1, 1, 1, 4,   17,   68});
  b->Args({1,  14,  14,  1,  1, 1, 1, 4,  136,   34});
  b->Args({1,  14,  14,  1,  1, 1, 1, 4,   34,  136});
  b->Args({1,   7,   7,  1,  1, 1, 1, 4,   34,  136});
  b->Args({1,   7,   7,  1,  1, 1, 1, 4,  272,   68});
  b->Args({1,   7,   7,  1,  1, 1, 1, 4,   68,  272});
}

/* ShuffleNet v1 with 8 groups */
static void ShuffleNetV1G8(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 224, 224,  3,  3, 2, 1, 1,    3,   24});
  b->Args({1,  56,  56,  1,  1, 1, 1, 8,    3,   11});
  b->Args({1,  28,  28,  1,  1, 1, 1, 8,   11,   45});
  b->Args({1,  28,  28,  1,  1, 1, 1, 8,   48,   12});
  b->Args({1,  28,  28,  1,  1, 1, 1, 8,   12,   48});
  b->Args({1,  14,  14,  1,  1, 1, 1, 8,   12,   48});
  b->Args({1,  14,  14,  1,  1, 1, 1, 8,   96,   24});
  b->Args({1,  14,  14,  1,  1, 1, 1, 8,   24,   96});
  b->Args({1,   7,   7,  1,  1, 1, 1, 8,   24,   96});
  b->Args({1,   7,   7,  1,  1, 1, 1, 8,  192,   48});
  b->Args({1,   7,   7,  1,  1, 1, 1, 8,   48,  192});
}

static void MobileNetV1(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /*       N   H    W   KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 224, 224,  3,  3, 2, 1,    1,    3,   32});
  b->Args({1, 112, 112,  3,  3, 1, 1,   32,    1,    1});
  b->Args({1, 112, 112,  1,  1, 1, 1,    1,   32,   64});
  b->Args({1, 112, 112,  3,  3, 2, 1,   64,    1,    1});
  b->Args({1,  56,  56,  1,  1, 1, 1,    1,   64,  128});
  b->Args({1,  56,  56,  3,  3, 1, 1,  128,    1,    1});
  b->Args({1,  56,  56,  1,  1, 1, 1,    1,  128,  128});
  b->Args({1,  56,  56,  3,  3, 2, 1,  128,    1,    1});
  b->Args({1,  28,  28,  1,  1, 1, 1,    1,  128,  256});
  b->Args({1,  28,  28,  3,  3, 1, 1,  256,    1,    1});
  b->Args({1,  28,  28,  1,  1, 1, 1,    1,  256,  256});
  b->Args({1,  28,  28,  3,  3, 2, 1,  256,    1,    1});
  b->Args({1,  14,  14,  1,  1, 1, 1,    1,  256,  512});
  b->Args({1,  14,  14,  3,  3, 1, 1,  512,    1,    1});
  b->Args({1,  14,  14,  1,  1, 1, 1,    1,  512,  512});
  b->Args({1,  14,  14,  3,  3, 2, 1,  512,    1,    1});
  b->Args({1,   7,   7,  1,  1, 1, 1,    1,  512, 1024});
  b->Args({1,   7,   7,  3,  3, 1, 1, 1024,    1,    1});
  b->Args({1,   7,   7,  1,  1, 1, 1,    1, 1024, 1024});
}

static void MobileNetV2(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
  b->Args({1, 224, 224,  3,  3, 2, 1,   1,    3,   32});

  /******************** Bottleneck 1 *******************/
  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
  b->Args({1, 112, 112,  3,  3, 1, 1,  32,    1,    1});
  b->Args({1, 112, 112,  1,  1, 1, 1,   1,   32,   16});

  /******************** Bottleneck 2 *******************/
  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
  b->Args({1, 112, 112,  1,  1, 1, 1,   1,   16,   96});
  b->Args({1, 112, 112,  3,  3, 2, 1,  96,    1,    1});
  b->Args({1,  56,  56,  1,  1, 1, 1,   1,   96,   24});
  b->Args({1,  56,  56,  1,  1, 1, 1,   1,   24,  144});
  b->Args({1,  56,  56,  3,  3, 1, 1, 144,    1,    1});
  b->Args({1,  56,  56,  1,  1, 1, 1,   1,  144,   24});

  /******************** Bottleneck 3 *******************/
  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
//b->Args({1,  56,  56,  1,  1, 1, 1,   1,   24,  144});
  b->Args({1,  56,  56,  3,  3, 2, 1, 144,    1,    1});
  b->Args({1,  28,  28,  1,  1, 1, 1,   1,  144,   32});
  b->Args({1,  28,  28,  1,  1, 1, 1,   1,   32,  192});
  b->Args({1,  28,  28,  3,  3, 1, 1, 192,    1,    1});
  b->Args({1,  28,  28,  1,  1, 1, 1,   1,  192,   32});
//b->Args({1,  28,  28,  1,  1, 1, 1,   1,   32,  192});
//b->Args({1,  28,  28,  3,  3, 1, 1, 192,    1,    1});
//b->Args({1,  28,  28,  1,  1, 1, 1,   1,  192,   32});

  /******************** Bottleneck 4 *******************/
  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
//b->Args({1,  28,  28,  1,  1, 1, 1,   1,   32,  192});
  b->Args({1,  28,  28,  3,  3, 2, 1, 192,    1,    1});
  b->Args({1,  14,  14,  1,  1, 1, 1,   1,  192,   64});
  b->Args({1,  14,  14,  1,  1, 1, 1,   1,   64,  384});
  b->Args({1,  14,  14,  3,  3, 1, 1, 384,    1,    1});
  b->Args({1,  14,  14,  1,  1, 1, 1,   1,  384,   64});
//b->Args({1,  14,  14,  1,  1, 1, 1,   1,   64,  384});
//b->Args({1,  14,  14,  3,  3, 1, 1, 384,    1,    1});
//b->Args({1,  14,  14,  1,  1, 1, 1,   1,  384,   64});
//b->Args({1,  14,  14,  1,  1, 1, 1,   1,   64,  384});
//b->Args({1,  14,  14,  3,  3, 1, 1, 384,    1,    1});
//b->Args({1,  14,  14,  1,  1, 1, 1,   1,  384,   64});

  /******************** Bottleneck 5 *******************/
  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
//b->Args({1,  14,  14,  1,  1, 1, 1,   1,   64,  384});
//b->Args({1,  14,  14,  3,  3, 1, 1, 384,    1,    1});
  b->Args({1,  14,  14,  1,  1, 1, 1,   1,  384,   96});
  b->Args({1,  14,  14,  1,  1, 1, 1,   1,   96,  576});
  b->Args({1,  14,  14,  3,  3, 1, 1, 576,    1,    1});
  b->Args({1,  14,  14,  1,  1, 1, 1,   1,  576,   96});
//b->Args({1,  14,  14,  1,  1, 1, 1,   1,   96,  576});
//b->Args({1,  14,  14,  3,  3, 1, 1, 576,    1,    1});
//b->Args({1,  14,  14,  1,  1, 1, 1,   1,  576,   96});

  /******************** Bottleneck 6 *******************/
  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
//b->Args({1,  14,  14,  1,  1, 1, 1,   1,   96,  576});
  b->Args({1,  14,  14,  3,  3, 2, 1, 576,    1,    1});
  b->Args({1,   7,   7,  1,  1, 1, 1,   1,  576,  160});
  b->Args({1,   7,   7,  1,  1, 1, 1,   1,  160,  960});
  b->Args({1,   7,   7,  3,  3, 1, 1, 960,    1,    1});
  b->Args({1,   7,   7,  1,  1, 1, 1,   1,  960,  160});
//b->Args({1,   7,   7,  1,  1, 1, 1,   1,  160,  960});
//b->Args({1,   7,   7,  3,  3, 1, 1, 960,    1,    1});
//b->Args({1,   7,   7,  1,  1, 1, 1,   1,  960,  160});

  /******************** Bottleneck 7 *******************/
  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
//b->Args({1,   7,   7,  1,  1, 1, 1,   1,  160,  960});
//b->Args({1,   7,   7,  3,  3, 1, 1, 960,    1,    1});
  b->Args({1,   7,   7,  1,  1, 1, 1,   1,  960,  320});

  /**************** Pre-pooling Conv2D *****************/
  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
  b->Args({1,   7,   7,  1,  1, 1, 1,   1,  320, 1280});
  /**************** Post-pooling Conv2D ****************/
  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
  b->Args({1,   1,   1,  1,  1, 1, 1,   1, 1280, 1000});
}

static void SqueezeNetV10(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /********************** Conv 1 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 224, 224,  7,  7, 2, 1, 1,    3,   96});
  /********************** Fire 2 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  55,  55,  1,  1, 1, 1, 1,   96,   16});
  b->Args({1,  55,  55,  1,  1, 1, 1, 1,   16,   64});
  b->Args({1,  55,  55,  3,  3, 1, 1, 1,   16,   64});
  /********************** Fire 3 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  56,  55,  1,  1, 1, 1, 1,  128,   16});
  b->Args({1,  55,  55,  1,  1, 1, 1, 1,   16,   64});
  b->Args({1,  55,  55,  3,  3, 1, 1, 1,   16,   64});
  /********************** Fire 4 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  55,  55,  1,  1, 1, 1, 1,  128,   32});
  b->Args({1,  55,  55,  1,  1, 1, 1, 1,   32,  128});
  b->Args({1,  55,  55,  3,  3, 1, 1, 1,   32,  128});
  /********************** Fire 5 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  27,  27,  1,  1, 1, 1, 1,  256,   32});
  b->Args({1,  27,  27,  1,  1, 1, 1, 1,   32,  128});
  b->Args({1,  27,  27,  3,  3, 1, 1, 1,   32,  128});
  /********************** Fire 6 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  27,  27,  1,  1, 1, 1, 1,  256,   48});
  b->Args({1,  27,  27,  1,  1, 1, 1, 1,   48,  192});
  b->Args({1,  27,  27,  3,  3, 1, 1, 1,   48,  192});
  /********************** Fire 7 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  27,  27,  1,  1, 1, 1, 1,  384,   48});
  b->Args({1,  27,  27,  1,  1, 1, 1, 1,   48,  192});
  b->Args({1,  27,  27,  3,  3, 1, 1, 1,   48,  192});
  /********************** Fire 8 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  27,  27,  1,  1, 1, 1, 1,  384,   64});
  b->Args({1,  27,  27,  1,  1, 1, 1, 1,   64,  256});
  b->Args({1,  27,  27,  3,  3, 1, 1, 1,   64,  256});
  /********************** Fire 9 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  13,  13,  1,  1, 1, 1, 1,  512,   64});
  b->Args({1,  13,  13,  1,  1, 1, 1, 1,   64,  256});
  b->Args({1,  13,  13,  3,  3, 1, 1, 1,   64,  256});
  /********************* Conv 10 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  13,  13,  1,  1, 1, 1, 1,  512, 1000});
}

static void SqueezeNetV11(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /********************** Conv 1 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 224, 224,  3,  3, 2, 1, 1,    3,   64});
  /********************** Fire 2 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  55,  55,  1,  1, 1, 1, 1,   64,   16});
  b->Args({1,  55,  55,  1,  1, 1, 1, 1,   16,   64});
  b->Args({1,  55,  55,  3,  3, 1, 1, 1,   16,   64});
  /********************** Fire 3 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  55,  55,  1,  1, 1, 1, 1,  128,   16});
  b->Args({1,  55,  55,  1,  1, 1, 1, 1,   16,   64});
  b->Args({1,  55,  55,  3,  3, 1, 1, 1,   16,   64});
  /********************** Fire 4 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  27,  27,  1,  1, 1, 1, 1,  128,   32});
  b->Args({1,  27,  27,  1,  1, 1, 1, 1,   32,  128});
  b->Args({1,  27,  27,  3,  3, 1, 1, 1,   32,  128});
  /********************** Fire 5 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  27,  27,  1,  1, 1, 1, 1,  256,   32});
  b->Args({1,  27,  27,  1,  1, 1, 1, 1,   32,  128});
  b->Args({1,  27,  27,  3,  3, 1, 1, 1,   32,  128});
  /********************** Fire 6 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  13,  13,  1,  1, 1, 1, 1,  256,   48});
  b->Args({1,  13,  13,  1,  1, 1, 1, 1,   48,  192});
  b->Args({1,  13,  13,  3,  3, 1, 1, 1,   48,  192});
  /********************** Fire 7 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  13,  13,  1,  1, 1, 1, 1,  384,   48});
  b->Args({1,  13,  13,  1,  1, 1, 1, 1,   48,  192});
  b->Args({1,  13,  13,  3,  3, 1, 1, 1,   48,  192});
  /********************** Fire 8 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  13,  13,  1,  1, 1, 1, 1,  384,   64});
  b->Args({1,  13,  13,  1,  1, 1, 1, 1,   64,  256});
  b->Args({1,  13,  13,  3,  3, 1, 1, 1,   64,  256});
  /********************** Fire 9 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  13,  13,  1,  1, 1, 1, 1,  512,   64});
  b->Args({1,  13,  13,  1,  1, 1, 1, 1,   64,  256});
  b->Args({1,  13,  13,  3,  3, 1, 1, 1,   64,  256});
  /********************* Conv 10 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  13,  13,  1,  1, 1, 1, 1,  512, 1000});
}

static void VGG(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /********************* Conv 1.1 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 224, 224,  3,  3, 1, 1, 1,    3,   64});
  /********************* Conv 1.2 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 224, 224,  3,  3, 1, 1, 1,   64,   64});

  /********************* Conv 2.1 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 112, 112,  3,  3, 1, 1, 1,   64,  128});
  /********************* Conv 2.2 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 112, 112,  3,  3, 1, 1, 1,  128,  128});

  /********************* Conv 3.1 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  56,  56,  3,  3, 1, 1, 1,  128,  256});
  /********************* Conv 3.2 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  56,  56,  3,  3, 1, 1, 1,  256,  256});
  /********************* Conv 3.3 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  56,  56,  1,  1, 1, 1, 1,  256,  256});

  /********************* Conv 4.1 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  28,  28,  3,  3, 1, 1, 1,  256,  512});
  /********************* Conv 4.2 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  28,  28,  3,  3, 1, 1, 1,  512,  512});
  /********************* Conv 4.3 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  28,  28,  1,  1, 1, 1, 1,  512,  512});

  /********************* Conv 5.X ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  14,  14,  3,  3, 1, 1, 1,  512,  512});
  /********************* Conv 5.3 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1,  14,  14,  1,  1, 1, 1, 1,  512,  512});
}

static void DWConv3x3(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /********************** 96 x 96 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 96, 96,  3,  3, 1, 1,  512,    1,    1});
  b->Args({1, 96, 96,  3,  3, 1, 1,  256,    1,    1});
  b->Args({1, 96, 96,  3,  3, 1, 1,  128,    1,    1});
  b->Args({1, 96, 96,  3,  3, 1, 1,   64,    1,    1});
  b->Args({1, 96, 96,  3,  3, 1, 1,   48,    1,    1});
  b->Args({1, 96, 96,  3,  3, 1, 1,   32,    1,    1});
  b->Args({1, 96, 96,  3,  3, 1, 1,   24,    1,    1});
  b->Args({1, 96, 96,  3,  3, 1, 1,   16,    1,    1});
  /********************** 32 x 32 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 32, 32,  3,  3, 1, 1,  768,    1,    1});
  b->Args({1, 32, 32,  3,  3, 1, 1,  512,    1,    1});
  b->Args({1, 32, 32,  3,  3, 1, 1,  256,    1,    1});
  b->Args({1, 32, 32,  3,  3, 1, 1,  128,    1,    1});
  b->Args({1, 32, 32,  3,  3, 1, 1,   64,    1,    1});
  b->Args({1, 32, 32,  3,  3, 1, 1,   48,    1,    1});
  b->Args({1, 32, 32,  3,  3, 1, 1,   32,    1,    1});
  b->Args({1, 32, 32,  3,  3, 1, 1,   24,    1,    1});
  b->Args({1, 32, 32,  3,  3, 1, 1,   16,    1,    1});
  /********************** 17 x 17 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 17, 17,  3,  3, 1, 1, 1024,    1,    1});
  b->Args({1, 17, 17,  3,  3, 1, 1,  768,    1,    1});
  b->Args({1, 17, 17,  3,  3, 1, 1,  512,    1,    1});
  b->Args({1, 17, 17,  3,  3, 1, 1,  384,    1,    1});
  b->Args({1, 17, 17,  3,  3, 1, 1,  256,    1,    1});
  b->Args({1, 17, 17,  3,  3, 1, 1,  128,    1,    1});
  b->Args({1, 17, 17,  3,  3, 1, 1,   64,    1,    1});
  b->Args({1, 17, 17,  3,  3, 1, 1,   32,    1,    1});
  b->Args({1, 17, 17,  3,  3, 1, 1,   16,    1,    1});
  /********************** 11 x 11 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 11, 11,  3,  3, 1, 1, 1024,    1,    1});
  b->Args({1, 11, 11,  3,  3, 1, 1,  768,    1,    1});
  b->Args({1, 11, 11,  3,  3, 1, 1,  512,    1,    1});
  b->Args({1, 11, 11,  3,  3, 1, 1,  384,    1,    1});
  b->Args({1, 11, 11,  3,  3, 1, 1,  256,    1,    1});
  b->Args({1, 11, 11,  3,  3, 1, 1,  192,    1,    1});
  b->Args({1, 11, 11,  3,  3, 1, 1,  128,    1,    1});
  b->Args({1, 11, 11,  3,  3, 1, 1,   64,    1,    1});
  b->Args({1, 11, 11,  3,  3, 1, 1,   32,    1,    1});
  b->Args({1, 11, 11,  3,  3, 1, 1,   16,    1,    1});
  /*********************** 7 x 7 **********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1,  7,  7,  3,  3, 1, 1, 1024,    1,    1});
  b->Args({1,  7,  7,  3,  3, 1, 1,  768,    1,    1});
  b->Args({1,  7,  7,  3,  3, 1, 1,  512,    1,    1});
  b->Args({1,  7,  7,  3,  3, 1, 1,  384,    1,    1});
  b->Args({1,  7,  7,  3,  3, 1, 1,  256,    1,    1});
  b->Args({1,  7,  7,  3,  3, 1, 1,  128,    1,    1});
  b->Args({1,  7,  7,  3,  3, 1, 1,   64,    1,    1});
  b->Args({1,  7,  7,  3,  3, 1, 1,   32,    1,    1});
  b->Args({1,  7,  7,  3,  3, 1, 1,   16,    1,    1});
}

static void DWConv3x3d2(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /********************** 96 x 96 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 96, 96,  3,  3, 1, 2,  512,    1,    1});
  b->Args({1, 96, 96,  3,  3, 1, 2,  256,    1,    1});
  b->Args({1, 96, 96,  3,  3, 1, 2,  128,    1,    1});
  b->Args({1, 96, 96,  3,  3, 1, 2,   64,    1,    1});
  b->Args({1, 96, 96,  3,  3, 1, 2,   48,    1,    1});
  b->Args({1, 96, 96,  3,  3, 1, 2,   32,    1,    1});
  b->Args({1, 96, 96,  3,  3, 1, 2,   24,    1,    1});
  b->Args({1, 96, 96,  3,  3, 1, 2,   16,    1,    1});
  /********************** 32 x 32 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 32, 32,  3,  3, 1, 2,  768,    1,    1});
  b->Args({1, 32, 32,  3,  3, 1, 2,  512,    1,    1});
  b->Args({1, 32, 32,  3,  3, 1, 2,  256,    1,    1});
  b->Args({1, 32, 32,  3,  3, 1, 2,  128,    1,    1});
  b->Args({1, 32, 32,  3,  3, 1, 2,   64,    1,    1});
  b->Args({1, 32, 32,  3,  3, 1, 2,   48,    1,    1});
  b->Args({1, 32, 32,  3,  3, 1, 2,   32,    1,    1});
  b->Args({1, 32, 32,  3,  3, 1, 2,   24,    1,    1});
  b->Args({1, 32, 32,  3,  3, 1, 2,   16,    1,    1});
  /********************** 17 x 17 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 17, 17,  3,  3, 1, 2, 1024,    1,    1});
  b->Args({1, 17, 17,  3,  3, 1, 2,  768,    1,    1});
  b->Args({1, 17, 17,  3,  3, 1, 2,  512,    1,    1});
  b->Args({1, 17, 17,  3,  3, 1, 2,  384,    1,    1});
  b->Args({1, 17, 17,  3,  3, 1, 2,  256,    1,    1});
  b->Args({1, 17, 17,  3,  3, 1, 2,  128,    1,    1});
  b->Args({1, 17, 17,  3,  3, 1, 2,   64,    1,    1});
  b->Args({1, 17, 17,  3,  3, 1, 2,   32,    1,    1});
  b->Args({1, 17, 17,  3,  3, 1, 2,   16,    1,    1});
  /********************** 11 x 11 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 11, 11,  3,  3, 1, 2, 1024,    1,    1});
  b->Args({1, 11, 11,  3,  3, 1, 2,  768,    1,    1});
  b->Args({1, 11, 11,  3,  3, 1, 2,  512,    1,    1});
  b->Args({1, 11, 11,  3,  3, 1, 2,  384,    1,    1});
  b->Args({1, 11, 11,  3,  3, 1, 2,  256,    1,    1});
  b->Args({1, 11, 11,  3,  3, 1, 2,  192,    1,    1});
  b->Args({1, 11, 11,  3,  3, 1, 2,  128,    1,    1});
  b->Args({1, 11, 11,  3,  3, 1, 2,   64,    1,    1});
  b->Args({1, 11, 11,  3,  3, 1, 2,   32,    1,    1});
  b->Args({1, 11, 11,  3,  3, 1, 2,   16,    1,    1});
  /*********************** 7 x 7 **********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1,  7,  7,  3,  3, 1, 2, 1024,    1,    1});
  b->Args({1,  7,  7,  3,  3, 1, 2,  768,    1,    1});
  b->Args({1,  7,  7,  3,  3, 1, 2,  512,    1,    1});
  b->Args({1,  7,  7,  3,  3, 1, 2,  384,    1,    1});
  b->Args({1,  7,  7,  3,  3, 1, 2,  256,    1,    1});
  b->Args({1,  7,  7,  3,  3, 1, 2,  128,    1,    1});
  b->Args({1,  7,  7,  3,  3, 1, 2,   64,    1,    1});
  b->Args({1,  7,  7,  3,  3, 1, 2,   32,    1,    1});
  b->Args({1,  7,  7,  3,  3, 1, 2,   16,    1,    1});
}

static void DWConv5x5(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /********************** 96 x 96 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 96, 96,  5,  5, 1, 1,  512,    1,    1});
  b->Args({1, 96, 96,  5,  5, 1, 1,  256,    1,    1});
  b->Args({1, 96, 96,  5,  5, 1, 1,  128,    1,    1});
  b->Args({1, 96, 96,  5,  5, 1, 1,   64,    1,    1});
  b->Args({1, 96, 96,  5,  5, 1, 1,   48,    1,    1});
  b->Args({1, 96, 96,  5,  5, 1, 1,   32,    1,    1});
  b->Args({1, 96, 96,  5,  5, 1, 1,   24,    1,    1});
  b->Args({1, 96, 96,  5,  5, 1, 1,   16,    1,    1});
  /********************** 32 x 32 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 32, 32,  5,  5, 1, 1,  768,    1,    1});
  b->Args({1, 32, 32,  5,  5, 1, 1,  512,    1,    1});
  b->Args({1, 32, 32,  5,  5, 1, 1,  256,    1,    1});
  b->Args({1, 32, 32,  5,  5, 1, 1,  128,    1,    1});
  b->Args({1, 32, 32,  5,  5, 1, 1,   64,    1,    1});
  b->Args({1, 32, 32,  5,  5, 1, 1,   48,    1,    1});
  b->Args({1, 32, 32,  5,  5, 1, 1,   32,    1,    1});
  b->Args({1, 32, 32,  5,  5, 1, 1,   24,    1,    1});
  b->Args({1, 32, 32,  5,  5, 1, 1,   16,    1,    1});
  /********************** 17 x 17 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 17, 17,  5,  5, 1, 1, 1024,    1,    1});
  b->Args({1, 17, 17,  5,  5, 1, 1,  768,    1,    1});
  b->Args({1, 17, 17,  5,  5, 1, 1,  512,    1,    1});
  b->Args({1, 17, 17,  5,  5, 1, 1,  384,    1,    1});
  b->Args({1, 17, 17,  5,  5, 1, 1,  256,    1,    1});
  b->Args({1, 17, 17,  5,  5, 1, 1,  128,    1,    1});
  b->Args({1, 17, 17,  5,  5, 1, 1,   64,    1,    1});
  b->Args({1, 17, 17,  5,  5, 1, 1,   32,    1,    1});
  b->Args({1, 17, 17,  5,  5, 1, 1,   16,    1,    1});
  /********************** 11 x 11 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 11, 11,  5,  5, 1, 1, 1024,    1,    1});
  b->Args({1, 11, 11,  5,  5, 1, 1,  768,    1,    1});
  b->Args({1, 11, 11,  5,  5, 1, 1,  512,    1,    1});
  b->Args({1, 11, 11,  5,  5, 1, 1,  384,    1,    1});
  b->Args({1, 11, 11,  5,  5, 1, 1,  256,    1,    1});
  b->Args({1, 11, 11,  5,  5, 1, 1,  128,    1,    1});
  b->Args({1, 11, 11,  5,  5, 1, 1,   64,    1,    1});
  b->Args({1, 11, 11,  5,  5, 1, 1,   32,    1,    1});
  b->Args({1, 11, 11,  5,  5, 1, 1,   16,    1,    1});
  /*********************** 7 x 7 **********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1,  7,  7,  5,  5, 1, 1, 1024,    1,    1});
  b->Args({1,  7,  7,  5,  5, 1, 1,  768,    1,    1});
  b->Args({1,  7,  7,  5,  5, 1, 1,  512,    1,    1});
  b->Args({1,  7,  7,  5,  5, 1, 1,  384,    1,    1});
  b->Args({1,  7,  7,  5,  5, 1, 1,  256,    1,    1});
  b->Args({1,  7,  7,  5,  5, 1, 1,  128,    1,    1});
  b->Args({1,  7,  7,  5,  5, 1, 1,   64,    1,    1});
  b->Args({1,  7,  7,  5,  5, 1, 1,   32,    1,    1});
  b->Args({1,  7,  7,  5,  5, 1, 1,   16,    1,    1});
}

BENCHMARK_DEFINE_F(Q8Convolution, run)(benchmark::State& state)
{
  for (auto _ : state) {
    qnnp_run_operator(convolutionObject(), nullptr /* thread pool */);
  }
}
BENCHMARK_REGISTER_F(Q8Convolution, run)->Apply(MobileNetV1);
BENCHMARK_REGISTER_F(Q8Convolution, run)->Apply(MobileNetV2);
BENCHMARK_REGISTER_F(Q8Convolution, run)->Apply(ShuffleNetV1G1);
BENCHMARK_REGISTER_F(Q8Convolution, run)->Apply(ShuffleNetV1G2);
BENCHMARK_REGISTER_F(Q8Convolution, run)->Apply(ShuffleNetV1G3);
BENCHMARK_REGISTER_F(Q8Convolution, run)->Apply(ShuffleNetV1G4);
BENCHMARK_REGISTER_F(Q8Convolution, run)->Apply(ShuffleNetV1G8);
BENCHMARK_REGISTER_F(Q8Convolution, run)->Apply(SqueezeNetV10);
BENCHMARK_REGISTER_F(Q8Convolution, run)->Apply(SqueezeNetV11);
BENCHMARK_REGISTER_F(Q8Convolution, run)->Apply(VGG);
BENCHMARK_REGISTER_F(Q8Convolution, run)->Apply(DWConv3x3);
BENCHMARK_REGISTER_F(Q8Convolution, run)->Apply(DWConv3x3d2);
BENCHMARK_REGISTER_F(Q8Convolution, run)->Apply(DWConv5x5);

#ifndef QNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
