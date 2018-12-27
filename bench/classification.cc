/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <qnnpack.h>

#include <benchmark/benchmark.h>


class ConvolutionArguments {
public:
  inline ConvolutionArguments& padding(uint32_t padding) {
    this->paddingTop_ = padding;
    this->paddingRight_ = padding;
    this->paddingBottom_ = padding;
    this->paddingLeft_ = padding;
    return *this;
  }

  inline ConvolutionArguments& padding(uint32_t paddingHeight, uint32_t paddingWidth) {
    this->paddingTop_ = paddingHeight;
    this->paddingRight_ = paddingWidth;
    this->paddingBottom_ = paddingHeight;
    this->paddingLeft_ = paddingWidth;
    return *this;
  }

  inline ConvolutionArguments& paddingHeight(uint32_t paddingHeight) {
    this->paddingTop_ = paddingHeight;
    this->paddingBottom_ = paddingHeight;
    return *this;
  }

  inline ConvolutionArguments& paddingWidth(uint32_t paddingWidth) {
    this->paddingRight_ = paddingWidth;
    this->paddingLeft_ = paddingWidth;
    return *this;
  }

  inline ConvolutionArguments& paddingTop(uint32_t paddingTop) {
    this->paddingTop_ = paddingTop;
    return *this;
  }

  inline uint32_t paddingTop() const {
    return this->paddingTop_;
  }

  inline ConvolutionArguments& paddingRight(uint32_t paddingRight) {
    this->paddingRight_ = paddingRight;
    return *this;
  }

  inline uint32_t paddingRight() const {
    return this->paddingRight_;
  }

  inline ConvolutionArguments& paddingBottom(uint32_t paddingBottom) {
    this->paddingBottom_ = paddingBottom;
    return *this;
  }

  inline uint32_t paddingBottom() const {
    return this->paddingBottom_;
  }

  inline ConvolutionArguments& paddingLeft(uint32_t paddingLeft) {
    this->paddingLeft_ = paddingLeft;
    return *this;
  }

  inline uint32_t paddingLeft() const {
    return this->paddingLeft_;
  }

  inline ConvolutionArguments& inputSize(uint32_t inputHeight, uint32_t inputWidth) {
    assert(inputHeight >= 1);
    assert(inputWidth >= 1);
    this->inputHeight_ = inputHeight;
    this->inputWidth_ = inputWidth;
    return *this;
  }

  inline ConvolutionArguments& inputHeight(uint32_t inputHeight) {
    assert(inputHeight >= 1);
    this->inputHeight_ = inputHeight;
    return *this;
  }

  inline uint32_t inputHeight() const {
    return this->inputHeight_;
  }

  inline ConvolutionArguments& inputWidth(uint32_t inputWidth) {
    assert(inputWidth >= 1);
    this->inputWidth_ = inputWidth;
    return *this;
  }

  inline uint32_t inputWidth() const {
    return this->inputWidth_;
  }

  inline ConvolutionArguments& groups(uint32_t groups) {
    assert(groups >= 1);
    this->groups_ = groups;
    return *this;
  }

  inline uint32_t groups() const {
    return this->groups_;
  }

  inline ConvolutionArguments& groupInputChannels(size_t groupInputChannels) {
    assert(groupInputChannels >= 1);
    this->groupInputChannels_ = groupInputChannels;
    return *this;
  }

  inline size_t groupInputChannels() const {
    return this->groupInputChannels_;
  }

  inline ConvolutionArguments& groupOutputChannels(size_t groupOutputChannels) {
    assert(groupOutputChannels >= 1);
    this->groupOutputChannels_ = groupOutputChannels;
    return *this;
  }

  inline size_t groupOutputChannels() const {
    return this->groupOutputChannels_;
  }

  inline ConvolutionArguments& batchSize(size_t batchSize) {
    assert(batchSize >= 1);
    this->batchSize_ = batchSize;
    return *this;
  }

  inline size_t batchSize() const {
    return this->batchSize_;
  }

  inline ConvolutionArguments& kernelSize(uint32_t kernelSize) {
    assert(kernelSize >= 1);
    this->kernelHeight_ = kernelSize;
    this->kernelWidth_ = kernelSize;
    return *this;
  }

  inline ConvolutionArguments& kernelSize(uint32_t kernelHeight, uint32_t kernelWidth) {
    assert(kernelHeight >= 1);
    assert(kernelWidth >= 1);
    this->kernelHeight_ = kernelHeight;
    this->kernelWidth_ = kernelWidth;
    return *this;
  }

  inline ConvolutionArguments& kernelHeight(uint32_t kernelHeight) {
    assert(kernelHeight >= 1);
    this->kernelHeight_ = kernelHeight;
    return *this;
  }

  inline uint32_t kernelHeight() const {
    return this->kernelHeight_;
  }

  inline ConvolutionArguments& kernelWidth(uint32_t kernelWidth) {
    assert(kernelWidth >= 1);
    this->kernelWidth_ = kernelWidth;
    return *this;
  }

  inline uint32_t kernelWidth() const {
    return this->kernelWidth_;
  }

  inline ConvolutionArguments& dilation(uint32_t dilation) {
    assert(dilation >= 1);
    this->dilationHeight_ = dilation;
    this->dilationWidth_ = dilation;
    return *this;
  }

  inline ConvolutionArguments& dilation(uint32_t dilationHeight, uint32_t dilationWidth) {
    assert(dilationHeight >= 1);
    assert(dilationWidth >= 1);
    this->dilationHeight_ = dilationHeight;
    this->dilationWidth_ = dilationWidth;
    return *this;
  }

  inline ConvolutionArguments& dilationHeight(uint32_t dilationHeight) {
    assert(dilationHeight >= 1);
    this->dilationHeight_ = dilationHeight;
    return *this;
  }

  inline uint32_t dilationHeight() const {
    return this->dilationHeight_;
  }

  inline ConvolutionArguments& dilationWidth(uint32_t dilationWidth) {
    assert(dilationWidth >= 1);
    this->dilationWidth_ = dilationWidth;
    return *this;
  }

  inline uint32_t dilationWidth() const {
    return this->dilationWidth_;
  }

  inline ConvolutionArguments& subsampling(uint32_t subsampling) {
    assert(subsampling >= 1);
    this->subsamplingHeight_ = subsampling;
    this->subsamplingWidth_ = subsampling;
    return *this;
  }

  inline ConvolutionArguments& subsampling(uint32_t subsamplingHeight, uint32_t subsamplingWidth) {
    assert(subsamplingHeight >= 1);
    assert(subsamplingWidth >= 1);
    this->subsamplingHeight_ = subsamplingHeight;
    this->subsamplingWidth_ = subsamplingWidth;
    return *this;
  }

  inline ConvolutionArguments& subsamplingHeight(uint32_t subsamplingHeight) {
    assert(subsamplingHeight >= 1);
    this->subsamplingHeight_ = subsamplingHeight;
    return *this;
  }

  inline uint32_t subsamplingHeight() const {
    return this->subsamplingHeight_;
  }

  inline ConvolutionArguments& subsamplingWidth(uint32_t subsamplingWidth) {
    assert(subsamplingWidth >= 1);
    this->subsamplingWidth_ = subsamplingWidth;
    return *this;
  }

  inline uint32_t subsamplingWidth() const {
    return this->subsamplingWidth_;
  }

  inline ConvolutionArguments& inputPixelStride(size_t inputPixelStride) {
    assert(inputPixelStride >= 1);
    this->inputPixelStride_ = inputPixelStride;
    return *this;
  }

  inline size_t inputPixelStride() const {
    if (this->inputPixelStride_ == 0) {
      return groupInputChannels() * groups();
    } else {
      assert(this->inputPixelStride_ >= groupInputChannels() * groups());
      return this->inputPixelStride_;
    }
  }

  inline ConvolutionArguments& outputPixelStride(size_t outputPixelStride) {
    assert(outputPixelStride >= 1);
    this->outputPixelStride_ = outputPixelStride;
    return *this;
  }

  inline size_t outputPixelStride() const {
    if (this->outputPixelStride_ == 0) {
      return groupOutputChannels() * groups();
    } else {
      assert(this->outputPixelStride_ >= groupOutputChannels() * groups());
      return this->outputPixelStride_;
    }
  }

  inline uint32_t dilatedKernelHeight() const {
    return (kernelHeight() - 1) * dilationHeight() + 1;
  }

  inline uint32_t dilatedKernelWidth() const {
    return (kernelWidth() - 1) * dilationWidth() + 1;
  }

  inline size_t outputHeight() const {
    const size_t paddedInputHeight = paddingTop() + inputHeight() + paddingBottom();
    if (paddedInputHeight <= dilatedKernelHeight()) {
      return 1;
    } else {
      return (paddedInputHeight - dilatedKernelHeight()) / subsamplingHeight() + 1;
    }
  }

  inline size_t outputWidth() const {
    const size_t paddedInputWidth = paddingLeft() + inputWidth() + paddingRight();
    if (paddedInputWidth <= dilatedKernelWidth()) {
      return 1;
    } else {
      return (paddedInputWidth - dilatedKernelWidth()) / subsamplingWidth() + 1;
    }
  }

private:
  uint32_t paddingTop_{0};
  uint32_t paddingRight_{0};
  uint32_t paddingBottom_{0};
  uint32_t paddingLeft_{0};
  size_t inputHeight_{1};
  size_t inputWidth_{1};
  uint32_t groups_{1};
  size_t groupInputChannels_{1};
  size_t inputPixelStride_{0};
  size_t groupOutputChannels_{1};
  size_t outputPixelStride_{0};
  size_t batchSize_{1};
  uint32_t kernelHeight_{1};
  uint32_t kernelWidth_{1};
  uint32_t dilationHeight_{1};
  uint32_t dilationWidth_{1};
  uint32_t subsamplingHeight_{1};
  uint32_t subsamplingWidth_{1};
};

class GlobalAveragePoolingArguments {
public:
  inline GlobalAveragePoolingArguments& inputSize(uint32_t inputHeight, uint32_t inputWidth) {
    assert(inputHeight >= 1);
    assert(inputWidth >= 1);
    this->inputHeight_ = inputHeight;
    this->inputWidth_ = inputWidth;
    return *this;
  }

  inline GlobalAveragePoolingArguments& inputHeight(uint32_t inputHeight) {
    assert(inputHeight >= 1);
    this->inputHeight_ = inputHeight;
    return *this;
  }

  inline uint32_t inputHeight() const {
    return this->inputHeight_;
  }

  inline GlobalAveragePoolingArguments& inputWidth(uint32_t inputWidth) {
    assert(inputWidth >= 1);
    this->inputWidth_ = inputWidth;
    return *this;
  }

  inline uint32_t inputWidth() const {
    return this->inputWidth_;
  }

  inline GlobalAveragePoolingArguments& channels(size_t channels) {
    assert(channels >= 1);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline GlobalAveragePoolingArguments& batchSize(size_t batchSize) {
    assert(batchSize >= 1);
    this->batchSize_ = batchSize;
    return *this;
  }

  inline size_t batchSize() const {
    return this->batchSize_;
  }

  inline GlobalAveragePoolingArguments& inputPixelStride(size_t inputPixelStride) {
    assert(inputPixelStride >= 1);
    this->inputPixelStride_ = inputPixelStride;
    return *this;
  }

  inline size_t inputPixelStride() const {
    if (this->inputPixelStride_ == 0) {
      return channels();
    } else {
      assert(this->inputPixelStride_ >= channels());
      return this->inputPixelStride_;
    }
  }

  inline GlobalAveragePoolingArguments& outputPixelStride(size_t outputPixelStride) {
    assert(outputPixelStride >= 1);
    this->outputPixelStride_ = outputPixelStride;
    return *this;
  }

  inline size_t outputPixelStride() const {
    if (this->outputPixelStride_ == 0) {
      return channels();
    } else {
      assert(this->outputPixelStride_ >= channels());
      return this->outputPixelStride_;
    }
  }

private:
  uint32_t paddingTop_{0};
  uint32_t paddingRight_{0};
  uint32_t paddingBottom_{0};
  uint32_t paddingLeft_{0};
  size_t inputHeight_{1};
  size_t inputWidth_{1};
  size_t channels_{1};
  size_t inputPixelStride_{0};
  size_t outputPixelStride_{0};
  size_t batchSize_{1};
};

class FullyConnectedArguments {
public:
  inline FullyConnectedArguments& inputChannels(size_t inputChannels) {
    assert(inputChannels >= 1);
    this->inputChannels_ = inputChannels;
    return *this;
  }

  inline size_t inputChannels() const {
    return this->inputChannels_;
  }

  inline FullyConnectedArguments& outputChannels(size_t outputChannels) {
    assert(outputChannels >= 1);
    this->outputChannels_ = outputChannels;
    return *this;
  }

  inline size_t outputChannels() const {
    return this->outputChannels_;
  }

  inline FullyConnectedArguments& batchSize(size_t batchSize) {
    assert(batchSize >= 1);
    this->batchSize_ = batchSize;
    return *this;
  }

  inline size_t batchSize() const {
    return this->batchSize_;
  }

  inline FullyConnectedArguments& inputStride(size_t inputStride) {
    assert(inputStride >= 1);
    this->inputStride_ = inputStride;
    return *this;
  }

  inline size_t inputStride() const {
    if (this->inputStride_ == 0) {
      return inputChannels();
    } else {
      assert(this->inputStride_ >= inputChannels());
      return this->inputStride_;
    }
  }

  inline FullyConnectedArguments& outputStride(size_t outputStride) {
    assert(outputStride >= 1);
    this->outputStride_ = outputStride;
    return *this;
  }

  inline size_t outputStride() const {
    if (this->outputStride_ == 0) {
      return outputChannels();
    } else {
      assert(this->outputStride_ >= outputChannels());
      return this->outputStride_;
    }
  }

private:
  size_t inputChannels_{1};
  size_t inputStride_{0};
  size_t outputChannels_{1};
  size_t outputStride_{0};
  size_t batchSize_{1};
};

class Convolution {
public:
  Convolution(const ConvolutionArguments& args) :
    args(args)
  {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> kernel(
      args.groups() * args.groupOutputChannels() * args.kernelHeight() * args.kernelWidth() * args.groupInputChannels());
    std::vector<int32_t> bias(args.groups() * args.groupOutputChannels());

    qnnp_status status = qnnp_initialize();
    assert(status == qnnp_status_success);

    status = qnnp_create_convolution2d_nhwc_q8(
      args.paddingTop(), args.paddingRight(), args.paddingBottom(), args.paddingLeft(),
      args.kernelHeight(), args.kernelWidth(),
      args.subsamplingHeight(), args.subsamplingWidth(),
      args.dilationHeight(), args.dilationWidth(),
      args.groups(), args.groupInputChannels(), args.groupOutputChannels(),
      127, 0.5f,
      127, 0.5f,
      kernel.data(), bias.data(),
      127, 0.5f, 0, 255,
      &operatorObject);
    assert(status == qnnp_status_success);
    assert(operatorObject != nullptr);
  }

  Convolution(Convolution&& convolution) :
    args(convolution.args),
    operatorObject(convolution.operatorObject)
  {
    convolution.operatorObject = nullptr;
  }

  Convolution(const Convolution&) = delete;

  size_t inputSize() const {
    return (args.batchSize() * args.inputHeight() * args.inputWidth() - 1) * args.inputPixelStride() + args.groups() * args.groupInputChannels();
  }

  size_t outputSize() const {
    return (args.batchSize() * args.outputHeight() * args.outputWidth() - 1) * args.outputPixelStride() + args.groups() * args.groupOutputChannels();
  }

  double multiplyAdds() const {
    return double(args.batchSize()) * args.outputHeight() * args.outputWidth() *
      args.kernelHeight() * args.kernelWidth() * args.groups() * args.groupInputChannels() * args.groupOutputChannels();
  }

  void setup(const void* input, void* output) {
    qnnp_status status = qnnp_setup_convolution2d_nhwc_q8(
      operatorObject,
      args.batchSize(), args.inputHeight(), args.inputWidth(),
      static_cast<const uint8_t*>(input), args.inputPixelStride(),
      static_cast<uint8_t*>(output), args.outputPixelStride(),
      nullptr /* thread pool */);
    assert(status == qnnp_status_success);
  }

  void run() {
    qnnp_status status = qnnp_run_operator(operatorObject, /* thread pool */ nullptr);
    assert(status == qnnp_status_success);
  }

  ~Convolution() {
    if (operatorObject != nullptr) {
      qnnp_status status = qnnp_delete_operator(operatorObject);
      assert(status == qnnp_status_success);
      operatorObject = nullptr;
    }
  }

private:
  ConvolutionArguments args;
  qnnp_operator_t operatorObject{nullptr};
};

class GlobalAveragePooling {
public:
  GlobalAveragePooling(const GlobalAveragePoolingArguments& args) :
    args(args)
  {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    qnnp_status status = qnnp_initialize();
    assert(status == qnnp_status_success);

    status = qnnp_create_global_average_pooling_nwc_q8(
      args.channels(),
      127 /* input zero point */, 1.0f /* input scale */,
      127 /* output zero point */, 1.0f /* output scale */,
      0 /* output min */, 255 /* output max */,
      &operatorObject);
    assert(status == qnnp_status_success);
    assert(operatorObject != nullptr);
  }

  GlobalAveragePooling(GlobalAveragePooling&& globalPooling) :
    args(globalPooling.args),
    operatorObject(globalPooling.operatorObject)
  {
    globalPooling.operatorObject = nullptr;
  }

  GlobalAveragePooling(const GlobalAveragePooling&) = delete;

  size_t inputSize() const {
    return (args.batchSize() * args.inputHeight() * args.inputWidth() - 1) * args.inputPixelStride() + args.channels();
  }

  size_t outputSize() const {
    return (args.batchSize() - 1) * args.outputPixelStride() + args.channels();
  }

  void setup(const void* input, void* output) {
    qnnp_status status = qnnp_setup_global_average_pooling_nwc_q8(
      operatorObject,
      args.batchSize(), args.inputHeight() * args.inputWidth(),
      static_cast<const uint8_t*>(input), args.inputPixelStride(),
      static_cast<uint8_t*>(output), args.outputPixelStride());
    assert(status == qnnp_status_success);
  }

  void run() {
    qnnp_status status = qnnp_run_operator(operatorObject, /* thread pool */ nullptr);
    assert(status == qnnp_status_success);
  }

  ~GlobalAveragePooling() {
    if (operatorObject != nullptr) {
      qnnp_status status = qnnp_delete_operator(operatorObject);
      assert(status == qnnp_status_success);
      operatorObject = nullptr;
    }
  }

private:
  GlobalAveragePoolingArguments args;
  qnnp_operator_t operatorObject{nullptr};
};

class FullyConnected {
public:
  FullyConnected(const FullyConnectedArguments& args) :
    args(args)
  {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> kernel(args.outputChannels() * args.inputChannels());
    std::vector<int32_t> bias(args.outputChannels());

    qnnp_status status = qnnp_initialize();
    assert(status == qnnp_status_success);

    status = qnnp_create_fully_connected_nc_q8(
      args.inputChannels(), args.outputChannels(),
      127 /* input zero point */, 0.5f /* input scale */,
      127 /* kernel zero point */, 0.5f /* kernel scale */,
      kernel.data(), bias.data(),
      127 /* output zero point */, 0.5f /* output scale */,
      0 /* output min */, 255 /* output max */,
      &operatorObject);
    assert(status == qnnp_status_success);
    assert(operatorObject != nullptr);
  }

  FullyConnected(FullyConnected&& fc) :
    args(fc.args),
    operatorObject(fc.operatorObject)
  {
    fc.operatorObject = nullptr;
  }

  FullyConnected(const FullyConnected&) = delete;

  size_t inputSize() const {
    return (args.batchSize() - 1) * args.inputStride() + args.inputChannels();
  }

  size_t outputSize() const {
    return (args.batchSize() - 1) * args.outputStride() + args.outputChannels();
  }

  double multiplyAdds() const {
    return double(args.batchSize()) * args.inputChannels() * args.outputChannels();
  }

  void setup(const void* input, void* output) {
    qnnp_status status = qnnp_setup_fully_connected_nc_q8(
      operatorObject,
      args.batchSize(),
      static_cast<const uint8_t*>(input), args.inputStride(),
      static_cast<uint8_t*>(output), args.outputStride(),
      nullptr /* thread pool */);
    assert(status == qnnp_status_success);
  }

  void run() {
    qnnp_status status = qnnp_run_operator(operatorObject, /* thread pool */ nullptr);
    assert(status == qnnp_status_success);
  }

  ~FullyConnected() {
    if (operatorObject != nullptr) {
      qnnp_status status = qnnp_delete_operator(operatorObject);
      assert(status == qnnp_status_success);
      operatorObject = nullptr;
    }
  }

private:
  FullyConnectedArguments args;
  qnnp_operator_t operatorObject{nullptr};
};

static void mobilenet_v1_q8(benchmark::State& state) {
  const size_t batchSize = static_cast<size_t>(state.range(0));

  std::random_device randomDevice;
  auto rng = std::mt19937(randomDevice());
  auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

  std::vector<Convolution> convolutions;
  /* Conv: 3x3 stride 2, 3 -> 32 channels */
  convolutions.emplace_back(
    ConvolutionArguments()
      .inputSize(224, 224)
      .groupInputChannels(3)
      .groupOutputChannels(32)
      .kernelSize(3, 3)
      .padding(1)
      .subsampling(2));
  /* Conv: 3x3 DW, 32 channels */
  convolutions.emplace_back(
    ConvolutionArguments()
      .inputSize(112, 112)
      .groups(32)
      .kernelSize(3, 3)
      .padding(1));
  /* Conv: 1x1, 32 -> 64 channels */
  convolutions.emplace_back(
    ConvolutionArguments()
      .inputSize(112, 112)
      .groupInputChannels(32)
      .groupOutputChannels(64));
  /* Conv: 3x3 stride 2 DW, 64 channels */
  convolutions.emplace_back(
    ConvolutionArguments()
      .inputSize(112, 112)
      .groups(64)
      .kernelSize(3, 3)
      .padding(1)
      .subsampling(2));
  /* Conv: 1x1, 64 -> 128 channels */
  convolutions.emplace_back(
    ConvolutionArguments()
      .inputSize(56, 56)
      .groupInputChannels(64)
      .groupOutputChannels(128));
  /* Conv: 3x3 DW, 128 channels */
  convolutions.emplace_back(
    ConvolutionArguments()
      .inputSize(56, 56)
      .groups(128)
      .kernelSize(3, 3)
      .padding(1));
  /* Conv: 1x1, 128 -> 128 channels */
  convolutions.emplace_back(
    ConvolutionArguments()
      .inputSize(56, 56)
      .groupInputChannels(128)
      .groupOutputChannels(128));
  /* Conv: 3x3 stride 2 DW, 128 channels */
  convolutions.emplace_back(
    ConvolutionArguments()
      .inputSize(56, 56)
      .groups(128)
      .kernelSize(3, 3)
      .padding(1)
      .subsampling(2));
  /* Conv: 1x1, 128 -> 256 channels */
  convolutions.emplace_back(
    ConvolutionArguments()
      .inputSize(28, 28)
      .groupInputChannels(128)
      .groupOutputChannels(256));
  /* Conv: 3x3 DW, 256 channels */
  convolutions.emplace_back(
    ConvolutionArguments()
      .inputSize(28, 28)
      .groups(256)
      .kernelSize(3, 3)
      .padding(1));
  /* Conv: 1x1, 256 -> 256 channels */
  convolutions.emplace_back(
    ConvolutionArguments()
      .inputSize(28, 28)
      .groupInputChannels(256)
      .groupOutputChannels(256));
  /* Conv: 3x3 stride 2 DW, 256 channels */
  convolutions.emplace_back(
    ConvolutionArguments()
      .inputSize(28, 28)
      .groups(256)
      .kernelSize(3, 3)
      .padding(1)
      .subsampling(2));
  /* Conv: 1x1, 256 -> 512 channels */
  convolutions.emplace_back(
    ConvolutionArguments()
      .inputSize(14, 14)
      .groupInputChannels(256)
      .groupOutputChannels(512));
  for (size_t i = 0; i < 5; i++) {
    /* Conv: 3x3 DW, 512 channels */
    convolutions.emplace_back(
      ConvolutionArguments()
        .inputSize(14, 14)
        .groups(512)
        .kernelSize(3, 3)
        .padding(1));
    /* Conv: 1x1, 512 -> 512 channels */
    convolutions.emplace_back(
      ConvolutionArguments()
        .inputSize(14, 14)
        .groupInputChannels(512)
        .groupOutputChannels(512));
  }
  /* Conv: 3x3 stride 2 DW, 512 channels */
  convolutions.emplace_back(
    ConvolutionArguments()
      .inputSize(14, 14)
      .groups(512)
      .kernelSize(3, 3)
      .padding(1)
      .subsampling(2));
  /* Conv: 1x1, 512 -> 1024 channels */
  convolutions.emplace_back(
    ConvolutionArguments()
      .inputSize(7, 7)
      .groupInputChannels(512)
      .groupOutputChannels(1024));
  /* Conv: 3x3 DW, 1024 channels */
  convolutions.emplace_back(
    ConvolutionArguments()
      .inputSize(7, 7)
      .groups(1024)
      .kernelSize(3, 3)
      .padding(1));
  /* Conv: 1x1, 1024 -> 1024 channels */
  convolutions.emplace_back(
    ConvolutionArguments()
      .inputSize(7, 7)
      .groupInputChannels(1024)
      .groupOutputChannels(1024));
  assert(convolutions.size() == 27);

  GlobalAveragePooling globalPooling(
    GlobalAveragePoolingArguments()
      .inputSize(7, 7)
      .channels(1024));

  FullyConnected fc(
    FullyConnectedArguments()
      .inputChannels(1024)
      .outputChannels(1000));

  size_t memorySize = 0;
  for (size_t i = 0; i < convolutions.size() - 1; i++) {
    memorySize = std::max(memorySize, convolutions[i].outputSize() + convolutions[i + 1].inputSize());
  }
  std::vector<uint8_t> workspace(memorySize);
  uint8_t* workspaceStart = workspace.data() + 8;
  uint8_t* workspaceEnd = workspace.data() + memorySize;

  uint8_t* data = workspaceStart;
  for (size_t i = 0; i < convolutions.size(); i++) {
    uint8_t* outputData;
    if (i % 2 == 1) {
      outputData = workspaceEnd - convolutions[i].outputSize();
    } else {
      outputData = workspaceStart;
    }
    convolutions[i].setup(data, outputData);
    data = outputData;
  }
  assert(data == workspaceStart);
  {
    uint8_t* outputData = data + convolutions.back().outputSize();
    globalPooling.setup(data, outputData);
    outputData = data;
  }
  {
    uint8_t* outputData = data + globalPooling.outputSize();
    assert(outputData + fc.outputSize() <= workspaceEnd);
    fc.setup(data, outputData);
  }

  double gflops = 0.0;
  for (const Convolution& convolution : convolutions) {
    gflops += 2.0e-9 * convolution.multiplyAdds();
  }

  for (auto _ : state) {
    for (Convolution& convolution : convolutions) {
      convolution.run();
    }
    globalPooling.run();
    fc.run();
  }

  state.counters["images"] =
    benchmark::Counter(batchSize * state.iterations(), benchmark::Counter::kIsRate);
  state.counters["GFLOPS"] =
    benchmark::Counter(gflops * batchSize * state.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK(mobilenet_v1_q8)->ArgNames({"N"})->Args({1})->Unit(benchmark::kMillisecond);

#ifndef QNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
