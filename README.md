# QNNPACK
QNNPACK (Quantized Neural Networks PACKage) is a mobile-optimized library for low-precision high-performance neural network inference. QNNPACK provides implementation of convolutional, deconvolutional, and fully connected neural network operators on quantized 8-bit tensors.

QNNPACK is not intended to be directly used by machine learning researchers; instead it provides low-level performance primitives for high-level deep learning frameworks. As of today, QNNPACK is integrated in [PyTorch 1.0](https://github.com/pytorch/pytorch) with Caffe2 graph representation.

## Building

QNNPACK provides standard CMake-based build scripts.

### Native compilation

Users are recommended to use `scripts/build-local.sh` script to build QNNPACK for the host machine. 

### Cross-compilation for Android

To cross-compile for Android, set `$ANDROID_NDK` environment variable (where `$ANDROID_NDK` is the path to Android NDK directorory, e.g. `/opt/android-ndk-r15c`) and use one of the scripts from the table below:

| ABI         | Build script                     | Restrictions               |
| ----------- | ---------------------------------| -------------------------- |
| armeabi-v7a | `scripts/build-android-armv7.sh` | Requires CPU with ARM NEON |
| arm64-v8a   | `scripts/build-android-arm64.sh` |                            |
| x86         | `scripts/build-android-x86.sh`   |                            |

Notes:
- On **armeabi-v7a** `qnnp_initialize` will fail with `qnnp_status_unsupported_hardware` if the mobile CPU does not support ARM NEON. Don't set `-DANDROID_ARM_NEON=1` for QNNPACK compilation as it can make `qnnp_initialize` crash on CPUs without ARM NEON.

### Cross-compilation for iOS

To cross-compile for iOS, clone [ios-cmake](https://github.com/leetal/ios-cmake), and set `$IOS_CMAKE_TOOLCHAIN_FILE` environment variable (where `$IOS_CMAKE_TOOLCHAIN_FILE` is the path to `ios.toolchain.cmake` file in [ios-cmake](https://github.com/leetal/ios-cmake)), and use one of the scripts from the table below:

| Architecture | Build script                  | Notes                     |
| ------------ | ----------------------------- | ------------------------- |
| armv7        | `scripts/build-ios-armv7.sh`  | iPhone 3GS/4/4S           |
| armv7        | `scripts/build-ios-armv7s.sh` | iPhone 5 and newer        |
| arm64        | `scripts/build-ios-arm64.sh`  | iPhone 5S and newer       |
| arm64e       | `scripts/build-ios-arm64e.sh` | iPhone XS/XR              |
| i386         | `scripts/build-ios-i386.sh`   | iPhone Simulator (32-bit) |
| x86_64       | `scripts/build-ios-x86_64.sh` | iPhone Simulator (64-bit) |

## End-to-End Benchmarking

Caffe2 backend of PyTorch 1.0 natively integrates QNNPACK, and provides a [pre-trained quantized MobileNet v2 model](https://github.com/caffe2/models/tree/master/mobilenet_v2_quantized). Below are instructions for benchmarking this model end-to-end with QNNPACK.

### ARMv7 (32-bit) Android

```bash
# Clone PyTorch 1.0 repo
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch

# Optional: update QNNPACK submodule to latest revision
git submodule update --remote third_party/QNNPACK

# Build Caffe2 (including binaries) for Android
scripts/build_android.sh -DANDROID_TOOLCHAIN=clang -DBUILD_BINARY=ON

# Download model weights and copy them to Android device
wget https://s3.amazonaws.com/download.caffe2.ai/models/mobilenet_v2_1.0_224_quant/init_net.pb
adb push init_net.pb /data/local/tmp/init_net.pb

# Download model graph and copy it to Android device
wget https://s3.amazonaws.com/download.caffe2.ai/models/mobilenet_v2_1.0_224_quant/predict_net.pb
adb push predict_net.pb /data/local/tmp/predict_net.pb

# Run speed benchmark with 50 warm-up iterations and 10 measurement iterations
adb shell /data/local/tmp/speed_benchmark \
	--net /data/local/tmp/predict_net.pb \
	--init_net /data/local/tmp/init_net.pb \
	--input data --input_dims 1,3,224,224 --input_type float \
	--warmup 50 --iter 10
```

### ARM64 (64-bit) Android

```bash
# Clone PyTorch 1.0 repo
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch

# Optional: update QNNPACK submodule to latest revision
git submodule update --remote third_party/QNNPACK

# Build Caffe2 (including binaries) for Android
scripts/build_android.sh -DANDROID_ABI=arm64-v8a -DANDROID_TOOLCHAIN=clang -DBUILD_BINARY=ON

# Download model weights and copy them to Android device
wget https://s3.amazonaws.com/download.caffe2.ai/models/mobilenet_v2_1.0_224_quant/init_net.pb
adb push init_net.pb /data/local/tmp/init_net.pb

# Download model graph and copy it to Android device
wget https://s3.amazonaws.com/download.caffe2.ai/models/mobilenet_v2_1.0_224_quant/predict_net.pb
adb push predict_net.pb /data/local/tmp/predict_net.pb

# Run speed benchmark with 50 warm-up iterations and 10 measurement iterations
adb shell /data/local/tmp/speed_benchmark \
	--net /data/local/tmp/predict_net.pb \
	--init_net /data/local/tmp/init_net.pb \
	--input data --input_dims 1,3,224,224 --input_type float \
	--warmup 50 --iter 10
```

## Acknowledgements

QNNPACK is developed by Marat Dukhan, Yiming Wu, Hao Lu, and Bert Maher. We thank Andrew Tulloch and Yangqing Jia for advice during the development of QNNPACK.

## License

QNNPACK is BSD licensed, as found in the [`LICENSE`](LICENSE) file.
