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

To cross-compile for iOS, use [ios-cmake](https://github.com/leetal/ios-cmake) to generate Xcode project files.

## Acknowledgements

QNNPACK is developed by Marat Dukhan, Yiming Wu, Hao Lu, and Bert Maher. We thank Andrew Tulloch and Yangqing Jia for advice during the development of QNNPACK.

## License

QNNPACK is BSD licensed, as found in the [`LICENSE`](LICENSE) file.
