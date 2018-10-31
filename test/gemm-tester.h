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
#include <cfloat>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <cpuinfo.h>
#include <qnnpack/AlignedAllocator.h>
#include <qnnpack/params.h>
#include <qnnpack/q8gemm.h>
#include <qnnpack/q8conv.h>
#include <qnnpack/scalar-utils.h>
#include <qnnpack/requantization.h>
#include <qnnpack/sgemm.h>

#include <fp16.h>


static inline int32_t ilog2(double x) {
  return log(x) * M_LOG2E;
}

class GemmTester {
 public:
  inline GemmTester& mr(size_t mr) {
    this->mr_ = mr;
    return *this;
  }

  inline size_t mr() const {
    return this->mr_;
  }

  inline GemmTester& nr(size_t nr) {
    this->nr_ = nr;
    return *this;
  }

  inline size_t nr() const {
    return this->nr_;
  }

  inline GemmTester& np(size_t np) {
    this->np_ = np;
    return *this;
  }

  inline size_t np() const {
    return this->np_;
  }

  inline GemmTester& kr(size_t kr) {
    this->kr_ = kr;
    return *this;
  }

  inline size_t kr() const {
    return this->kr_;
  }

  inline GemmTester& m(size_t m) {
    this->m_ = m;
    return *this;
  }

  inline size_t m() const {
    return this->m_;
  }

  inline GemmTester& n(size_t n) {
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline GemmTester& k(size_t k) {
    this->k_ = k;
    return *this;
  }

  inline size_t k() const {
    return this->k_;
  }

  inline GemmTester& ks(size_t ks) {
    this->ks_ = ks;
    return *this;
  }

  inline size_t ks() const {
    return this->ks_;
  }

  inline size_t packedK() const {
    return k() % kr() == 0 ? k() : (k() / kr() + 1) * kr();
  }

  inline size_t packedN() const {
    return n() % np() == 0 ? n() : (n() / np() + 1) * np();
  }

  inline GemmTester& aStride(size_t aStride) {
    this->aStride_ = aStride;
    return *this;
  }

  inline size_t aStride() const {
    return this->aStride_ == 0 ? k() : this->aStride_;
  }

  inline GemmTester& cStride(size_t cStride) {
    this->cStride_ = cStride;
    return *this;
  }

  inline size_t cStride() const {
    return this->cStride_ == 0 ? n() : this->cStride_;
  }

  inline GemmTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline GemmTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline GemmTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void testMicroKernel(q8gemm_ukernel_function qgemm) const {
    ASSERT_LE(m(), mr());
    ASSERT_LE(n(), nr());
    ASSERT_GE(k(), kr());

    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    const uint8_t aZeroPoint = 127;
    const uint8_t bZeroPoint = 127;

    std::vector<uint8_t> a((m() - 1) * aStride() + k() + 8);
    std::vector<uint8_t> b(n() * k());
    std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> packedB(packedN() * packedK());
    std::vector<int32_t> bias(nr());
    std::vector<uint8_t> c((m() - 1) * cStride() + n());
    std::vector<int32_t> acc(m() * n());
    std::vector<uint8_t> cRef(m() * n());

    const uint8_t* aPtr = a.data() + 8;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(u8rng));
      std::generate(b.begin(), b.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(c.begin(), c.end(), 0xA5);

      std::fill(packedB.begin(), packedB.end(), bZeroPoint);
      pack_q8gemm_b(n(), k(), np(), kr(), b.data(), k(), packedB.data());

      ASSERT_NE(*std::max_element(a.cbegin(), a.cend()), *std::min_element(a.cbegin(), a.cend()));
      ASSERT_NE(*std::max_element(b.cbegin(), b.cend()), *std::min_element(b.cbegin(), b.cend()));

      /* Compute 32-bit results and output quantization arguments */
      std::fill(acc.begin(), acc.end(), 0);
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          for (size_t kBlockStart = 0; kBlockStart < k(); kBlockStart += kr()) {
            for (size_t kBlockOffset = 0; kBlockOffset < std::min(k() - kBlockStart, kr()); kBlockOffset++) {
              ASSERT_LE(n(), packedN());
              ASSERT_LT(mIndex * n() + nIndex, acc.size());
              ASSERT_LT(mIndex * k() + kBlockStart + kBlockOffset, a.size());
              ASSERT_LT(kBlockStart * np() + nIndex * kr() + kBlockOffset, packedB.size());
              acc[mIndex * n() + nIndex] +=
                  (int32_t(aPtr[mIndex * aStride() + kBlockStart + kBlockOffset]) - int32_t(aZeroPoint)) *
                  (int32_t(packedB[kBlockStart * packedN() + nIndex * kr() + kBlockOffset]) - int32_t(bZeroPoint));
            }
          }
          acc[mIndex * n() + nIndex] += bias[nIndex];
        }
      }

      const int32_t accMin = *std::min_element(acc.cbegin(), acc.cend());
      const int32_t accMax = *std::max_element(acc.cbegin(), acc.cend());
      if (m() * n() >= 3) {
        ASSERT_NE(accMax, accMin)
            << "Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }

      const double cScale = uint32_t(accMax - accMin) >= 256 ? double(uint32_t(accMax - accMin)) / 255.0 : 1.00001;
      const uint8_t cZeroPoint = uint8_t(std::max(std::min(
        lrint(127.5 - 0.5 * double(accMin + accMax) / cScale),
        long(std::numeric_limits<uint8_t>::max())), long(std::numeric_limits<uint8_t>::min())));

      const float requantizationScale = 1.0f / float(cScale);
      const union qnnp_conv_quantization_params quantizationParams =
        qnnp_compute_conv_quantization_params(
          aZeroPoint, bZeroPoint,
          requantizationScale, cZeroPoint, qmin(), qmax());
      const union qnnp_q31_requantization_params scalarRequantizationParams =
        qnnp_compute_scalar_requantization_params(
          requantizationScale, cZeroPoint, qmin(), qmax());

      qgemm(
        m(), n(), k(),
        aPtr, aStride() * sizeof(uint8_t),
        packedB.data(), bias.data(),
        c.data(), cStride() * sizeof(uint8_t),
        &quantizationParams);

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          cRef[mIndex * n() + nIndex] = qnnp_q31_requantize(acc[mIndex * n() + nIndex], scalarRequantizationParams);
        }
      }

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          ASSERT_LE(uint32_t(c[mIndex * cStride() + nIndex]), uint32_t(qmax()));
          ASSERT_GE(uint32_t(c[mIndex * cStride() + nIndex]), uint32_t(qmin()));
          ASSERT_EQ(uint32_t(c[mIndex * cStride() + nIndex]), uint32_t(cRef[mIndex * n() + nIndex]))
              << "at " << mIndex << ", " << nIndex << ": reference = " << (uint32_t) cRef[mIndex * n() + nIndex]
              << " (accumulator = " << acc[mIndex * n() + nIndex]
              << "), optimized = " << (uint32_t) c[mIndex * cStride() + nIndex] << ", Mr x Nr x Kr = " << mr() << " x "
              << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k()
              << ", requantization scale = " << requantizationScale << ", output zero point = " << int32_t(cZeroPoint);
        }
      }
    }
  }

  void testMicroKernel(q8conv_ukernel_function qconv) const {
    ASSERT_LE(m(), mr());
    ASSERT_LE(n(), nr());
    ASSERT_GE(k(), kr());

    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    const uint8_t aZeroPoint = 127;
    const uint8_t bZeroPoint = 127;

    std::vector<uint8_t> a((mr() - 1) * aStride() + k() + 8);
    std::vector<uint8_t> b(n() * ks() * k());
    std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> packedB(packedN() * ks() * packedK());
    std::vector<int32_t> bias(nr());
    std::vector<uint8_t> c((m() - 1) * cStride() + n());
    std::vector<int32_t> acc(m() * n());
    std::vector<uint8_t> cRef(m() * n());
    std::vector<const uint8_t*> im2col(mr() * ks());

    const uint8_t* aPtr = a.data() + 8;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(u8rng));
      std::generate(b.begin(), b.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(c.begin(), c.end(), 0xA5);

      std::fill(packedB.begin(), packedB.end(), bZeroPoint);
      pack_q8conv_b(n(), ks(), k(), np(), kr(), b.data(), packedB.data());

      ASSERT_NE(*std::max_element(a.cbegin(), a.cend()), *std::min_element(a.cbegin(), a.cend()));
      ASSERT_NE(*std::max_element(b.cbegin(), b.cend()), *std::min_element(b.cbegin(), b.cend()));

      for (size_t ksIndex = 0; ksIndex < ks(); ksIndex++) {
        for (size_t mIndex = 0; mIndex < mr(); mIndex++) {
          im2col[ksIndex * mr() + mIndex] = aPtr + aStride() * mIndex;
        }
      }
      std::shuffle(im2col.begin(), im2col.end(), rng);
      for (size_t ksIndex = 0; ksIndex < ks(); ksIndex++) {
        for (size_t mIndex = m(); mIndex < mr(); mIndex++) {
          im2col[ksIndex * mr() + mIndex] = im2col[ksIndex * mr() + m() - 1];
        }
      }

      /* Compute 32-bit results and output quantization arguments */
      std::fill(acc.begin(), acc.end(), 0);
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          for (size_t ksIndex = 0; ksIndex < ks(); ksIndex++) {
            for (size_t kBlockStart = 0; kBlockStart < k(); kBlockStart += kr()) {
              for (size_t kBlockOffset = 0; kBlockOffset < std::min(k() - kBlockStart, kr()); kBlockOffset++) {
                ASSERT_LT(ksIndex * mr() + mIndex, im2col.size());
                ASSERT_LT(kBlockStart + kBlockOffset, k());
                ASSERT_LT(kBlockStart + kBlockOffset, aStride());
                ASSERT_LT(kBlockStart * ks() * packedN() + (ksIndex * nr() + nIndex) * kr() + kBlockOffset, packedB.size());

                acc[mIndex * n() + nIndex] +=
                  (int32_t(im2col[ksIndex * mr() + mIndex][kBlockStart + kBlockOffset]) - int32_t(aZeroPoint)) *
                  (int32_t(packedB[kBlockStart * ks() * packedN() + (ksIndex * nr() + nIndex) * kr() + kBlockOffset]) - int32_t(bZeroPoint));
              }
            }
          }
          acc[mIndex * n() + nIndex] += bias[nIndex];
        }
      }

      const int32_t accMin = *std::min_element(acc.cbegin(), acc.cend());
      const int32_t accMax = *std::max_element(acc.cbegin(), acc.cend());
      if (m() * n() >= 3) {
        ASSERT_NE(accMax, accMin)
            << "Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n()
            << " x " << k();
      }

      const double cScale = uint32_t(accMax - accMin) >= 256 ? double(uint32_t(accMax - accMin)) / 255.0 : 1.00001;
      const uint8_t cZeroPoint = uint8_t(std::max(std::min(
        lrint(127.5 - 0.5 * double(accMin + accMax) / cScale),
        long(std::numeric_limits<uint8_t>::max())), long(std::numeric_limits<uint8_t>::min())));

      const float requantizationScale = 1.0f / float(cScale);
      const union qnnp_conv_quantization_params quantizationParams =
        qnnp_compute_conv_quantization_params(
          aZeroPoint, bZeroPoint,
          requantizationScale, cZeroPoint, qmin(), qmax());
      const union qnnp_q31_requantization_params scalarRequantizationParams =
        qnnp_compute_scalar_requantization_params(
          requantizationScale, cZeroPoint, qmin(), qmax());

      qconv(
        m(), n(), k(), ks(),
        im2col.data(), packedB.data(), bias.data(),
        c.data(), cStride() * sizeof(uint8_t),
        &quantizationParams);

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          cRef[mIndex * n() + nIndex] = qnnp_q31_requantize(acc[mIndex * n() + nIndex], scalarRequantizationParams);
        }
      }

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          ASSERT_LE(uint32_t(c[mIndex * cStride() + nIndex]), uint32_t(qmax()));
          ASSERT_GE(uint32_t(c[mIndex * cStride() + nIndex]), uint32_t(qmin()));
          ASSERT_EQ(uint32_t(c[mIndex * cStride() + nIndex]), uint32_t(cRef[mIndex * n() + nIndex]))
              << "at " << mIndex << ", " << nIndex << ": reference = " << uint32_t(cRef[mIndex * n() + nIndex])
              << " (accumulator = " << acc[mIndex * n() + nIndex]
              << "), optimized = " << uint32_t(c[mIndex * cStride() + nIndex]) << ", Mr x Nr x Kr = " << mr() << " x "
              << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k()
              << ", requantization scale = " << requantizationScale << ", output zero point = " << int32_t(cZeroPoint);
        }
      }
    }
  }

  void pack_sgemm_b(size_t n, size_t k, size_t np, size_t kr, const float* b, size_t b_stride, float* packed_b) const
  {
    const size_t k_stride = (k + (kr - 1)) & -kr;
    for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += np) {
      const size_t nr_block_size = std::min(n - nr_block_start, np);
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
        for (size_t kr_block_start = 0; kr_block_start < k; kr_block_start += kr) {
          const size_t kr_block_size = std::min(k - kr_block_start, kr);
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
            packed_b[nr_block_start * k_stride + kr_block_start * np + nr_block_offset * kr + kr_block_offset] =
              b[(nr_block_start + nr_block_offset) * b_stride + (kr_block_start + kr_block_offset)];
          }
        }
      }
    }
  }

  void pack_hgemm_b(size_t n, size_t k, size_t np, size_t kr, const uint16_t* b, size_t b_stride, uint16_t* packed_b) const
  {
    const size_t k_stride = (k + (kr - 1)) & -kr;
    for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += np) {
      const size_t nr_block_size = std::min(n - nr_block_start, np);
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
        for (size_t kr_block_start = 0; kr_block_start < k; kr_block_start += kr) {
          const size_t kr_block_size = std::min(k - kr_block_start, kr);
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
            packed_b[nr_block_start * k_stride + kr_block_start * np + nr_block_offset * kr + kr_block_offset] =
              b[(nr_block_start + nr_block_offset) * b_stride + (kr_block_start + kr_block_offset)];
          }
        }
      }
    }
  }

  void pack_q8gemm_b(size_t n, size_t k, size_t np, size_t kr, const uint8_t* b, size_t b_stride, uint8_t* packed_b) const
  {
    const size_t k_stride = (k + (kr - 1)) & -kr;
    for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += np) {
      const size_t nr_block_size = std::min(n - nr_block_start, np);
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
        for (size_t kr_block_start = 0; kr_block_start < k; kr_block_start += kr) {
          const size_t kr_block_size = std::min(k - kr_block_start, kr);
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
            packed_b[nr_block_start * k_stride + kr_block_start * np + nr_block_offset * kr + kr_block_offset] =
              b[(nr_block_start + nr_block_offset) * b_stride + (kr_block_start + kr_block_offset)];
          }
        }
      }
    }
  }

  void pack_q8conv_b(size_t n, size_t ks, size_t kc, size_t np, size_t kr, const uint8_t* b, uint8_t* packed_b) const
  {
    const size_t kc_stride = (kc + (kr - 1)) & -kr;
    for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += np) {
      const size_t nr_block_size = std::min(n - nr_block_start, np);
      for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
        const size_t kr_block_size = std::min(kc - kr_block_start, kr);
        for (size_t ki = 0; ki < ks; ki++) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
              packed_b[(nr_block_start * ks + ki * np) * kc_stride + kr_block_start * np + nr_block_offset * kr + kr_block_offset] =
                b[((nr_block_start + nr_block_offset) * ks + ki) * kc + (kr_block_start + kr_block_offset)];
            }
          }
        }
      }
    }
  }

  void pack_q8gemm_b_diagonal(
      size_t n,
      size_t k,
      uint32_t nr,
      uint32_t kr,
      uint32_t kc,
      const uint8_t* b,
      uint8_t* packed_b) const
  {
    const size_t k_stride = (k + (kr - 1)) & -kr;
    for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
      /* Pack b first in big chunk of size nr x kc,
       *  within the block of size nr x kc, pack b diagonally in size of kr.
       * kc (power of 2) must be multiples of kr (power of 2) */
      const size_t nr_block_size = std::min(n - nr_block_start, size_t(nr));
      size_t kr_chunk_start = 0;
      for (; kr_chunk_start < k / kc * kc; kr_chunk_start += kc) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
            for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
              /* When kc is power of 2, x % kc == x & (kc - 1) */
              packed_b[nr_block_start * k_stride + (kr_chunk_start + kr_block_start) * nr + nr_block_offset * kr + kr_block_offset]
                = b[(nr_block_start + nr_block_offset) * k + kr_chunk_start + ((kr_block_start + nr_block_offset * kr + kr_block_offset) & (kc - 1))];
            }
          }
        }
      }

      /* for the remaining k (< kc), pack it in the same way as pack_q8gemm_b */
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
        for (size_t kr_block_start = kr_chunk_start; kr_block_start < k; kr_block_start += kr) {
          const size_t kr_block_size = std::min(k - kr_block_start, size_t(kr));
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
            packed_b[nr_block_start * k_stride + kr_block_start * nr + nr_block_offset * kr + kr_block_offset] =
                b[(nr_block_start + nr_block_offset) * k + (kr_block_start + kr_block_offset)];
          }
        }
      }
    }
  }

  static void q8gemm_compute_row_sum(
    const uint8_t* a,
    size_t m,
    size_t k,
    size_t stride,
    const int32_t multiplier,
    int32_t* row_sum,
    q8sum_rows_ukernel_function q8sum_rows) {
    const size_t block_size = 4;
    for (size_t block_start = 0; block_start < m; block_start += block_size) {
      q8sum_rows(
          a + block_start * stride,
          std::min(block_size, m - block_start),
          k,
          stride,
          multiplier,
          row_sum + block_start);
    }
  }

  void testMicroKernel(q8gemm_xzp_ukernel_function qgemm, q8sum_rows_ukernel_function q8sum_rows) const
  {
    ASSERT_LE(m(), mr());
    ASSERT_LE(n(), nr());
    ASSERT_GE(k(), kr());

    std::random_device rng_device;
    auto rng = std::bind(std::uniform_int_distribution<uint8_t>(), std::mt19937(rng_device()));

    const uint8_t aZeroPoint = 0x80;
    const uint8_t bZeroPoint = 0x80;

    std::vector<uint8_t> a((m() - 1) * aStride() + k() + 8);
    std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> b(n() * k());
    std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> packedB(packedN() * packedK());
    std::vector<int32_t> bias(nr());
    std::vector<int32_t> as(m());
    std::vector<int32_t> asRef(m());
    std::vector<int32_t> biasBuf(nr());
    std::vector<int32_t> biasRef(nr());
    std::vector<uint8_t> c((m() - 1) * cStride() + n());
    std::vector<int32_t> cAcc(m() * n());
    std::vector<uint8_t> cRef(m() * n());

    const uint8_t* aPtr = a.data() + 8;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(rng));
      std::generate(b.begin(), b.end(), std::ref(rng));
      std::generate(bias.begin(), bias.end(), std::ref(rng));

      memcpy(biasBuf.data(), bias.data(), bias.size() * sizeof(int32_t));
      memcpy(biasRef.data(), bias.data(), bias.size() * sizeof(int32_t));

      std::fill(packedB.begin(), packedB.end(), 0);
      pack_q8gemm_b_diagonal(n(), k(), np(), kr(), 8, b.data(), packedB.data());

      std::fill(as.begin(), as.end(), 0);
      std::fill(asRef.begin(), asRef.end(), 0);
      std::fill(c.begin(), c.end(), 0xA5);
      std::fill(cAcc.begin(), cAcc.end(), 0);

      ASSERT_NE(*std::max_element(a.cbegin(), a.cend()), *std::min_element(a.cbegin(), a.cend()));
      ASSERT_NE(*std::max_element(b.cbegin(), b.cend()), *std::min_element(b.cbegin(), b.cend()));

      /* compute row sums of a and b */
      q8gemm_compute_row_sum(aPtr, m(), k(), aStride(), -bZeroPoint, as.data(), q8sum_rows);
      q8gemm_compute_row_sum(b.data(), n(), k(), k(), -aZeroPoint, biasBuf.data(), q8sum_rows);
      const int32_t zeropoint_product = k() * aZeroPoint * bZeroPoint;
      for (size_t nIndex = 0; nIndex < n(); nIndex++) {
        biasBuf[nIndex] += bias[nIndex] + zeropoint_product;
      }

      /* Compute 32-bit results and output quantization arguments */
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          int32_t bAcc = 0;
          for (size_t kBlockStart = 0; kBlockStart < k(); kBlockStart += kr()) {
            for (size_t kBlockOffset = 0; kBlockOffset < std::min(k() - kBlockStart, kr()); kBlockOffset++) {
              ASSERT_LE(n(), packedN());
              ASSERT_LT(mIndex * n() + nIndex, cAcc.size());
              ASSERT_LT(mIndex * k() + kBlockStart + kBlockOffset, a.size());
              ASSERT_LT(kBlockStart * np() + nIndex * kr() + kBlockOffset, packedB.size());
              if (nIndex == 0) {
                asRef[mIndex] += (int32_t) aPtr[mIndex * aStride() + kBlockStart + kBlockOffset];
              }
              if (mIndex == 0) {
                bAcc += (int32_t) b[nIndex * k() + kBlockStart + kBlockOffset];
              }
              cAcc[mIndex * n() + nIndex] +=
                  ((int32_t) aPtr[mIndex * aStride() + kBlockStart + kBlockOffset] - (int32_t) aZeroPoint) *
                  ((int32_t) b[nIndex * k() + kBlockStart + kBlockOffset] - (int32_t) bZeroPoint);
            }
          }
          if (mIndex == 0) {
            biasRef[nIndex] += bAcc * (-aZeroPoint) + k() * aZeroPoint * bZeroPoint;
          }
          cAcc[mIndex * n() + nIndex] += bias[nIndex];
        }
        asRef[mIndex] *= -bZeroPoint;
      }

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        ASSERT_EQ(as[mIndex], asRef[mIndex])
            << "at " << mIndex << ": reference = " << asRef[mIndex] << ", optimized = " << as[mIndex]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n()
            << " x " << k();
      }

      for (size_t nIndex = 0; nIndex < n(); nIndex++) {
        ASSERT_EQ(biasBuf[nIndex], biasRef[nIndex])
            << "at " << nIndex << ": reference = " << biasRef[nIndex] << ", optimized = " << biasBuf[nIndex]
            << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n()
            << " x " << k();
      }

      const int32_t accMin = *std::min_element(cAcc.cbegin(), cAcc.cend());
      const int32_t accMax = *std::max_element(cAcc.cbegin(), cAcc.cend());
      if (m() * n() >= 3) {
        ASSERT_NE(accMax, accMin) << "Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
                                  << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }

      const double cScale = uint32_t(accMax - accMin) >= 256 ? double(uint32_t(accMax - accMin)) / 255.0 : 1.00001;
      const uint8_t cZeroPoint = uint8_t(std::max(
          std::min(lrint(127.5 - 0.5 * double(accMin + accMax) / cScale), long(std::numeric_limits<uint8_t>::max())),
          long(std::numeric_limits<uint8_t>::min())));

      const float requantizationScale = 1.0f / float(cScale);
      const union qnnp_q31_requantization_params requantizationParams =
          qnnp_compute_requantization_params(requantizationScale, cZeroPoint, qmin(), qmax());
      const union qnnp_q31_requantization_params scalarRequantizationParams =
          qnnp_compute_scalar_requantization_params(requantizationScale, cZeroPoint, qmin(), qmax());

      qgemm(
          m(),
          n(),
          k(),
          aPtr,
          aStride(),
          packedB.data(),
          biasBuf.data(),
          c.data(),
          cStride(),
          as.data(),
          &requantizationParams);

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          cRef[mIndex * n() + nIndex] = qnnp_q31_requantize(cAcc[mIndex * n() + nIndex], scalarRequantizationParams);
        }
      }

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          ASSERT_LE(uint32_t(c[mIndex * cStride() + nIndex]), uint32_t(qmax()));
          ASSERT_GE(uint32_t(c[mIndex * cStride() + nIndex]), uint32_t(qmin()));
          ASSERT_EQ(c[mIndex * cStride() + nIndex], cRef[mIndex * n() + nIndex])
              << "at " << mIndex << ", " << nIndex << ": reference = " << (uint32_t) cRef[mIndex * n() + nIndex]
              << ", optimized = " << (uint32_t) c[mIndex * cStride() + nIndex] << ", Mr x Nr x Kr = " << mr() << " x "
              << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
    }
  }

  void testMicroKernel(hgemm_ukernel_function hgemm) const
  {
    if(!cpuinfo_initialize() || !cpuinfo_has_arm_neon_fp16_arith()) {
      return;
    }

    ASSERT_LE(m(), mr());
    ASSERT_LE(n(), nr());
    ASSERT_GE(k(), kr());
    ASSERT_GE(aStride(), k());
    ASSERT_GE(cStride(), n());

    std::random_device randomDevice;
    auto rng = std::bind(fp16_ieee_from_fp32_value, std::bind(std::uniform_real_distribution<float>(), std::mt19937(randomDevice())));

    std::vector<uint16_t> a((m() - 1) * aStride() + k() + 4);
    std::vector<uint16_t> b(n() * k());
    std::vector<uint16_t, AlignedAllocator<uint16_t, 32>> packedB(packedN() * packedK());
    std::vector<uint16_t> bias(nr());
    std::vector<uint16_t> c((mr() - 1) * cStride() + nr());
    std::vector<float> cRef(m() * n());

    const uint16_t* aPtr = a.data() + 4;

    struct qnnp_fp16_clamping_params clampingParams;
    clampingParams.scale = UINT16_C(0x3C00) /* 1.0 */;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(rng));
      std::generate(b.begin(), b.end(), std::ref(rng));
      std::generate(bias.begin(), bias.end(), std::ref(rng));
      std::fill(c.begin(), c.end(), UINT16_C(0x7E00) /* NaN */);
      std::fill(cRef.begin(), cRef.end(), 0.0f);

      std::fill(packedB.begin(), packedB.end(), 0);
      pack_hgemm_b(n(), k(), np(), kr(), b.data(), k(), packedB.data());

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          for (size_t kBlockStart = 0; kBlockStart < k(); kBlockStart += kr()) {
            for (size_t kBlockOffset = 0; kBlockOffset < std::min(k() - kBlockStart, kr()); kBlockOffset++) {
              ASSERT_LE(n(), packedN());
              ASSERT_LT(mIndex * n() + nIndex, cRef.size());
              ASSERT_LT(mIndex * k() + kBlockStart + kBlockOffset, a.size());
              ASSERT_LT(kBlockStart * np() + nIndex * kr() + kBlockOffset, packedB.size());
              cRef[mIndex * n() + nIndex] +=
                fp16_ieee_to_fp32_value(aPtr[mIndex * aStride() + kBlockStart + kBlockOffset]) *
                fp16_ieee_to_fp32_value(packedB[kBlockStart * packedN() + nIndex * kr() + kBlockOffset]);
            }
          }
          cRef[mIndex * n() + nIndex] += fp16_ieee_to_fp32_value(bias[nIndex]);
        }
      }

      const float accMin = *std::min_element(cRef.cbegin(), cRef.cend());
      const float accMax = *std::max_element(cRef.cbegin(), cRef.cend());
      const float cMin = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accMin + (accMax - accMin) / 255.0f * float(qmin())));
      const float cMax = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accMax - (accMax - accMin) / 255.0f * float(255 - qmax())));
      clampingParams.max = fp16_ieee_from_fp32_value(cMax);
      clampingParams.min = fp16_ieee_from_fp32_value(cMin);

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          cRef[mIndex * n() + nIndex] = std::max(std::min(cRef[mIndex * n() + nIndex], cMax), cMin);
        }
      }

      hgemm(m(), n(), k(),
        aPtr, aStride() * sizeof(uint16_t),
        packedB.data(), bias.data(),
        c.data(), cStride() * sizeof(uint16_t),
        &clampingParams);

      /* Validate micro-kernel outputs */
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          ASSERT_NEAR(
              fp16_ieee_to_fp32_value(c[mIndex * cStride() + nIndex]),
              cRef[mIndex * n() + nIndex],
              std::abs(cRef[mIndex * n() + nIndex]) * 1.0e-2f)
              << "at " << mIndex << ", " << nIndex << ": reference = " << cRef[mIndex * n() + nIndex]
              << ", optimized = " << fp16_ieee_to_fp32_value(c[mIndex * cStride() + nIndex]) << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
      /* Check that micro-kernel did not overwrite data beyond bounds */
      for (size_t mIndex = 0; mIndex < m() - 1; mIndex++) {
        for (size_t nIndex = n(); nIndex < cStride(); nIndex++) {
          ASSERT_EQ(UINT16_C(0x7E00) /* NaN */, c[mIndex * cStride() + nIndex])
            << "at " << mIndex << ", " << nIndex
            << ": Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
      for (size_t i = (m() - 1) * cStride() + n(); i < c.size(); i++) {
        ASSERT_EQ(UINT16_C(0x7E00) /* NaN */, c[i])
          << "at i = " << i << ", Mr x Nr x Kr = " << mr() << " x " << nr()
          << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }

  void testMicroKernel(sgemm_ukernel_function sgemm) const {
    ASSERT_LE(m(), mr());
    ASSERT_LE(n(), nr());
    ASSERT_GE(k(), kr());
    ASSERT_GE(aStride(), k());
    ASSERT_GE(cStride(), n());

    std::random_device randomDevice;
    auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(randomDevice()));

    std::vector<float> a((m() - 1) * aStride() + k() + 4);
    std::vector<float> b(n() * k());
    std::vector<float, AlignedAllocator<float, 32>> packedB(packedN() * packedK());
    std::vector<float> bias(nr());
    std::vector<float> c((mr() - 1) * cStride() + nr());
    std::vector<float> cRef(m() * n());

    const float* aPtr = a.data() + 4;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(rng));
      std::generate(b.begin(), b.end(), std::ref(rng));
      std::generate(bias.begin(), bias.end(), std::ref(rng));
      std::fill(c.begin(), c.end(), nanf(""));
      std::fill(cRef.begin(), cRef.end(), 0.0f);

      std::fill(packedB.begin(), packedB.end(), 0);
      pack_sgemm_b(n(), k(), np(), kr(), b.data(), k(), packedB.data());

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          for (size_t kBlockStart = 0; kBlockStart < k(); kBlockStart += kr()) {
            for (size_t kBlockOffset = 0; kBlockOffset < std::min(k() - kBlockStart, kr()); kBlockOffset++) {
              ASSERT_LE(n(), packedN());
              ASSERT_LT(mIndex * n() + nIndex, cRef.size());
              ASSERT_LT(mIndex * k() + kBlockStart + kBlockOffset, a.size());
              ASSERT_LT(kBlockStart * np() + nIndex * kr() + kBlockOffset, packedB.size());
              cRef[mIndex * n() + nIndex] +=
                aPtr[mIndex * aStride() + kBlockStart + kBlockOffset] *
                packedB[kBlockStart * packedN() + nIndex * kr() + kBlockOffset];
            }
          }
          cRef[mIndex * n() + nIndex] += bias[nIndex];
        }
      }

      const float accMin = *std::min_element(cRef.cbegin(), cRef.cend());
      const float accMax = *std::max_element(cRef.cbegin(), cRef.cend());
      const float cMin = accMin + (accMax - accMin) / 255.0f * float(qmin());
      const float cMax = accMax - (accMax - accMin) / 255.0f * float(255 - qmax());
      struct qnnp_fp32_clamping_params clampingParams = {
        .max = cMax,
        .min = cMin,
      };

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          cRef[mIndex * n() + nIndex] = std::max(std::min(cRef[mIndex * n() + nIndex], cMax), cMin);
        }
      }

      sgemm(m(), n(), k(),
        aPtr, aStride() * sizeof(float),
        packedB.data(), bias.data(),
        c.data(), cStride() * sizeof(float),
        &clampingParams);

      /* Validate micro-kernel outputs */
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          ASSERT_NEAR(
              c[mIndex * cStride() + nIndex],
              cRef[mIndex * n() + nIndex],
              std::abs(cRef[mIndex * n() + nIndex]) * 1.0e-6f)
              << "at " << mIndex << ", " << nIndex << ": reference = " << cRef[mIndex * n() + nIndex]
              << ", optimized = " << c[mIndex * cStride() + nIndex] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
      /* Check that micro-kernel did not overwrite data beyond bounds */
      for (size_t mIndex = 0; mIndex < m() - 1; mIndex++) {
        for (size_t nIndex = n(); nIndex < cStride(); nIndex++) {
          ASSERT_TRUE(std::isnan(c[mIndex * cStride() + nIndex]))
            << "at " << mIndex << ", " << nIndex
            << ": Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
      for (size_t i = (m() - 1) * cStride() + n(); i < c.size(); i++) {
        ASSERT_TRUE(std::isnan(c[i]))
          << "at i = " << i << ", Mr x Nr x Kr = " << mr() << " x " << nr()
          << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }
    }
  }

 private:
  size_t mr_{1};
  size_t nr_{1};
  size_t np_{1};
  size_t kr_{1};
  size_t m_{1};
  size_t n_{1};
  size_t k_{1};
  size_t ks_{1};
  size_t aStride_{0};
  size_t cStride_{0};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
