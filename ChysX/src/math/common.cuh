// SPDX-License-Identifier: Apache-2.0
//
// Shared host/device macros and small numeric helpers used by the rest
// of the chysx::math library.
//
// The CHYSX_HD* macros let the same templated code compile both on the
// CPU (e.g. unit tests, debug tools) and inside __global__/__device__
// kernels.  When the file is consumed by a non-CUDA TU the qualifiers
// just disappear.

#pragma once

#include <cmath>

#if defined(__CUDACC__)
    #define CHYSX_HD  __host__ __device__
    #define CHYSX_HDI __host__ __device__ inline
    #define CHYSX_DI  __device__ __forceinline__
#else
    #define CHYSX_HD
    #define CHYSX_HDI inline
    #define CHYSX_DI  inline
#endif

namespace chysx {
namespace math {

// Small scalar helpers.  We deliberately *do not* import std::min /
// std::max / std::abs into this namespace because the names are reused
// below by Vec/Mat overloads — keeping the scalar versions here lets
// callers write `using namespace chysx::math;` without ambiguity.

template <typename T>
CHYSX_HDI T abs(T a) {
    return a < T{0} ? -a : a;
}

template <typename T>
CHYSX_HDI T min(T a, T b) {
    return a < b ? a : b;
}

template <typename T>
CHYSX_HDI T max(T a, T b) {
    return a < b ? b : a;
}

template <typename T>
CHYSX_HDI T clamp(T x, T lo, T hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

template <typename T>
CHYSX_HDI void swap(T& a, T& b) {
    T t = a;
    a = b;
    b = t;
}

template <typename T>
CHYSX_HDI int sign(T a) {
    return a < T{0} ? -1 : (a > T{0} ? 1 : 0);
}

}  // namespace math
}  // namespace chysx
