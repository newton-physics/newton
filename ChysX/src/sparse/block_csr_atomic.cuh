// SPDX-License-Identifier: Apache-2.0
//
// Device-side helpers used by constraint kernels to scatter local 3x3
// Hessian blocks into a `chysx::sparse::BlockCSR3`.  The split-storage
// matrix uses a single signed integer slot per block:
//
//   slot < 0  : target is `A.diag[-slot - 1]`        (diagonal block)
//   slot >= 0 : target is `A.values[slot]`           (off-diagonal block)
//
// Callers precompute the per-block slot tables on the host (see
// `BlockCSR3::resolve_slots` and the default `ConstraintN<N>::
// bind_hessian_layout`) and pass `(diag, values, slots)` to their
// kernel.  This header provides the inner atomic-add primitive both
// `accumulate_hessian` overrides funnel through.

#pragma once

#include <cuda_runtime.h>

#include "../math/matrix.cuh"

namespace chysx {
namespace sparse {

// Atomically add the 3x3 block `blk` into either `diag[-slot - 1]` or
// `values[slot]`, chosen by the sign of `slot`.  The two destination
// pointers come from the caller's `BlockCSR3` view; passing both is
// cheaper than dereferencing a struct on the device.
//
// Mat3f stores its 9 elements as a flat row-major `data[9]`, so we
// can issue 9 plain `atomicAdd<float>` calls without juggling row/col
// indices.
__device__ __forceinline__ void scatter_hessian_block(
    int slot,
    math::Mat3f* __restrict__ diag,
    math::Mat3f* __restrict__ values,
    const math::Mat3f& blk) {
    math::Mat3f* target = (slot < 0) ? &diag[-slot - 1] : &values[slot];
    float* dst = target->data;
    const float* src = blk.data;
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        atomicAdd(&dst[k], src[k]);
    }
}

}  // namespace sparse
}  // namespace chysx
