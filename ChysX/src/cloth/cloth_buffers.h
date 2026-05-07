// SPDX-License-Identifier: Apache-2.0
//
// Buffer container for the chysx cloth simulator.
//
// Two flavours of memory live here side by side, distinguished in the
// type system so we can't accidentally cudaFree somebody else's
// tensor:
//
//   * Externally-owned views (`pos`, `vel`, `inv_mass`) are
//     DeviceSpan<T> — typed (T*, count) pairs that DO NOT free their
//     underlying memory on destruction.  Use these for buffers that
//     live in the host framework (Newton / Warp) and are merely
//     passed to ChysX through `wp.array.ptr`.
//
//   * Internally-owned buffers (`tris`, `edges`, `rest_lengths`) are
//     CudaArray<T> — RAII containers ChysX itself allocates and
//     manages.  Their destructors call cudaFree.
//
// Storing Newton's pointers in a CudaArray would let our destructor
// free Newton's memory; that's exactly the bug DeviceSpan exists to
// rule out.

#pragma once

#include "../memory/cuda_array.h"      // also pulls in device_span.h
#include "../math/vec.cuh"

namespace chysx {
namespace cloth {

struct ClothBuffers {
    // ---- externally owned (Newton / Warp / user) -----------------------
    //
    // `pos.size()` doubles as the particle count.

    // Particle positions [m].
    DeviceSpan<math::Vec3f> pos;
    // Particle velocities [m/s].
    DeviceSpan<math::Vec3f> vel;
    // Per-particle inverse masses [1/kg].  Optional; the free-fall
    // path leaves it empty.
    DeviceSpan<float> inv_mass;

    int particle_count() const noexcept {
        return static_cast<int>(pos.size());
    }

    // ---- internally owned (ChysX) --------------------------------------

    // Triangle topology: vertex index triples.
    CudaArray<math::Vec3i> tris;
    // Edge topology: vertex index pairs (sorted), parallel to rest_lengths.
    CudaArray<math::Vec2i> edges;
    // Rest length [m] of each edge in `edges`.
    CudaArray<float> rest_lengths;

    // Forget all externally-owned views without touching the
    // internally-owned CudaArrays.  Useful between Newton substeps
    // when the host might reallocate its own tensors and pass new
    // addresses in.
    void clear_external() noexcept {
        pos.reset();
        vel.reset();
        inv_mass.reset();
    }
};

}  // namespace cloth
}  // namespace chysx
