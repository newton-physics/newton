// SPDX-License-Identifier: Apache-2.0
//
// chysx::cloth::ClothSimulator
//
// Glue class that pairs a ClothMaterial (parameters) with a
// ClothBuffers (memory) and exposes a single `step()` entry point.
//
// Lifecycle:
//
//   1. Construct.
//   2. Push parameters in via `set_material()`.  ChysX takes a copy;
//      the caller can throw the source ClothMaterial away.
//   3. Push raw device pointers in via `set_external_buffers()` each
//      step (or once if they don't change between steps).  ChysX does
//      not take ownership.
//   4. Call `step(dt)` to integrate; ChysX launches its own kernels
//      against the externally-owned pointers.
//
// The header deliberately does not pull in <cuda_runtime.h>; all CUDA
// runtime calls are confined to cloth_simulator.cu.

#pragma once

#include <cstdint>

#include "cloth_buffers.h"
#include "cloth_material.h"

namespace chysx {
namespace cloth {

class ClothSimulator {
public:
    ClothSimulator() = default;

    // Move-only.  Two simulators with the same external pointers would
    // step the same particles twice, which is almost never desired.
    ClothSimulator(const ClothSimulator&) = delete;
    ClothSimulator& operator=(const ClothSimulator&) = delete;
    ClothSimulator(ClothSimulator&&) noexcept = default;
    ClothSimulator& operator=(ClothSimulator&&) noexcept = default;

    // ---- material -----------------------------------------------------

    // Copy `m` into this simulator.  After the call, mutating `m` no
    // longer affects the simulator state.
    void set_material(const ClothMaterial& m) noexcept { material_ = m; }

    ClothMaterial& material() noexcept { return material_; }
    const ClothMaterial& material() const noexcept { return material_; }

    // ---- buffers ------------------------------------------------------

    // Stash externally-owned CUDA device pointers.  The simulator does
    // not copy or free them; they must remain valid until step()
    // returns (or until they are replaced by a later call here).
    //
    // `pos_ptr` and `vel_ptr` are required (cast cudaMalloc'd
    // addresses to uintptr_t).  `inv_mass_ptr` is optional and
    // currently unused by the free-fall path.
    void set_external_buffers(std::uintptr_t pos_ptr,
                              std::uintptr_t vel_ptr,
                              int particle_count,
                              std::uintptr_t inv_mass_ptr = 0) noexcept;

    ClothBuffers& buffers() noexcept { return buffers_; }
    const ClothBuffers& buffers() const noexcept { return buffers_; }

    // ---- stepping -----------------------------------------------------

    // Advance the simulation by `dt` seconds using the currently set
    // material and external buffers.  For now this is a single
    // semi-implicit Euler free-fall update with optional velocity
    // damping; later we'll add elastic + bending contributions behind
    // the same entry point so callers don't need to re-wire anything.
    //
    // `cuda_stream`: cudaStream_t handle cast to uintptr_t. 0 = the
    // default stream (the call returns synchronously in that case).
    void step(float dt, std::uintptr_t cuda_stream = 0);

private:
    ClothMaterial material_;
    ClothBuffers buffers_;
};

}  // namespace cloth
}  // namespace chysx
