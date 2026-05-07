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

#include "../constraint/pin_constraint.h"
#include "../constraint/spring_constraint.h"
#include "../constraint/triangle_stretch_constraint.h"
#include "../math/matrix.cuh"
#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "../solver/pcg_solver.h"
#include "../sparse/block_csr.h"
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

    // ---- pinned particles --------------------------------------------
    //
    // While the linear-system solve isn't wired into the step yet, the
    // freefall integrator hard-clamps every pinned particle's position
    // back to its target and zeroes its velocity at the end of the
    // step.  That gives the same visual result as a penalty pin with
    // very large stiffness (which is what `PinConstraint` will
    // contribute once we run PCG inside step()).

    // Install `n` pins.  `host_indices[c]` is the global particle
    // index of pin c; `host_targets[c]` is its target world-space
    // position.  Replaces any previously installed pins.  `stiffness`
    // is stored on the constraint for future PCG-based solves; the
    // current freefall step ignores it (hard clamp).
    void set_pins(const int* host_indices,
                  const math::Vec3f* host_targets,
                  int n,
                  float stiffness = 1.0e6f);

    // Drop all pins.
    void clear_pins() noexcept;

    constraint::PinConstraint& pins() noexcept { return pins_; }
    const constraint::PinConstraint& pins() const noexcept { return pins_; }

    // ---- mesh topology + springs --------------------------------------
    //
    // `set_mesh` uploads a triangle index list (Vec3i triples) into
    // ChysX-owned memory and extracts the unique edge list on the host
    // (host-side dedup; cheap for the cloth sizes we target).  Call
    // `build_springs_from_current_positions` afterwards to install one
    // SpringConstraint instance per unique edge with rest length =
    // current edge length read from the externally-bound positions.
    void set_mesh(const math::Vec3i* host_triangles, int n_triangles);

    // Install one spring per mesh edge.  Rest length is taken from the
    // current `buffers_.pos` configuration (which must be set first
    // via `set_external_buffers`).  Replaces any previously installed
    // springs.
    void build_springs_from_current_positions(
        float stiffness,
        std::uintptr_t cuda_stream = 0);

    constraint::SpringConstraint& springs() noexcept { return springs_; }
    const constraint::SpringConstraint& springs() const noexcept {
        return springs_;
    }

    // ---- FEM triangle stretch (Baraff-Witkin) -------------------------
    //
    // Install one TriangleStretchConstraint instance per face in the
    // current `buffers_.tris` table.  The reference shape (Dm_inv,
    // rest area) is computed from the *current* externally-bound
    // positions, so call this once after `set_external_buffers` /
    // `set_mesh` and before stepping.
    void build_fem_stretch_from_current_positions(
        float stiffness,
        std::uintptr_t cuda_stream = 0);

    constraint::TriangleStretchConstraint& fem_stretch() noexcept {
        return fem_stretch_;
    }
    const constraint::TriangleStretchConstraint& fem_stretch() const noexcept {
        return fem_stretch_;
    }

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

    // ---- solver tuning ------------------------------------------------

    void set_pcg_iterations(int max_iter) noexcept {
        pcg_max_iterations_ = (max_iter > 0) ? max_iter : 1;
    }
    int pcg_iterations() const noexcept { return pcg_max_iterations_; }

    // Toggle the PCG solver's CUDA-Graph capture path.  Default ON;
    // disable if you want to run kernel-by-kernel for debugging or
    // for nsys profiles where you'd rather see every individual
    // kernel launch instead of one fused `cudaGraphLaunch`.
    void set_graph_enabled(bool enabled) { pcg_.set_graph_enabled(enabled); }
    bool graph_enabled() const noexcept { return pcg_.graph_enabled(); }

private:
    // Lazy-resize all per-particle work buffers to length `n`.  No-op
    // if every buffer already matches.
    void resize_work_buffers(int n);

    // Rebuild `H_`'s topology from the currently installed
    // constraints (off-diagonal pairs only — pins are diagonal-only
    // and add nothing structurally) and bind every constraint's
    // Hessian-slot LUT against it.  Called lazily by `step()` when
    // `topology_dirty_` is set; flagged by anything that changes the
    // mesh, springs, or FEM stretch instance set.
    void ensure_hessian_topology();

    ClothMaterial material_;
    ClothBuffers buffers_;
    constraint::PinConstraint pins_;
    constraint::SpringConstraint springs_;
    constraint::TriangleStretchConstraint fem_stretch_;

    // ---- implicit-Euler PCG step working state ----------------------
    //
    // Per-particle workspace (length = particle_count):
    //
    //   x_n      : positions at the start of the step (snapshot)
    //   x_tilde  : inertial predicted positions (x_n + dt v + dt^2 g)
    //   dx       : PCG solution (displacement correction)
    //   rhs      : right-hand side b = -grad E(x_tilde)
    //   mass     : per-particle scalar mass (lazy from inv_mass)
    CudaArray<math::Vec3f> x_n_;
    CudaArray<math::Vec3f> x_tilde_;
    CudaArray<math::Vec3f> dx_;
    CudaArray<math::Vec3f> rhs_;
    CudaArray<float>       mass_;

    sparse::BlockCSR3 H_;
    int H_num_block_rows_ = 0;       // last N H_ was built for
    bool topology_dirty_ = true;     // mesh / springs / fem changed?
    solver::PCGSolver pcg_;
    int pcg_max_iterations_ = 50;
};

}  // namespace cloth
}  // namespace chysx
