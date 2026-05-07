// SPDX-License-Identifier: Apache-2.0
//
// chysx::constraint::Constraint — abstract base for elastic /
// kinematic constraints contributing to a global Newton-style implicit
// solve.
//
// A "constraint" here is a *batch* of homogeneous constraint
// instances of one fixed type (e.g. a batch of triangle FEM elements,
// or a batch of pin constraints).  Each instance touches a small
// fixed number N ∈ {1, 2, 3, 4} of particles.  Concrete subclasses
// derive from ConstraintN<N> (defined in constraint_n.h) and plug in
// the N-specific physics:
//
//   * elastic energy        E(x)        : ℝ^{3 n_particles} → ℝ
//   * gradient              g = ∂E/∂x   : ℝ^{3 n_particles} → ℝ^{3 n_particles}
//   * Hessian               H = ∂²E/∂x² : symmetric block-sparse
//
// Conventions
// -----------
// * Particles live in 3D.  Vertex `i` owns DOF range [3i, 3i+3).
// * `size()` is the number of constraint *instances* in the batch
//   (e.g. number of triangles for a triangle FEM batch).
// * `num_vertices_per_constraint()` is the constraint type's arity
//   (a small fixed integer 1..4), not the batch size.
// * `accumulate_gradient` adds (atomically) into a global per-particle
//   gradient buffer; multiple constraints sharing a particle compose
//   without races.
// * Hessian scatter is a two-stage protocol against a
//   `chysx::sparse::BlockCSR3` whose topology is built once for the
//   simulation and reused frame-to-frame:
//     - `bind_hessian_layout(A)` (one-time) walks this batch's
//       vertex tuples on the host, looks up the per-block slot in
//       `A.diag` / `A.values`, and stores the result in a device-
//       resident slot table.
//     - `accumulate_hessian(positions, A)` (each step) launches one
//       kernel that reads positions, recomputes the local 3x3
//       Hessian blocks, and atomicAdd's each block into the slot
//       computed above.  No host triplets, no per-step CSR build.

#pragma once

#include <cstdint>

#include "../math/matrix.cuh"
#include "../math/vec.cuh"
#include "../memory/cuda_array.h"  // also brings in DeviceSpan
#include "../sparse/block_csr.h"

namespace chysx {
namespace constraint {

class Constraint {
public:
    virtual ~Constraint() = default;

    Constraint(const Constraint&) = delete;
    Constraint& operator=(const Constraint&) = delete;

    // ---- shape (pure-virtual; ConstraintN<N> implements them) -------

    // Number of particles each constraint instance touches.  A
    // compile-time constant on subclasses (1, 2, 3, or 4).
    virtual int num_vertices_per_constraint() const noexcept = 0;

    // Number of constraint instances currently held in this batch.
    virtual int size() const noexcept = 0;

    // ---- shape helpers (non-virtual, derived from the two above) ----

    // Local DOF count per instance.
    int local_dof() const noexcept {
        return 3 * num_vertices_per_constraint();
    }

    // Total atomic-add operations accumulate_gradient() will perform.
    // = size() * num_vertices_per_constraint()
    int total_gradient_entries() const noexcept {
        return size() * num_vertices_per_constraint();
    }

    // Total 3x3 block triplets accumulate_hessian() will emit.
    // Each instance contributes an N x N grid of 3x3 blocks, so:
    // = size() * num_vertices_per_constraint()^2
    int total_hessian_blocks() const noexcept {
        const int n = num_vertices_per_constraint();
        return size() * n * n;
    }

    // ---- physics interface (pure-virtual, subclass plugs in kernels)

    // Sum of elastic energies over every instance in this batch [J].
    //
    // `positions` is the global per-particle position buffer (Vec3f
    // per particle); the kernel reads it through this batch's
    // `indices()`.  The call may launch a CUDA reduction kernel and
    // synchronously copy the scalar back; pass a non-default stream
    // if you need it asynchronous.
    virtual float compute_energy(
        DeviceSpan<math::Vec3f> positions,
        std::uintptr_t cuda_stream = 0) const = 0;

    // Atomically accumulate ∂E/∂x_i into `out_grad[i]` (a Vec3f per
    // particle).  `out_grad` has one entry per *global* particle, not
    // per-constraint.  Caller is responsible for zeroing it before
    // the first batch contributes.
    virtual void accumulate_gradient(
        DeviceSpan<math::Vec3f> positions,
        DeviceSpan<math::Vec3f> out_grad,
        std::uintptr_t cuda_stream = 0) const = 0;

    // ---- Hessian scatter -------------------------------------------
    //
    // The matrix `A` stores a *split* representation: per-particle
    // diagonal blocks in `A.diag`, off-diagonal blocks in CSR form.
    // Scattering Hessian blocks is therefore a two-stage protocol:
    //
    //   1. (one-time) `bind_hessian_layout(A)` precomputes a per-block
    //      slot index based on this constraint's vertex tuples and
    //      A's topology.  Slot encoding:
    //          slot < 0  -> diagonal block, target is `diag[-slot - 1]`
    //          slot >= 0 -> off-diag,       target is `values[slot]`
    //      The slot table lives on the constraint as a CudaArray<int>
    //      of length `total_hessian_blocks()`.
    //
    //   2. (each step) `accumulate_hessian(positions, A, stream)`
    //      launches one kernel per constraint instance which reads
    //      its slot table and atomicAdd's each of the N^2 3x3 blocks
    //      into A.diag / A.values.  Caller must clear A before the
    //      first contribution of the step (e.g. `A.set_zero()`).

    // One-time setup against `A`'s topology.  Must be called whenever
    // either `A`'s topology changes or this constraint's vertex
    // tuples change.  The constraint takes a host-side snapshot of
    // `A.row_offsets` / `A.col_indices` (already kept inside `A` for
    // free) to produce per-block slots.
    virtual void bind_hessian_layout(const sparse::BlockCSR3& A) = 0;

    // Scatter per-instance local Hessian blocks into `A` using the
    // slot table set up by `bind_hessian_layout`.  No host work, no
    // device-host sync inside.
    virtual void accumulate_hessian(
        DeviceSpan<math::Vec3f> positions,
        sparse::BlockCSR3& A,
        std::uintptr_t cuda_stream = 0) const = 0;

protected:
    // Only subclasses can construct.  This keeps the abstract base
    // out of accidental `Constraint c;` instantiations.
    Constraint() = default;
};

}  // namespace constraint
}  // namespace chysx
