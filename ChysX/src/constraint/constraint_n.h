// SPDX-License-Identifier: Apache-2.0
//
// chysx::constraint::ConstraintN<N>
//
// Mid-level base for any constraint that touches exactly N particles
// (N ∈ {1, 2, 3, 4}).  Handles vertex-index storage, arity reporting,
// batch sizing, plus a default `bind_hessian_layout` that produces
// per-block slot tables against a `BlockCSR3` matrix.  Concrete
// subclasses (PinConstraint, SpringConstraint, FemTriangleConstraint,
// ...) only need to plug in the physics:
//
//   * compute_energy(positions)
//   * accumulate_gradient(positions, out_grad)
//   * accumulate_hessian(positions, A)        // uses hessian_slots_
//
// Storage
// -------
// `indices_` is a `CudaArray<IndexTuple>` where IndexTuple matches the
// arity:  N=1 → int, N=2 → Vec2i, N=3 → Vec3i, N=4 → Vec4i.  This way
// kernels can load all N indices of one instance in a single 32-bit /
// vector load, and the format lines up with how `TriangleMesh` already
// stores its triangle table.

#pragma once

#include <vector>

#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "../sparse/block_csr.h"
#include "constraint.h"

namespace chysx {
namespace constraint {

namespace detail {

template <int N>
struct IndexTuple;

template <> struct IndexTuple<1> { using type = int; };
template <> struct IndexTuple<2> { using type = math::Vec2i; };
template <> struct IndexTuple<3> { using type = math::Vec3i; };
template <> struct IndexTuple<4> { using type = math::Vec4i; };

}  // namespace detail

template <int N>
class ConstraintN : public Constraint {
    static_assert(N >= 1 && N <= 4,
                  "ConstraintN<N> currently supports N in {1, 2, 3, 4}");

public:
    // ---- compile-time shape ------------------------------------------

    static constexpr int kNumVertices       = N;
    static constexpr int kLocalDOF          = 3 * N;
    static constexpr int kLocalHessianSize  = kLocalDOF * kLocalDOF;

    // For N==1 this is `int`; for N≥2 it is `VecNi`.
    using IndexTuple = typename detail::IndexTuple<N>::type;

    ~ConstraintN() override = default;

    // ---- shape (Constraint overrides) --------------------------------

    int num_vertices_per_constraint() const noexcept override {
        return N;
    }

    int size() const noexcept override {
        // One IndexTuple per constraint instance, so the device-side
        // count is the instance count directly.
        return static_cast<int>(indices_.gpu_size());
    }

    // ---- topology buffer --------------------------------------------

    // indices_[c] holds the N vertex indices of constraint instance c.
    // Owned by ChysX (RAII via CudaArray); subclasses populate it
    // during construction (e.g. from a TriangleMesh).
    CudaArray<IndexTuple>& indices() noexcept { return indices_; }
    const CudaArray<IndexTuple>& indices() const noexcept { return indices_; }

    // Convenience: typed device pointer for use inside kernels.
    IndexTuple* indices_device() noexcept { return indices_.gpu_data(); }
    const IndexTuple* indices_device() const noexcept {
        return indices_.gpu_data();
    }

    // Allocate space for `n` constraint instances on both host and
    // device.  Subclasses typically call this after gathering topology
    // on the host, then fill `indices().cpu_data()` and
    // `indices().copy_to_device()`.
    void resize(int n) {
        indices_.resize(static_cast<std::size_t>(n));
    }

    // ---- Hessian-slot bookkeeping (default implementation) ----------
    //
    // For an instance with vertex tuple (i_0, ..., i_{N-1}), the
    // local Hessian is an N x N grid of 3x3 blocks H_{a,b}.  We
    // scatter these into the global block-CSR matrix `A` using a
    // per-block slot table:
    //
    //     slot[c * N^2 + a * N + b] = encoding of (i_a, i_b) into A
    //
    // where the encoding follows BlockCSR3::resolve_slots:
    //   * slot < 0  ->  diag[-slot - 1]
    //   * slot >= 0 ->  values[slot]
    //
    // The default implementation walks `indices_` on the host and
    // calls `A.resolve_slots(...)`.  Subclasses can override it if
    // they have a more compact layout.
    void bind_hessian_layout(const sparse::BlockCSR3& A) override {
        const int n_inst = size();
        if (n_inst == 0) {
            hessian_slots_.resize(0);
            return;
        }

        const int blocks_per_inst = N * N;
        const int total_blocks = n_inst * blocks_per_inst;

        if (indices_.cpu_data() == nullptr) {
            // Host mirror went stale (e.g. caller only updated the
            // device side).  The bind step is rare and host-only, so
            // this should not normally fire.
            indices_.copy_to_host();
        }

        std::vector<int> rows(static_cast<std::size_t>(total_blocks));
        std::vector<int> cols(static_cast<std::size_t>(total_blocks));

        const IndexTuple* tuples = indices_.cpu_data();
        for (int c = 0; c < n_inst; ++c) {
            const IndexTuple t = tuples[c];
            int verts[N];
            unpack_indices(t, verts);
            const int base = c * blocks_per_inst;
            for (int a = 0; a < N; ++a) {
                for (int b = 0; b < N; ++b) {
                    rows[base + a * N + b] = verts[a];
                    cols[base + a * N + b] = verts[b];
                }
            }
        }

        hessian_slots_.resize(static_cast<std::size_t>(total_blocks));
        A.resolve_slots(rows.data(), cols.data(),
                        hessian_slots_.cpu_data(), total_blocks);
        hessian_slots_.copy_to_device();
    }

    // Read-only access to the slot table for subclass kernels.
    const int* hessian_slots_device() const noexcept {
        return hessian_slots_.gpu_data();
    }

protected:
    ConstraintN() = default;

    // Helper used by `bind_hessian_layout`'s default implementation.
    // Specialised below for each N.
    static void unpack_indices(const IndexTuple& t, int* out);

    CudaArray<IndexTuple> indices_;
    CudaArray<int>        hessian_slots_;  // length = size() * N^2
};

// Per-N specialisations of `unpack_indices`.

template <> inline void ConstraintN<1>::unpack_indices(
    const ConstraintN<1>::IndexTuple& t, int* out) {
    out[0] = t;
}
template <> inline void ConstraintN<2>::unpack_indices(
    const ConstraintN<2>::IndexTuple& t, int* out) {
    out[0] = t.x; out[1] = t.y;
}
template <> inline void ConstraintN<3>::unpack_indices(
    const ConstraintN<3>::IndexTuple& t, int* out) {
    out[0] = t.x; out[1] = t.y; out[2] = t.z;
}
template <> inline void ConstraintN<4>::unpack_indices(
    const ConstraintN<4>::IndexTuple& t, int* out) {
    out[0] = t.x; out[1] = t.y; out[2] = t.z; out[3] = t.w;
}

// Convenience aliases used by concrete subclasses.

using Constraint1 = ConstraintN<1>;
using Constraint2 = ConstraintN<2>;
using Constraint3 = ConstraintN<3>;
using Constraint4 = ConstraintN<4>;

}  // namespace constraint
}  // namespace chysx
