// SPDX-License-Identifier: Apache-2.0
//
// chysx::collision::UntangleDetector
//
// 5-vertex Edge-Face (EF) tangle contact detector.  Implements the
// ICM (Intersection Contour Minimization) algorithm of Volino &
// Magnenat-Thalmann, SIGGRAPH 2006, "Resolving Surface Collisions
// through Intersection Contour Minimization".
//
// Reference implementation we follow: `newton/_src/solvers/style3d/
// collision/kernels.py::solve_untangling_kernel` (chosen over
// cuda-cloth's `kernel_cull_EF_pairs` because cuda-cloth omits the
// edge-vs-face force-direction sign flip and averages -- rather
// than accumulates+normalises -- the per-adjacent-face gradient
// contributions, which can starve the restoring force on dense
// pre-tangled meshes).
//
// Picks up the already-computed EF candidate stream from
// `SelfCollisionDetector`'s BVH (so we don't pay for a second
// broadphase) and runs a per-pair ray-triangle intersection.
// Whenever the edge actually pierces the face, we emit a 5-vertex
// contact -- two edge endpoints + three face vertices -- together
// with:
//
//   * face barycentric weights  w = (w0, w1, w2)
//   * edge-side weights         u = (u0, u1) = (|d2|, |d1|) / (|d1|+|d2|)
//                               (where d1, d2 are the signed normal-
//                               distances of the edge endpoints from
//                               the face plane)
//   * a 3-vector "G" direction (Volino's intersection-gradient
//     vector, accumulated from the edge's two adjacent-face normals
//     then renormalised to a unit vector)
//   * a depth = `thickness` (the apply kernel multiplies through
//     `disp = 2 * thickness` per Volino's displacement-target
//     scheme; the data field stores the user-set thickness)
//
// Why a separate detector from `SelfCollisionDetector`?
//
//   * The geometric test is different: VF/EE proximity uses
//     closest-point + distance < thickness; tangle detection uses
//     ray-triangle intersection.  An edge that has already crossed
//     all the way through a face has *zero* shortest-distance
//     between any pair of (edge endpoint, face) points but is the
//     entire population we want to repair.
//   * The output stream has a different arity (5 vs 4) and a
//     different normal/weight layout that doesn't fit
//     `ContactWeights`.  Trying to unify them would force every
//     downstream kernel to branch on contact arity per-thread,
//     which is the opposite of what we want.
//   * The Hessian contribution is diagonal-only by design (no
//     cross-particle off-diagonal blocks), so untangle never needs
//     a `ContactSpMVOp` sidecar — it stays out of the PCG SpMV.
//
// The detector reuses the BVH that `SelfCollisionDetector` already
// built for the proximity broadphase, via `ef_candidates_pairs/
// count/max` accessors on `SelfCollisionDetector`.  In the cloth
// simulator step() ordering, untangle therefore runs strictly after
// self-collision detect, in the same step.

#pragma once

#include <cstdint>

#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "../memory/device_span.h"
#include "mesh_topology.h"

namespace chysx {
namespace collision {

class UntangleDetector {
public:
    UntangleDetector() = default;

    UntangleDetector(const UntangleDetector&) = delete;
    UntangleDetector& operator=(const UntangleDetector&) = delete;
    UntangleDetector(UntangleDetector&&) noexcept = default;
    UntangleDetector& operator=(UntangleDetector&&) noexcept = default;

    // Allocate result buffers for up to `max_contacts` 5-vertex
    // tangles per detection pass.  Idempotent.  Bumping the cap
    // grows the buffers; shrinking it does NOT release memory but
    // does cap subsequent kernel writes (ignored beyond cap).
    void reserve(int max_contacts);

    // Bind the static topology used by the cull kernel for
    // edge / face / edge2face / opposite-vertex lookups.  Lifetime
    // of `topology` must outlive any later call to `detect()`.
    // No per-face cache is allocated -- adjacent-face normals are
    // recomputed inside the cull kernel from the edge's two
    // opposite vertices each pass.
    void bind_topology(const MeshTopology* topology);

    // Single detection pass.  `ef_pairs_dev` / `ef_count_dev` /
    // `ef_max` come from `SelfCollisionDetector::ef_candidates_*`,
    // which expose the BVH's broadphase output without re-running
    // it.  `positions` are the cloth particle positions to test
    // against (typically `x_n` from the cloth simulator).
    //
    // After the call, `pairs() / weights() / normals() / depths()`
    // contain the tangle contacts and `count()` / `count_device_ptr()`
    // hold their number.
    void detect(DeviceSpan<math::Vec3f> positions,
                const math::Vec2i*      ef_pairs_dev,
                const int*              ef_count_dev,
                int                     ef_max,
                float                   thickness,
                std::uintptr_t          cuda_stream = 0);

    // ---- result accessors --------------------------------------------

    int max_contacts() const noexcept { return max_contacts_; }

    // Synchronous host-side count read.  Avoid in hot loops.
    int count(std::uintptr_t cuda_stream = 0);

    // 5 ints per contact: e0, e1, f0, f1, f2.
    const CudaArray<int>& pairs() const noexcept { return pairs_; }
    CudaArray<int>&       pairs() noexcept       { return pairs_; }

    // 5 floats per contact: u0, u1, w0, w1, w2.
    const CudaArray<float>& weights() const noexcept { return weights_; }
    CudaArray<float>&       weights() noexcept       { return weights_; }

    // One Vec3f per contact: the unit "G" untangle direction.
    const CudaArray<math::Vec3f>& normals() const noexcept { return normals_; }
    CudaArray<math::Vec3f>&       normals() noexcept       { return normals_; }

    // One float per contact: depth (= thickness today, but the
    // detector keeps it per-contact so future variants can use a
    // tangle-depth proxy without changing the constraint API).
    const CudaArray<float>& depths() const noexcept { return depths_; }
    CudaArray<float>&       depths() noexcept       { return depths_; }

    int*       count_device_ptr()       noexcept { return count_.gpu_data(); }
    const int* count_device_ptr() const noexcept { return count_.gpu_data(); }

private:
    int max_contacts_ = 0;
    const MeshTopology* topology_ = nullptr;

    // Output stream.  Five ids + five weights per contact, plus a
    // unit "G" direction and the user-set thickness in the depth
    // slot.  Layout (per contact c):
    //
    //     pairs_  [5*c + 0..4] = (e0, e1, f0, f1, f2)
    //     weights_[5*c + 0..4] = (edge_w0, edge_w1, face_w0, face_w1, face_w2)
    //     normals_[c]          = unit Volino gradient direction G
    //     depths_ [c]          = thickness  (apply kernel uses 2*thickness)
    //
    // All five weights are NON-negative; the opposite-direction
    // forces on edge vs face vertices are applied by the constraint
    // kernel via a hard-coded sign flip on indices [2..4].
    CudaArray<int>         pairs_;
    CudaArray<float>       weights_;
    CudaArray<math::Vec3f> normals_;
    CudaArray<float>       depths_;
    CudaArray<int>         count_;
};

}  // namespace collision
}  // namespace chysx
