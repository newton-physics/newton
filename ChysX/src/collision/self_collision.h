// SPDX-License-Identifier: Apache-2.0
//
// chysx::collision::SelfCollisionDetector
//
// Detect-and-emit self-contacts on a triangle mesh.  Mirrors cuda-cloth's
// `SelfCollisionBvhDcd` pipeline:
//
//   1. Compute per-frame face AABBs + face centers and edge AABBs.
//   2. Linear BVH refit over the faces (Karras 2012 binary-radix tree).
//   3. Self-EF broadphase: for every edge AABB, query the face BVH and
//      collect (edge_id, face_id) candidates -- with covertex(EF)
//      filtering at the leaf so 1-ring EF hits get dropped on emit.
//   4. EF -> {VF, EE}: each EF candidate spawns
//        - one VF test (vertex `vert_in_edge[edge_id]` vs the candidate face)
//        - up to three EE tests (input edge vs each of the face's 3 edges)
//      Both VF and EE results land in the same flat `Vec4i + ContactWeights`
//      stream so the downstream Hessian / SpMV kernels can treat them uniformly.
//   5. Adjacent EE pass: a precomputed list of 1-ring-adjacent edge
//      pairs (`MeshTopology::adj_ee_pairs`) gets EE-tested directly, since
//      the BVH AABB filter cannot tell those apart.
//
// Output layout (unchanged across the pipeline rewrite):
//
//   pairs[c]    = (i0, i1, i2, i3)         four particle indices
//   weights[c]  = (w0, w1, w2, w3,         barycentric / line weights
//                  nx, ny, nz,             unit contact normal
//                  thickness - d)          penetration depth (positive = penetrating)
//
// VF case:  pairs = (v, f.x, f.y, f.z),  w = (1, -wf.x, -wf.y, -wf.z)
// EE case:  pairs = (a0, a1, b0, b1),    w = (c1, 1-c1, -c2, -1+c2)
//
// where (wf.x, wf.y, wf.z) is the closest-point barycentric, c1/c2 are
// the line parameters of the closest points on the two edges, n is the
// unit vector pointing from the contact point on the second primitive
// toward the first, and `d` is the contact-point distance.

#pragma once

#include <cstdint>

#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "../memory/device_span.h"
#include "aabb.cuh"
#include "mesh_topology.h"
#include "quant_bvh.h"

namespace chysx {
namespace collision {

// Per-contact weight and contact-frame normal, packed to 8 floats per
// contact so the constraint kernel can read it as two `float4` loads.
// Layout matches cuda-cloth's `float8`.
struct alignas(16) ContactWeights {
    float w0, w1, w2, w3;     // raw[0..3] : barycentric / line weights
    float nx, ny, nz, depth;  // raw[4..6] : unit normal, raw[7] : penetration
};

class SelfCollisionDetector {
public:
    SelfCollisionDetector() = default;

    SelfCollisionDetector(const SelfCollisionDetector&) = delete;
    SelfCollisionDetector& operator=(const SelfCollisionDetector&) = delete;
    SelfCollisionDetector(SelfCollisionDetector&&) noexcept = default;
    SelfCollisionDetector& operator=(SelfCollisionDetector&&) noexcept =
        default;

    // Allocate the per-frame scratch + result buffers.  `max_contacts`
    // bounds the number of contacts we will emit across both the
    // EF->VFEE pass and the adjacent-EE pass; overflow is dropped.
    // `max_ef_candidates` bounds the (edge_id, face_id) pair list out
    // of the BVH; for typical cloth ~8x edge_count is generous.
    //
    // Idempotent: re-calling with the same sizes is a no-op.
    void reserve(int max_contacts, int max_ef_candidates);

    // Bind static-topology tables built once at setup time.  The
    // detector keeps a non-owning pointer; the caller (ClothSimulator)
    // must keep `topology` alive for as long as the detector is in
    // use.  Calling `bind_topology` on a different topology forces a
    // BVH rebuild on the next `detect` (since n_leaves changes).
    void bind_topology(const MeshTopology* topology);

    // Run a single detection pass at `positions` against the bound
    // mesh.  After the call, `pairs()`, `weights()` and `count()`
    // reflect the contacts found this pass.  The previous pass's
    // contacts are discarded.
    //
    // `thickness` is the contact distance; pairs with d < thickness
    // are emitted with penetration depth = thickness - d.
    void detect(DeviceSpan<math::Vec3f> positions,
                float thickness,
                std::uintptr_t cuda_stream = 0);

    // ---- result accessors --------------------------------------------

    int max_contacts() const noexcept { return max_contacts_; }

    // Number of contacts emitted by the last `detect()` call.  Reads
    // `count_` device-side -> host (synchronous).
    int count(std::uintptr_t cuda_stream = 0);

    CudaArray<math::Vec4i>& pairs() noexcept { return pairs_; }
    const CudaArray<math::Vec4i>& pairs() const noexcept { return pairs_; }

    CudaArray<ContactWeights>& weights() noexcept { return weights_; }
    const CudaArray<ContactWeights>& weights() const noexcept { return weights_; }

    // Device pointer to the int32 contact counter; the constraint
    // kernel reads this to bound its loop.
    int*       count_device_ptr()       noexcept { return count_.gpu_data(); }
    const int* count_device_ptr() const noexcept { return count_.gpu_data(); }

private:
    int max_contacts_ = 0;
    int max_ef_candidates_ = 0;

    // Static topology (non-owning).  May be null until bind_topology.
    const MeshTopology* topology_ = nullptr;

    // Per-frame primitive AABBs / centers and the BVH built on top.
    //
    // `BvhImpl` is a one-line swap between the two broadphases we
    // ship: `LinearBvh` (Karras 2012 LBVH with KittenGpuLBVH-style
    // 64-byte nodes) and `QuantBvh` (cuda-cloth's 16-byte quantized
    // stackless BVH).  Both expose the same `build / refit /
    // query_self_ef / query_pairs_dev / query_count_dev` surface, so
    // the rest of the pipeline doesn't care which one is in use.
    using BvhImpl = QuantBvh;
    CudaArray<Aabb>        face_aabbs_;
    CudaArray<math::Vec3f> face_centers_;
    BvhImpl                bvh_;

    // Output stream (unified VF + EE 4-particle contacts).
    CudaArray<math::Vec4i>     pairs_;
    CudaArray<ContactWeights>  weights_;
    CudaArray<int>             count_;  // single-element counter
};

}  // namespace collision
}  // namespace chysx
