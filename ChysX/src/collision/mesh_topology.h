// SPDX-License-Identifier: Apache-2.0
//
// chysx::collision::MeshTopology
//
// Static (rest-pose) connectivity tables that the self-collision
// pipeline needs.  All of these can be computed from the triangle
// list alone, so we do it once on the host at setup time and ship the
// results to the GPU.  Refit / query / narrow-phase only ever read
// these arrays.
//
// Output tables (mirroring cuda-cloth's `ClothBaseBuffer`):
//
//   edges          [n_edges]   (v0, v1)            unordered, deduplicated
//   edge2face      [n_edges]   (f0, f1)            -1 for boundary side
//   vert_in_edge   [n_edges]   v in [-1, n_verts)  one designated vertex
//                                                  per global vertex (pick
//                                                  the lowest-indexed edge
//                                                  containing v); -1 elsewhere
//   edge_in_face   [n_faces]   (e0, e1, e2)        edge ids of the 3 face edges,
//                                                  ordered (v0,v1) (v1,v2) (v2,v0)
//   adj_ee_pairs   [n_adj_ee]  (a0, a1, b0, b1)    1-ring-adjacent edge pairs
//                                                  (sharing exactly one vertex)
//                                                  pre-flattened to 4 vertex
//                                                  ids so the narrow-phase can
//                                                  reuse the unified VF/EE
//                                                  4-particle Hessian path.
//
// `vert_in_edge` is the trick cuda-cloth uses to avoid duplicate VF
// emissions during the EF broadphase: each global vertex is "owned"
// by exactly one edge, so the VF test fires exactly once per
// (vertex, candidate face).

#pragma once

#include <cstdint>
#include <vector>

#include "../math/vec.cuh"
#include "../memory/cuda_array.h"

namespace chysx {
namespace collision {

class MeshTopology {
public:
    MeshTopology() = default;

    MeshTopology(const MeshTopology&) = delete;
    MeshTopology& operator=(const MeshTopology&) = delete;
    MeshTopology(MeshTopology&&) noexcept = default;
    MeshTopology& operator=(MeshTopology&&) noexcept = default;

    // Build everything from `tris` (host-side triangle list, length n_tris)
    // and the total particle count `n_verts`.  The host-side connectivity
    // is computed in O(n_tris log n_tris) time and the device-side arrays
    // are populated via a synchronous H->D copy.
    void build(const std::vector<math::Vec3i>& tris,
               int n_verts);

    // Are we set up for a non-trivial mesh?
    bool valid() const noexcept { return n_faces_ > 0; }

    // ---- sizes --------------------------------------------------------

    int n_verts() const noexcept { return n_verts_; }
    int n_faces() const noexcept { return n_faces_; }
    int n_edges() const noexcept { return n_edges_; }
    int n_adj_ee() const noexcept { return n_adj_ee_; }

    // ---- device-resident tables --------------------------------------

    const CudaArray<math::Vec3i>& faces() const noexcept       { return faces_; }
    const CudaArray<math::Vec2i>& edges() const noexcept       { return edges_; }
    const CudaArray<math::Vec2i>& edge2face() const noexcept   { return edge2face_; }
    const CudaArray<int>&         vert_in_edge() const noexcept{ return vert_in_edge_; }
    const CudaArray<math::Vec3i>& edge_in_face() const noexcept{ return edge_in_face_; }
    const CudaArray<math::Vec4i>& adj_ee_pairs() const noexcept{ return adj_ee_pairs_; }

private:
    int n_verts_ = 0;
    int n_faces_ = 0;
    int n_edges_ = 0;
    int n_adj_ee_ = 0;

    CudaArray<math::Vec3i> faces_;          // [n_faces] triangle vertex ids
    CudaArray<math::Vec2i> edges_;
    CudaArray<math::Vec2i> edge2face_;
    CudaArray<int>         vert_in_edge_;
    CudaArray<math::Vec3i> edge_in_face_;
    CudaArray<math::Vec4i> adj_ee_pairs_;
};

}  // namespace collision
}  // namespace chysx
