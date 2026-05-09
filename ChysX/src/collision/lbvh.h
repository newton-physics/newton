// SPDX-License-Identifier: Apache-2.0
//
// chysx::collision::LinearBvh
//
// Karras 2012 / Apetrei "linear BVH" implementation.
//
//   build(...)        sized for n_leaves; allocates persistent node /
//                     scratch buffers.  Cheap re-call when the leaf
//                     count is unchanged.
//   refit(...)        per-frame: compute Morton codes from
//                     primitive centroids, radix-sort by key, build
//                     the binary radix tree, refit AABBs bottom-up.
//   query_self_ef(..) one thread per query AABB; emits a flat list of
//                     (edge_id, face_id) candidate pairs subject to
//                     `!covertex_EF` filtering (drops the trivial
//                     "edge incident to face" hits the AABB test
//                     wouldn't tell us about).
//
// Storage layout: 2N-1 nodes.  Leaves live at indices [N-1, 2N-1) and
// internal nodes at [0, N-1).  This matches cuda-cloth's `bvhNode`
// layout so the kernels are 1:1 ports modulo the cub-vs-thrust sort.

#pragma once

#include <cstdint>

#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "aabb.cuh"

namespace chysx {
namespace collision {

struct BvhNode {
    int parent;
    int left;
    int right;

    // Marker for "uninitialised" (matches cuda-cloth's cudaMemset(-1)).
    static constexpr int kInvalid = -1;
};

class LinearBvh {
public:
    LinearBvh() = default;

    LinearBvh(const LinearBvh&) = delete;
    LinearBvh& operator=(const LinearBvh&) = delete;
    LinearBvh(LinearBvh&&) noexcept = default;
    LinearBvh& operator=(LinearBvh&&) noexcept = default;

    // Allocate / re-allocate persistent buffers for `n_leaves`.
    // Idempotent for the same leaf count.
    void build(int n_leaves, int max_query_pairs);

    // Per-frame refit driven by `leaf_aabbs` and `leaf_centers`,
    // both length == n_leaves.  Inputs are read-only.
    void refit(const Aabb*       leaf_aabbs,
               const math::Vec3f* leaf_centers,
               std::uintptr_t     cuda_stream = 0);

    // Self-EF query: for each query AABB (length `n_queries`), traverse
    // the tree from root and emit (query_id, leaf_id) pairs whose AABBs
    // overlap, skipping pairs that share at least one vertex (covertex
    // filter, given the edge & face arrays from MeshTopology).  Result
    // count + flat list live in `query_count_dev_` / `query_pairs_`.
    void query_self_ef(const Aabb*           query_aabbs,
                       int                   n_queries,
                       const math::Vec2i*    edges,
                       const math::Vec3i*    faces,
                       std::uintptr_t        cuda_stream = 0);

    // ---- accessors ----------------------------------------------------

    int n_leaves() const noexcept { return n_leaves_; }
    int max_query_pairs() const noexcept { return max_query_pairs_; }

    // Sorted leaf -> original primitive id.  Length n_leaves.
    const int* sorted_id_dev() const noexcept { return sorted_id_.gpu_data(); }

    // Tree topology + per-node AABB.  Length 2*n_leaves - 1.
    const BvhNode* nodes_dev() const noexcept { return nodes_.gpu_data(); }
    const Aabb*    node_aabbs_dev() const noexcept { return node_aabbs_.gpu_data(); }

    // Flat list of (query_id, leaf_id) emitted by the last query.
    // The first `*query_count_dev_` entries are valid.
    int*  query_count_dev() noexcept { return query_count_.gpu_data(); }
    const math::Vec2i* query_pairs_dev() const noexcept { return query_pairs_.gpu_data(); }
    math::Vec2i* query_pairs_dev() noexcept { return query_pairs_.gpu_data(); }

private:
    int n_leaves_ = 0;
    int max_query_pairs_ = 0;
    std::size_t cub_temp_bytes_ = 0;

    // Per-frame scratch.
    CudaArray<std::uint64_t> morton_keys_in_;
    CudaArray<std::uint64_t> morton_keys_out_;
    CudaArray<int>           sorted_id_in_;
    CudaArray<int>           sorted_id_;     // values out of cub sort
    CudaArray<unsigned int>  refit_flag_;    // length n_leaves - 1
    CudaArray<std::uint8_t>  cub_temp_;      // raw bytes for cub::DeviceRadixSort

    // Scene-bbox scratch (length 1) and per-block partial reductions
    // for the two-pass scene-AABB reduce.
    CudaArray<Aabb> scene_bbox_;
    CudaArray<Aabb> scene_partial_;

    // Tree.
    CudaArray<BvhNode> nodes_;       // length 2 * n_leaves - 1
    CudaArray<Aabb>    node_aabbs_;  // length 2 * n_leaves - 1

    // Query result.
    CudaArray<int>         query_count_;     // single int
    CudaArray<math::Vec2i> query_pairs_;     // length max_query_pairs
};

}  // namespace collision
}  // namespace chysx
