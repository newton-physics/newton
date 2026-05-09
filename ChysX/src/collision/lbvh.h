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
// Storage layout (modeled on KittenGpuLBVH for cache-friendly traversal
// and cheap atomic-light refit):
//
//   `nodes_`         : `n_leaves - 1` internal nodes.  Each node fits
//                      in exactly one 64-byte cache line and embeds the
//                      AABBs of *both* its children, so a query thread
//                      sitting on an internal node only needs one cache
//                      line to decide whether to descend left, right,
//                      both, or neither.
//   `leaf_parents_`  : `n_leaves` `uint32_t` slots; `leaf_parents_[i]`
//                      is the internal-node index that owns leaf i,
//                      with the MSB set when leaf i is the *right*
//                      child of that parent.
//   `sorted_id_`     : `n_leaves` int slots; the post-Morton-sort
//                      mapping (sorted leaf order -> original primitive
//                      id).  Used at leaf hits to translate the BVH's
//                      internal sorted index back to the caller's
//                      primitive id (e.g. a face index).
//   `leaf_aabbs`     : *external* pointer (passed into `refit`).  The
//                      BVH does not own the per-leaf AABB buffer; each
//                      refit reads from it and pushes those bounds up
//                      into the per-internal-node `bounds[2]` slots.
//
// Index encoding (MSB-tagged uint32_t throughout):
//
//   - Internal node child links (`BvhNode::left`, `BvhNode::right`):
//       MSB = 0 -> internal node index (in `nodes_`)
//       MSB = 1 -> sorted leaf index   (in `sorted_id_`)
//   - Internal node parent link (`BvhNode::parent`):
//       MSB = 0 -> "I am my parent's left child"
//       MSB = 1 -> "I am my parent's right child"
//   - Leaf-to-parent link (`leaf_parents_[i]`):
//       lower 31 bits = internal node index
//       MSB = 0 / 1 same convention as `BvhNode::parent`
//
// `fence` per internal node is the index of the "far end" of the leaf
// range its subtree covers (the other end is the node's own index when
// you reinterpret it as a sorted-leaf coordinate).  Plumbed through so
// future self-queries can dedupe pairs (`max(idx, fence) <= q` skip);
// the current cross-query EF traversal does not need it.

#pragma once

#include <cstdint>

#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "aabb.cuh"

namespace chysx {
namespace collision {

// Packed AABB used inside `BvhNode`.  Identical numerical content to
// `Aabb` but without the `alignas(16)` qualifier, so an array of two
// of them fits in 48 bytes flat -- the rest of the 64-byte node is the
// 4 uint32_t link fields.  Cast to/from `Aabb` is a 24-byte memcpy.
struct AabbPacked {
    math::Vec3f mn;
    math::Vec3f mx;
};
static_assert(sizeof(AabbPacked) == 24,
              "AabbPacked must be exactly 24 bytes for BvhNode layout");

// One node per *internal* tree position.  Leaves are stored implicitly
// through `leaf_parents_` + `sorted_id_`; the leaf's own AABB lives in
// the external `leaf_aabbs` buffer the caller hands `refit(...)`.
//
// Layout (64 bytes, single L1 cache line):
//
//   [ 0..3 ]  parent  (uint32: MSB=isRightChild, lower 31 bits=parent index)
//   [ 4..7 ]  left    (uint32: MSB=isLeaf,       lower 31 bits=child index)
//   [ 8..11]  right   (uint32: MSB=isLeaf,       lower 31 bits=child index)
//   [12..15]  fence   (uint32: far end of subtree's leaf range)
//   [16..39]  bounds[0]  (24-byte AABB of the LEFT child)
//   [40..63]  bounds[1]  (24-byte AABB of the RIGHT child)
struct alignas(64) BvhNode {
    std::uint32_t parent;
    std::uint32_t left;
    std::uint32_t right;
    std::uint32_t fence;
    AabbPacked    bounds[2];

    // Top-bit tag helpers (kept here so users do not bake the magic
    // constant 0x80000000 into call sites).
    static constexpr std::uint32_t kLeafBit  = 0x80000000u;
    static constexpr std::uint32_t kRightBit = 0x80000000u;
    static constexpr std::uint32_t kIdxMask  = 0x7FFFFFFFu;
};
static_assert(sizeof(BvhNode) == 64,
              "BvhNode must be 64 bytes (one L1 cache line)");

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
    // both length == n_leaves.  Inputs are read-only and only used
    // during this call; the BVH copies what it needs into its own
    // node buffers, then `query_self_ef` traverses without going
    // back to `leaf_aabbs`.
    void refit(const Aabb*       leaf_aabbs,
               const math::Vec3f* leaf_centers,
               std::uintptr_t     cuda_stream = 0);

    // Self-EF query: for each query AABB (length `n_queries`), traverse
    // the tree from root and emit (query_id, leaf_id) pairs whose AABBs
    // overlap, skipping pairs that share at least one vertex (covertex
    // filter, given the edge & face arrays from MeshTopology).  Result
    // count + flat list live in `query_count_` / `query_pairs_`.
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

    // Tree topology with embedded child AABBs.  Length n_leaves - 1.
    const BvhNode* nodes_dev() const noexcept { return nodes_.gpu_data(); }

    // Flat list of (query_id, leaf_id) emitted by the last query.
    // The first `*query_count_dev_` entries are valid.
    int*  query_count_dev() noexcept { return query_count_.gpu_data(); }
    const math::Vec2i* query_pairs_dev() const noexcept { return query_pairs_.gpu_data(); }
    math::Vec2i* query_pairs_dev() noexcept { return query_pairs_.gpu_data(); }

private:
    int n_leaves_ = 0;
    int max_query_pairs_ = 0;
    std::size_t cub_temp_bytes_ = 0;

    // Maximum DFS stack size needed by `query_self_ef`.  Computed
    // during `refit` (the merge-up kernel writes the answer into
    // `refit_flag_[0]` when it lands on the root); read back to host
    // so the templated query kernel can be dispatched with the
    // smallest sufficient stack size.
    int max_stack_size_ = 1;

    // Per-frame Morton scratch.
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

    // Tree topology (with embedded child AABBs) + per-leaf parent link.
    CudaArray<BvhNode>      nodes_;          // length n_leaves - 1
    CudaArray<std::uint32_t> leaf_parents_;  // length n_leaves

    // Query result.
    CudaArray<int>         query_count_;     // single int
    CudaArray<math::Vec2i> query_pairs_;     // length max_query_pairs
};

}  // namespace collision
}  // namespace chysx
