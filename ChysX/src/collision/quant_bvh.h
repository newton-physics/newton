// SPDX-License-Identifier: Apache-2.0
//
// chysx::collision::QuantBvh
//
// Quantized stackless BVH, ported 1:1 from cuda-cloth's `QuanBvh.cu`.
// Drop-in replacement for `LinearBvh` -- exposes the same `build /
// refit / query_self_ef` triplet and the same `query_pairs_dev /
// query_count_dev` accessors so `SelfCollisionDetector` can swap
// implementations by changing one type alias.
//
// What the "quant stackless" trick buys us over the LBVH:
//
//   - **16-byte nodes** (`ulonglong2`), four per L1 cache line, vs the
//     LBVH's 64-byte node.  Each node packs:
//       node.x = (left_child_index : 22) << 42 | (quant_min_xyz : 42)
//       node.y = (escape_index     : 22) << 42 | (quant_max_xyz : 42)
//     where `quant_min/max_xyz` are 14-bit integer quantizations of
//     the AABB bounds along each axis.
//
//   - **Stackless traversal** via DFS-order reordering + escape
//     pointers.  After build, the internal nodes are reordered so
//     that node[i]'s left child is always at node[i+1] (the next
//     sequential index).  Traversal reduces to:
//
//         st = 0;
//         do {
//             node = nodes[st];
//             if (overlaps(node, query))
//                 st = is_leaf(node) ? handle_leaf(), node.escape : node.left;
//             else
//                 st = node.escape;
//         } while (st != MaxIndex);
//
//     Zero stack memory, zero register-stack overhead, sequential
//     reads on left descents.
//
//   - **Integer-only overlap test**.  The query AABB is quantized
//     into a `intAABB` once at the top of the kernel; every node
//     intersection is then six 14-bit integer comparisons -- no
//     floating-point loads or compares inside the inner loop.
//
//   - **Fused build+refit**.  Each call to `refit()` recomputes the
//     scene bbox, Morton codes, sorted leaf order, AND the entire
//     internal tree topology + per-node AABBs in a single bottom-up
//     atomicAdd-flag walk-up.  No separate refit pass.
//
// Reference:
//   - cuda-cloth `src/Collision/BoundingVolume/QuanBvh.cuh / .cu`
//   - Apetrei, "Fast and Simple Agglomerative LBVH Construction" (2014)

#pragma once

#include <cstdint>

#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "aabb.cuh"

namespace chysx {
namespace collision {

// 16-byte packed-node POD, equivalent to CUDA's `ulonglong2` but with
// no dependence on `vector_types.h` -- keeps the header includable
// from plain C++ translation units.
struct alignas(16) Ull2 {
    std::uint64_t x;
    std::uint64_t y;
};
static_assert(sizeof(Ull2) == 16, "Ull2 must be 16 bytes (ulonglong2 layout)");

// 16-byte packed leaf payload: face vertex indices + the original
// (pre-sort) face id, all four reachable in a single 128-bit
// `LD.E.128` from the query kernel.  `alignas(16)` is what makes
// NVCC emit the vectorised load -- without it the read decays to
// four separate 32-bit fetches and we lose most of the speedup the
// pack was meant to buy.
struct alignas(16) PackedFace {
    int x;   // face vertex 0
    int y;   // face vertex 1
    int z;   // face vertex 2
    int w;   // original face id (== `idx` in the unsorted face array)
};
static_assert(sizeof(PackedFace) == 16, "PackedFace must be 16 bytes (int4 layout)");

class QuantBvh {
public:
    // ---- packing constants (also referenced inside the kernels) ------
    //
    // 14-bit quantization per axis, leaving 22 bits per index = up to
    // 4_194_303 nodes.  Plenty for any cloth scene we ship.
    static constexpr int aabb_bits  = 14;
    static constexpr int index_bits = 64 - 3 * aabb_bits;     // 22
    static constexpr int offset3    = aabb_bits * 3;          // 42
    static constexpr int offset2    = aabb_bits * 2;          // 28
    static constexpr int offset1    = aabb_bits * 1;          // 14
    static constexpr std::uint64_t index_mask =
        std::uint64_t(0xFFFFFFFFFFFFFFFFull) << offset3;      // top 22 bits
    static constexpr std::uint32_t aabb_mask =
        std::uint32_t(0xFFFFFFFFu) >> (32 - aabb_bits);       // 0x3FFF
    // Sentinel for "leaf" (in node.x's index slot) and for "stop"
    // (in node.y's escape slot).  Lower 22 bits all 1s.  Computed as
    // a 64-bit shift then truncated -- a bare `(uint32 >> 42)` would
    // be UB.
    static constexpr std::uint32_t max_index = static_cast<std::uint32_t>(
        std::uint64_t(0xFFFFFFFFFFFFFFFFull) >> offset3);     // 0x3FFFFF

    QuantBvh() = default;

    QuantBvh(const QuantBvh&) = delete;
    QuantBvh& operator=(const QuantBvh&) = delete;
    QuantBvh(QuantBvh&&) noexcept = default;
    QuantBvh& operator=(QuantBvh&&) noexcept = default;

    // Allocate persistent buffers for `n_leaves`.  Idempotent for the
    // same leaf count.  Mirrors `LinearBvh::build` so the two BVHs
    // share the same lifecycle in `SelfCollisionDetector`.
    void build(int n_leaves, int max_query_pairs);

    // Per-frame fused build-and-refit.  Recomputes the scene bbox,
    // sorts leaves by Morton code, builds the internal tree topology
    // and AABB hierarchy, then quantizes everything into the final
    // `nodes_` array used by `query_self_ef`.
    //
    // `faces` is needed so that the sorted-leaf payload can be
    // packed into a single 16-byte `PackedFace` per leaf, which the
    // query kernel reads with one 128-bit load (saves two dependent
    // global loads per leaf hit vs storing only the original face id
    // and indirecting through `faces[]`).
    void refit(const Aabb*        leaf_aabbs,
               const math::Vec3f* leaf_centers,
               const math::Vec3i* faces,
               std::uintptr_t     cuda_stream = 0);

    // Self-EF query: for each edge `e` in `edges[0..n_edges)` build
    // its (thickness-enlarged) AABB *inside* the kernel from
    // `verts[]`, do a stackless traversal of the tree, and emit
    // (edge_id, original_face_id) pairs whose AABBs overlap,
    // dropping pairs that share at least one vertex (covertex(EF)
    // filter, performed inline against the packed leaf payload --
    // no extra `faces[]` indirection).
    void query_self_ef(const math::Vec2i*    edges,
                       const math::Vec3f*    verts,
                       int                   n_edges,
                       float                 thickness,
                       std::uintptr_t        cuda_stream = 0);

    // ---- accessors (same shape as LinearBvh) -------------------------

    int n_leaves() const noexcept { return n_leaves_; }
    int max_query_pairs() const noexcept { return max_query_pairs_; }

    int*       query_count_dev() noexcept       { return query_count_.gpu_data(); }
    const int* query_count_dev() const noexcept { return query_count_.gpu_data(); }
    const math::Vec2i* query_pairs_dev() const noexcept { return query_pairs_.gpu_data(); }
    math::Vec2i* query_pairs_dev() noexcept { return query_pairs_.gpu_data(); }

private:
    int n_leaves_ = 0;
    int max_query_pairs_ = 0;
    std::size_t cub_sort_bytes_ = 0;
    std::size_t cub_scan_bytes_ = 0;

    // Scene-bbox scratch (length 1) and per-block partial reductions
    // for the two-pass scene-AABB reduce.
    CudaArray<Aabb> scene_bbox_;
    CudaArray<Aabb> scene_partial_;

    // Sort scratch (cub::DeviceRadixSort).
    CudaArray<std::uint32_t> morton_in_;
    CudaArray<std::uint32_t> morton_out_;
    CudaArray<int>           sorted_id_in_;
    CudaArray<int>           sorted_id_;
    CudaArray<int>           prim_map_;       // inverse of sorted_id_
    CudaArray<std::uint8_t>  cub_sort_temp_;

    // Per-leaf scratch (length N).
    CudaArray<Aabb>          ext_box_;
    // 16-byte packed leaf payload (sorted-leaf order):
    //   .x/.y/.z = the three vertex indices of the original face
    //   .w       = the original (pre-sort) face id
    // Replaces the previous `int ext_idx_[]` (4 B/leaf, original id
    // only).  Loaded with a single `LD.E.128` per leaf hit in
    // `query_self_ef_kernel`.
    CudaArray<PackedFace>    ext_face_;
    CudaArray<int>           ext_lca_;        // length N+1; encodes escape pointers
    CudaArray<std::uint32_t> ext_par_;
    CudaArray<std::uint32_t> ext_mark_;
    CudaArray<int>           metric_;
    CudaArray<std::uint32_t> count_;          // depth count per leaf, fed to scan
    CudaArray<std::uint32_t> offset_table_;   // exclusive scan of count_
    CudaArray<int>           tk_map_;         // internal-node DFS reorder map
    CudaArray<std::uint8_t>  cub_scan_temp_;

    // Per-internal-node scratch (length N - 1).
    CudaArray<int>           int_lc_;
    CudaArray<int>           int_rc_;
    CudaArray<int>           int_par_;
    CudaArray<int>           range_x_;
    CudaArray<int>           range_y_;
    CudaArray<std::uint32_t> int_mark_;
    CudaArray<Aabb>          int_box_;
    CudaArray<std::uint32_t> flag_;           // first/second-arrival counter

    // Final packed quantized tree, length 2N - 1.  Internal nodes at
    // [0, N-1), leaves at [N-1, 2N-1).  This is the only buffer the
    // query kernel touches.
    CudaArray<Ull2>          nodes_;

    // Query result.
    CudaArray<int>           query_count_;
    CudaArray<math::Vec2i>   query_pairs_;
};

}  // namespace collision
}  // namespace chysx
