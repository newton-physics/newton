// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::collision::QuantBvh.
//
// 1:1 port of cuda-cloth's `QuanBvh.cu`, with surrounding plumbing
// adapted to chysx's types (`Aabb` instead of `aabb`, `Vec3i` /
// `Vec2i` instead of `vec3i` / `vec2i`, `CudaArray` instead of
// `DeviceHostVector`, cub instead of thrust where it's a one-line
// swap).
//
// Pipeline per `refit()` call:
//
//   1. **scene_aabb (two-pass reduce)**     -> `scene_bbox_[0]`
//   2. **morton32 + id seeding**            -> `morton_in_[]`, `sorted_id_in_[]`
//   3. **cub::DeviceRadixSort (32-bit)**    -> `morton_out_[]`, `sorted_id_[]`
//   4. **inverse_mapping**                  -> `prim_map_[]`  (= inverse of sorted_id_)
//   5. **build_primitives_from_box**        -> `ext_box_[]`, `ext_idx_[]` (sorted-leaf storage)
//   6. **calc_split_metric**                -> `metric_[]`   (Apetrei split bit positions)
//   7. **build_int_nodes (atomicAdd flag)** -> `int_lc/rc/par/range_x/range_y/mark/box[]`,
//                                              `ext_par/lca/count[]`
//      Per-leaf walk-up: builds tree topology AND merges AABBs in one pass.
//   8. **cub::DeviceScan::ExclusiveSum**    -> `offset_table_[]` (DFS reorder offsets)
//   9. **calc_int_node_orders**             -> `tk_map_[]`  (internal-node DFS index)
//  10. **update_bvh_ext_links**             -> rewrites `ext_par_/ext_lca_` in DFS space
//  11. **reorder_quantized_node**           -> packs the final `ulonglong2[]` query tree
//
// `query_self_ef` is then a pure stackless do/while loop over
// `nodes_`, doing 14-bit integer overlap tests and emitting hits via
// a per-block shared-memory bucket so the global counter only sees
// one atomicAdd per block per flush.

#include "quant_bvh.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

namespace chysx {
namespace collision {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("chysx::collision::QuantBvh: ") +
                                 what + " failed: " + cudaGetErrorString(err));
    }
}

// cuda-cloth uses 256-thread blocks for the build pipeline and the
// query kernel.  Keeping the same numeric so the warp-shuffle reduce
// math stays identical.
constexpr int kBlockDim = 256;
inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// ============================================================================
// 32-bit morton code (10 bits per axis -- matches cuda-cloth's `morton3D`).
//
// QuantBvh uses 32-bit Morton codes (NOT chysx's 64-bit `morton3d`)
// because the Apetrei split metric `metric[i] = 32 - clz(code[i] ^
// code[i+1])` is calibrated for 32-bit codes (sentinel value 33).
// 10 bits per axis = 1024 buckets per axis is plenty given the
// downstream 14-bit AABB quantization grid.
// ============================================================================

__device__ __forceinline__ std::uint32_t expand_bits10(std::uint32_t v) {
    v &= 0x3FFu;
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __forceinline__ std::uint32_t morton3d_32(float fx, float fy, float fz) {
    fx = fx * 1024.0f;
    fy = fy * 1024.0f;
    fz = fz * 1024.0f;
    fx = fminf(fmaxf(fx, 0.0f), 1023.0f);
    fy = fminf(fmaxf(fy, 0.0f), 1023.0f);
    fz = fminf(fmaxf(fz, 0.0f), 1023.0f);
    return (expand_bits10((std::uint32_t)fx) << 2) |
           (expand_bits10((std::uint32_t)fy) << 1) |
            expand_bits10((std::uint32_t)fz);
}

// ============================================================================
// Scene-AABB reduce (two-pass, identical to LinearBvh's version)
// ============================================================================

__global__ void scene_aabb_pass1(
    const Aabb* __restrict__ leaves,
    int n,
    Aabb*       __restrict__ out) {
    __shared__ Aabb tile[kBlockDim];

    Aabb local;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        local.add(leaves[i]);
    }
    tile[threadIdx.x] = local;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) tile[threadIdx.x].add(tile[threadIdx.x + offset]);
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = tile[0];
}

__global__ void scene_aabb_pass2(
    const Aabb* __restrict__ in,
    int n,
    Aabb*       __restrict__ scene) {
    __shared__ Aabb tile[kBlockDim];
    Aabb local;
    for (int i = threadIdx.x; i < n; i += blockDim.x) local.add(in[i]);
    tile[threadIdx.x] = local;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) tile[threadIdx.x].add(tile[threadIdx.x + offset]);
        __syncthreads();
    }
    if (threadIdx.x == 0) scene[0] = tile[0];
}

// ============================================================================
// Morton + id seed
// ============================================================================

__global__ void compute_morton_and_id(
    const math::Vec3f* __restrict__ centers,
    const Aabb*        __restrict__ scene,
    int n,
    std::uint32_t* __restrict__ codes,
    int*           __restrict__ ids) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const Aabb sb = scene[0];
    const float w = fmaxf(sb.mx.x - sb.mn.x, 1e-30f);
    const float h = fmaxf(sb.mx.y - sb.mn.y, 1e-30f);
    const float d = fmaxf(sb.mx.z - sb.mn.z, 1e-30f);
    const math::Vec3f c = centers[idx];
    codes[idx] = morton3d_32((c.x - sb.mn.x) / w,
                             (c.y - sb.mn.y) / h,
                             (c.z - sb.mn.z) / d);
    ids[idx] = idx;
}

// ============================================================================
// Primitive scattering into sorted slots
// ============================================================================

__global__ void inverse_mapping_kernel(int n,
                                       const int* __restrict__ map,
                                       int*       __restrict__ inv_map) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    inv_map[map[idx]] = idx;
}

__global__ void build_primitives_from_box(
    int n,
    PackedFace*         __restrict__ prim_face,
    Aabb*               __restrict__ prim_box,
    const int*          __restrict__ prim_map,
    const Aabb*         __restrict__ box,
    const math::Vec3i*  __restrict__ faces) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int new_idx = prim_map[idx];
    // Pack the sorted-leaf payload as one 16-byte struct so the
    // query kernel can pull face vertices + original id with a
    // single `LD.E.128`.
    const math::Vec3i f = faces[idx];
    PackedFace p;
    p.x = f.x;
    p.y = f.y;
    p.z = f.z;
    p.w = idx;        // original (pre-sort) face id
    prim_face[new_idx] = p;
    prim_box[new_idx]  = box[idx];
}

// ============================================================================
// Apetrei split metric: bit-position of disagreement between adjacent
// sorted Morton codes.  Sentinel `33` for the rightmost slot so it
// always loses the "smaller metric wins" tiebreak.
// ============================================================================

__global__ void calc_split_metric(int n,
                                  const std::uint32_t* __restrict__ codes,
                                  int*                 __restrict__ metric) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    metric[idx] = (idx != n - 1) ? (32 - __clz(codes[idx] ^ codes[idx + 1])) : 33;
}

// ============================================================================
// build_int_nodes -- the heart of the algorithm
//
// One thread per leaf.  Walks toward the root via atomicAdd-flag
// gating (first arrival returns; second arrival merges and continues).
// Computes the internal tree topology AND the per-internal-node AABBs
// in a single bottom-up pass (no separate refit kernel).
//
// Marks (3 LSBs of `int_mark[i]`):
//   bit 0: left  child of internal node `i` is a LEAF
//   bit 1: right child of internal node `i` is a LEAF
//   bit 2: internal node `i` is the RIGHT child of its parent
//
// `ext_lca[k+1]` records, for each leaf `k` (k from 0..N-1), the
// internal-node index of the LCA between leaves k and k+1.  This is
// what the final `reorder_quantized_node` kernel uses to compute
// each node's escape pointer.  `count[k+1]` is the number of internal
// nodes on the LCA chain from leaf `k+1` upward through that leaf's
// "leftmost-leaf" subtrees -- prefix-summed later to get the DFS
// position assignment.
// ============================================================================

__global__ void build_int_nodes_kernel(
    int                  n,
    std::uint32_t*       __restrict__ count,
    int*                 __restrict__ ext_lca,
    const int*           __restrict__ ext_metric,
    std::uint32_t*       __restrict__ ext_par,
    std::uint32_t*       __restrict__ ext_mark,
    const Aabb*          __restrict__ ext_box,
    int*                 __restrict__ int_rc,
    int*                 __restrict__ int_lc,
    int*                 __restrict__ int_range_y,
    int*                 __restrict__ int_range_x,
    std::uint32_t*       __restrict__ int_mark,
    Aabb*                __restrict__ int_box,
    std::uint32_t*       __restrict__ flag,
    int*                 __restrict__ int_par) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Reset ext_lca[idx] / count[idx] -- one entry per thread.
    ext_lca[idx] = -1;
    count[idx]   = 0u;

    int  l = idx - 1;
    int  r = idx;
    bool mark = (l >= 0) ? (ext_metric[l] < ext_metric[r]) : false;
    int  cur  = mark ? l : r;

    ext_par[idx] = static_cast<std::uint32_t>(cur);
    if (mark) {
        int_rc[cur] = idx;
        int_range_y[cur] = idx;
        atomicOr(&int_mark[cur], 0x00000002u);
        ext_mark[idx] = 0x00000007u;
    } else {
        int_lc[cur] = idx;
        int_range_x[cur] = idx;
        atomicOr(&int_mark[cur], 0x00000001u);
        ext_mark[idx] = 0x00000003u;
    }

    while (atomicAdd(&flag[cur], 1u) == 1u) {
        // Second arrival: merge child AABBs into `cur`.
        const int   chl       = int_lc[cur];
        const int   chr       = int_rc[cur];
        const std::uint32_t m = int_mark[cur];

        Aabb merged = (m & 1u) ? ext_box[chl] : int_box[chl];
        if (m & 2u) merged.add(ext_box[chr]);
        else        merged.add(int_box[chr]);
        int_box[cur] = merged;

        // Sanitize mark to its 3 lowest bits (clearing any racy bits
        // from an unlucky atomicOr/atomicAnd interleaving).
        int_mark[cur] &= 0x00000007u;

        // Determine `cur`'s own parent based on the metrics of the
        // leaves immediately to the left (l) and right (r) of `cur`'s
        // covered range.
        l = int_range_x[cur] - 1;
        r = int_range_y[cur];
        ext_lca[l + 1] = cur;
        count[l + 1]   = count[l + 1] + 1u;  // safe: only one writer per `l+1`

        bool mark_up = (l >= 0) ? (ext_metric[l] < ext_metric[r]) : false;

        if (l + 1 == 0 && r == n - 1) {
            // Reached the root -- no further parent.
            int_par[cur] = -1;
            int_mark[cur] &= 0xFFFFFFFBu;
            break;
        }

        const int par = mark_up ? l : r;
        int_par[cur] = par;
        if (mark_up) {
            int_rc[par] = cur;
            int_range_y[par] = r;
            atomicAnd(&int_mark[par], 0xFFFFFFFDu);  // clear bit 1
            int_mark[cur] |= 0x00000004u;            // self is right child
        } else {
            int_lc[par] = cur;
            int_range_x[par] = l + 1;
            atomicAnd(&int_mark[par], 0xFFFFFFFEu);  // clear bit 0
            int_mark[cur] &= 0xFFFFFFFBu;            // self is left child
        }
        __threadfence();
        cur = par;
    }
}

// ============================================================================
// DFS reorder: turn `int_lc/rc/par/mark/box` (in build order) into
// `nodes_` (in DFS / left-spine order).
//
// `tk_map[i]` is the new (DFS-order) index for the internal node that
// was originally at position `i`.  Computed by walking the LCA chain
// from each leaf, depositing consecutive offsets.
// ============================================================================

__global__ void calc_int_node_orders(
    int                  n,
    const int*           __restrict__ int_lc,
    const int*           __restrict__ ext_lca,
    const std::uint32_t* __restrict__ count,
    const std::uint32_t* __restrict__ offset_table,
    int*                 __restrict__ tk_map) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int node = ext_lca[idx];
    int depth = static_cast<int>(count[idx]);
    int id    = static_cast<int>(offset_table[idx]);
    if (node != -1) {
        for (; depth--; node = int_lc[node]) {
            tk_map[node] = id++;
        }
    }
}

// Update `ext_par_` and `ext_lca_` to use the new DFS-ordered indices,
// and pack `(idx, isLeaf)` into the LSB of `ext_lca_` so the
// reorder kernel can recover the leaf/internal flag in one shift.
__global__ void update_bvh_ext_links(
    int                  n,
    const int*           __restrict__ tk_map,
    int*                 __restrict__ ext_lca,
    std::uint32_t*       __restrict__ ext_par) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    ext_par[idx] = static_cast<std::uint32_t>(tk_map[ext_par[idx]]);
    int ori = ext_lca[idx];
    if (ori != -1) ext_lca[idx] = tk_map[ori] << 1;       // bit0 = 0 -> internal
    else           ext_lca[idx] = (idx << 1) | 1;         // bit0 = 1 -> leaf, idx is the leaf
}

// Pack a single AABB into the 42 low bits of (q.x, q.y).  Call sites
// pre-OR in the index field (top 22 bits).
__device__ __forceinline__ void quantize_aabb(
    Ull2& q, const Aabb& b,
    const math::Vec3f& origin, const math::Vec3f& delta) {
    const float idx = 1.0f / fmaxf(delta.x, 1e-30f);
    const float idy = 1.0f / fmaxf(delta.y, 1e-30f);
    const float idz = 1.0f / fmaxf(delta.z, 1e-30f);
    q.x |= static_cast<std::uint64_t>((b.mn.x - origin.x) * idx) << QuantBvh::offset2;
    q.x |= static_cast<std::uint64_t>((b.mn.y - origin.y) * idy) << QuantBvh::offset1;
    q.x |= static_cast<std::uint64_t>((b.mn.z - origin.z) * idz);
    q.y |= static_cast<std::uint64_t>(ceilf((b.mx.x - origin.x) * idx)) << QuantBvh::offset2;
    q.y |= static_cast<std::uint64_t>(ceilf((b.mx.y - origin.y) * idy)) << QuantBvh::offset1;
    q.y |= static_cast<std::uint64_t>(ceilf((b.mx.z - origin.z) * idz));
}

__global__ void reorder_quantized_nodes(
    int                  int_size,                 // = n_leaves - 1
    const int*           __restrict__ tk_map,
    const int*           __restrict__ ext_lca,     // length n_leaves + 1, with leaf/internal flag in LSB
    const Aabb*          __restrict__ ext_box,
    const int*           __restrict__ unord_int_lc,
    const std::uint32_t* __restrict__ unord_int_mark,
    const int*           __restrict__ unord_int_range_y,
    const Aabb*          __restrict__ unord_int_box,
    const Aabb*          __restrict__ scene_box,
    Ull2*                __restrict__ nodes) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= int_size + 1) return;

    const Aabb sb = scene_box[0];
    const math::Vec3f origin = sb.mn;
    math::Vec3f delta(sb.mx.x - sb.mn.x,
                      sb.mx.y - sb.mn.y,
                      sb.mx.z - sb.mn.z);
    // Match cuda-cloth's quantization: divide the scene extent into
    // (1<<aabb_bits) - 2 buckets, leaving a slack bucket on each side
    // for ceilf rounding to never overflow.
    const float bucket = static_cast<float>((1 << QuantBvh::aabb_bits) - 2);
    delta.x /= bucket;
    delta.y /= bucket;
    delta.z /= bucket;

    // ---- LEAF: write to nodes[int_size + idx] ------------------------
    {
        Ull2 node = {0ull, 0ull};
        quantize_aabb(node, ext_box[idx], origin, delta);
        node.x |= QuantBvh::index_mask;  // top 22 bits = MaxIndex -> leaf marker

        const int escape = ext_lca[idx + 1];
        if (escape == -1) {
            node.y |= QuantBvh::index_mask;  // stop traversal
        } else {
            const int b_leaf = escape & 1;
            int e = escape >> 1;
            if (b_leaf) e += int_size;
            node.y |= static_cast<std::uint64_t>(static_cast<std::uint32_t>(e))
                     << QuantBvh::offset3;
        }
        nodes[idx + int_size] = node;
    }

    // ---- INTERNAL: write to nodes[tk_map[idx]] -----------------------
    if (idx >= int_size) return;
    const int new_id = tk_map[idx];
    const std::uint32_t mark = unord_int_mark[idx];

    Ull2 node = {0ull, 0ull};
    quantize_aabb(node, unord_int_box[idx], origin, delta);

    const int lc_raw = unord_int_lc[idx];
    const int lc = (mark & 1u) ? (lc_raw + int_size) : tk_map[lc_raw];
    node.x |= static_cast<std::uint64_t>(static_cast<std::uint32_t>(lc))
             << QuantBvh::offset3;

    const int int_escape = ext_lca[unord_int_range_y[idx] + 1];
    if (int_escape == -1) {
        node.y |= QuantBvh::index_mask;
    } else {
        const int b_leaf = int_escape & 1;
        int e = int_escape >> 1;
        if (b_leaf) e += int_size;
        node.y |= static_cast<std::uint64_t>(static_cast<std::uint32_t>(e))
                 << QuantBvh::offset3;
    }
    nodes[new_id] = node;
}

// ============================================================================
// Stackless EF query
//
// Each thread owns one edge.  At kernel start it reads `edges[tid]`
// into a register, fetches the two endpoint positions from
// `verts[]`, builds the thickness-enlarged AABB on the fly, and
// quantises it to integer space -- no precomputed `query_aabbs[]`
// buffer, so we save one full kernel launch + n_edges * 24 B of
// DRAM traffic upstream.
//
// Each step:
//   - Load the (16-byte) node.
//   - Extract lc (top 22 bits of node.x) and escape (top 22 bits of node.y).
//   - Quant-AABB overlap test (six 14-bit integer comparisons).
//   - Hit on internal (lc != MaxIndex)            -> st = lc       (= next sequential index!)
//   - Hit on leaf     (lc == MaxIndex)            -> single 128-bit
//                                                     load of the
//                                                     packed
//                                                     `(face.xyz,
//                                                     orig_id)` -->
//                                                     inline
//                                                     covertex
//                                                     test --> emit
//                                                     -> st = escape
//   - Miss                                         -> st = escape
//   - Termination when st == MaxIndex.
//
// Hits go through a per-block shared-memory bucket so the global pair
// counter only sees one atomicAdd per block per flush.
// ============================================================================

// Packed integer AABB (6 ints).
struct IntAabb {
    int mn_x, mn_y, mn_z;
    int mx_x, mx_y, mx_z;
};

// Test the 6 packed 14-bit components of a node against an int query
// AABB.  Layout-aware: extracts (min_x, min_y, min_z) from node.x and
// (max_x, max_y, max_z) from node.y in the same 14-bit slot order
// used by `quantize_aabb`.
__device__ __forceinline__ bool overlaps_ull2_int(const Ull2& a, const IntAabb& b) {
    constexpr std::uint64_t MASK = QuantBvh::aabb_mask;
    int v;
    v = (int)((a.x >> QuantBvh::offset2) & MASK);  if (v > b.mx_x) return false;  // node.min.x > query.max.x
    v = (int)((a.y >> QuantBvh::offset2) & MASK);  if (v < b.mn_x) return false;  // node.max.x < query.min.x
    v = (int)((a.x >> QuantBvh::offset1) & MASK);  if (v > b.mx_y) return false;
    v = (int)((a.y >> QuantBvh::offset1) & MASK);  if (v < b.mn_y) return false;
    v = (int)( a.x                       & MASK);  if (v > b.mx_z) return false;
    v = (int)( a.y                       & MASK);  if (v < b.mn_z) return false;
    return true;
}

// 1024 pairs * 8 bytes = 8 KB shared per block.  Same as cuda-cloth.
constexpr int kMaxResPerBlock = 1024;

__global__ void query_self_ef_kernel(
    const math::Vec2i*   __restrict__ edges,
    const math::Vec3f*   __restrict__ verts,
    int                  n_edges,
    float                thickness,
    int                  int_size,           // = n_leaves - 1
    const PackedFace*    __restrict__ ext_face,
    const Aabb*          __restrict__ scene_box,
    const Ull2*          __restrict__ nodes,
    int*                 __restrict__ pair_count,
    math::Vec2i*         __restrict__ pair_list,
    int                  max_pairs) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = tid < n_edges;

    // ---- scene-box constants (cheap, identical across threads) -----
    const Aabb sb = scene_box[0];
    const math::Vec3f origin = sb.mn;
    const float bucket = static_cast<float>((1 << QuantBvh::aabb_bits) - 2);
    // Reciprocals of the per-axis quantisation step.  Using `*inv`
    // instead of `/delta` matches what NVCC would generate anyway
    // and shaves one DIV per axis on the slower SASS paths.
    const float dx = (sb.mx.x - sb.mn.x) / bucket;
    const float dy = (sb.mx.y - sb.mn.y) / bucket;
    const float dz = (sb.mx.z - sb.mn.z) / bucket;
    const float idx_q = 1.0f / fmaxf(dx, 1e-30f);
    const float idy_q = 1.0f / fmaxf(dy, 1e-30f);
    const float idz_q = 1.0f / fmaxf(dz, 1e-30f);

    // ---- build & quantise this thread's edge box on the fly --------
    //
    // Stays in registers all the way through: no `query_aabbs[]`
    // round-trip through DRAM.  `edge` itself stays in registers
    // for the whole kernel so the inline covertex test below pays
    // zero memory traffic for it (vs the old `edges[tid]` reload
    // per leaf hit).
    math::Vec2i edge(0, 0);
    IntAabb bv;
    if (active) {
        edge = edges[tid];
        const math::Vec3f a = verts[edge.x];
        const math::Vec3f b = verts[edge.y];
        const float mn_x = fminf(a.x, b.x) - thickness;
        const float mn_y = fminf(a.y, b.y) - thickness;
        const float mn_z = fminf(a.z, b.z) - thickness;
        const float mx_x = fmaxf(a.x, b.x) + thickness;
        const float mx_y = fmaxf(a.y, b.y) + thickness;
        const float mx_z = fmaxf(a.z, b.z) + thickness;
        bv.mn_x = (int)((mn_x - origin.x) * idx_q);
        bv.mn_y = (int)((mn_y - origin.y) * idy_q);
        bv.mn_z = (int)((mn_z - origin.z) * idz_q);
        bv.mx_x = (int)ceilf((mx_x - origin.x) * idx_q);
        bv.mx_y = (int)ceilf((mx_y - origin.y) * idy_q);
        bv.mx_z = (int)ceilf((mx_z - origin.z) * idz_q);
    }

    __shared__ math::Vec2i s_buf[kMaxResPerBlock];
    __shared__ int         s_counter;
    __shared__ int         s_global_off;
    if (threadIdx.x == 0) s_counter = 0;

    std::uint32_t st = 0u;

    while (true) {
        __syncthreads();

        if (active) {
            while (st != QuantBvh::max_index) {
                Ull2 node = nodes[st];
                const std::uint32_t lc     = (std::uint32_t)(node.x >> QuantBvh::offset3);
                const std::uint32_t escape = (std::uint32_t)(node.y >> QuantBvh::offset3);

                if (overlaps_ull2_int(node, bv)) {
                    if (lc == QuantBvh::max_index) {
                        // Leaf hit.  Single 128-bit load gives us
                        // both the three face vertex ids (for the
                        // covertex test) and the original face id
                        // (for the output pair) -- no
                        // `ext_idx` -> `faces[]` indirection chain.
                        const int leaf_slot = static_cast<int>(st) - int_size;
                        const PackedFace fd = ext_face[leaf_slot];
                        const bool covertex =
                            (edge.x == fd.x) | (edge.x == fd.y) | (edge.x == fd.z) |
                            (edge.y == fd.x) | (edge.y == fd.y) | (edge.y == fd.z);
                        if (!covertex) {
                            const int s_idx = atomicAdd(&s_counter, 1);
                            if (s_idx >= kMaxResPerBlock) {
                                // Bucket full: drop this hit (st was
                                // about to advance to escape anyway,
                                // and we'll re-emit it next iteration
                                // by rewinding st).  Match the cuda-
                                // cloth behaviour exactly: just break
                                // and let the block flush.
                                break;
                            }
                            s_buf[s_idx] = math::Vec2i(tid, fd.w);
                        }
                        st = escape;
                    } else {
                        // Internal hit -> descend into the left
                        // child.  Because the tree was DFS-reordered,
                        // `lc` is always st+1 for "happy path"
                        // descents -- the cleanest possible memory
                        // access pattern for the next iteration.
                        st = lc;
                    }
                } else {
                    // Miss -> follow the escape pointer.
                    st = escape;
                }
            }
        }

        // ----- flush bucket --------------------------------------------
        __syncthreads();
        int total = s_counter;
        if (total > kMaxResPerBlock) total = kMaxResPerBlock;

        if (threadIdx.x == 0)
            s_global_off = atomicAdd(pair_count, total);

        __syncthreads();
        const int g_off = s_global_off;

        if (g_off >= max_pairs || total == 0) return;
        if (threadIdx.x == 0) s_counter = 0;

        bool done = (total < kMaxResPerBlock);
        if (g_off + total > max_pairs) {
            total = max_pairs - g_off;
            done = true;
        }

        for (int i = threadIdx.x; i < total; i += blockDim.x)
            pair_list[g_off + i] = s_buf[i];

        if (done) break;
    }
}

// ============================================================================
// Tiny utilities
// ============================================================================

__global__ void clear_int_kernel(int* p) { *p = 0; }

__global__ void memset_uint_kernel(std::uint32_t* p, std::uint32_t v, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

__global__ void memset_int_kernel(int* p, int v, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

}  // namespace

// ============================================================================
// QuantBvh
// ============================================================================

void QuantBvh::build(int n_leaves, int max_query_pairs) {
    if (n_leaves <= 1) {
        n_leaves_ = n_leaves;
        max_query_pairs_ = max_query_pairs;
        return;
    }
    if (n_leaves_ == n_leaves && max_query_pairs_ == max_query_pairs) {
        return;
    }

    n_leaves_ = n_leaves;
    max_query_pairs_ = max_query_pairs;

    const std::size_t N = static_cast<std::size_t>(n_leaves);

    scene_bbox_.resize(1);
    scene_partial_.resize(1024);

    morton_in_.resize(N);
    morton_out_.resize(N);
    sorted_id_in_.resize(N);
    sorted_id_.resize(N);
    prim_map_.resize(N);

    ext_box_.resize(N);
    ext_face_.resize(N);
    ext_lca_.resize(N + 1);
    ext_par_.resize(N);
    ext_mark_.resize(N);
    metric_.resize(N);
    count_.resize(N);
    offset_table_.resize(N);
    tk_map_.resize(N);

    int_lc_.resize(N - 1);
    int_rc_.resize(N - 1);
    int_par_.resize(N - 1);
    range_x_.resize(N - 1);
    range_y_.resize(N - 1);
    int_mark_.resize(N - 1);
    int_box_.resize(N - 1);
    flag_.resize(N - 1);

    nodes_.resize(2 * N - 1);

    query_count_.resize(1);
    query_pairs_.resize(static_cast<std::size_t>(max_query_pairs));

    // cub temp-storage sniff (sort + scan).
    {
        std::size_t bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr, bytes,
            morton_in_.gpu_data(), morton_out_.gpu_data(),
            sorted_id_in_.gpu_data(), sorted_id_.gpu_data(),
            n_leaves_);
        cub_sort_bytes_ = bytes;
        cub_sort_temp_.resize(bytes);
    }
    {
        std::size_t bytes = 0;
        cub::DeviceScan::ExclusiveSum(
            nullptr, bytes,
            count_.gpu_data(), offset_table_.gpu_data(),
            n_leaves_);
        cub_scan_bytes_ = bytes;
        cub_scan_temp_.resize(bytes);
    }
}

void QuantBvh::refit(const Aabb*        leaf_aabbs,
                     const math::Vec3f* leaf_centers,
                     const math::Vec3i* faces,
                     std::uintptr_t     cuda_stream) {
    if (n_leaves_ <= 1) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    const int N = n_leaves_;
    Aabb* d_scene = scene_bbox_.gpu_data();

    // 1. Scene AABB (two-pass parallel reduce).
    {
        const int blocks = std::min(grid_for(N), 1024);
        scene_aabb_pass1<<<blocks, kBlockDim, 0, stream>>>(
            leaf_aabbs, N, scene_partial_.gpu_data());
        scene_aabb_pass2<<<1, kBlockDim, 0, stream>>>(
            scene_partial_.gpu_data(), blocks, d_scene);
        check_cuda(cudaGetLastError(), "scene_aabb reduce");
    }

    // 2. Morton + id seed.
    compute_morton_and_id<<<grid_for(N), kBlockDim, 0, stream>>>(
        leaf_centers, d_scene, N,
        morton_in_.gpu_data(), sorted_id_in_.gpu_data());
    check_cuda(cudaGetLastError(), "compute_morton_and_id");

    // 3. Sort by morton (32-bit key, int value).
    cub::DeviceRadixSort::SortPairs(
        cub_sort_temp_.gpu_data(), cub_sort_bytes_,
        morton_in_.gpu_data(), morton_out_.gpu_data(),
        sorted_id_in_.gpu_data(), sorted_id_.gpu_data(),
        N, 0, sizeof(std::uint32_t) * 8, stream);

    // 4. Inverse mapping (sorted_id -> prim_map).
    inverse_mapping_kernel<<<grid_for(N), kBlockDim, 0, stream>>>(
        N, sorted_id_.gpu_data(), prim_map_.gpu_data());
    check_cuda(cudaGetLastError(), "inverse_mapping_kernel");

    // 5. Scatter input AABBs + packed face payload into sorted slots.
    build_primitives_from_box<<<grid_for(N), kBlockDim, 0, stream>>>(
        N, ext_face_.gpu_data(), ext_box_.gpu_data(),
        prim_map_.gpu_data(), leaf_aabbs, faces);
    check_cuda(cudaGetLastError(), "build_primitives_from_box");

    // 6. Apetrei split metric.
    calc_split_metric<<<grid_for(N), kBlockDim, 0, stream>>>(
        N, morton_out_.gpu_data(), metric_.gpu_data());
    check_cuda(cudaGetLastError(), "calc_split_metric");

    // 7. Build internal nodes (walk-up + AABB merge in one pass).
    //
    // Reset all the buffers `build_int_nodes_kernel` reads/writes.
    memset_uint_kernel<<<grid_for(N), kBlockDim, 0, stream>>>(
        ext_mark_.gpu_data(), 7u, N);
    memset_int_kernel<<<grid_for(N), kBlockDim, 0, stream>>>(
        reinterpret_cast<int*>(ext_par_.gpu_data()), -1, N);
    memset_int_kernel<<<grid_for(N + 1), kBlockDim, 0, stream>>>(
        ext_lca_.gpu_data(), -1, N + 1);
    memset_uint_kernel<<<grid_for(N - 1), kBlockDim, 0, stream>>>(
        flag_.gpu_data(), 0u, N - 1);
    memset_uint_kernel<<<grid_for(N - 1), kBlockDim, 0, stream>>>(
        int_mark_.gpu_data(), 0u, N - 1);

    build_int_nodes_kernel<<<grid_for(N), kBlockDim, 0, stream>>>(
        N, count_.gpu_data(), ext_lca_.gpu_data(), metric_.gpu_data(),
        ext_par_.gpu_data(), ext_mark_.gpu_data(), ext_box_.gpu_data(),
        int_rc_.gpu_data(), int_lc_.gpu_data(),
        range_y_.gpu_data(), range_x_.gpu_data(),
        int_mark_.gpu_data(), int_box_.gpu_data(),
        flag_.gpu_data(), int_par_.gpu_data());
    check_cuda(cudaGetLastError(), "build_int_nodes_kernel");

    // 8. Scan `count_` to get DFS-order offsets.
    cub::DeviceScan::ExclusiveSum(
        cub_scan_temp_.gpu_data(), cub_scan_bytes_,
        count_.gpu_data(), offset_table_.gpu_data(),
        N, stream);

    // 9. Compute the per-internal-node DFS index (`tk_map_`).
    calc_int_node_orders<<<grid_for(N), kBlockDim, 0, stream>>>(
        N, int_lc_.gpu_data(), ext_lca_.gpu_data(),
        count_.gpu_data(), offset_table_.gpu_data(),
        tk_map_.gpu_data());
    check_cuda(cudaGetLastError(), "calc_int_node_orders");

    // 10. Reset ext_lca[N] = -1 so the leaf at the right edge gets a
    // "stop" escape pointer.
    memset_int_kernel<<<1, 1, 0, stream>>>(
        ext_lca_.gpu_data() + N, -1, 1);

    // 11. Update ext_par_ / ext_lca_ to DFS-space, with leaf flag in LSB.
    update_bvh_ext_links<<<grid_for(N), kBlockDim, 0, stream>>>(
        N, tk_map_.gpu_data(), ext_lca_.gpu_data(), ext_par_.gpu_data());
    check_cuda(cudaGetLastError(), "update_bvh_ext_links");

    // 12. Pack into the final stackless quantized tree.
    reorder_quantized_nodes<<<grid_for(N), kBlockDim, 0, stream>>>(
        N - 1, tk_map_.gpu_data(), ext_lca_.gpu_data(), ext_box_.gpu_data(),
        int_lc_.gpu_data(), int_mark_.gpu_data(),
        range_y_.gpu_data(), int_box_.gpu_data(),
        scene_bbox_.gpu_data(), nodes_.gpu_data());
    check_cuda(cudaGetLastError(), "reorder_quantized_nodes");
}

void QuantBvh::query_self_ef(const math::Vec2i*    edges,
                             const math::Vec3f*    verts,
                             int                   n_edges,
                             float                 thickness,
                             std::uintptr_t        cuda_stream) {
    if (n_leaves_ <= 1 || n_edges <= 0) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    clear_int_kernel<<<1, 1, 0, stream>>>(query_count_.gpu_data());
    query_self_ef_kernel<<<grid_for(n_edges), kBlockDim, 0, stream>>>(
        edges, verts, n_edges, thickness,
        n_leaves_ - 1,
        ext_face_.gpu_data(),
        scene_bbox_.gpu_data(),
        nodes_.gpu_data(),
        query_count_.gpu_data(), query_pairs_.gpu_data(),
        max_query_pairs_);
    check_cuda(cudaGetLastError(), "query_self_ef_kernel");
}

}  // namespace collision
}  // namespace chysx
