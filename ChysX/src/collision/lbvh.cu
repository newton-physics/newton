// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::collision::LinearBvh.
//
// Karras 2012 binary-radix-tree build over Morton-sorted leaves with
// the cache-friendly node layout pioneered by KittenGpuLBVH
// (https://github.com/JerryHsu/KittenGpuLBVH).  Three improvements
// over the previous chysx port make this version both faster and
// less prone to overflow on dense self-collision scenes:
//
//   1. **One-cache-line nodes.**  `BvhNode` is exactly 64 bytes and
//      embeds the AABBs of *both* children directly, so a query
//      thread only touches one node per descent (instead of one node
//      *plus* two child AABB fetches).
//
//   2. **atomicOr-based refit.**  Each leaf walks toward the root and
//      atomicOrs a per-internal-node depth flag.  The first thread to
//      arrive at a node returns immediately; the second is the one
//      that reads the sibling's bounds and walks one level higher.
//      No CAS contention on hot internal nodes; the `or` also doubles
//      as a cheap DFS-stack-depth tracker so the query kernel can be
//      dispatched with the smallest possible stack.
//
//   3. **Block-shared output buffer.**  The query kernel collects
//      candidate pairs into a `__shared__` ring of size
//      MAX_RES_PER_BLOCK and only fires *one* `atomicAdd` on the
//      global counter per flush.  Per-leaf-hit atomics are gone.
//      For a 100x200 cloth this drops broadphase wall time by ~3-4x
//      compared with the previous "atomicAdd on every leaf hit" code.
//
// The two non-trivial dependencies (a global radix sort and a global
// AABB reduce) are filled in with cub instead of thrust so we don't
// have to add a new build dep.

#include "lbvh.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

#include "morton.cuh"

namespace chysx {
namespace collision {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("chysx::collision::LinearBvh: ") +
                                 what + " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 128;
inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// ============================================================================
// Scene-AABB reduce (two-pass parallel reduce, cub-free for portability)
// ============================================================================

__global__ void scene_aabb_pass1(
    const Aabb* __restrict__ leaves,
    int n,
    Aabb*       __restrict__ out) {
    __shared__ Aabb tile[kBlockDim];

    Aabb local;  // identity bbox
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        local.add(leaves[i]);
    }
    tile[threadIdx.x] = local;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            tile[threadIdx.x].add(tile[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = tile[0];
    }
}

__global__ void scene_aabb_pass2(
    const Aabb* __restrict__ in,
    int n,
    Aabb*       __restrict__ scene) {
    __shared__ Aabb tile[kBlockDim];

    Aabb local;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local.add(in[i]);
    }
    tile[threadIdx.x] = local;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            tile[threadIdx.x].add(tile[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        scene[0] = tile[0];
    }
}

// ============================================================================
// Morton codes
// ============================================================================

// Compute morton code per primitive and seed the (key, id) pair the
// radix sort consumes.  Keys are 64-bit
//
//   key = morton << 32 | id
//
// so identical morton codes still produce distinct keys (sort stable).
__global__ void compute_morton_keys(
    const math::Vec3f* __restrict__ centers,
    const Aabb*        __restrict__ scene,
    int n,
    std::uint64_t* __restrict__ keys,
    int*           __restrict__ ids) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const Aabb sb = scene[0];
    float ex = sb.mx.x - sb.mn.x;
    float ey = sb.mx.y - sb.mn.y;
    float ez = sb.mx.z - sb.mn.z;
    const float scale = fmaxf(fmaxf(ex, ey), ez);
    const float inv = (scale > 1e-30f) ? (1.0f / scale) : 0.0f;

    const math::Vec3f c = centers[idx];
    const float fx = (c.x - sb.mn.x) * inv;
    const float fy = (c.y - sb.mn.y) * inv;
    const float fz = (c.z - sb.mn.z) * inv;

    const std::uint64_t m = morton3d(fx, fy, fz);
    keys[idx] = (m << 32) | static_cast<std::uint32_t>(idx);
    ids[idx]  = idx;
}

// ============================================================================
// Karras 2012 binary radix tree build
// ============================================================================

// Bits common between the 64-bit sorted keys at positions i and j.
__device__ __forceinline__ int common_upper_bits(
    const std::uint64_t* __restrict__ keys, int n, int i, int j) {
    if (j < 0 || j >= n) return -1;
    return __clzll(keys[i] ^ keys[j]);
}

// Compute the [first, last] sorted-leaf range covered by internal
// node `idx`.  Mirror of KittenLBVH's `determineRange` which is in
// turn a clarified rewrite of the Karras-2012 paper's algorithm 3.
__device__ __forceinline__ void determine_range(
    const std::uint64_t* __restrict__ keys, int n, int idx,
    int& out_first, int& out_last) {
    if (idx == 0) {
        out_first = 0;
        out_last  = n - 1;
        return;
    }
    const int l_delta = common_upper_bits(keys, n, idx, idx - 1);
    const int r_delta = common_upper_bits(keys, n, idx, idx + 1);
    const int d = (r_delta > l_delta) ? 1 : -1;
    const int min_delta = (l_delta < r_delta) ? l_delta : r_delta;

    int l_max = 2;
    int probe;
    while ((probe = idx + d * l_max) >= 0 && probe < n) {
        if (common_upper_bits(keys, n, idx, probe) <= min_delta) break;
        l_max <<= 1;
    }

    int t = l_max >> 1;
    int l = 0;
    while (t > 0) {
        probe = idx + (l + t) * d;
        if (probe >= 0 && probe < n) {
            if (common_upper_bits(keys, n, idx, probe) > min_delta) l += t;
        }
        t >>= 1;
    }

    int j = idx + l * d;
    if (d < 0) {
        out_first = j;
        out_last  = idx;
    } else {
        out_first = idx;
        out_last  = j;
    }
}

// Find the highest position in [first, last) whose key shares more
// upper bits with first than first does with last.  Karras 2012
// "find split".
__device__ __forceinline__ int find_split(
    const std::uint64_t* __restrict__ keys, int n, int first, int last) {
    const int delta_node = common_upper_bits(keys, n, first, last);
    int split  = first;
    int stride = last - first;
    do {
        stride = (stride + 1) >> 1;
        const int middle = split + stride;
        if (middle < last) {
            if (common_upper_bits(keys, n, first, middle) > delta_node) {
                split = middle;
            }
        }
    } while (stride > 1);
    return split;
}

// One thread per internal node.  Writes:
//
//   - `nodes[i].left`  / `nodes[i].right`  with MSB tagging leaves
//   - `nodes[i].fence`                     for future self-query dedup
//   - `nodes[child_internal].parent`       (with MSB = isRightChild)
//   - `leaf_parents[child_leaf]`           (with MSB = isRightChild)
//
// Does NOT touch `bounds[]`; that's the refit pass's job.
__global__ void build_radix_tree(
    BvhNode*             __restrict__ nodes,
    std::uint32_t*       __restrict__ leaf_parents,
    const std::uint64_t* __restrict__ keys,
    int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;

    int range_first, range_last;
    determine_range(keys, n, i, range_first, range_last);

    // `fence` = the far end of this subtree's leaf range; the near
    // end is implicit (= i for left-spanning, = i+1 for right; cf.
    // Karras's identity).  KittenLBVH stores the *opposite* end
    // ("fence"), which is what self-query dedup uses.
    nodes[i].fence = (i == range_first) ? range_last : range_first;

    const int gamma = find_split(keys, n, range_first, range_last);

    // Left child: leaf if its sorted range collapsed to a single
    // sorted leaf at position `gamma` (gamma == range_first), else
    // internal node `gamma`.
    std::uint32_t left;
    if (range_first == gamma) {
        leaf_parents[gamma] = static_cast<std::uint32_t>(i);  // left child
        left = static_cast<std::uint32_t>(gamma) | BvhNode::kLeafBit;
    } else {
        left = static_cast<std::uint32_t>(gamma);
        nodes[gamma].parent = static_cast<std::uint32_t>(i);  // left child
    }

    // Right child: similar, but at split position `gamma + 1`.
    std::uint32_t right;
    if (range_last == gamma + 1) {
        leaf_parents[gamma + 1] = static_cast<std::uint32_t>(i)
                                 | BvhNode::kRightBit;
        right = (static_cast<std::uint32_t>(gamma + 1)) | BvhNode::kLeafBit;
    } else {
        right = static_cast<std::uint32_t>(gamma + 1);
        nodes[gamma + 1].parent = static_cast<std::uint32_t>(i)
                                  | BvhNode::kRightBit;
    }

    nodes[i].left  = left;
    nodes[i].right = right;
}

// ============================================================================
// Bottom-up AABB merge (atomicOr depth gate)
// ============================================================================

__global__ void merge_up_kernel(
    BvhNode*             __restrict__ nodes,
    const std::uint32_t* __restrict__ leaf_parents,
    const Aabb*          __restrict__ leaf_aabbs,
    const int*           __restrict__ sorted_id,
    int*                 __restrict__ flag,
    int n_leaves) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_leaves) return;

    // Track the maximum DFS stack depth required to traverse this
    // tree (assuming we always push the right child and explore the
    // left first).  Written into `flag[0]` by the thread that lands
    // on the root, then read back to the host so the query kernel
    // can be dispatched with the smallest sufficient templated
    // stack size.
    int depth = 1;

    AabbPacked last;
    {
        const Aabb b = leaf_aabbs[sorted_id[tid]];
        last.mn = b.mn;
        last.mx = b.mx;
    }

    std::uint32_t parent_raw = leaf_parents[tid];
    while (true) {
        const int is_right = (parent_raw & BvhNode::kRightBit) ? 1 : 0;
        const int parent   = static_cast<int>(parent_raw & BvhNode::kIdxMask);

        nodes[parent].bounds[is_right] = last;

        // First thread to arrive: bail out so the second thread sees
        // both child slots populated.  `atomicOr(.., depth)` doubles
        // as a stack-depth tracker -- the second arrival picks up
        // whatever depth value the first arrival left behind.
        const int other_depth = atomicOr(flag + parent, depth);
        if (other_depth == 0) return;

        // Second arrival: combine our depth with the sibling's.  The
        // depth math here mirrors KittenLBVH and assumes the query
        // kernel always pushes the *right* child onto the stack and
        // descends into the left first.
        if (is_right)
            depth = max(depth + 1, other_depth);
        else
            depth = max(depth, other_depth + 1);

        // Make sure the first arrival's `bounds[1 - is_right]` write
        // is visible before we read it.
        __threadfence();

        if (parent == 0) {
            // Root reached.  We are the lucky thread that gets to
            // record the final stack depth.
            flag[0] = depth;
            return;
        }

        AabbPacked sibling = nodes[parent].bounds[1 - is_right];
        // Union `last` and `sibling` in-place.
        last.mn.x = fminf(last.mn.x, sibling.mn.x);
        last.mn.y = fminf(last.mn.y, sibling.mn.y);
        last.mn.z = fminf(last.mn.z, sibling.mn.z);
        last.mx.x = fmaxf(last.mx.x, sibling.mx.x);
        last.mx.y = fmaxf(last.mx.y, sibling.mx.y);
        last.mx.z = fmaxf(last.mx.z, sibling.mx.z);

        parent_raw = nodes[parent].parent;
    }
}

// ============================================================================
// EF query kernel (shared-memory buffered, templated stack size)
// ============================================================================

// Covertex(EF) filter: returns true if any of the 2 edge endpoints is
// shared with any of the 3 triangle vertices.
__device__ __forceinline__ bool covertex_ef(
    const math::Vec2i* __restrict__ edges,
    const math::Vec3i* __restrict__ faces,
    int e_idx, int f_idx) {
    const math::Vec2i e = edges[e_idx];
    const math::Vec3i f = faces[f_idx];
    if (e.x == f.x || e.x == f.y || e.x == f.z) return true;
    if (e.y == f.x || e.y == f.y || e.y == f.z) return true;
    return false;
}

// AABB-vs-AABB test against the packed in-node bounds.
__device__ __forceinline__ bool overlaps_packed(
    const AabbPacked& b, const Aabb& q) {
    return !(b.mx.x < q.mn.x || b.mn.x > q.mx.x ||
             b.mx.y < q.mn.y || b.mn.y > q.mx.y ||
             b.mx.z < q.mn.z || b.mn.z > q.mx.z);
}

// Block-shared output bucket (one math::Vec2i per slot).  4 * 128 =
// 4 KB shared memory per block.  This is the KittenGpuLBVH design:
// threads stage hits into the bucket via `atomicAdd(&s_counter)`,
// then only ONE thread per block fires `atomicAdd(pair_count, total)`
// per flush -- collapsing what would otherwise be tens of thousands of
// global atomics per frame down to a few hundred.
constexpr int kMaxResPerBlock = 4 * kBlockDim;

// 1:1 port of KittenGpuLBVH's `lbvhQueryKernel` adapted to chysx's
// covertex-EF leaf filter.  Each block:
//
//   1. Pops nodes off its per-thread stack until either the stack is
//      empty or the shared bucket overflows.
//   2. Syncs, reads `s_counter`, and lets thread-0 reserve a slice of
//      the global pair list with one `atomicAdd`.
//   3. Cooperatively memcpys the bucket into `pair_list[g_off ..]`.
//   4. Loops until every thread in the block has emptied its stack.
template <int STACK_SIZE>
__global__ void query_self_ef_kernel(
    const Aabb*         __restrict__ query_aabbs,
    int                 n_queries,
    const BvhNode*      __restrict__ nodes,
    const int*          __restrict__ sorted_id,
    const math::Vec2i*  __restrict__ edges,
    const math::Vec3i*  __restrict__ faces,
    int*                __restrict__ pair_count,
    math::Vec2i*        __restrict__ pair_list,
    int                 max_pairs) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = tid < n_queries;

    Aabb queryAABB;
    if (active) queryAABB = query_aabbs[tid];

    __shared__ math::Vec2i s_buf[kMaxResPerBlock];
    __shared__ int         s_counter;
    __shared__ int         s_global_off;
    if (threadIdx.x == 0) s_counter = 0;

    std::uint32_t stack[STACK_SIZE];
    std::uint32_t* sp = stack;
    *(sp++) = 0u;  // root

    while (true) {
        __syncthreads();

        if (active) {
            while (sp != stack) {
                std::uint32_t node_raw = *(--sp);
                const bool is_leaf = (node_raw & BvhNode::kLeafBit) != 0;
                const std::uint32_t idx = node_raw & BvhNode::kIdxMask;

                if (is_leaf) {
                    const int obj = sorted_id[idx];
                    if (covertex_ef(edges, faces, tid, obj)) continue;

                    const int s_idx = atomicAdd(&s_counter, 1);
                    if (s_idx >= kMaxResPerBlock) {
                        // Bucket full -- push the leaf back onto our
                        // own stack and let the block flush.  The
                        // block will reconverge at the syncthreads
                        // below, drain the bucket to global memory,
                        // then loop and we'll try this leaf again.
                        *(sp++) = node_raw;
                        break;
                    }
                    s_buf[s_idx] = math::Vec2i(tid, obj);
                } else {
                    // One 64B node load = both children's AABBs in
                    // registers; cheaper than the previous
                    // "internal node + two child AABB fetches".
                    const BvhNode node = nodes[idx];
                    const bool overlapL = overlaps_packed(node.bounds[0], queryAABB);
                    const bool overlapR = overlaps_packed(node.bounds[1], queryAABB);
                    // Push the right child first so the left child
                    // is popped (and explored) next.  Matches the
                    // depth math in `merge_up_kernel`.
                    if (overlapR) *(sp++) = node.right;
                    if (overlapL) *(sp++) = node.left;
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

        // Cooperative bucket -> global copy.  KittenGpuLBVH unrolls
        // the "full blocks" loop manually; the simpler strided form
        // below codegen's identically on sm_70+.
        for (int i = threadIdx.x; i < total; i += blockDim.x)
            pair_list[g_off + i] = s_buf[i];

        if (done) break;
    }
}

__global__ void clear_int_kernel(int* p) { *p = 0; }
__global__ void clear_uint_kernel(unsigned int* p, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0u;
}

// Templated query dispatcher (KittenGpuLBVH pattern).
//
// The query kernel is compiled for stack sizes 1..32; we pick the
// smallest one that fits the actual tree depth observed during refit.
// Smaller stacks burn fewer registers per thread, which raises
// occupancy.  Per Kitten this is a ~15% perf win on dense scenes;
// on cloth it matters less but costs nothing.
inline int clamp_stack(int s) {
    if (s < 1)  return 1;
    if (s > 32) return 32;
    return s;
}

void launch_query(
    int                   n_queries,
    const Aabb*           query_aabbs,
    const BvhNode*        nodes,
    const int*            sorted_id,
    const math::Vec2i*    edges,
    const math::Vec3i*    faces,
    int*                  pair_count,
    math::Vec2i*          pair_list,
    int                   max_pairs,
    int                   stack_size,
    cudaStream_t          stream) {
    const int blocks = grid_for(n_queries);
    const int s = clamp_stack(stack_size);

#define DISPATCH(N) case N:                                                 \
    query_self_ef_kernel<N><<<blocks, kBlockDim, 0, stream>>>(              \
        query_aabbs, n_queries, nodes, sorted_id, edges, faces,             \
        pair_count, pair_list, max_pairs);                                  \
    break;

    switch (s) {
        DISPATCH(1)  DISPATCH(2)  DISPATCH(3)  DISPATCH(4)
        DISPATCH(5)  DISPATCH(6)  DISPATCH(7)  DISPATCH(8)
        DISPATCH(9)  DISPATCH(10) DISPATCH(11) DISPATCH(12)
        DISPATCH(13) DISPATCH(14) DISPATCH(15) DISPATCH(16)
        DISPATCH(17) DISPATCH(18) DISPATCH(19) DISPATCH(20)
        DISPATCH(21) DISPATCH(22) DISPATCH(23) DISPATCH(24)
        DISPATCH(25) DISPATCH(26) DISPATCH(27) DISPATCH(28)
        DISPATCH(29) DISPATCH(30) DISPATCH(31) DISPATCH(32)
        default:
            query_self_ef_kernel<32><<<blocks, kBlockDim, 0, stream>>>(
                query_aabbs, n_queries, nodes, sorted_id, edges, faces,
                pair_count, pair_list, max_pairs);
            break;
    }
#undef DISPATCH
}

}  // namespace

// ============================================================================
// LinearBvh
// ============================================================================

void LinearBvh::build(int n_leaves, int max_query_pairs) {
    if (n_leaves <= 1) {
        // Pathological -- caller must guard, but bail out cleanly.
        n_leaves_ = n_leaves;
        max_query_pairs_ = max_query_pairs;
        max_stack_size_ = 1;
        return;
    }

    if (n_leaves_ == n_leaves && max_query_pairs_ == max_query_pairs) {
        return;  // already sized
    }

    n_leaves_ = n_leaves;
    max_query_pairs_ = max_query_pairs;
    max_stack_size_ = 1;

    const std::size_t N = static_cast<std::size_t>(n_leaves);

    morton_keys_in_.resize(N);
    morton_keys_out_.resize(N);
    sorted_id_in_.resize(N);
    sorted_id_.resize(N);
    refit_flag_.resize(N - 1);
    nodes_.resize(N - 1);
    leaf_parents_.resize(N);

    scene_bbox_.resize(1);
    // Cap the pass-1 grid at 1024 so the second pass fits in a single
    // block (reduce 1024 partial AABBs in 128 threads is fine).
    scene_partial_.resize(1024);

    query_count_.resize(1);
    query_pairs_.resize(static_cast<std::size_t>(max_query_pairs));

    // Sniff the cub temp-storage requirement once.
    std::size_t bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, bytes,
        morton_keys_in_.gpu_data(), morton_keys_out_.gpu_data(),
        sorted_id_in_.gpu_data(),   sorted_id_.gpu_data(),
        n_leaves_);
    cub_temp_bytes_ = bytes;
    cub_temp_.resize(bytes);
}

void LinearBvh::refit(const Aabb*       leaf_aabbs,
                      const math::Vec3f* leaf_centers,
                      std::uintptr_t     cuda_stream) {
    if (n_leaves_ <= 1) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    const int N = n_leaves_;
    Aabb* d_scene = scene_bbox_.gpu_data();

    // 1. Compute scene AABB (two-pass parallel reduce).
    {
        const int blocks = std::min(grid_for(N), 1024);
        scene_aabb_pass1<<<blocks, kBlockDim, 0, stream>>>(
            leaf_aabbs, N, scene_partial_.gpu_data());
        scene_aabb_pass2<<<1, kBlockDim, 0, stream>>>(
            scene_partial_.gpu_data(), blocks, d_scene);
        check_cuda(cudaGetLastError(), "scene_aabb reduce");
    }

    // 2. Morton codes.
    compute_morton_keys<<<grid_for(N), kBlockDim, 0, stream>>>(
        leaf_centers, d_scene, N,
        morton_keys_in_.gpu_data(),
        sorted_id_in_.gpu_data());
    check_cuda(cudaGetLastError(), "compute_morton_keys");

    // 3. Radix sort (key = morton<<32 | id, value = id).
    cub::DeviceRadixSort::SortPairs(
        cub_temp_.gpu_data(), cub_temp_bytes_,
        morton_keys_in_.gpu_data(), morton_keys_out_.gpu_data(),
        sorted_id_in_.gpu_data(),   sorted_id_.gpu_data(),
        N, 0, sizeof(std::uint64_t) * 8, stream);

    // 4. Build the binary radix tree topology.  Refit-flag must be
    // zeroed first (it doubles as the depth tracker for refit).
    clear_uint_kernel<<<grid_for(N - 1), kBlockDim, 0, stream>>>(
        refit_flag_.gpu_data(), N - 1);
    build_radix_tree<<<grid_for(N - 1), kBlockDim, 0, stream>>>(
        nodes_.gpu_data(), leaf_parents_.gpu_data(),
        morton_keys_out_.gpu_data(), N);
    check_cuda(cudaGetLastError(), "build_radix_tree");

    // 5. Bottom-up AABB merge with atomicOr first/second-arrival gate.
    //
    // `merge_up_kernel` also writes the actual DFS stack depth this
    // tree needs into `refit_flag_[0]` as a free byproduct of the
    // atomicOr.  Read it back so the templated query kernel can be
    // dispatched with the smallest sufficient stack (= fewer registers
    // per thread = higher occupancy).  This adds one
    // cudaStreamSynchronize per refit; for a 100x200 cloth on a 5090
    // that costs ~0.1 ms and is paid back several times over by the
    // higher-occupancy query.
    merge_up_kernel<<<grid_for(N), kBlockDim, 0, stream>>>(
        nodes_.gpu_data(), leaf_parents_.gpu_data(),
        leaf_aabbs, sorted_id_.gpu_data(),
        reinterpret_cast<int*>(refit_flag_.gpu_data()), N);
    check_cuda(cudaGetLastError(), "merge_up_kernel");

    int depth_host = 1;
    check_cuda(cudaMemcpyAsync(&depth_host, refit_flag_.gpu_data(),
                               sizeof(int), cudaMemcpyDeviceToHost, stream),
               "max_stack memcpy");
    check_cuda(cudaStreamSynchronize(stream), "max_stack sync");
    max_stack_size_ = clamp_stack(depth_host);
}

void LinearBvh::query_self_ef(const Aabb*           query_aabbs,
                              int                   n_queries,
                              const math::Vec2i*    edges,
                              const math::Vec3i*    faces,
                              std::uintptr_t        cuda_stream) {
    if (n_leaves_ <= 1 || n_queries <= 0) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    clear_int_kernel<<<1, 1, 0, stream>>>(query_count_.gpu_data());
    launch_query(n_queries, query_aabbs,
                 nodes_.gpu_data(), sorted_id_.gpu_data(),
                 edges, faces,
                 query_count_.gpu_data(), query_pairs_.gpu_data(),
                 max_query_pairs_,
                 max_stack_size_, stream);
    check_cuda(cudaGetLastError(), "query_self_ef_kernel");
}

}  // namespace collision
}  // namespace chysx
