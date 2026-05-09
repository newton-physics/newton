// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::collision::LinearBvh.
//
// Karras 2012 binary-radix-tree build over Morton-sorted leaves, with
// a stack-based traversal kernel for self-EF queries.  Logic ported
// from cuda-cloth's `LinearBvh.cu`; the two non-trivial dependencies
// (a global radix sort and a global AABB reduce) are filled in with
// cub instead of thrust so we don't have to add a new build dep.

#include "lbvh.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
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
// Kernels
// ============================================================================

// Reduce: per-block AABB union, written into `out` (length `gridDim.x`).
// The host launches a second pass that reduces those into `final`.
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

// Compute morton code per primitive and seed the (key, id) pair that
// the radix sort consumes.  Keys are 64-bit
//   key = morton << 32 | id
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

// Karras 2012: one thread per internal node, builds the binary radix
// tree.  Also seeds leaf AABBs into the tail of `node_aabbs` so the
// subsequent refit pass can reduce upward in a single sweep.
__device__ __forceinline__ int delta(
    const std::uint64_t* __restrict__ keys, int n, int i, int j) {
    if (j < 0 || j >= n) return -1;
    return __clzll(keys[i] ^ keys[j]);
}

__global__ void build_radix_tree(
    BvhNode*       __restrict__ nodes,
    Aabb*          __restrict__ node_aabbs,
    const Aabb*    __restrict__ leaf_aabbs_in,
    const std::uint64_t* __restrict__ keys,
    const int*     __restrict__ sorted_id,
    int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Place the i-th sorted leaf's AABB into the leaf range [n-1, 2n-1).
    node_aabbs[i + n - 1] = leaf_aabbs_in[sorted_id[i]];

    if (i >= n - 1) return;  // only the first n-1 threads build internals

    const int dright = delta(keys, n, i, i + 1);
    const int dleft  = delta(keys, n, i, i - 1);
    const int d = (dright > dleft) ? 1 : -1;
    const int min_delta = delta(keys, n, i, i - d);

    int lmax = 2;
    while (delta(keys, n, i, i + lmax * d) > min_delta) lmax *= 2;

    int len = 0;
    for (int t = lmax / 2; t > 0; t >>= 1) {
        if (delta(keys, n, i, i + (len + t) * d) > min_delta) len += t;
    }
    const int j = i + len * d;

    int s = 0;
    const int delta_node = delta(keys, n, i, j);
    int t = (len + 1) / 2;
    while (true) {
        if (delta(keys, n, i, i + (s + t) * d) > delta_node) s += t;
        if (t == 1) break;
        t = (t + 1) / 2;
    }
    const int gamma = i + s * d + min(d, 0);

    const int lo = min(i, j);
    const int hi = max(i, j);
    const int left_idx  = (lo == gamma)     ? (gamma + n - 1)     : gamma;
    const int right_idx = (hi == gamma + 1) ? (gamma + n)         : (gamma + 1);

    nodes[i].left  = left_idx;
    nodes[i].right = right_idx;
    nodes[left_idx].parent  = i;
    nodes[right_idx].parent = i;
}

// Bottom-up AABB refit.  Each leaf walks toward the root; the second
// thread to arrive at each internal node is the one that does the
// merge (atomicCAS gate).
__global__ void refit_aabbs(
    Aabb*           __restrict__ node_aabbs,
    const BvhNode*  __restrict__ nodes,
    unsigned int*   __restrict__ flag,
    int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int idx = nodes[i + n - 1].parent;
    while (idx != -1 && idx != BvhNode::kInvalid) {
        const unsigned int old = atomicCAS(&flag[idx], 0u, 1u);
        if (old == 0u) return;  // first arrival; second arrival merges

        const int  l = nodes[idx].left;
        const int  r = nodes[idx].right;
        Aabb merged = node_aabbs[l];
        merged.add(node_aabbs[r]);
        node_aabbs[idx] = merged;

        idx = nodes[idx].parent;
    }
}

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

// One thread per query AABB.  Stack-based BVH traversal; on each leaf
// hit emit (query_id, leaf_id) into the global pair list, dropping
// covertex hits.
__global__ void query_self_ef_kernel(
    const Aabb*         __restrict__ query_aabbs,
    int                 n_queries,
    const BvhNode*      __restrict__ nodes,
    const Aabb*         __restrict__ node_aabbs,
    const int*          __restrict__ sorted_id,
    int                 n_leaves,
    const math::Vec2i*  __restrict__ edges,
    const math::Vec3i*  __restrict__ faces,
    int*                __restrict__ pair_count,
    math::Vec2i*        __restrict__ pair_list,
    int                 max_pairs) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_queries) return;

    const Aabb queryAABB = query_aabbs[q];
    int stack[64];
    int sp = 0;
    int idx = 0;  // root

    while (idx != -1) {
        const int idxL = nodes[idx].left;
        const int idxR = nodes[idx].right;
        const bool overlapL = queryAABB.overlaps(node_aabbs[idxL]);
        const bool overlapR = queryAABB.overlaps(node_aabbs[idxR]);
        const bool leafL = (idxL >= n_leaves - 1);
        const bool leafR = (idxR >= n_leaves - 1);

        if (overlapL && leafL) {
            const int obj = sorted_id[idxL - (n_leaves - 1)];
            if (!covertex_ef(edges, faces, q, obj)) {
                const int pid = atomicAdd(pair_count, 1);
                if (pid < max_pairs) {
                    pair_list[pid] = math::Vec2i(q, obj);
                }else{
                    printf("pid >= max_pairs\n");
                }
            }
        }
        if (overlapR && leafR) {
            const int obj = sorted_id[idxR - (n_leaves - 1)];
            if (!covertex_ef(edges, faces, q, obj)) {
                const int pid = atomicAdd(pair_count, 1);
                if (pid < max_pairs) {
                    pair_list[pid] = math::Vec2i(q, obj);
                }else{
                    printf("pid >= max_pairs\n");
                }
            }
        }

        const bool traverseL = overlapL && !leafL;
        const bool traverseR = overlapR && !leafR;
        if (!traverseL && !traverseR) {
            idx = (sp == 0) ? -1 : stack[--sp];
        } else {
            idx = traverseL ? idxL : idxR;
            if (traverseL && traverseR) {
                if (sp < 64) stack[sp++] = idxR;
            }
        }
    }
}

__global__ void clear_int_kernel(int* p) { *p = 0; }

__global__ void clear_uint_kernel(unsigned int* p, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0u;
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
        return;
    }

    if (n_leaves_ == n_leaves && max_query_pairs_ == max_query_pairs) {
        return;  // already sized
    }

    n_leaves_ = n_leaves;
    max_query_pairs_ = max_query_pairs;

    const std::size_t N = static_cast<std::size_t>(n_leaves);
    const std::size_t total_nodes = 2 * N - 1;

    morton_keys_in_.resize(N);
    morton_keys_out_.resize(N);
    sorted_id_in_.resize(N);
    sorted_id_.resize(N);
    refit_flag_.resize(N - 1);
    nodes_.resize(total_nodes);
    node_aabbs_.resize(total_nodes);

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

    // Mark every node parent (and child links) as -1.  Subsequent
    // build_radix_tree calls overwrite all of these except the root's
    // parent, which must remain -1 so refit_aabbs knows when to stop.
    check_cuda(cudaMemset(nodes_.gpu_data(), -1,
                          total_nodes * sizeof(BvhNode)),
               "memset nodes");
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

    // 4. Build the binary radix tree.  Refit-flag must be zeroed first.
    clear_uint_kernel<<<grid_for(N - 1), kBlockDim, 0, stream>>>(
        refit_flag_.gpu_data(), N - 1);
    build_radix_tree<<<grid_for(N), kBlockDim, 0, stream>>>(
        nodes_.gpu_data(), node_aabbs_.gpu_data(),
        leaf_aabbs, morton_keys_out_.gpu_data(),
        sorted_id_.gpu_data(), N);
    check_cuda(cudaGetLastError(), "build_radix_tree");

    // 5. Bottom-up AABB merge.
    refit_aabbs<<<grid_for(N), kBlockDim, 0, stream>>>(
        node_aabbs_.gpu_data(), nodes_.gpu_data(),
        refit_flag_.gpu_data(), N);
    check_cuda(cudaGetLastError(), "refit_aabbs");
}

void LinearBvh::query_self_ef(const Aabb*           query_aabbs,
                              int                   n_queries,
                              const math::Vec2i*    edges,
                              const math::Vec3i*    faces,
                              std::uintptr_t        cuda_stream) {
    if (n_leaves_ <= 1 || n_queries <= 0) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    clear_int_kernel<<<1, 1, 0, stream>>>(query_count_.gpu_data());
    query_self_ef_kernel<<<grid_for(n_queries), kBlockDim, 0, stream>>>(
        query_aabbs, n_queries,
        nodes_.gpu_data(), node_aabbs_.gpu_data(),
        sorted_id_.gpu_data(), n_leaves_,
        edges, faces,
        query_count_.gpu_data(),
        query_pairs_.gpu_data(),
        max_query_pairs_);
    check_cuda(cudaGetLastError(), "query_self_ef_kernel");
}

}  // namespace collision
}  // namespace chysx
