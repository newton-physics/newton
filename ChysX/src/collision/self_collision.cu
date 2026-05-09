// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::collision::SelfCollisionDetector.
//
// Pipeline (matches cuda-cloth's SelfCollisionBvhDcd):
//
//   build face/edge AABBs   (kernel)
//   refit face LBVH         (LinearBvh::refit)
//   self-EF broadphase      (LinearBvh::query_self_ef)
//   cull EF -> VF + EE      (cull_ef_to_vfee_kernel)
//   cull adjacent EE        (cull_ee_adjacent_kernel)
//
// Output is the unified `(Vec4i, ContactWeights)` stream consumed by
// `ContactSpMVOp` and `SelfCollisionConstraint`.

#include "self_collision.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace chysx {
namespace collision {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("chysx::collision::SelfCollisionDetector: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 128;
inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// ============================================================================
// Geometry primitives (host+device)
// ============================================================================

// Closest point on triangle (a, b, c) to query p.  Returns the point
// in `out_q` and barycentric weights (w_a, w_b, w_c) that sum to 1.
//
// Adapted from Ericson, Real-Time Collision Detection, Sec. 5.1.5.
__device__ __forceinline__ void closest_point_tri(
    const math::Vec3f& p,
    const math::Vec3f& a, const math::Vec3f& b, const math::Vec3f& c,
    math::Vec3f& out_q, math::Vec3f& out_w) {
    const math::Vec3f ab = b - a;
    const math::Vec3f ac = c - a;
    const math::Vec3f ap = p - a;
    const float d1 = ab.x*ap.x + ab.y*ap.y + ab.z*ap.z;
    const float d2 = ac.x*ap.x + ac.y*ap.y + ac.z*ap.z;
    if (d1 <= 0.0f && d2 <= 0.0f) {
        out_q = a; out_w = math::Vec3f(1.0f, 0.0f, 0.0f); return;
    }

    const math::Vec3f bp = p - b;
    const float d3 = ab.x*bp.x + ab.y*bp.y + ab.z*bp.z;
    const float d4 = ac.x*bp.x + ac.y*bp.y + ac.z*bp.z;
    if (d3 >= 0.0f && d4 <= d3) {
        out_q = b; out_w = math::Vec3f(0.0f, 1.0f, 0.0f); return;
    }

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        out_q = a + v * ab;
        out_w = math::Vec3f(1.0f - v, v, 0.0f);
        return;
    }

    const math::Vec3f cp = p - c;
    const float d5 = ab.x*cp.x + ab.y*cp.y + ab.z*cp.z;
    const float d6 = ac.x*cp.x + ac.y*cp.y + ac.z*cp.z;
    if (d6 >= 0.0f && d5 <= d6) {
        out_q = c; out_w = math::Vec3f(0.0f, 0.0f, 1.0f); return;
    }

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        out_q = a + w * ac;
        out_w = math::Vec3f(1.0f - w, 0.0f, w);
        return;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        out_q = b + w * (c - b);
        out_w = math::Vec3f(0.0f, 1.0f - w, w);
        return;
    }

    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    out_q = a + ab * v + ac * w;
    out_w = math::Vec3f(1.0f - v - w, v, w);
}

// Closest points between two segments p0-p1 and q0-q1.
// `s, t` get the line parameters in [0, 1] and `cp`, `cq` the points.
//
// Numerical fallback: parallel segments collapse to s = 0, then `t` is
// solved on q from p[s].
__device__ __forceinline__ void closest_point_seg_seg(
    const math::Vec3f& p0, const math::Vec3f& p1,
    const math::Vec3f& q0, const math::Vec3f& q1,
    float& s, float& t,
    math::Vec3f& cp, math::Vec3f& cq) {
    const math::Vec3f d1 = p1 - p0;
    const math::Vec3f d2 = q1 - q0;
    const math::Vec3f r  = p0 - q0;
    const float a = d1.x*d1.x + d1.y*d1.y + d1.z*d1.z;
    const float e = d2.x*d2.x + d2.y*d2.y + d2.z*d2.z;
    const float f = d2.x*r.x  + d2.y*r.y  + d2.z*r.z;

    const float kEps = 1e-20f;
    if (a <= kEps && e <= kEps) {
        s = 0.0f; t = 0.0f; cp = p0; cq = q0; return;
    }
    if (a <= kEps) {
        s = 0.0f;
        t = f / e;
        t = (t < 0.0f) ? 0.0f : (t > 1.0f ? 1.0f : t);
    } else {
        const float c = d1.x*r.x + d1.y*r.y + d1.z*r.z;
        if (e <= kEps) {
            t = 0.0f;
            s = -c / a;
            s = (s < 0.0f) ? 0.0f : (s > 1.0f ? 1.0f : s);
        } else {
            const float b = d1.x*d2.x + d1.y*d2.y + d1.z*d2.z;
            const float denom = a * e - b * b;
            if (denom != 0.0f) {
                s = (b * f - c * e) / denom;
                s = (s < 0.0f) ? 0.0f : (s > 1.0f ? 1.0f : s);
            } else {
                s = 0.0f;
            }
            t = (b * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = -c / a;
                s = (s < 0.0f) ? 0.0f : (s > 1.0f ? 1.0f : s);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = (b - c) / a;
                s = (s < 0.0f) ? 0.0f : (s > 1.0f ? 1.0f : s);
            }
        }
    }
    cp = p0 + d1 * s;
    cq = q0 + d2 * t;
}

// ============================================================================
// AABB / center kernels
// ============================================================================

__global__ void face_aabb_center_kernel(
    const math::Vec3i* __restrict__ faces, int n_faces,
    const math::Vec3f* __restrict__ pos,
    float thickness,
    Aabb*        __restrict__ aabbs,
    math::Vec3f* __restrict__ centers) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_faces) return;
    const math::Vec3i f = faces[idx];
    const math::Vec3f a = pos[f.x];
    const math::Vec3f b = pos[f.y];
    const math::Vec3f c = pos[f.z];
    Aabb box;
    box.set(a, b);
    box.add(c);
    box.enlarge(thickness);
    aabbs[idx] = box;
    centers[idx] = math::Vec3f(0.5f * (box.mn.x + box.mx.x),
                               0.5f * (box.mn.y + box.mx.y),
                               0.5f * (box.mn.z + box.mx.z));
}

__global__ void edge_aabb_kernel(
    const math::Vec2i* __restrict__ edges, int n_edges,
    const math::Vec3f* __restrict__ pos,
    float thickness,
    Aabb* __restrict__ aabbs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_edges) return;
    const math::Vec2i e = edges[idx];
    Aabb box;
    box.set(pos[e.x], pos[e.y]);
    box.enlarge(thickness);
    aabbs[idx] = box;
}

__global__ void clear_int_kernel(int* p) { *p = 0; }

// ============================================================================
// Narrow-phase kernels
// ============================================================================

// `ef_pairs[i] = (edge_id, face_id)`.  For each candidate, run one VF
// test (via vert_in_edge[edge_id]) and up to three EE tests against
// the face's edges.  Each successful test atomically appends one
// 4-particle contact to the unified output stream.
__global__ void cull_ef_to_vfee_kernel(
    const math::Vec3f*  __restrict__ pos,
    const math::Vec2i*  __restrict__ edges,
    const math::Vec3i*  __restrict__ faces,
    const int*          __restrict__ vert_in_edge,
    const math::Vec3i*  __restrict__ edge_in_face,
    const math::Vec2i*  __restrict__ ef_pairs,
    const int*          __restrict__ ef_count,
    int                  ef_max,
    float                thickness,
    int                  max_contacts,
    int*                 __restrict__ out_count,
    math::Vec4i*         __restrict__ out_pairs,
    ContactWeights*      __restrict__ out_weights) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_raw = *ef_count;
    const int n = (n_raw < ef_max) ? n_raw : ef_max;
    if (idx >= n) return;

    const math::Vec2i ef = ef_pairs[idx];
    const int eid = ef.x;
    const int fid = ef.y;

    // ---- VF: the edge's "owner" vertex against the face --------------

    const int vid = vert_in_edge[eid];
    if (vid >= 0) {
        const math::Vec3i f = faces[fid];
        // Skip if the owner vertex is one of the face's verts (1-ring).
        if (vid != f.x && vid != f.y && vid != f.z) {
            const math::Vec3f v  = pos[vid];
            const math::Vec3f t0 = pos[f.x];
            const math::Vec3f t1 = pos[f.y];
            const math::Vec3f t2 = pos[f.z];
            math::Vec3f q, w;
            closest_point_tri(v, t0, t1, t2, q, w);
            const math::Vec3f diff = v - q;
            const float d = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
            if (d > 1e-12f && d < thickness &&
                w.x > -1e-3f && w.y > -1e-3f && w.z > -1e-3f) {
                const float inv = 1.0f / d;
                const float nx = diff.x * inv;
                const float ny = diff.y * inv;
                const float nz = diff.z * inv;
                const int cid = atomicAdd(out_count, 1);
                if (cid < max_contacts) {
                    out_pairs[cid] = math::Vec4i(vid, f.x, f.y, f.z);
                    ContactWeights cw;
                    cw.w0 = 1.0f;  cw.w1 = -w.x;  cw.w2 = -w.y;  cw.w3 = -w.z;
                    cw.nx = nx;    cw.ny = ny;    cw.nz = nz;
                    cw.depth = thickness - d;
                    out_weights[cid] = cw;
                }
            }
        }
    }

    // ---- EE: input edge against each of the face's 3 edges ----------

    const math::Vec3i e3 = edge_in_face[fid];
    const math::Vec2i ea = edges[eid];

    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int oeid = e3.data[j];
        if (oeid < 0)   continue;
        if (oeid == eid) continue;
        // Avoid duplicate EE: only emit when input edge id < other edge id.
        if (oeid >= eid) continue;

        const math::Vec2i eb = edges[oeid];
        // Skip 1-ring shared-vertex (degenerate EE).
        if (ea.x == eb.x || ea.x == eb.y ||
            ea.y == eb.x || ea.y == eb.y) continue;

        const math::Vec3f p0 = pos[ea.x];
        const math::Vec3f p1 = pos[ea.y];
        const math::Vec3f q0 = pos[eb.x];
        const math::Vec3f q1 = pos[eb.y];
        float s, t;
        math::Vec3f cp, cq;
        closest_point_seg_seg(p0, p1, q0, q1, s, t, cp, cq);
        const math::Vec3f diff = cp - cq;
        const float d = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
        if (d > 1e-12f && d < thickness &&
            s > -1e-3f && s < 1.003f && t > -1e-3f && t < 1.003f) {
            const float inv = 1.0f / d;
            const int cid = atomicAdd(out_count, 1);
            if (cid < max_contacts) {
                out_pairs[cid] = math::Vec4i(ea.x, ea.y, eb.x, eb.y);
                // EE weight, byte-for-byte cuda-cloth convention:
                //   weights = (s, 1-s, -s, s-1)
                // Pairs the same `s` parameter to both edges (asymmetric
                // closest-point parameterisation; not the textbook
                // symmetric (1-s, s, -(1-t), -t) form).  See the
                // CullEF2VFEE / CullEEAdjacent kernels in cuda-cloth's
                // SelfCollisionBvhDcd.cu for the original write.
                ContactWeights cw;
                cw.w0 = s;
                cw.w1 = 1.0f - s;
                cw.w2 = -s;
                cw.w3 = s - 1.0f;
                cw.nx = diff.x * inv;
                cw.ny = diff.y * inv;
                cw.nz = diff.z * inv;
                cw.depth = thickness - d;
                out_weights[cid] = cw;
            }
        }
    }
}

// Adjacent-EE: run EE between the precomputed 1-ring-adjacent edge
// pairs (sharing exactly one vertex).  These pairs the AABB-driven
// broadphase wouldn't separate cleanly, so we test them unconditionally.
__global__ void cull_ee_adjacent_kernel(
    const math::Vec3f*  __restrict__ pos,
    const math::Vec4i*  __restrict__ adj_pairs,
    int                  n_adj,
    float                thickness,
    int                  max_contacts,
    int*                 __restrict__ out_count,
    math::Vec4i*         __restrict__ out_pairs,
    ContactWeights*      __restrict__ out_weights) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_adj) return;
    const math::Vec4i ids = adj_pairs[idx];

    const math::Vec3f p0 = pos[ids.x];
    const math::Vec3f p1 = pos[ids.y];
    const math::Vec3f q0 = pos[ids.z];
    const math::Vec3f q1 = pos[ids.w];
    float s, t;
    math::Vec3f cp, cq;
    closest_point_seg_seg(p0, p1, q0, q1, s, t, cp, cq);
    const math::Vec3f diff = cp - cq;
    const float d = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);

    // Same tolerance / weight convention as the EF->VFEE EE branch
    // and as cuda-cloth's `CullEEAdjacent`.  The MeshTopology builder
    // already filters out the degenerate "edges share a vertex" pairs,
    // so we don't need a stricter interior filter here.
    if (d > 1e-12f && d < thickness &&
        s > -1e-3f && s < 1.003f &&
        t > -1e-3f && t < 1.003f) {
        const float inv = 1.0f / d;
        const int cid = atomicAdd(out_count, 1);
        if (cid < max_contacts) {
            out_pairs[cid] = ids;
            ContactWeights cw;
            cw.w0 = s;
            cw.w1 = 1.0f - s;
            cw.w2 = -s;
            cw.w3 = s - 1.0f;
            cw.nx = diff.x * inv;
            cw.ny = diff.y * inv;
            cw.nz = diff.z * inv;
            cw.depth = thickness - d;
            out_weights[cid] = cw;
        }
    }
}

}  // namespace

// ============================================================================
// SelfCollisionDetector
// ============================================================================

void SelfCollisionDetector::reserve(int max_contacts, int max_ef_candidates) {
    if (max_contacts < 1) max_contacts = 1;
    if (max_ef_candidates < 1) max_ef_candidates = 1;
    max_contacts_ = max_contacts;
    max_ef_candidates_ = max_ef_candidates;

    pairs_.resize(static_cast<std::size_t>(max_contacts));
    weights_.resize(static_cast<std::size_t>(max_contacts));
    if (count_.gpu_size() == 0) count_.resize(1);

    // If a topology is already bound, re-bind it so the BVH picks up
    // the new EF-candidate cap.  (`bind_topology` is idempotent for
    // unchanged sizes; the LBVH internally no-ops when nothing grew.)
    if (topology_ != nullptr && topology_->valid()) {
        bind_topology(topology_);
    }
}

void SelfCollisionDetector::bind_topology(const MeshTopology* topology) {
    topology_ = topology;
    if (topology_ == nullptr || !topology_->valid()) return;
    if (max_ef_candidates_ <= 0) {
        // Caps not configured yet -- defer the BVH allocation until
        // `reserve(...)` runs, at which point `bind_topology` is
        // re-invoked from there.
        return;
    }

    const int n_faces = topology_->n_faces();
    const int n_edges = topology_->n_edges();

    face_aabbs_.resize(static_cast<std::size_t>(n_faces));
    face_centers_.resize(static_cast<std::size_t>(n_faces));
    edge_aabbs_.resize(static_cast<std::size_t>(n_edges));

    bvh_.build(n_faces, max_ef_candidates_);
}

void SelfCollisionDetector::detect(DeviceSpan<math::Vec3f> positions,
                                   float thickness,
                                   std::uintptr_t cuda_stream) {
    if (topology_ == nullptr || !topology_->valid()) return;
    if (max_contacts_ <= 0) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    const int n_faces = topology_->n_faces();
    const int n_edges = topology_->n_edges();
    const int n_adj   = topology_->n_adj_ee();

    // Reset the global contact counter.
    clear_int_kernel<<<1, 1, 0, stream>>>(count_.gpu_data());

    // 1. Build face/edge AABBs and face centers from current positions.
    face_aabb_center_kernel<<<grid_for(n_faces), kBlockDim, 0, stream>>>(
        topology_->faces().gpu_data(), n_faces,
        positions.data(), thickness,
        face_aabbs_.gpu_data(), face_centers_.gpu_data());
    check_cuda(cudaGetLastError(), "face_aabb_center_kernel");

    edge_aabb_kernel<<<grid_for(n_edges), kBlockDim, 0, stream>>>(
        topology_->edges().gpu_data(), n_edges,
        positions.data(), thickness,
        edge_aabbs_.gpu_data());
    check_cuda(cudaGetLastError(), "edge_aabb_kernel");

    // 2. LBVH refit on faces.
    bvh_.refit(face_aabbs_.gpu_data(), face_centers_.gpu_data(),
               cuda_stream);

    // 3. Self-EF broadphase.
    bvh_.query_self_ef(edge_aabbs_.gpu_data(), n_edges,
                       topology_->edges().gpu_data(),
                       topology_->faces().gpu_data(),
                       cuda_stream);

    // 4. Cull EF -> {VF, EE}.
    cull_ef_to_vfee_kernel<<<grid_for(max_ef_candidates_), kBlockDim, 0, stream>>>(
        positions.data(),
        topology_->edges().gpu_data(),
        topology_->faces().gpu_data(),
        topology_->vert_in_edge().gpu_data(),
        topology_->edge_in_face().gpu_data(),
        bvh_.query_pairs_dev(),
        bvh_.query_count_dev(),
        max_ef_candidates_,
        thickness,
        max_contacts_,
        count_.gpu_data(),
        pairs_.gpu_data(),
        weights_.gpu_data());
    check_cuda(cudaGetLastError(), "cull_ef_to_vfee_kernel");

    // 5. Adjacent EE pass (precomputed 1-ring edge pairs).
    if (n_adj > 0) {
        cull_ee_adjacent_kernel<<<grid_for(n_adj), kBlockDim, 0, stream>>>(
            positions.data(),
            topology_->adj_ee_pairs().gpu_data(),
            n_adj,
            thickness,
            max_contacts_,
            count_.gpu_data(),
            pairs_.gpu_data(),
            weights_.gpu_data());
        check_cuda(cudaGetLastError(), "cull_ee_adjacent_kernel");
    }
}

int SelfCollisionDetector::count(std::uintptr_t cuda_stream) {
    if (count_.gpu_size() == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    int host_count = 0;
    check_cuda(cudaMemcpyAsync(&host_count, count_.gpu_data(), sizeof(int),
                               cudaMemcpyDeviceToHost, stream),
               "count() memcpy");
    check_cuda(cudaStreamSynchronize(stream), "count() sync");
    if (host_count > max_contacts_) host_count = max_contacts_;
    if (host_count < 0) host_count = 0;
    return host_count;
}

}  // namespace collision
}  // namespace chysx
