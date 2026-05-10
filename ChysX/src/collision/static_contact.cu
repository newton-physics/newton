// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::collision::StaticContactSet.
//
// One thread per particle; the per-particle work is to walk the
// (small) static-shape table, pick the deepest penetration, cache
// (depth, normal), and have a follow-up scatter pass push the
// resulting penalty contributions into the cloth simulator's `rhs`
// and `H_.diag` arrays.

#include "static_contact.h"

#include <cuda_runtime.h>
#include <vector_types.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace chysx {
namespace collision {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("chysx::collision::StaticContactSet: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 256;
inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// Plane SDF.  `n` is unit; signed distance is `dot(n, x) + d`,
// outward normal is just `n` (independent of x).
__device__ inline float plane_signed_dist(const math::Vec3f& p,
                                          const PlaneShape&  pl,
                                          math::Vec3f&       out_n) {
    out_n = pl.n;
    return dot(pl.n, p) + pl.d;
}

// OBB SDF.  Returns positive distance outside, negative inside,
// and writes the unit world-space outward normal into `out_n` (for
// points strictly inside, this points toward the closest face, which
// is what the penalty wants — it's the direction we want to push the
// particle to leave the box).
//
// Standard formula from Ericson, "Real-Time Collision Detection".
__device__ inline float box_signed_dist(const math::Vec3f& p,
                                        const BoxShape&    b,
                                        math::Vec3f&       out_n) {
    const math::Vec3f rel = p - b.center;
    const float qx = dot(rel, b.ex);
    const float qy = dot(rel, b.ey);
    const float qz = dot(rel, b.ez);

    const float dx = fabsf(qx) - b.half_ext.x;
    const float dy = fabsf(qy) - b.half_ext.y;
    const float dz = fabsf(qz) - b.half_ext.z;

    if (dx <= 0.0f && dy <= 0.0f && dz <= 0.0f) {
        // Inside the box.  All dᵢ are ≤ 0; the closest face is the
        // axis with the *largest* dᵢ (smallest absolute penetration).
        // signed_dist is that dᵢ (negative).
        if (dx >= dy && dx >= dz) {
            out_n = (qx >= 0.0f) ? b.ex : (b.ex * -1.0f);
            return dx;
        } else if (dy >= dz) {
            out_n = (qy >= 0.0f) ? b.ey : (b.ey * -1.0f);
            return dy;
        } else {
            out_n = (qz >= 0.0f) ? b.ez : (b.ez * -1.0f);
            return dz;
        }
    }

    // Outside the box.  Distance to surface = ‖max(d, 0)‖₂.  The
    // outward normal in box-local coords is (sign(qᵢ) * max(dᵢ, 0)) /
    // dist; rotate back into world via R.
    const float cx = (dx > 0.0f) ? dx : 0.0f;
    const float cy = (dy > 0.0f) ? dy : 0.0f;
    const float cz = (dz > 0.0f) ? dz : 0.0f;
    const float dist = sqrtf(cx * cx + cy * cy + cz * cz);
    const float inv_d = 1.0f / fmaxf(dist, 1.0e-30f);
    const float sx = (qx > 0.0f) ?  cx : -cx;
    const float sy = (qy > 0.0f) ?  cy : -cy;
    const float sz = (qz > 0.0f) ?  cz : -cz;
    out_n = b.ex * (sx * inv_d) + b.ey * (sy * inv_d) + b.ez * (sz * inv_d);
    return dist;
}

// One thread per particle.  Walks every plane and box, picks the
// deepest penetration (largest `thickness - signed_dist`), and writes
// the resulting (n, depth) into `contacts[p]`.  depth ≤ 0 means
// "no active contact" and the downstream scatter passes skip it.
__global__ void detect_kernel(
    const float3* __restrict__       positions,
    int                              n_particles,
    const PlaneShape* __restrict__   planes,
    int                              n_planes,
    const BoxShape* __restrict__     boxes,
    int                              n_boxes,
    float                            thickness,
    math::Vec4f* __restrict__        contacts) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_particles) return;

    const float3 pf = positions[p];
    const math::Vec3f x(pf.x, pf.y, pf.z);

    float       best_depth = 0.0f;
    math::Vec3f best_n(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < n_planes; ++i) {
        math::Vec3f n;
        const float d = plane_signed_dist(x, planes[i], n);
        const float depth = thickness - d;
        if (depth > best_depth) {
            best_depth = depth;
            best_n = n;
        }
    }

    for (int i = 0; i < n_boxes; ++i) {
        math::Vec3f n;
        const float d = box_signed_dist(x, boxes[i], n);
        const float depth = thickness - d;
        if (depth > best_depth) {
            best_depth = depth;
            best_n = n;
        }
    }

    contacts[p] = math::Vec4f(best_n.x, best_n.y, best_n.z, best_depth);
}

// rhs[p] += -k * depth * n.   No atomic — each particle gets exactly
// one contact at most so there is no cross-thread contention; the
// pre-existing rhs value (gradient from elastic / pin / etc.) gets
// read once and written once per particle.
__global__ void scatter_gradient_kernel(
    const math::Vec4f* __restrict__ contacts,
    int                             n_particles,
    float                           stiffness,
    math::Vec3f* __restrict__       rhs) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_particles) return;

    const math::Vec4f c = contacts[p];
    const float depth = c.w;
    if (depth <= 0.0f) return;

    const float kd = -stiffness * depth;
    rhs[p].x += kd * c.x;
    rhs[p].y += kd * c.y;
    rhs[p].z += kd * c.z;
}

// diag[p] += k_n * (n n^T)  +  (μ_v / dt) * (I - n n^T).
//
//  - Normal block:   Gauss-Newton block of the penalty energy
//    `(1/2) k_n (h - d)^2`, see static_contact.h.
//  - Tangential block: implicit-Euler viscous friction
//    `F_friction = -μ_v * P_t * v_{n+1}`, with v_{n+1} = dx / dt
//    in the IE update, so the contribution lands purely on A
//    (the unknown is dx).
//
// Same single-writer-per-particle property as the gradient pass --
// no atomics needed.  We bake the full 3x3 rather than only the upper
// triangle so the BlockCSR3 SpMV path doesn't need to know about
// symmetry.  The friction term is skipped at compile-time-friendly
// runtime cost when `mu_over_dt == 0`.
__global__ void bake_diag_kernel(
    const math::Vec4f* __restrict__ contacts,
    int                             n_particles,
    float                           stiffness,
    float                           mu_over_dt,
    math::Mat3f* __restrict__       diag) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_particles) return;

    const math::Vec4f c = contacts[p];
    const float depth = c.w;
    if (depth <= 0.0f) return;

    const float k  = stiffness;
    const float nx = c.x;
    const float ny = c.y;
    const float nz = c.z;

    // Normal Gauss-Newton block: k_n * (n n^T).
    float a00 = k * nx * nx;
    float a01 = k * nx * ny;
    float a02 = k * nx * nz;
    float a11 = k * ny * ny;
    float a12 = k * ny * nz;
    float a22 = k * nz * nz;

    // Tangential viscous-friction block: (μ_v / dt) * (I - n n^T).
    // Folded into the same 3x3 so we only touch `diag[p]` once.
    if (mu_over_dt > 0.0f) {
        const float m = mu_over_dt;
        a00 += m * (1.0f - nx * nx);
        a01 += m * (-nx * ny);
        a02 += m * (-nx * nz);
        a11 += m * (1.0f - ny * ny);
        a12 += m * (-ny * nz);
        a22 += m * (1.0f - nz * nz);
    }

    math::Mat3f& A = diag[p];
    A.data[0] += a00;
    A.data[1] += a01;
    A.data[2] += a02;
    A.data[3] += a01;       // symmetric
    A.data[4] += a11;
    A.data[5] += a12;
    A.data[6] += a02;       // symmetric
    A.data[7] += a12;       // symmetric
    A.data[8] += a22;
}

}  // namespace

void StaticContactSet::clear() {
    n_planes_     = 0;
    n_boxes_      = 0;
    shapes_dirty_ = true;
    planes_.clear();
    boxes_.clear();
}

void StaticContactSet::add_plane(const PlaneShape& p) {
    // Tiny shape counts (typically <10) make full reallocation cheap
    // and avoid the need for a custom growable container.
    const int new_n = n_planes_ + 1;
    CudaArray<PlaneShape> next(static_cast<std::size_t>(new_n));
    if (n_planes_ > 0) {
        std::memcpy(next.cpu_data(), planes_.cpu_data(),
                    static_cast<std::size_t>(n_planes_) * sizeof(PlaneShape));
    }
    next[static_cast<std::size_t>(n_planes_)] = p;
    planes_ = std::move(next);
    n_planes_ = new_n;
    shapes_dirty_ = true;
}

void StaticContactSet::add_box(const BoxShape& b) {
    const int new_n = n_boxes_ + 1;
    CudaArray<BoxShape> next(static_cast<std::size_t>(new_n));
    if (n_boxes_ > 0) {
        std::memcpy(next.cpu_data(), boxes_.cpu_data(),
                    static_cast<std::size_t>(n_boxes_) * sizeof(BoxShape));
    }
    next[static_cast<std::size_t>(n_boxes_)] = b;
    boxes_ = std::move(next);
    n_boxes_ = new_n;
    shapes_dirty_ = true;
}

void StaticContactSet::upload_shapes_() {
    if (n_planes_ > 0) planes_.copy_to_device();
    if (n_boxes_ > 0)  boxes_.copy_to_device();
    shapes_dirty_ = false;
}

void StaticContactSet::detect(const math::Vec3f* positions,
                              int                n_particles,
                              std::uintptr_t     cuda_stream) {
    if (!active() || n_particles <= 0) return;
    if (positions == nullptr) {
        throw std::invalid_argument(
            "chysx::collision::StaticContactSet::detect: positions must be "
            "non-null");
    }

    if (shapes_dirty_) {
        upload_shapes_();
    }
    if (cached_n_particles_ != n_particles) {
        contacts_.allocate_device(static_cast<std::size_t>(n_particles));
        cached_n_particles_ = n_particles;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    detect_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
        reinterpret_cast<const float3*>(positions),
        n_particles,
        planes_.gpu_data(),
        n_planes_,
        boxes_.gpu_data(),
        n_boxes_,
        thickness_,
        contacts_.gpu_data());
    check_cuda(cudaGetLastError(), "detect_kernel launch");
}

void StaticContactSet::accumulate_gradient(math::Vec3f*    rhs,
                                           int             n_particles,
                                           std::uintptr_t  cuda_stream) const {
    if (!active() || n_particles <= 0) return;
    if (cached_n_particles_ != n_particles) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    scatter_gradient_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
        contacts_.gpu_data(), n_particles, stiffness_, rhs);
    check_cuda(cudaGetLastError(), "scatter_gradient_kernel launch");
}

void StaticContactSet::bake_diag(math::Mat3f*   diag,
                                 int             n_particles,
                                 float           dt,
                                 std::uintptr_t  cuda_stream) const {
    if (!active() || n_particles <= 0) return;
    if (cached_n_particles_ != n_particles) return;

    // Friction contribution is only meaningful for `dt > 0`; guard
    // against an upstream caller passing dt = 0 (which would blow up
    // the (μ_v / dt) multiplier).  In that case we silently disable
    // friction for this call and only bake the normal block.
    const float mu_over_dt =
        (friction_ > 0.0f && dt > 0.0f) ? (friction_ / dt) : 0.0f;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    bake_diag_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
        contacts_.gpu_data(), n_particles, stiffness_, mu_over_dt, diag);
    check_cuda(cudaGetLastError(), "bake_diag_kernel launch");
}

}  // namespace collision
}  // namespace chysx
