// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::constraint::BendingConstraint.

#include "bending_constraint.h"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

#include "../sparse/block_csr_atomic.cuh"

namespace chysx {
namespace constraint {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("chysx::constraint::BendingConstraint: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 256;
inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// Numerical floor used to bail out on degenerate dihedrals (collapsed
// edge or a triangle with no height over its hinge edge).  Same order
// of magnitude cuda-cloth uses for its own normalisations.
constexpr float kMinNorm = 1.0e-12f;

// ---------------------------------------------------------------------------
// Geometry helper used by every kernel.
//
// Computes everything the BW98 / Bridson dihedral element needs from a
// single (v0, v1, v2, v3) load:
//   * theta              : current dihedral angle in [-pi, pi]
//   * a[0..3]            : d theta / d x_i  (four 3-vectors, == "shape grads")
//   * `valid` is set false if any of the geometry pieces is degenerate
//     (zero-length edge / triangle with no height).  Callers should
//     skip the dihedral in that case to avoid emitting NaNs into A_diag
//     / A_values, which would poison the entire PCG solve.
// ---------------------------------------------------------------------------
struct DihedralFrame {
    float       theta;
    math::Vec3f a[4];
    bool        valid;
};

__device__ __forceinline__ DihedralFrame compute_dihedral_frame(
    const math::Vec3f& v0,
    const math::Vec3f& v1,
    const math::Vec3f& v2,
    const math::Vec3f& v3) {
    DihedralFrame F;
    F.valid = false;

    const math::Vec3f e10 = v1 - v0;
    const math::Vec3f e20 = v2 - v0;
    const math::Vec3f e30 = v3 - v0;

    const float l = math::length(e10);
    if (l < kMinNorm) return F;

    const float inv_l = 1.0f / l;
    const math::Vec3f e10_hat = e10 * inv_l;

    const math::Vec3f n1_raw = math::cross(e20, e10);
    const math::Vec3f n2_raw = math::cross(e10, e30);
    const float n1_len = math::length(n1_raw);
    const float n2_len = math::length(n2_raw);
    if (n1_len < kMinNorm || n2_len < kMinNorm) return F;
    const math::Vec3f n1 = n1_raw * (1.0f / n1_len);
    const math::Vec3f n2 = n2_raw * (1.0f / n2_len);

    // Signed dihedral angle.  Sign comes from the projection of n1 x n2
    // onto the (oriented) shared edge.  At a flat configuration n1 == n2
    // so theta = 0 by construction.
    float sin_theta = math::dot(math::cross(n1, n2), e10_hat);
    if (sin_theta >  1.0f) sin_theta =  1.0f;
    if (sin_theta < -1.0f) sin_theta = -1.0f;
    const float cos_theta = math::dot(n1, n2);
    F.theta = atan2f(sin_theta, cos_theta);

    // Bridson's shape-vector derivation (reused by cuda-cloth):
    //   omega_k = e10·e_k0 / |e10|^2 is the parameter where the foot of
    //   the perpendicular from v_{k+1} (k=1,2) lands on the v0->v1 line.
    //   h_k is the perpendicular distance.  Then a_i = dtheta/dx_i is
    //   the linear combination t1[i] n1 + t2[i] n2 below.
    const float inv_l2 = inv_l * inv_l;
    const float omega1 = math::dot(e10, e20) * inv_l2;
    const float omega2 = math::dot(e10, e30) * inv_l2;

    const math::Vec3f e20_perp = e20 - e10 * omega1;
    const math::Vec3f e30_perp = e30 - e10 * omega2;
    const float h1 = math::length(e20_perp);
    const float h2 = math::length(e30_perp);
    if (h1 < kMinNorm || h2 < kMinNorm) return F;

    const float inv_h1 = 1.0f / h1;
    const float inv_h2 = 1.0f / h2;
    const float t1[4] = {
        (omega1 - 1.0f) * inv_h1,
        -omega1 * inv_h1,
         1.0f * inv_h1,
         0.0f,
    };
    const float t2[4] = {
        (omega2 - 1.0f) * inv_h2,
        -omega2 * inv_h2,
         0.0f,
         1.0f * inv_h2,
    };

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        F.a[i] = n1 * t1[i] + n2 * t2[i];
    }

    F.valid = true;
    return F;
}

// ---------------------------------------------------------------------------
// Rest-angle init kernel
// ---------------------------------------------------------------------------

__global__ void bending_rest_angle_kernel(
    const math::Vec4i* __restrict__ verts,
    const math::Vec3f* __restrict__ positions,
    float* __restrict__ rest_angles,
    int n) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;

    const math::Vec4i id = verts[e];
    const math::Vec3f v0 = positions[id.x];
    const math::Vec3f v1 = positions[id.y];
    const math::Vec3f v2 = positions[id.z];
    const math::Vec3f v3 = positions[id.w];

    const DihedralFrame F = compute_dihedral_frame(v0, v1, v2, v3);
    rest_angles[e] = F.valid ? F.theta : 0.0f;
}

// ---------------------------------------------------------------------------
// Energy
// ---------------------------------------------------------------------------

__global__ void bending_energy_kernel(
    const math::Vec4i* __restrict__ verts,
    const float*       __restrict__ rest_angles,
    const math::Vec3f* __restrict__ positions,
    float k,
    float* __restrict__ out,
    int n) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;

    const math::Vec4i id = verts[e];
    const DihedralFrame F = compute_dihedral_frame(
        positions[id.x], positions[id.y],
        positions[id.z], positions[id.w]);
    if (!F.valid) return;

    const float dtheta = F.theta - rest_angles[e];
    atomicAdd(out, 0.5f * k * dtheta * dtheta);
}

// ---------------------------------------------------------------------------
// Gradient scatter
//
//   out_grad[v_i] += k * (theta - theta_rest) * a_i.
// ---------------------------------------------------------------------------

__global__ void bending_gradient_kernel(
    const math::Vec4i* __restrict__ verts,
    const float*       __restrict__ rest_angles,
    const math::Vec3f* __restrict__ positions,
    float k,
    math::Vec3f* __restrict__ out_grad,
    int n) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;

    const math::Vec4i id = verts[e];
    const DihedralFrame F = compute_dihedral_frame(
        positions[id.x], positions[id.y],
        positions[id.z], positions[id.w]);
    if (!F.valid) return;

    const float coeff = k * (F.theta - rest_angles[e]);
    const int idx[4] = { id.x, id.y, id.z, id.w };

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const math::Vec3f g = F.a[i] * coeff;
        atomicAdd(&out_grad[idx[i]].x, g.x);
        atomicAdd(&out_grad[idx[i]].y, g.y);
        atomicAdd(&out_grad[idx[i]].z, g.z);
    }
}

// ---------------------------------------------------------------------------
// Hessian scatter
//
//   H_{i,j} = k * outer(a_i, a_j)         (Gauss-Newton, PSD by construction)
//
// `slots[16 e + 4 i + j]` follows the ConstraintN<4>::bind_hessian_layout
// encoding (negative = diag, non-negative = off-diag values index).
// ---------------------------------------------------------------------------

__global__ void bending_hessian_scatter_kernel(
    const math::Vec4i* __restrict__ verts,
    const math::Vec3f* __restrict__ positions,
    float k,
    const int* __restrict__ slots,
    math::Mat3f* __restrict__ A_diag,
    math::Mat3f* __restrict__ A_values,
    int n) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;

    const math::Vec4i id = verts[e];
    const DihedralFrame F = compute_dihedral_frame(
        positions[id.x], positions[id.y],
        positions[id.z], positions[id.w]);
    if (!F.valid) return;

    const int base = 16 * e;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const math::Mat3f H = math::outer(F.a[i], F.a[j]) * k;
            sparse::scatter_hessian_block(slots[base + 4 * i + j],
                                          A_diag, A_values, H);
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// BendingConstraint
// ---------------------------------------------------------------------------

void BendingConstraint::set_dihedrals_from_positions(
    const math::Vec4i* host_dihedrals,
    int n,
    DeviceSpan<math::Vec3f> positions,
    std::uintptr_t cuda_stream) {
    if (n < 0) {
        throw std::invalid_argument(
            "BendingConstraint::set_dihedrals_from_positions: negative count");
    }

    indices_.resize(static_cast<std::size_t>(n));
    rest_angles_.resize(static_cast<std::size_t>(n));

    if (n == 0) return;

    std::memcpy(indices_.cpu_data(), host_dihedrals, n * sizeof(math::Vec4i));
    indices_.copy_to_device(cuda_stream);

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    bending_rest_angle_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        indices_.gpu_data(),
        positions.data(),
        rest_angles_.gpu_data(),
        n);

    if (cuda_stream == 0) {
        check_cuda(cudaStreamSynchronize(stream),
                   "cudaStreamSynchronize after bending_rest_angle_kernel");
    }
}

float BendingConstraint::compute_energy(DeviceSpan<math::Vec3f> positions,
                                        std::uintptr_t cuda_stream) const {
    const int n = size();
    if (n == 0) return 0.0f;

    if (energy_buffer_.gpu_size() < 1) {
        energy_buffer_.resize(1);
    }

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    check_cuda(cudaMemsetAsync(energy_buffer_.gpu_data(), 0, sizeof(float),
                               stream),
               "cudaMemsetAsync(energy)");

    bending_energy_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        indices_.gpu_data(),
        rest_angles_.gpu_data(),
        positions.data(),
        stiffness_,
        energy_buffer_.gpu_data(),
        n);

    energy_buffer_.copy_to_host(cuda_stream);
    return energy_buffer_.cpu_data()[0];
}

void BendingConstraint::accumulate_gradient(DeviceSpan<math::Vec3f> positions,
                                            DeviceSpan<math::Vec3f> out_grad,
                                            std::uintptr_t cuda_stream) const {
    const int n = size();
    if (n == 0) return;

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    bending_gradient_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        indices_.gpu_data(),
        rest_angles_.gpu_data(),
        positions.data(),
        stiffness_,
        out_grad.data(),
        n);

    if (cuda_stream == 0) {
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
}

void BendingConstraint::accumulate_hessian(DeviceSpan<math::Vec3f> positions,
                                           sparse::BlockCSR3& A,
                                           std::uintptr_t cuda_stream) const {
    const int n = size();
    if (n == 0) return;

    if (static_cast<int>(hessian_slots_.gpu_size()) < 16 * n) {
        throw std::runtime_error(
            "BendingConstraint::accumulate_hessian: bind_hessian_layout(A) "
            "must be called before stepping");
    }

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    bending_hessian_scatter_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        indices_.gpu_data(),
        positions.data(),
        stiffness_,
        hessian_slots_.gpu_data(),
        A.diag.gpu_data(),
        A.values.gpu_data(),
        n);

    if (cuda_stream == 0) {
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
}

}  // namespace constraint
}  // namespace chysx
