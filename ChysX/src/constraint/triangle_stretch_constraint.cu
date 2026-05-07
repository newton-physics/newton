// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::constraint::TriangleStretchConstraint.
//
// Math derivations follow Baraff & Witkin's 1998 cloth paper, with
// the same PSD filtering trick used by chysx's SpringConstraint:
// drop the (cu * d^2 c_u/dx^2) Hessian term whenever cu < 0
// (compressed) and likewise for cv, so each per-element block is
// guaranteed to be positive semi-definite for PCG to converge.

#include "triangle_stretch_constraint.h"

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
            std::string("chysx::constraint::TriangleStretchConstraint: ") +
            what + " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 256;
inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// Avoid divide-by-zero when wu / wv collapse (e.g. degenerate triangle).
constexpr float kMinNorm = 1.0e-6f;

// ---------------------------------------------------------------------------
// Reference shape extraction
//
// Builds the 2D rest configuration of triangle t in its own UV frame
// (placing v_a at the origin, v_b on the +U axis, v_c above), inverts
// the resulting 2x2 edge matrix, and stores the rest area for use as a
// per-element stiffness weight.
// ---------------------------------------------------------------------------

__global__ void triangle_rest_shape_kernel(
    const math::Vec3i* __restrict__ indices,
    const math::Vec3f* __restrict__ positions,
    math::Mat2f*       __restrict__ Dm_inv,
    float*             __restrict__ areas,
    int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;

    const math::Vec3i tri = indices[t];
    const math::Vec3f a = positions[tri.x];
    const math::Vec3f b = positions[tri.y];
    const math::Vec3f c = positions[tri.z];

    const math::Vec3f ab = b - a;
    const math::Vec3f ac = c - a;

    const float lab = math::length(ab);
    const float lac = math::length(ac);

    // Project ac onto the AB axis and recover the perpendicular height.
    const math::Vec3f ab_hat = (lab > kMinNorm) ? (ab * (1.0f / lab))
                                                : math::Vec3f(1.0f, 0.0f, 0.0f);
    const float u_c = math::dot(ab_hat, ac);
    const float v_c2 = lac * lac - u_c * u_c;
    const float v_c = (v_c2 > 0.0f) ? sqrtf(v_c2) : 0.0f;

    // Dm = [UV(b) - UV(a) | UV(c) - UV(a)] = [[ lab, u_c ], [ 0, v_c ]].
    const math::Mat2f Dm(lab, u_c,
                         0.0f, v_c);

    // Rest area = (lab * v_c) / 2  (the parallelogram half-area).
    areas[t] = 0.5f * lab * v_c;

    // Guard the inverse against degenerate triangles by clamping the
    // determinant; the Dm.inverse() else-branch would NaN otherwise.
    const float det = lab * v_c;
    if (det > kMinNorm) {
        Dm_inv[t] = math::inverse(Dm);
    } else {
        Dm_inv[t] = math::Mat2f::identity();
    }
}

// ---------------------------------------------------------------------------
// Energy reduction
// ---------------------------------------------------------------------------

__global__ void triangle_energy_kernel(
    const math::Vec3i* __restrict__ indices,
    const math::Mat2f* __restrict__ Dm_inv_arr,
    const float*       __restrict__ areas,
    const math::Vec3f* __restrict__ positions,
    float k,
    float* __restrict__ out,
    int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;

    const math::Vec3i tri = indices[t];
    const math::Vec3f x0 = positions[tri.x];
    const math::Vec3f x1 = positions[tri.y];
    const math::Vec3f x2 = positions[tri.z];

    const math::Mat2f Dm_inv = Dm_inv_arr[t];

    const math::Vec3f a = x1 - x0;
    const math::Vec3f b = x2 - x0;

    const math::Vec3f wu = a * Dm_inv(0, 0) + b * Dm_inv(1, 0);
    const math::Vec3f wv = a * Dm_inv(0, 1) + b * Dm_inv(1, 1);

    const float lu = math::length(wu);
    const float lv = math::length(wv);
    const float cu = lu - 1.0f;
    const float cv = lv - 1.0f;

    const float k_eff = areas[t] * k;
    atomicAdd(out, 0.5f * k_eff * (cu * cu + cv * cv));
}

// ---------------------------------------------------------------------------
// Gradient scatter (atomic into the global per-particle gradient)
// ---------------------------------------------------------------------------

__global__ void triangle_gradient_kernel(
    const math::Vec3i* __restrict__ indices,
    const math::Mat2f* __restrict__ Dm_inv_arr,
    const float*       __restrict__ areas,
    const math::Vec3f* __restrict__ positions,
    float k,
    math::Vec3f* __restrict__ out_grad,
    int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;

    const math::Vec3i tri = indices[t];
    const math::Vec3f x0 = positions[tri.x];
    const math::Vec3f x1 = positions[tri.y];
    const math::Vec3f x2 = positions[tri.z];

    const math::Mat2f Dm_inv = Dm_inv_arr[t];

    const math::Vec3f a = x1 - x0;
    const math::Vec3f b = x2 - x0;

    math::Vec3f wu = a * Dm_inv(0, 0) + b * Dm_inv(1, 0);
    math::Vec3f wv = a * Dm_inv(0, 1) + b * Dm_inv(1, 1);

    float lu = math::length(wu);
    float lv = math::length(wv);
    if (lu < kMinNorm) lu = kMinNorm;
    if (lv < kMinNorm) lv = kMinNorm;
    const float cu = lu - 1.0f;
    const float cv = lv - 1.0f;
    const math::Vec3f wu_hat = wu * (1.0f / lu);
    const math::Vec3f wv_hat = wv * (1.0f / lv);

    // dw[a, *] = S * Dm_inv,  with S = [[-1,-1],[1,0],[0,1]].
    const float dwu[3] = {
        -Dm_inv(0, 0) - Dm_inv(1, 0),
         Dm_inv(0, 0),
         Dm_inv(1, 0),
    };
    const float dwv[3] = {
        -Dm_inv(0, 1) - Dm_inv(1, 1),
         Dm_inv(0, 1),
         Dm_inv(1, 1),
    };

    const float k_eff = areas[t] * k;

    // g_a = k_eff * (cu * dcu/dx_a + cv * dcv/dx_a)
    //     = k_eff * (cu * dwu[a] * wu_hat + cv * dwv[a] * wv_hat)
    const math::Vec3f g0 = (wu_hat * (cu * dwu[0]) + wv_hat * (cv * dwv[0])) * k_eff;
    const math::Vec3f g1 = (wu_hat * (cu * dwu[1]) + wv_hat * (cv * dwv[1])) * k_eff;
    const math::Vec3f g2 = (wu_hat * (cu * dwu[2]) + wv_hat * (cv * dwv[2])) * k_eff;

    atomicAdd(&out_grad[tri.x].x, g0.x);
    atomicAdd(&out_grad[tri.x].y, g0.y);
    atomicAdd(&out_grad[tri.x].z, g0.z);

    atomicAdd(&out_grad[tri.y].x, g1.x);
    atomicAdd(&out_grad[tri.y].y, g1.y);
    atomicAdd(&out_grad[tri.y].z, g1.z);

    atomicAdd(&out_grad[tri.z].x, g2.x);
    atomicAdd(&out_grad[tri.z].y, g2.y);
    atomicAdd(&out_grad[tri.z].z, g2.z);
}

// ---------------------------------------------------------------------------
// Hessian (PSD-projected, Baraff-Witkin)
//
// Writes 9 triplets at consecutive offsets [9 t, 9 t + 9):
//
//     for a in 0..3:
//         for b in 0..3:
//             out[9 t + 3 a + b] = (i_a, i_b, H_{a,b})
//
// where H_{a,b} = ∂^2 E / ∂x_{i_a} ∂x_{i_b} is the 3x3 block.
// ---------------------------------------------------------------------------

__global__ void triangle_hessian_scatter_kernel(
    const math::Vec3i* __restrict__ indices,
    const math::Mat2f* __restrict__ Dm_inv_arr,
    const float*       __restrict__ areas,
    const math::Vec3f* __restrict__ positions,
    float k,
    const int*   __restrict__ slots,        // length 9*n
    math::Mat3f* __restrict__ A_diag,
    math::Mat3f* __restrict__ A_values,
    int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;

    const math::Vec3i tri = indices[t];

    const math::Vec3f x0 = positions[tri.x];
    const math::Vec3f x1 = positions[tri.y];
    const math::Vec3f x2 = positions[tri.z];

    const math::Mat2f Dm_inv = Dm_inv_arr[t];

    const math::Vec3f a = x1 - x0;
    const math::Vec3f b = x2 - x0;

    math::Vec3f wu = a * Dm_inv(0, 0) + b * Dm_inv(1, 0);
    math::Vec3f wv = a * Dm_inv(0, 1) + b * Dm_inv(1, 1);

    float lu = math::length(wu);
    float lv = math::length(wv);
    if (lu < kMinNorm) lu = kMinNorm;
    if (lv < kMinNorm) lv = kMinNorm;
    const float inv_lu = 1.0f / lu;
    const float inv_lv = 1.0f / lv;
    const float cu = lu - 1.0f;
    const float cv = lv - 1.0f;
    const math::Vec3f wu_hat = wu * inv_lu;
    const math::Vec3f wv_hat = wv * inv_lv;

    const float dwu[3] = {
        -Dm_inv(0, 0) - Dm_inv(1, 0),
         Dm_inv(0, 0),
         Dm_inv(1, 0),
    };
    const float dwv[3] = {
        -Dm_inv(0, 1) - Dm_inv(1, 1),
         Dm_inv(0, 1),
         Dm_inv(1, 1),
    };

    const float k_eff = areas[t] * k;

    const math::Mat3f I3 = math::Mat3f::identity();
    const math::Mat3f Pu = I3 - math::outer(wu_hat, wu_hat);
    const math::Mat3f Pv = I3 - math::outer(wv_hat, wv_hat);

    math::Vec3f dcudx[3];
    math::Vec3f dcvdx[3];
    for (int aa = 0; aa < 3; ++aa) {
        dcudx[aa] = wu_hat * dwu[aa];
        dcvdx[aa] = wv_hat * dwv[aa];
    }

    const int base = 9 * t;
    for (int aa = 0; aa < 3; ++aa) {
        for (int bj = 0; bj < 3; ++bj) {
            math::Mat3f H_ab = math::outer(dcudx[aa], dcudx[bj])
                             + math::outer(dcvdx[aa], dcvdx[bj]);
            H_ab *= k_eff;

            if (cu > 0.0f) {
                H_ab += Pu * (k_eff * cu * inv_lu * dwu[aa] * dwu[bj]);
            }
            if (cv > 0.0f) {
                H_ab += Pv * (k_eff * cv * inv_lv * dwv[aa] * dwv[bj]);
            }

            const int slot = slots[base + 3 * aa + bj];
            sparse::scatter_hessian_block(slot, A_diag, A_values, H_ab);
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// TriangleStretchConstraint
// ---------------------------------------------------------------------------

void TriangleStretchConstraint::set_triangles_from_positions(
    const math::Vec3i* host_triangles,
    int n,
    DeviceSpan<math::Vec3f> positions,
    std::uintptr_t cuda_stream) {
    if (n < 0) {
        throw std::invalid_argument(
            "TriangleStretchConstraint::set_triangles_from_positions: "
            "negative count");
    }

    indices_.resize(static_cast<std::size_t>(n));
    Dm_inv_.resize(static_cast<std::size_t>(n));
    areas_.resize(static_cast<std::size_t>(n));

    if (n == 0) return;

    std::memcpy(indices_.cpu_data(), host_triangles, n * sizeof(math::Vec3i));
    indices_.copy_to_device(cuda_stream);

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    triangle_rest_shape_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        indices_.gpu_data(),
        positions.data(),
        Dm_inv_.gpu_data(),
        areas_.gpu_data(),
        n);

    if (cuda_stream == 0) {
        check_cuda(cudaStreamSynchronize(stream),
                   "cudaStreamSynchronize after triangle_rest_shape_kernel");
    }
}

float TriangleStretchConstraint::compute_energy(
    DeviceSpan<math::Vec3f> positions,
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

    triangle_energy_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        indices_.gpu_data(),
        Dm_inv_.gpu_data(),
        areas_.gpu_data(),
        positions.data(),
        stiffness_,
        energy_buffer_.gpu_data(),
        n);

    energy_buffer_.copy_to_host(cuda_stream);
    return energy_buffer_.cpu_data()[0];
}

void TriangleStretchConstraint::accumulate_gradient(
    DeviceSpan<math::Vec3f> positions,
    DeviceSpan<math::Vec3f> out_grad,
    std::uintptr_t cuda_stream) const {
    const int n = size();
    if (n == 0) return;

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    triangle_gradient_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        indices_.gpu_data(),
        Dm_inv_.gpu_data(),
        areas_.gpu_data(),
        positions.data(),
        stiffness_,
        out_grad.data(),
        n);

    if (cuda_stream == 0) {
        check_cuda(cudaStreamSynchronize(stream),
                   "cudaStreamSynchronize(triangle_gradient_kernel)");
    }
}

void TriangleStretchConstraint::accumulate_hessian(
    DeviceSpan<math::Vec3f> positions,
    sparse::BlockCSR3& A,
    std::uintptr_t cuda_stream) const {
    const int n = size();
    if (n == 0) return;

    if (static_cast<int>(hessian_slots_.gpu_size()) < 9 * n) {
        throw std::runtime_error(
            "TriangleStretchConstraint::accumulate_hessian: "
            "bind_hessian_layout(A) must be called before stepping");
    }

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    triangle_hessian_scatter_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        indices_.gpu_data(),
        Dm_inv_.gpu_data(),
        areas_.gpu_data(),
        positions.data(),
        stiffness_,
        hessian_slots_.gpu_data(),
        A.diag.gpu_data(),
        A.values.gpu_data(),
        n);

    if (cuda_stream == 0) {
        check_cuda(cudaStreamSynchronize(stream),
                   "cudaStreamSynchronize(triangle_hessian_scatter_kernel)");
    }
}

}  // namespace constraint
}  // namespace chysx
