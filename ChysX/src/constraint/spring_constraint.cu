// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::constraint::SpringConstraint.

#include "spring_constraint.h"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../sparse/block_csr_atomic.cuh"

namespace chysx {
namespace constraint {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("chysx::constraint::SpringConstraint: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 256;
inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// Avoid divide-by-zero when two particles overlap.  This is the same
// guard cuda-cloth uses; tweaked higher than 1e-15 because the kernel
// then divides 1.0/l once and reuses it three times.
constexpr float kMinLength = 1.0e-6f;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void spring_rest_length_kernel(
    const math::Vec2i* __restrict__ edges,
    const math::Vec3f* __restrict__ positions,
    float* __restrict__ rest_lengths,
    int n) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;

    const math::Vec2i ij = edges[e];
    const math::Vec3f d = positions[ij.x] - positions[ij.y];
    rest_lengths[e] = math::length(d);
}

// Per-spring elastic energy reduction.
//   E += (k/2) * (|x_a - x_b| - L)^2
__global__ void spring_energy_kernel(
    const math::Vec2i* __restrict__ edges,
    const float* __restrict__ rest_lengths,
    const math::Vec3f* __restrict__ positions,
    float k,
    float* __restrict__ out,
    int n) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;

    const math::Vec2i ij = edges[e];
    const math::Vec3f d = positions[ij.x] - positions[ij.y];
    const float l = math::length(d);
    const float dl = l - rest_lengths[e];
    atomicAdd(out, 0.5f * k * dl * dl);
}

// Per-spring gradient scatter:
//   out_grad[a] +=  k * (l - L) * d̂
//   out_grad[b] += -k * (l - L) * d̂
__global__ void spring_gradient_kernel(
    const math::Vec2i* __restrict__ edges,
    const float* __restrict__ rest_lengths,
    const math::Vec3f* __restrict__ positions,
    float k,
    math::Vec3f* __restrict__ out_grad,
    int n) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;

    const math::Vec2i ij = edges[e];
    const math::Vec3f d = positions[ij.x] - positions[ij.y];

    float l = math::length(d);
    if (l < kMinLength) l = kMinLength;
    const float inv_l = 1.0f / l;
    const math::Vec3f dhat = d * inv_l;

    const float coeff = k * (l - rest_lengths[e]);
    const math::Vec3f g = dhat * coeff;

    atomicAdd(&out_grad[ij.x].x,  g.x);
    atomicAdd(&out_grad[ij.x].y,  g.y);
    atomicAdd(&out_grad[ij.x].z,  g.z);
    atomicAdd(&out_grad[ij.y].x, -g.x);
    atomicAdd(&out_grad[ij.y].y, -g.y);
    atomicAdd(&out_grad[ij.y].z, -g.z);
}

// Per-spring block-Hessian scatter (PSD-projected, Baraff-Witkin).
//
// `slots[4 e + k]` for k = 0..3 encodes the destination for the
// (a,a), (a,b), (b,a), (b,b) blocks respectively (slot < 0 -> diag,
// slot >= 0 -> off-diag values).
__global__ void spring_hessian_scatter_kernel(
    const math::Vec2i* __restrict__ edges,
    const float* __restrict__ rest_lengths,
    const math::Vec3f* __restrict__ positions,
    float k,
    const int* __restrict__ slots,
    math::Mat3f* __restrict__ A_diag,
    math::Mat3f* __restrict__ A_values,
    int n) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;

    const math::Vec2i ij = edges[e];
    const float L = rest_lengths[e];
    const math::Vec3f d = positions[ij.x] - positions[ij.y];

    float l = math::length(d);
    if (l < kMinLength) l = kMinLength;
    const float inv_l = 1.0f / l;
    const math::Vec3f dhat = d * inv_l;

    // a = k * L / l.  In the stretched branch H_blk = (k - a) I + a d̂d̂ᵀ.
    // In the compressed branch we drop the (k - a) I term (it becomes
    // negative-definite when a > k) and keep a d̂d̂ᵀ alone.
    const float a = k * L * inv_l;
    const math::Mat3f outer_dd = math::outer(dhat, dhat);

    math::Mat3f H_blk;
    if (l < L) {
        H_blk = outer_dd * a;
    } else {
        const math::Mat3f I3 = math::Mat3f::identity();
        H_blk = I3 * (k - a) + outer_dd * a;
    }
    const math::Mat3f neg_H = H_blk * (-1.0f);

    const int base = 4 * e;
    sparse::scatter_hessian_block(slots[base + 0], A_diag, A_values, H_blk);
    sparse::scatter_hessian_block(slots[base + 1], A_diag, A_values, neg_H);
    sparse::scatter_hessian_block(slots[base + 2], A_diag, A_values, neg_H);
    sparse::scatter_hessian_block(slots[base + 3], A_diag, A_values, H_blk);
}

}  // namespace

// ---------------------------------------------------------------------------
// SpringConstraint
// ---------------------------------------------------------------------------

void SpringConstraint::set_springs(const math::Vec2i* host_edges,
                                   const float* host_rest_lengths,
                                   int n) {
    if (n < 0) {
        throw std::invalid_argument("SpringConstraint::set_springs: negative count");
    }

    indices_.resize(static_cast<std::size_t>(n));
    rest_lengths_.resize(static_cast<std::size_t>(n));

    if (n == 0) return;

    std::memcpy(indices_.cpu_data(), host_edges, n * sizeof(math::Vec2i));
    std::memcpy(rest_lengths_.cpu_data(), host_rest_lengths, n * sizeof(float));
    indices_.copy_to_device();
    rest_lengths_.copy_to_device();
}

void SpringConstraint::set_springs_from_positions(
    const math::Vec2i* host_edges,
    int n,
    DeviceSpan<math::Vec3f> positions,
    std::uintptr_t cuda_stream) {
    if (n < 0) {
        throw std::invalid_argument(
            "SpringConstraint::set_springs_from_positions: negative count");
    }

    indices_.resize(static_cast<std::size_t>(n));
    rest_lengths_.resize(static_cast<std::size_t>(n));

    if (n == 0) return;

    std::memcpy(indices_.cpu_data(), host_edges, n * sizeof(math::Vec2i));
    indices_.copy_to_device(cuda_stream);

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    spring_rest_length_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        indices_.gpu_data(),
        positions.data(),
        rest_lengths_.gpu_data(),
        n);

    if (cuda_stream == 0) {
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
}

float SpringConstraint::compute_energy(DeviceSpan<math::Vec3f> positions,
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

    spring_energy_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        indices_.gpu_data(),
        rest_lengths_.gpu_data(),
        positions.data(),
        stiffness_,
        energy_buffer_.gpu_data(),
        n);

    energy_buffer_.copy_to_host(cuda_stream);
    return energy_buffer_.cpu_data()[0];
}

void SpringConstraint::accumulate_gradient(DeviceSpan<math::Vec3f> positions,
                                           DeviceSpan<math::Vec3f> out_grad,
                                           std::uintptr_t cuda_stream) const {
    const int n = size();
    if (n == 0) return;

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    spring_gradient_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        indices_.gpu_data(),
        rest_lengths_.gpu_data(),
        positions.data(),
        stiffness_,
        out_grad.data(),
        n);

    if (cuda_stream == 0) {
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
}

void SpringConstraint::accumulate_hessian(DeviceSpan<math::Vec3f> positions,
                                          sparse::BlockCSR3& A,
                                          std::uintptr_t cuda_stream) const {
    const int n = size();
    if (n == 0) return;

    if (static_cast<int>(hessian_slots_.gpu_size()) < 4 * n) {
        throw std::runtime_error(
            "SpringConstraint::accumulate_hessian: bind_hessian_layout(A) "
            "must be called before stepping");
    }

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    spring_hessian_scatter_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        indices_.gpu_data(),
        rest_lengths_.gpu_data(),
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
