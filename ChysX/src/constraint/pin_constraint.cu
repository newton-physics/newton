// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::constraint::PinConstraint.

#include "pin_constraint.h"

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
            std::string("chysx::constraint::PinConstraint: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 256;

inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// One thread per pin.
//
// E += (k/2) * |x[i_c] - t_c|^2.  `half_k` is `0.5f * stiffness_`,
// folded in on the host so the inner loop stays tight.
__global__ void pin_energy_kernel(const int* __restrict__ indices,
                                  const math::Vec3f* __restrict__ targets,
                                  const math::Vec3f* __restrict__ positions,
                                  float half_k,
                                  float* __restrict__ out,
                                  int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;

    const int i = indices[c];
    const math::Vec3f d = positions[i] - targets[c];
    const float e = half_k * (d.x * d.x + d.y * d.y + d.z * d.z);
    atomicAdd(out, e);
}

// out_grad[i_c] += k * (x[i_c] - t_c)
//
// Multiple constraints can target the same particle, hence the
// componentwise atomicAdd.  Vec3f has the layout
// `union { struct { float x, y, z; }; float data[3]; }`, so &v.x is a
// well-formed float* address for atomicAdd().
__global__ void pin_gradient_kernel(const int* __restrict__ indices,
                                    const math::Vec3f* __restrict__ targets,
                                    const math::Vec3f* __restrict__ positions,
                                    float k,
                                    math::Vec3f* __restrict__ out_grad,
                                    int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;

    const int i = indices[c];
    const math::Vec3f g = (positions[i] - targets[c]) * k;
    atomicAdd(&out_grad[i].x, g.x);
    atomicAdd(&out_grad[i].y, g.y);
    atomicAdd(&out_grad[i].z, g.z);
}

// AtomicAdd k*I_3 into A.diag[i_c] (or A.values, but pin slots are
// always diagonal so the slot is < 0 for every instance).
__global__ void pin_hessian_scatter_kernel(
    const int* __restrict__ slots,        // length n; slot[c] < 0 (diag-encoded)
    float k,
    math::Mat3f* __restrict__ A_diag,
    math::Mat3f* __restrict__ A_values,
    int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;

    const math::Mat3f H_blk(k,    0.0f, 0.0f,
                            0.0f, k,    0.0f,
                            0.0f, 0.0f, k);
    sparse::scatter_hessian_block(slots[c], A_diag, A_values, H_blk);
}

}  // namespace

// ---------------------------------------------------------------------------
// PinConstraint
// ---------------------------------------------------------------------------

void PinConstraint::set_pins(const int* host_indices,
                             const math::Vec3f* host_targets,
                             int n) {
    if (n < 0) {
        throw std::invalid_argument("PinConstraint::set_pins: negative count");
    }

    // ConstraintN<1>::indices_ already has the right element type (int),
    // so reuse it for the particle indices and keep targets locally.
    indices_.resize(static_cast<std::size_t>(n));
    targets_.resize(static_cast<std::size_t>(n));

    if (n == 0) return;

    std::memcpy(indices_.cpu_data(), host_indices, n * sizeof(int));
    std::memcpy(targets_.cpu_data(), host_targets, n * sizeof(math::Vec3f));
    indices_.copy_to_device();
    targets_.copy_to_device();
}

float PinConstraint::compute_energy(DeviceSpan<math::Vec3f> positions,
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

    pin_energy_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        indices_.gpu_data(),
        targets_.gpu_data(),
        positions.data(),
        0.5f * stiffness_,
        energy_buffer_.gpu_data(),
        n);

    // copy_to_host blocks on `cuda_stream` for non-default streams,
    // and synchronously runs on the default stream when stream == 0,
    // so the float we read back is always finalised.
    energy_buffer_.copy_to_host(cuda_stream);
    return energy_buffer_.cpu_data()[0];
}

void PinConstraint::accumulate_gradient(DeviceSpan<math::Vec3f> positions,
                                        DeviceSpan<math::Vec3f> out_grad,
                                        std::uintptr_t cuda_stream) const {
    const int n = size();
    if (n == 0) return;

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    pin_gradient_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        indices_.gpu_data(),
        targets_.gpu_data(),
        positions.data(),
        stiffness_,
        out_grad.data(),
        n);

    if (cuda_stream == 0) {
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
}

void PinConstraint::accumulate_hessian(DeviceSpan<math::Vec3f> /*positions*/,
                                       sparse::BlockCSR3& A,
                                       std::uintptr_t cuda_stream) const {
    const int n = size();
    if (n == 0) return;

    if (static_cast<int>(hessian_slots_.gpu_size()) < n) {
        throw std::runtime_error(
            "PinConstraint::accumulate_hessian: bind_hessian_layout(A) "
            "must be called before stepping");
    }

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    pin_hessian_scatter_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        hessian_slots_.gpu_data(),
        stiffness_,
        A.diag.gpu_data(),
        A.values.gpu_data(),
        n);

    if (cuda_stream == 0) {
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
}

}  // namespace constraint
}  // namespace chysx
