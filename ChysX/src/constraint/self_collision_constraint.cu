// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::constraint::SelfCollisionConstraint.

#include "self_collision_constraint.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace chysx {
namespace constraint {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("chysx::constraint::SelfCollisionConstraint: ") +
            what + " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 256;
inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// Sign convention.  Penalty energy
//
//     E_c  =  (k/2) * (thickness - dot(g(x), n))^2,
//
// where `g(x) = sum_i w_i * x_i` is the contact's signed-distance
// constraint vector (== `x_v - x_face_cp` for a VF contact).  Then
//
//     d E_c / d x_j  =  -k * (thickness - dot(g, n)) * w_j * n
//                    =  -k * depth * w_j * n.
//
// chysx accumulates `+grad E` into `out_grad` and subtracts at the end
// (`assemble_rhs_kernel`), so we add `-k * depth * w_j * n` here.  The
// sign flips relative to cuda-cloth's `KernelComputeCollisionHessianAndForce_4`
// because cuda-cloth stores `+force = -grad E` directly in its `f`
// buffer.
__global__ void scatter_gradient_kernel(
    const math::Vec4i* __restrict__ pairs,
    const collision::ContactWeights* __restrict__ weights,
    const int* __restrict__ count_ptr,
    int max_contacts,
    float stiffness,
    math::Vec3f* __restrict__ out_grad) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_raw = *count_ptr;
    const int n = (n_raw < max_contacts) ? n_raw : max_contacts;
    if (c >= n) return;

    const math::Vec4i ids = pairs[c];
    const collision::ContactWeights w = weights[c];

    const float kd = -stiffness * w.depth;  // -k * depth (chysx +grad sign)
    const math::Vec3f g_unit(kd * w.nx, kd * w.ny, kd * w.nz);

    const int idxs[4] = {ids.x, ids.y, ids.z, ids.w};
    const float ws[4] = {w.w0, w.w1, w.w2, w.w3};

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float s = ws[i];
        if (s == 0.0f) continue;  // bary-on-edge corner case
        const int idx = idxs[i];
        atomicAdd(&out_grad[idx].x, g_unit.x * s);
        atomicAdd(&out_grad[idx].y, g_unit.y * s);
        atomicAdd(&out_grad[idx].z, g_unit.z * s);
    }
}

}  // namespace

collision::ContactSpMVOp SelfCollisionConstraint::make_spmv_op(
    const collision::SelfCollisionDetector& detector) const noexcept {
    collision::ContactSpMVOp op;
    op.pairs        = detector.pairs().gpu_data();
    op.weights      = detector.weights().gpu_data();
    op.count_dev    = detector.count_device_ptr();
    op.max_contacts = detector.max_contacts();
    op.stiffness    = stiffness_;
    return op;
}

void SelfCollisionConstraint::accumulate_gradient(
    const collision::SelfCollisionDetector& detector,
    DeviceSpan<math::Vec3f> out_grad,
    std::uintptr_t cuda_stream) const {
    const int cap = detector.max_contacts();
    if (cap <= 0 || stiffness_ == 0.0f) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    scatter_gradient_kernel<<<grid_for(cap), kBlockDim, 0, stream>>>(
        detector.pairs().gpu_data(),
        detector.weights().gpu_data(),
        detector.count_device_ptr(),
        cap,
        stiffness_,
        out_grad.data());
    check_cuda(cudaGetLastError(), "scatter_gradient_kernel launch");
}

}  // namespace constraint
}  // namespace chysx
