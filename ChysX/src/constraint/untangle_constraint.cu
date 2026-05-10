// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::constraint::UntangleConstraint.

#include "untangle_constraint.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace chysx {
namespace constraint {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("chysx::constraint::UntangleConstraint: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 256;
inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// Sign + magnitude convention, mirroring style3d's
// `solve_untangling_kernel` (Volino 2006 ICM).
//
// For each tangle contact c with unit gradient `G`, weights
// `(u0, u1, w0, w1, w2)` and stored thickness `t`:
//
//   * The "displacement target" is `disp = 2 * t` -- we want to
//     push the edge endpoints by `disp` past the face plane so the
//     un-tangled side ends up with `t` of clearance.
//   * Force magnitude per contact: `F0 = k * disp = 2 * k * t`.
//   * Edge endpoints (i = 0, 1) get pushed in `+G * u_i`; face
//     vertices (i = 2, 3, 4) get pushed in `-G * w_i`.  Opposite
//     signs are what makes the contact open up rather than just
//     translating the whole 5-vertex group together.
//
// chysx accumulates `+grad E` into `out_grad` and subtracts it at
// the end of step block 3 (`assemble_rhs_kernel`), so the gradient
// signs are the negation of the force signs:
//
//     out_grad[edge_i] += -F0 * u_i * G   (force = +F0 * u_i * G,
//                                          push edge along +G)
//     out_grad[face_j] += +F0 * w_j * G   (force = -F0 * w_j * G,
//                                          push face along -G)
//
// One-to-one with style3d's kernel:
//
//     force0 += force * edge_bary[0]                ->  -F0 * u0 * G
//     wp.atomic_add(forces, face[0], -force * w0)  ->  +F0 * w0 * G
//
// (where their `forces` is `-grad E`, i.e. signs are flipped wrt
// our `out_grad`).
__global__ void scatter_gradient_5_kernel(
    const int*         __restrict__ pairs5,    // length 5 * max_contacts
    const float*       __restrict__ weights5,  // length 5 * max_contacts
    const math::Vec3f* __restrict__ normals,   // length max_contacts
    const float*       __restrict__ depths,    // length max_contacts
    const int*         __restrict__ count_ptr,
    int                              max_contacts,
    float                            stiffness,
    math::Vec3f*       __restrict__  out_grad) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_raw = *count_ptr;
    const int n = (n_raw < max_contacts) ? n_raw : max_contacts;
    if (c >= n) return;

    const math::Vec3f G = normals[c];
    const float depth   = depths[c];
    // Volino "displacement target": we want a `2 * thickness` push
    // per crossing, regardless of how deep the edge has actually
    // gone -- the algorithm keeps applying this target every step
    // until the contact disengages.
    const float force_mag = 2.0f * stiffness * depth;

    const int*   ids = pairs5   + 5 * c;
    const float* ws  = weights5 + 5 * c;

    // Edge endpoints (i = 0, 1): out_grad += -force_mag * u_i * G.
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float s = -force_mag * ws[i];
        if (s == 0.0f) continue;
        const int idx = ids[i];
        atomicAdd(&out_grad[idx].x, s * G.x);
        atomicAdd(&out_grad[idx].y, s * G.y);
        atomicAdd(&out_grad[idx].z, s * G.z);
    }

    // Face vertices (i = 2, 3, 4): out_grad += +force_mag * w_i * G.
    #pragma unroll
    for (int i = 2; i < 5; ++i) {
        const float s = +force_mag * ws[i];
        if (s == 0.0f) continue;
        const int idx = ids[i];
        atomicAdd(&out_grad[idx].x, s * G.x);
        atomicAdd(&out_grad[idx].y, s * G.y);
        atomicAdd(&out_grad[idx].z, s * G.z);
    }
}

// One thread per contact.  Each thread does up to 5 atomic-3x3
// updates into `diag_blocks`.  Diagonal-only -- mirrors cuda-cloth's
// `KernelComputeCollisionHessianAndForce_EF` Hessian half (no
// off-diagonal sidecar by design, see header).
__global__ void bake_diag_5_kernel(
    const int*         __restrict__ pairs5,
    const float*       __restrict__ weights5,
    const math::Vec3f* __restrict__ normals,
    const int*         __restrict__ count_ptr,
    int                              max_contacts,
    float                            stiffness,
    math::Mat3f*       __restrict__  diag_blocks) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_raw = *count_ptr;
    const int n = (n_raw < max_contacts) ? n_raw : max_contacts;
    if (c >= n) return;

    const math::Vec3f G = normals[c];
    const float nx = G.x, ny = G.y, nz = G.z;

    // outer(G, G) * stiffness
    const float h00 = stiffness * nx * nx;
    const float h01 = stiffness * nx * ny;
    const float h02 = stiffness * nx * nz;
    const float h11 = stiffness * ny * ny;
    const float h12 = stiffness * ny * nz;
    const float h22 = stiffness * nz * nz;

    const int*   ids = pairs5   + 5 * c;
    const float* ws  = weights5 + 5 * c;

    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        const float wi = ws[i];
        const float ww = wi * wi;
        if (ww == 0.0f) continue;
        const int idx = ids[i];
        float* dst = diag_blocks[idx].data;
        atomicAdd(&dst[0], ww * h00);
        atomicAdd(&dst[1], ww * h01);
        atomicAdd(&dst[2], ww * h02);
        atomicAdd(&dst[3], ww * h01);
        atomicAdd(&dst[4], ww * h11);
        atomicAdd(&dst[5], ww * h12);
        atomicAdd(&dst[6], ww * h02);
        atomicAdd(&dst[7], ww * h12);
        atomicAdd(&dst[8], ww * h22);
    }
}

}  // namespace

void UntangleConstraint::accumulate_gradient(
    const collision::UntangleDetector& detector,
    DeviceSpan<math::Vec3f>            out_grad,
    std::uintptr_t                     cuda_stream) const {
    const int cap = detector.max_contacts();
    if (cap <= 0 || stiffness_ == 0.0f) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    scatter_gradient_5_kernel<<<grid_for(cap), kBlockDim, 0, stream>>>(
        detector.pairs().gpu_data(),
        detector.weights().gpu_data(),
        detector.normals().gpu_data(),
        detector.depths().gpu_data(),
        detector.count_device_ptr(),
        cap,
        stiffness_,
        out_grad.data());
    check_cuda(cudaGetLastError(), "scatter_gradient_5_kernel launch");
}

void UntangleConstraint::bake_diag(
    const collision::UntangleDetector& detector,
    math::Mat3f*                       diag_blocks,
    int                                /*n_particles*/,
    std::uintptr_t                     cuda_stream) const {
    const int cap = detector.max_contacts();
    if (cap <= 0 || stiffness_ == 0.0f || diag_blocks == nullptr) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    bake_diag_5_kernel<<<grid_for(cap), kBlockDim, 0, stream>>>(
        detector.pairs().gpu_data(),
        detector.weights().gpu_data(),
        detector.normals().gpu_data(),
        detector.count_device_ptr(),
        cap,
        stiffness_,
        diag_blocks);
    check_cuda(cudaGetLastError(), "bake_diag_5_kernel launch");
}

}  // namespace constraint
}  // namespace chysx
