// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::collision::apply_contact_spmv.

#include "contact_spmv.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace chysx {
namespace collision {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("chysx::collision::apply_contact_spmv: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 256;
inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// One thread per contact; each thread does up to 4 atomicAdd-mat3
// updates into `diag_blocks`.  Mirrors cuda-cloth's
// `KernelComputeCollisionHessianAndForce_4` MINUS the force-side write
// (chysx folds the force into the RHS via the gradient pathway --
// `SelfCollisionConstraint::accumulate_gradient` -- so we only bake
// the Hessian-diagonal half here).
__global__ void bake_contact_diag_kernel(
    const math::Vec4i* __restrict__ pairs,
    const ContactWeights* __restrict__ weights,
    const int* __restrict__ count_ptr,
    int max_contacts,
    float k_alpha,
    math::Mat3f* __restrict__ diag_blocks) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_raw = *count_ptr;
    const int n = (n_raw < max_contacts) ? n_raw : max_contacts;
    if (c >= n) return;

    const math::Vec4i ids = pairs[c];
    const ContactWeights w = weights[c];
    const float nx = w.nx, ny = w.ny, nz = w.nz;
    const int idxs[4] = {ids.x, ids.y, ids.z, ids.w};
    const float ws[4] = {w.w0, w.w1, w.w2, w.w3};

    // outer(n, n) * k_alpha
    const float h00 = k_alpha * nx * nx;
    const float h01 = k_alpha * nx * ny;
    const float h02 = k_alpha * nx * nz;
    const float h11 = k_alpha * ny * ny;
    const float h12 = k_alpha * ny * nz;
    const float h22 = k_alpha * nz * nz;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float ww = ws[i] * ws[i];
        if (ww == 0.0f) continue;
        const int idx = idxs[i];
        // Mat3f stores 9 floats row-major in `data[]`.
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

// One thread per contact; each thread does the FULL 4x4 minus its
// diagonal.  Layout mirrors cuda-cloth's `CollisionSpmv_4`
// (SolverUtils.cu): for every i in 0..3 sum_{j != i} w_i*w_j*x[id_j],
// then accumulate `H * temp` into `y[id_i]` with `H = k * (n n^T)`.
//
// The diagonal `i == j` term is INTENTIONALLY dropped because the
// caller already baked it into the BlockCSR3's `diag` array via
// `bake_contact_diag` -- it's covered by the regular CSR SpMV.
__global__ void apply_contact_spmv_kernel(
    const math::Vec4i* __restrict__ pairs,
    const ContactWeights* __restrict__ weights,
    const int* __restrict__ count_ptr,
    int max_contacts,
    float k_alpha,
    const math::Vec3f* __restrict__ x,
    math::Vec3f* __restrict__ y) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_raw = *count_ptr;
    const int n = (n_raw < max_contacts) ? n_raw : max_contacts;
    if (c >= n) return;

    const math::Vec4i ids = pairs[c];
    const ContactWeights w = weights[c];
    const math::Vec3f normal(w.nx, w.ny, w.nz);
    const int idxs[4] = {ids.x, ids.y, ids.z, ids.w};
    const float ws[4] = {w.w0, w.w1, w.w2, w.w3};

    // Cache neighbour positions once -- four reads, then four reuses.
    const math::Vec3f xs[4] = {
        x[idxs[0]], x[idxs[1]], x[idxs[2]], x[idxs[3]],
    };

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float wi = ws[i];
        if (wi == 0.0f) continue;
        // temp = sum_{j != i} w_j * x_j
        float tx = 0.0f, ty = 0.0f, tz = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j == i) continue;
            const float wj = ws[j];
            tx += wj * xs[j].x;
            ty += wj * xs[j].y;
            tz += wj * xs[j].z;
        }
        // H * temp = k * (n n^T) * temp = k * (n . temp) * n
        const float dn = normal.x * tx + normal.y * ty + normal.z * tz;
        const float scale = k_alpha * wi * dn;
        atomicAdd(&y[idxs[i]].x, scale * normal.x);
        atomicAdd(&y[idxs[i]].y, scale * normal.y);
        atomicAdd(&y[idxs[i]].z, scale * normal.z);
    }
}

}  // namespace

void bake_contact_diag(math::Mat3f* diag_blocks,
                       int /*n_particles*/,
                       const ContactSpMVOp& op,
                       float alpha,
                       std::uintptr_t cuda_stream) {
    if (!op.active() || diag_blocks == nullptr) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    const float k_alpha = alpha * op.stiffness;
    bake_contact_diag_kernel<<<grid_for(op.max_contacts), kBlockDim, 0, stream>>>(
        op.pairs,
        op.weights,
        op.count_dev,
        op.max_contacts,
        k_alpha,
        diag_blocks);
    check_cuda(cudaGetLastError(), "bake_contact_diag_kernel launch");
}

void apply_contact_spmv(const ContactSpMVOp& op,
                        const math::Vec3f* x,
                        math::Vec3f* y,
                        int /*n_particles*/,
                        float alpha,
                        std::uintptr_t cuda_stream) {
    if (!op.active() || x == nullptr || y == nullptr) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    const float k_alpha = alpha * op.stiffness;
    apply_contact_spmv_kernel<<<grid_for(op.max_contacts), kBlockDim, 0, stream>>>(
        op.pairs,
        op.weights,
        op.count_dev,
        op.max_contacts,
        k_alpha,
        x,
        y);
    check_cuda(cudaGetLastError(), "apply_contact_spmv_kernel launch");
}

}  // namespace collision
}  // namespace chysx
