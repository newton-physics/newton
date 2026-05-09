// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::solver::PCGSolver.

#include "pcg_solver.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "../profile/nvtx_range.h"

namespace chysx {
namespace solver {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("chysx::solver::PCGSolver: ") +
                                 what + " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 256;

inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// M_inv[i] = inverse(A_diag[i]).  Diagonal blocks are SPD in the
// problems we target (mass + stiffness), but we don't enforce that
// here — calling code must guarantee invertibility.
__global__ void invert_diag_kernel(const math::Mat3f* __restrict__ A_diag,
                                   math::Mat3f* __restrict__ M_inv,
                                   int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    M_inv[i] = math::inverse(A_diag[i]);
}

// out[i] = M_inv[i] * in[i]   (block-diagonal preconditioner application)
__global__ void apply_jacobi_kernel(const math::Mat3f* __restrict__ M_inv,
                                    const math::Vec3f* __restrict__ in,
                                    math::Vec3f* __restrict__ out,
                                    int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = M_inv[i] * in[i];
}

// Block reduction returning the full sum in thread 0.
template <int BLOCK>
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[BLOCK / 32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    constexpr int kNumWarps = BLOCK / 32;
    val = (threadIdx.x < kNumWarps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) {
        #pragma unroll
        for (int offset = kNumWarps / 2; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
    }
    return val;  // valid in threadIdx.x == 0
}

// out += sum_i dot(a[i], b[i])
template <int BLOCK>
__global__ void dot_kernel(const math::Vec3f* __restrict__ a,
                           const math::Vec3f* __restrict__ b,
                           float* __restrict__ out,
                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float local = 0.0f;
    if (i < n) {
        const math::Vec3f av = a[i];
        const math::Vec3f bv = b[i];
        local = av.x * bv.x + av.y * bv.y + av.z * bv.z;
    }
    const float bsum = block_reduce_sum<BLOCK>(local);
    if (threadIdx.x == 0) atomicAdd(out, bsum);
}

// y = alpha * x + y
//
// `alpha` is read from device memory through a 1-element pointer so
// callers can chain a divide kernel that produces alpha into this one
// without ever staging through host memory.
__global__ void axpy_dev_kernel(const float* __restrict__ alpha_ptr,
                                const math::Vec3f* __restrict__ x,
                                math::Vec3f* __restrict__ y,
                                int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float alpha = *alpha_ptr;
    y[i] = y[i] + x[i] * alpha;
}

// y = -alpha * x + y     (used to update r := r - alpha Ap)
__global__ void naxpy_dev_kernel(const float* __restrict__ alpha_ptr,
                                 const math::Vec3f* __restrict__ x,
                                 math::Vec3f* __restrict__ y,
                                 int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float alpha = *alpha_ptr;
    y[i] = y[i] - x[i] * alpha;
}

// p = z + beta * p   (CG direction update)
__global__ void update_p_kernel(const float* __restrict__ beta_ptr,
                                const math::Vec3f* __restrict__ z,
                                math::Vec3f* __restrict__ p,
                                int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float beta = *beta_ptr;
    p[i] = z[i] + p[i] * beta;
}

// Single-thread helpers that do scalar arithmetic on the device, so we
// can avoid host round-trips inside the iteration.
__global__ void scalar_div_kernel(const float* a, const float* b, float* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const float bv = *b;
        // Match the reference: if denominator is essentially zero,
        // the iteration is meaningless — emit zero rather than NaN.
        *out = (bv > 1e-37f || bv < -1e-37f) ? (*a / bv) : 0.0f;
    }
}

__global__ void scalar_copy_kernel(const float* src, float* dst) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *dst = *src;
}

}  // namespace

// ---------------------------------------------------------------------------
// PCGSolver
// ---------------------------------------------------------------------------

void PCGSolver::initialize(int num_block_rows) {
    if (num_block_rows < 0) {
        throw std::invalid_argument("PCGSolver::initialize: negative size");
    }
    if (num_block_rows == num_block_rows_) return;

    r_.resize(num_block_rows);
    p_.resize(num_block_rows);
    z_.resize(num_block_rows);
    Ap_.resize(num_block_rows);
    M_inv_.resize(num_block_rows);
    coeff_.resize(4);

    num_block_rows_ = num_block_rows;
}

int PCGSolver::solve(const sparse::BlockCSR3& A,
                     DeviceSpan<math::Vec3f> b,
                     DeviceSpan<math::Vec3f> x,
                     const PCGParams& params,
                     std::uintptr_t cuda_stream,
                     const collision::ContactSpMVOp* contact) {
    const int n = A.num_block_rows();
    if (n == 0) return 0;

    if (static_cast<int>(b.size()) < n || static_cast<int>(x.size()) < n) {
        throw std::invalid_argument("PCGSolver::solve: b/x shorter than A rows");
    }
    if (static_cast<int>(A.diag.gpu_size()) < n) {
        throw std::invalid_argument(
            "PCGSolver::solve: A.diag has fewer than A.num_block_rows() entries; "
            "call A.build_topology(...) before solving");
    }

    if (n != num_block_rows_) {
        initialize(n);
    }

    CHYSX_NVTX_RANGE_COLOUR("pcg::solve", 0xfff1c40f);

    auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    const int grid = grid_for(n);
    const auto stream_uintptr = cuda_stream;

    // M_inv = inverse(diag).  A.diag's contents are refreshed every
    // step by the cloth solver before this call.  Note that the
    // preconditioner is built only from A's diagonal -- contact
    // contributions live in the COO sidecar and are NOT folded into
    // M_inv.  At large contact stiffness this can slow convergence;
    // if it becomes a problem we can build a "contact diag lump"
    // buffer and add it to A.diag before inverting.  For now
    // elastic-only Jacobi is plenty.
    invert_diag_kernel<<<grid, kBlockDim, 0, stream>>>(
        A.diag.gpu_data(), M_inv_.gpu_data(), n);

    // We DO NOT touch `x` here.  Whatever the caller put in `x` is the
    // initial guess for the CG iteration; the cloth simulator passes
    // the previous frame's `dx` so that we warm-start from a solution
    // that was already close to the new one (cloth state changes
    // smoothly between consecutive timesteps).  This typically halves
    // the iteration count needed to drive the residual to a given
    // tolerance compared with the cold-start `x = 0`.
    //
    // Initial residual r_0 = b - (A + C) x_0.  Implemented as:
    //   r_0 = b                         (memcpy)
    //   r_0 = -A x_0 + 1 * r_0          (spmv with alpha=-1, beta=1)
    //   r_0 += -C x_0                   (apply_contact_spmv, alpha=-1)
    check_cuda(cudaMemcpyAsync(r_.gpu_data(), b.data(),
                               n * sizeof(math::Vec3f),
                               cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpyAsync(r=b)");
    sparse::spmv(A, x,
                 DeviceSpan<math::Vec3f>::from(r_),
                 -1.0f, 1.0f,
                 stream_uintptr);
    if (contact != nullptr && contact->active()) {
        collision::apply_contact_spmv(*contact, x.data(), r_.gpu_data(),
                                      n, -1.0f, stream_uintptr);
    }

    apply_jacobi_kernel<<<grid, kBlockDim, 0, stream>>>(
        M_inv_.gpu_data(), r_.gpu_data(), z_.gpu_data(), n);

    check_cuda(cudaMemcpyAsync(p_.gpu_data(), z_.gpu_data(),
                               n * sizeof(math::Vec3f),
                               cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpyAsync(p=z)");

    check_cuda(cudaMemsetAsync(coeff_.gpu_data(), 0, 4 * sizeof(float), stream),
               "cudaMemsetAsync(coeff)");
    dot_kernel<kBlockDim><<<grid, kBlockDim, 0, stream>>>(
        r_.gpu_data(), z_.gpu_data(), &coeff_.gpu_data()[0], n);

    for (int iter = 0; iter < params.max_iterations; ++iter) {
        // Ap = (A + C) * p
        sparse::spmv(A,
                     DeviceSpan<math::Vec3f>::from(p_),
                     DeviceSpan<math::Vec3f>::from(Ap_),
                     1.0f, 0.0f,
                     stream_uintptr);
        if (contact != nullptr && contact->active()) {
            collision::apply_contact_spmv(*contact, p_.gpu_data(),
                                          Ap_.gpu_data(), n, 1.0f,
                                          stream_uintptr);
        }

        // sigma = <p, Ap>  -> coeff_[1]
        check_cuda(cudaMemsetAsync(&coeff_.gpu_data()[1], 0, sizeof(float),
                                   stream),
                   "cudaMemsetAsync(coeff[1])");
        dot_kernel<kBlockDim><<<grid, kBlockDim, 0, stream>>>(
            p_.gpu_data(), Ap_.gpu_data(), &coeff_.gpu_data()[1], n);

        scalar_div_kernel<<<1, 1, 0, stream>>>(
            &coeff_.gpu_data()[0], &coeff_.gpu_data()[1], &coeff_.gpu_data()[3]);

        axpy_dev_kernel<<<grid, kBlockDim, 0, stream>>>(
            &coeff_.gpu_data()[3], p_.gpu_data(), x.data(), n);

        naxpy_dev_kernel<<<grid, kBlockDim, 0, stream>>>(
            &coeff_.gpu_data()[3], Ap_.gpu_data(), r_.gpu_data(), n);

        apply_jacobi_kernel<<<grid, kBlockDim, 0, stream>>>(
            M_inv_.gpu_data(), r_.gpu_data(), z_.gpu_data(), n);

        check_cuda(cudaMemsetAsync(&coeff_.gpu_data()[2], 0, sizeof(float),
                                   stream),
                   "cudaMemsetAsync(coeff[2])");
        dot_kernel<kBlockDim><<<grid, kBlockDim, 0, stream>>>(
            r_.gpu_data(), z_.gpu_data(), &coeff_.gpu_data()[2], n);

        scalar_div_kernel<<<1, 1, 0, stream>>>(
            &coeff_.gpu_data()[2], &coeff_.gpu_data()[0], &coeff_.gpu_data()[3]);

        update_p_kernel<<<grid, kBlockDim, 0, stream>>>(
            &coeff_.gpu_data()[3], z_.gpu_data(), p_.gpu_data(), n);

        scalar_copy_kernel<<<1, 1, 0, stream>>>(
            &coeff_.gpu_data()[2], &coeff_.gpu_data()[0]);
    }

    return params.max_iterations;
}

float PCGSolver::last_residual() {
    if (coeff_.gpu_size() == 0) return 0.0f;
    coeff_.copy_to_host();
    return coeff_.cpu_data()[0];
}

}  // namespace solver
}  // namespace chysx
