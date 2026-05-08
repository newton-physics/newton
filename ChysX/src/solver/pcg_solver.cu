// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::solver::PCGSolver.

#include "pcg_solver.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

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

// ---------------------------------------------------------------------------
// PCGSolver — body submission (shared by direct + capture paths)
// ---------------------------------------------------------------------------

void PCGSolver::submit_solve_ops_(cudaStream_t stream,
                                  const sparse::BlockCSR3& A,
                                  DeviceSpan<math::Vec3f> b,
                                  DeviceSpan<math::Vec3f> x,
                                  const PCGParams& params) {
    const int n = num_block_rows_;
    const int grid = grid_for(n);

    // M_inv = inverse(diag).  A.diag's contents are refreshed every
    // step by the cloth solver before this call; the kernel reads
    // those latest values whether we run it directly or through a
    // captured graph.
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
    // Initial residual r_0 = b - A x_0.  Implemented as:
    //   r_0 = b                         (memcpy)
    //   r_0 = -A x_0 + 1 * r_0          (spmv with alpha=-1, beta=1)
    check_cuda(cudaMemcpyAsync(r_.gpu_data(), b.data(),
                               n * sizeof(math::Vec3f),
                               cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpyAsync(r=b)");
    sparse::spmv(A, x,
                 DeviceSpan<math::Vec3f>::from(r_),
                 -1.0f, 1.0f,
                 reinterpret_cast<std::uintptr_t>(stream));

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
        // Ap = A * p
        sparse::spmv(A,
                     DeviceSpan<math::Vec3f>::from(p_),
                     DeviceSpan<math::Vec3f>::from(Ap_),
                     1.0f, 0.0f,
                     reinterpret_cast<std::uintptr_t>(stream));

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
}

// ---------------------------------------------------------------------------
// PCGSolver — graph capture / replay
// ---------------------------------------------------------------------------

void PCGSolver::ensure_graph_resources_() {
    if (graph_stream_ != nullptr) return;

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags(graph_stream)");
    graph_stream_ = stream;

    cudaEvent_t e_pre = nullptr;
    cudaEvent_t e_post = nullptr;
    // `cudaEventDisableTiming` lets the driver use the cheap
    // synchronisation-only event implementation; we never read these
    // for timing.
    check_cuda(cudaEventCreateWithFlags(&e_pre, cudaEventDisableTiming),
               "cudaEventCreate(pre_event)");
    check_cuda(cudaEventCreateWithFlags(&e_post, cudaEventDisableTiming),
               "cudaEventCreate(post_event)");
    pre_event_ = e_pre;
    post_event_ = e_post;
}

void PCGSolver::capture_graph_(const sparse::BlockCSR3& A,
                               DeviceSpan<math::Vec3f> b,
                               DeviceSpan<math::Vec3f> x,
                               const PCGParams& params) {
    CHYSX_NVTX_RANGE_COLOUR("pcg::graph_capture", 0xfff1c40f);

    if (graph_exec_ != nullptr) {
        // Drop the previous executable before building a new one.
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }

    cudaStream_t s = graph_stream_;

    // ThreadLocal mode keeps the capture confined to this stream on
    // this thread — other Warp / pybind code running concurrently
    // (e.g. Newton viewer kernels) won't accidentally end up in our
    // graph.
    check_cuda(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal),
               "cudaStreamBeginCapture");

    submit_solve_ops_(s, A, b, x, params);

    cudaGraph_t graph = nullptr;
    check_cuda(cudaStreamEndCapture(s, &graph), "cudaStreamEndCapture");

    cudaGraphExec_t exec = nullptr;
    // 12.x: cudaGraphInstantiate(&exec, graph, 0).  No log-buffer
    // params required; if the driver rejects a node we let the
    // exception propagate.
    cudaError_t inst_err = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    check_cuda(inst_err, "cudaGraphInstantiate");

    graph_exec_ = exec;
}

// ---------------------------------------------------------------------------
// PCGSolver — public entry points
// ---------------------------------------------------------------------------

PCGSolver::PCGSolver(PCGSolver&& other) noexcept
    : num_block_rows_(other.num_block_rows_),
      r_(std::move(other.r_)),
      p_(std::move(other.p_)),
      z_(std::move(other.z_)),
      Ap_(std::move(other.Ap_)),
      M_inv_(std::move(other.M_inv_)),
      coeff_(std::move(other.coeff_)),
      graph_enabled_(other.graph_enabled_),
      graph_stream_(other.graph_stream_),
      pre_event_(other.pre_event_),
      post_event_(other.post_event_),
      graph_exec_(other.graph_exec_),
      cached_key_(other.cached_key_) {
    other.num_block_rows_ = 0;
    other.graph_stream_ = nullptr;
    other.pre_event_ = nullptr;
    other.post_event_ = nullptr;
    other.graph_exec_ = nullptr;
    other.cached_key_ = GraphKey{};
}

PCGSolver& PCGSolver::operator=(PCGSolver&& other) noexcept {
    if (this == &other) return *this;

    // Tear down our own resources before stealing.
    invalidate_graph();
    if (graph_stream_) {
        cudaStreamDestroy(graph_stream_);
        graph_stream_ = nullptr;
    }
    if (pre_event_)  { cudaEventDestroy(pre_event_);  pre_event_  = nullptr; }
    if (post_event_) { cudaEventDestroy(post_event_); post_event_ = nullptr; }

    num_block_rows_ = other.num_block_rows_;
    r_     = std::move(other.r_);
    p_     = std::move(other.p_);
    z_     = std::move(other.z_);
    Ap_    = std::move(other.Ap_);
    M_inv_ = std::move(other.M_inv_);
    coeff_ = std::move(other.coeff_);
    graph_enabled_ = other.graph_enabled_;
    graph_stream_  = other.graph_stream_;
    pre_event_     = other.pre_event_;
    post_event_    = other.post_event_;
    graph_exec_    = other.graph_exec_;
    cached_key_    = other.cached_key_;

    other.num_block_rows_ = 0;
    other.graph_stream_ = nullptr;
    other.pre_event_ = nullptr;
    other.post_event_ = nullptr;
    other.graph_exec_ = nullptr;
    other.cached_key_ = GraphKey{};
    return *this;
}

PCGSolver::~PCGSolver() {
    // Destructors must not throw — swallow CUDA errors here.  Any
    // runtime that's already shut down (e.g. process teardown after a
    // device reset) will return errors that don't matter at this
    // point.
    if (graph_exec_)  cudaGraphExecDestroy(graph_exec_);
    if (graph_stream_) cudaStreamDestroy(graph_stream_);
    if (pre_event_)   cudaEventDestroy(pre_event_);
    if (post_event_)  cudaEventDestroy(post_event_);
}

void PCGSolver::set_graph_enabled(bool enabled) {
    if (graph_enabled_ == enabled) return;
    graph_enabled_ = enabled;
    if (!enabled) invalidate_graph();
}

void PCGSolver::invalidate_graph() {
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    cached_key_ = GraphKey{};
}

int PCGSolver::solve(const sparse::BlockCSR3& A,
                     DeviceSpan<math::Vec3f> b,
                     DeviceSpan<math::Vec3f> x,
                     const PCGParams& params,
                     std::uintptr_t cuda_stream) {
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
        // r_/p_/z_/Ap_/M_inv_/coeff_ pointers may have moved — drop
        // the cached graph since it referenced the old ones.
        invalidate_graph();
    }

    auto caller_stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    if (!graph_enabled_) {
        // Legacy path — submit every kernel directly to the caller's
        // stream.  Useful for debugging or when the caller has already
        // wrapped solve() inside its own CUDA Graph capture.
        submit_solve_ops_(caller_stream, A, b, x, params);
        return params.max_iterations;
    }

    ensure_graph_resources_();

    GraphKey key{};
    key.n            = n;
    key.max_iter     = params.max_iterations;
    key.A_diag       = A.diag.gpu_data();
    key.A_values     = A.values.gpu_data();
    key.A_row_offsets = A.row_offsets.gpu_data();
    key.A_col_indices = A.col_indices.gpu_data();
    key.b            = b.data();
    key.x            = x.data();

    if (graph_exec_ == nullptr || cached_key_ != key) {
        capture_graph_(A, b, x, params);
        cached_key_ = key;
    }

    // Rendezvous: caller_stream → graph_stream_ → caller_stream.
    //
    // Without these events the captured graph would race with
    // upstream work the caller queued on caller_stream (e.g. the
    // Hessian scatter that just filled `b = rhs_`) and with anything
    // the caller queues afterwards (e.g. `finalize_step_kernel`).
    check_cuda(cudaEventRecord(pre_event_, caller_stream),
               "cudaEventRecord(pre_event)");
    check_cuda(cudaStreamWaitEvent(graph_stream_, pre_event_, 0),
               "cudaStreamWaitEvent(graph_stream <- pre)");

    check_cuda(cudaGraphLaunch(graph_exec_, graph_stream_),
               "cudaGraphLaunch");

    check_cuda(cudaEventRecord(post_event_, graph_stream_),
               "cudaEventRecord(post_event)");
    check_cuda(cudaStreamWaitEvent(caller_stream, post_event_, 0),
               "cudaStreamWaitEvent(caller_stream <- post)");

    return params.max_iterations;
}

float PCGSolver::last_residual() {
    if (coeff_.gpu_size() == 0) return 0.0f;
    coeff_.copy_to_host();
    return coeff_.cpu_data()[0];
}

}  // namespace solver
}  // namespace chysx
