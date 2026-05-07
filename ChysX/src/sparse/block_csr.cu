// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::sparse::BlockCSR3 + spmv().
//
// The matrix splits into a diagonal `diag[N]` and an off-diagonal CSR
// (`row_offsets`, `col_indices`, `values`).  Topology is supposed to
// be set up once per simulation (mesh + spring + FEM topology stays
// constant from frame to frame), so the build path lives entirely on
// the host and uses std::sort / std::unique — there is no per-step
// host work.

#include "block_csr.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace chysx {
namespace sparse {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("chysx::sparse::BlockCSR3: ") +
                                 what + " failed: " + cudaGetErrorString(err));
    }
}

template <typename T>
void upload_vector(CudaArray<T>& dst, const std::vector<T>& src) {
    const std::size_t n = src.size();
    dst.resize(n);
    if (n == 0) return;
    std::memcpy(dst.cpu_data(), src.data(), n * sizeof(T));
    dst.copy_to_device();
}

}  // namespace

// ----------------------------------------------------------------------------
// CUDA kernels
// ----------------------------------------------------------------------------

namespace {

// y_i = alpha * (diag[i] x_i + sum_{k in row i} values[k] x[col[k]])
//        + beta * y_i.
__global__ void spmv_kernel(int n_block_rows,
                            const math::Mat3f* __restrict__ diag,
                            const int* __restrict__ row_offsets,
                            const int* __restrict__ col_indices,
                            const math::Mat3f* __restrict__ values,
                            const math::Vec3f* __restrict__ x,
                            math::Vec3f* __restrict__ y,
                            float alpha,
                            float beta) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_block_rows) return;

    // Diagonal contribution is always present.
    math::Vec3f acc = diag[row] * x[row];

    // Off-diagonal contributions (may be zero entries if row has no
    // neighbours, in which case beg == end and the loop is skipped).
    const int beg = row_offsets[row];
    const int end = row_offsets[row + 1];
    for (int k = beg; k < end; ++k) {
        const int col = col_indices[k];
        acc += values[k] * x[col];
    }

    if (beta == 0.0f) {
        y[row] = acc * alpha;
    } else {
        y[row] = y[row] * beta + acc * alpha;
    }
}

}  // namespace

// ----------------------------------------------------------------------------
// BlockCSR3
// ----------------------------------------------------------------------------

void BlockCSR3::build_topology(int num_block_rows,
                               const int* host_rows,
                               const int* host_cols,
                               int num_pairs) {
    if (num_block_rows < 0 || num_pairs < 0) {
        throw std::invalid_argument("BlockCSR3::build_topology: negative size");
    }

    num_block_rows_ = num_block_rows;

    // Allocate diag (length N) and zero on device.  This is the only
    // place the diagonal is sized; it's untouched by set_zero() but
    // its values are reset to zero there.
    diag.resize(static_cast<std::size_t>(num_block_rows));
    if (num_block_rows > 0) {
        check_cuda(cudaMemset(diag.gpu_data(), 0,
                              num_block_rows * sizeof(math::Mat3f)),
                   "cudaMemset(diag)");
    }

    // Filter out diagonal pairs and dedupe off-diagonal pairs.
    std::vector<std::pair<int, int>> tmp;
    tmp.reserve(static_cast<std::size_t>(num_pairs));
    for (int k = 0; k < num_pairs; ++k) {
        const int r = host_rows[k];
        const int c = host_cols[k];
        if (r < 0 || r >= num_block_rows || c < 0 || c >= num_block_rows) {
            throw std::out_of_range("BlockCSR3::build_topology: pair out of range");
        }
        if (r == c) continue;  // diagonal — handled by `diag` separately
        tmp.emplace_back(r, c);
    }
    std::sort(tmp.begin(), tmp.end());
    tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());

    // Build CSR row_offsets from the (sorted) row indices.
    host_row_offsets_.assign(static_cast<std::size_t>(num_block_rows) + 1, 0);
    host_col_indices_.resize(tmp.size());
    for (std::size_t i = 0; i < tmp.size(); ++i) {
        host_row_offsets_[tmp[i].first + 1] += 1;
        host_col_indices_[i] = tmp[i].second;
    }
    for (int i = 0; i < num_block_rows; ++i) {
        host_row_offsets_[i + 1] += host_row_offsets_[i];
    }

    upload_vector(row_offsets, host_row_offsets_);
    upload_vector(col_indices, host_col_indices_);

    // Allocate `values` to nnz_off, zeroed on device.
    values.resize(tmp.size());
    if (!tmp.empty()) {
        check_cuda(cudaMemset(values.gpu_data(), 0,
                              tmp.size() * sizeof(math::Mat3f)),
                   "cudaMemset(values)");
    }
}

void BlockCSR3::set_zero(std::uintptr_t cuda_stream) {
    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    if (num_block_rows_ > 0) {
        check_cuda(cudaMemsetAsync(diag.gpu_data(), 0,
                                   num_block_rows_ * sizeof(math::Mat3f),
                                   stream),
                   "cudaMemsetAsync(diag)");
    }
    const int nnz_off = num_off_diag_blocks();
    if (nnz_off > 0) {
        check_cuda(cudaMemsetAsync(values.gpu_data(), 0,
                                   nnz_off * sizeof(math::Mat3f), stream),
                   "cudaMemsetAsync(values)");
    }
    if (cuda_stream == 0) {
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
}

void BlockCSR3::resolve_slots(const int* host_rows,
                              const int* host_cols,
                              int* out_slots,
                              int num_pairs) const {
    if (num_pairs < 0) {
        throw std::invalid_argument("BlockCSR3::resolve_slots: negative count");
    }
    if (num_pairs > 0 &&
        (host_rows == nullptr || host_cols == nullptr || out_slots == nullptr)) {
        throw std::invalid_argument("BlockCSR3::resolve_slots: null pointer");
    }

    for (int k = 0; k < num_pairs; ++k) {
        const int r = host_rows[k];
        const int c = host_cols[k];
        if (r < 0 || r >= num_block_rows_ || c < 0 || c >= num_block_rows_) {
            throw std::out_of_range("BlockCSR3::resolve_slots: pair out of range");
        }
        if (r == c) {
            // Diagonal entry encoded as -r - 1 (so any negative slot
            // means "scatter into diag[-slot - 1]" inside a kernel).
            out_slots[k] = -r - 1;
            continue;
        }
        // Binary-search col_indices in the slice [beg, end) for `c`.
        const int beg = host_row_offsets_[r];
        const int end = host_row_offsets_[r + 1];
        const auto first = host_col_indices_.begin() + beg;
        const auto last  = host_col_indices_.begin() + end;
        const auto it = std::lower_bound(first, last, c);
        if (it == last || *it != c) {
            throw std::out_of_range(
                "BlockCSR3::resolve_slots: off-diagonal (row, col) not in topology");
        }
        out_slots[k] = static_cast<int>(it - host_col_indices_.begin());
    }
}

// ----------------------------------------------------------------------------
// SPMV
// ----------------------------------------------------------------------------

void spmv(const BlockCSR3& A,
          DeviceSpan<math::Vec3f> x,
          DeviceSpan<math::Vec3f> y,
          float alpha,
          float beta,
          std::uintptr_t cuda_stream) {
    const int n_rows = A.num_block_rows();
    if (n_rows == 0) return;

    if (static_cast<int>(x.size()) < n_rows) {
        throw std::invalid_argument("spmv: x is shorter than num_block_rows");
    }
    if (static_cast<int>(y.size()) < n_rows) {
        throw std::invalid_argument("spmv: y is shorter than num_block_rows");
    }

    const int block = 256;
    const int grid = (n_rows + block - 1) / block;
    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    spmv_kernel<<<grid, block, 0, stream>>>(
        n_rows,
        A.diag.gpu_data(),
        A.row_offsets.gpu_data(),
        A.col_indices.gpu_data(),
        A.values.gpu_data(),
        x.data(),
        y.data(),
        alpha, beta);

    if (cuda_stream == 0) {
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
}

}  // namespace sparse
}  // namespace chysx
