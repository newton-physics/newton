// SPDX-License-Identifier: Apache-2.0
//
// chysx::sparse::BlockCSR3
//
// Square block-sparse matrix for the implicit-Euler linear system.
// Every block is a 3x3 single-precision matrix (math::Mat3f), and a
// "block row / column" corresponds to one particle (so the scalar
// dimension of the matrix is 3 * num_block_rows).
//
// Storage
// -------
// The matrix is split into a *diagonal* part and an *off-diagonal*
// part stored separately:
//
//   diag         : Mat3f[N]                  (one block per particle)
//   row_offsets  : int[N + 1]                (CSR row pointer, off-diag)
//   col_indices  : int[nnz_off]              (block column index)
//   values       : Mat3f[nnz_off]            (off-diag blocks only)
//
// The split is what cuda-cloth calls `A_diag` + `A_values`, and it
// has two big payoffs for an implicit-Euler cloth solver:
//
//   * the per-particle inertia + pin contribution (diagonal-only)
//     touches only `diag`, so those updates are an O(N) memset+kernel,
//     not a sparse-matrix update;
//
//   * the block-Jacobi preconditioner used by `chysx::solver::PCGSolver`
//     reduces to "invert each block of `diag`" — no need for a separate
//     `extract_diagonal` pass that scans the CSR structure looking for
//     (i, i) entries.
//
// Build flow
// ----------
// 1. Call `build_topology(N, host_rows, host_cols, num_pairs)` once
//    with the off-diagonal (i, j) pairs your constraints will write
//    into.  Pairs with i == j are dropped (they belong to `diag`).
//    The build sorts and dedupes in-place; both directions of an
//    edge must be supplied (the matrix is symmetric in our use, but
//    the storage is not).
//
// 2. Per timestep call `set_zero()` to clear `diag` and `values`,
//    then have each constraint scatter its local Hessian blocks
//    directly into the matrix using slot indices computed by
//    `resolve_slots()` (or the per-constraint `bind_hessian_layout`
//    helper that wraps it).

#pragma once

#include <cstdint>
#include <vector>

#include "../math/matrix.cuh"
#include "../math/vec.cuh"
#include "../memory/cuda_array.h"

namespace chysx {
namespace sparse {

class BlockCSR3 {
public:
    BlockCSR3() = default;

    // Move-only.  Copying a sparse matrix would silently double GPU
    // memory traffic and is almost never what callers want.
    BlockCSR3(const BlockCSR3&) = delete;
    BlockCSR3& operator=(const BlockCSR3&) = delete;
    BlockCSR3(BlockCSR3&&) noexcept = default;
    BlockCSR3& operator=(BlockCSR3&&) noexcept = default;

    // ---- topology ----------------------------------------------------

    int num_block_rows() const noexcept { return num_block_rows_; }
    int num_off_diag_blocks() const noexcept {
        return static_cast<int>(values.gpu_size());
    }

    // ---- data buffers -----------------------------------------------

    CudaArray<math::Mat3f>  diag;          // size = num_block_rows
    CudaArray<int>          row_offsets;   // size = num_block_rows + 1
    CudaArray<int>          col_indices;   // size = nnz_off
    CudaArray<math::Mat3f>  values;        // size = nnz_off

    // ---- builders ---------------------------------------------------

    // Allocate `diag` (zeroed) and build the off-diagonal CSR
    // structure from a host-side list of (row, col) pairs.  Pairs
    // with row == col are dropped; duplicates are merged.
    //
    // Both directions of every off-diagonal entry must be supplied
    // because we don't know in advance which side a constraint will
    // contribute to (the matrix is structurally symmetric for our
    // cloth physics, but storage is general).
    //
    // After this call:
    //   * `diag` is allocated to length N and zero-initialised on the
    //     device;
    //   * `row_offsets` / `col_indices` are populated on both host and
    //     device, with `values` allocated and zero-initialised;
    //   * a host-side copy of `row_offsets` / `col_indices` is cached
    //     internally so subsequent `resolve_slots()` calls don't have
    //     to round-trip through the device.
    void build_topology(int num_block_rows,
                        const int* host_rows,
                        const int* host_cols,
                        int num_pairs);

    // Zero `diag` and `values` in place.  Topology untouched.  Cheap
    // (two cudaMemsetAsync calls) — call it once per step before any
    // constraint scatters its Hessian into this matrix.
    void set_zero(std::uintptr_t cuda_stream = 0);

    // Look up the slot of every (row, col) pair on the host side.
    //
    //   * If row == col, encodes the diagonal entry as `-row - 1`
    //     (so callers can use one branch:
    //         `slot < 0 ? diag[-slot - 1] : values[slot]`).
    //   * Otherwise, returns the index in `values` of (row, col).
    //
    // Throws if any off-diagonal pair is not in topology.  Bounds-
    // checks every (row, col) against `num_block_rows()`.
    void resolve_slots(const int* host_rows,
                       const int* host_cols,
                       int* out_slots,
                       int num_pairs) const;

private:
    int num_block_rows_ = 0;

    // Host-side copies kept after `build_topology` so that
    // `resolve_slots()` can binary-search without a device round-
    // trip.  These mirror `row_offsets` / `col_indices` exactly.
    std::vector<int> host_row_offsets_;
    std::vector<int> host_col_indices_;
};

// y = alpha * A * x + beta * y
//
// `x` and `y` both have length A.num_block_rows() (the matrix is
// square in our usage).  beta == 0 is special-cased (no read of y).
// When `cuda_stream != 0` the launch is asynchronous on that stream;
// otherwise it queues on the default stream.
void spmv(const BlockCSR3& A,
          DeviceSpan<math::Vec3f> x,
          DeviceSpan<math::Vec3f> y,
          float alpha = 1.0f,
          float beta  = 0.0f,
          std::uintptr_t cuda_stream = 0);

}  // namespace sparse
}  // namespace chysx
