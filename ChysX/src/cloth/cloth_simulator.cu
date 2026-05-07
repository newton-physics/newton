// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::cloth::ClothSimulator.
//
// Each `step()` performs one Quasi-Newton iteration of the implicit
// Euler problem
//
//     min_x  Phi(x) = 1/(2 dt^2) (x - x_tilde)^T M (x - x_tilde) + E(x)
//
// where  x_tilde = x_n + dt v_n + dt^2 g  is the inertial predictor.
// Linearising around x_tilde gives
//
//     (M/dt^2 + H_E(x_tilde))  dx  =  -grad E(x_tilde)
//
// which we assemble as a block-CSR matrix (3x3 blocks per particle
// pair) by accumulating triplets from every constraint plus a
// diagonal `M[i,i]/dt^2 * I` block per particle, then solve with
// PCG.  The displacement `dx` is added to `x_tilde` to produce the
// new positions and finite-differenced into the new velocities.
//
// Pin constraints participate naturally through this same machinery:
// `(k I)` on the diagonal + `k (target - x_tilde)` on the RHS, with
// k chosen large enough to act as a hard pin.

#include "cloth_simulator.h"

#include <cuda_runtime.h>
#include <vector_types.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../profile/nvtx_range.h"

namespace chysx {
namespace cloth {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("chysx::cloth: ") + what +
                                 " failed: " + cudaGetErrorString(err));
    }
}

// Convert per-particle inverse mass into per-particle mass.
//
// Newton's convention encodes "infinite-mass / kinematic" particles
// as `inv_mass == 0`.  We map that to a finite-but-huge mass so the
// inertia diagonal block dominates the linear system and the PCG
// solve effectively pins those particles in place — same behaviour
// users get when they pass an explicit `PinConstraint`.
//
// `inv_mass_ptr` may be null; in that case every mass is set to 1.0
// kg (matches Newton's default `add_cloth_grid` mass spec for the
// cases we care about: each particle ends up well-defined and the
// simulation degenerates gracefully if the host forgot to wire
// inv_mass through).
__global__ void mass_from_inv_mass_kernel(const float* __restrict__ inv_mass,
                                          float* __restrict__ mass,
                                          int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (inv_mass == nullptr) {
        mass[i] = 1.0f;
        return;
    }
    const float w = inv_mass[i];
    mass[i] = (w > 1.0e-12f) ? (1.0f / w) : 1.0e8f;
}

// Inertial predictor: x_tilde = x_n + dt * v_n + dt^2 * g.
__global__ void compute_x_tilde_kernel(const float3* __restrict__ pos,
                                       const float3* __restrict__ vel,
                                       float3* __restrict__ x_tilde,
                                       int n,
                                       float3 g,
                                       float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float3 p = pos[i];
    const float3 v = vel[i];
    x_tilde[i] = make_float3(p.x + dt * v.x + dt * dt * g.x,
                             p.y + dt * v.y + dt * dt * g.y,
                             p.z + dt * v.z + dt * dt * g.z);
}

// Negate every Vec3 in place (used to flip grad -> RHS).
__global__ void negate_vec3_kernel(float3* __restrict__ v, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float3 a = v[i];
    v[i] = make_float3(-a.x, -a.y, -a.z);
}

// Add the inertia diagonal block (m_i / dt^2 * I_3) to `A.diag[i]`.
//
// Other constraint kernels also scatter into `A.diag` via atomicAdd,
// but they all live in separate kernel launches on the same stream,
// so within this kernel each thread is the unique writer of its row
// and we can use a plain += instead of atomicAdd (cheap win since the
// load/store is non-atomic and uncontended).
__global__ void add_inertia_diag_kernel(math::Mat3f* __restrict__ A_diag,
                                        const float* __restrict__ mass,
                                        int n,
                                        float inv_dt2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float d = mass[i] * inv_dt2;
    // Mat3f stores its 9 elements row-major in `data[9]`.
    A_diag[i].data[0] += d;
    A_diag[i].data[4] += d;
    A_diag[i].data[8] += d;
}

// Final position / velocity update.
//
//     x_{n+1} = x_tilde + dx
//     v_{n+1} = (x_{n+1} - x_n) / dt        (with optional damping)
//
// `damping` follows the existing freefall convention: an exponential
// per-second decay applied as `exp(-damping * dt)`.
__global__ void finalize_step_kernel(float3* __restrict__ pos,
                                     float3* __restrict__ vel,
                                     const float3* __restrict__ x_n,
                                     const float3* __restrict__ x_tilde,
                                     const float3* __restrict__ dx,
                                     int n,
                                     float dt,
                                     float damping) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float3 xt = x_tilde[i];
    const float3 d  = dx[i];
    const float3 xn = x_n[i];
    const float3 xnew = make_float3(xt.x + d.x, xt.y + d.y, xt.z + d.z);
    float3 vnew = make_float3((xnew.x - xn.x) / dt,
                              (xnew.y - xn.y) / dt,
                              (xnew.z - xn.z) / dt);
    if (damping > 0.0f) {
        const float decay = expf(-damping * dt);
        vnew.x *= decay;
        vnew.y *= decay;
        vnew.z *= decay;
    }
    pos[i] = xnew;
    vel[i] = vnew;
}

}  // namespace

void ClothSimulator::set_external_buffers(std::uintptr_t pos_ptr,
                                          std::uintptr_t vel_ptr,
                                          int particle_count,
                                          std::uintptr_t inv_mass_ptr) noexcept {
    const auto n = static_cast<std::size_t>(particle_count);
    buffers_.pos = DeviceSpan<math::Vec3f>::from_raw(pos_ptr, n);
    buffers_.vel = DeviceSpan<math::Vec3f>::from_raw(vel_ptr, n);
    buffers_.inv_mass = DeviceSpan<float>::from_raw(inv_mass_ptr, n);
}

void ClothSimulator::set_pins(const int* host_indices,
                              const math::Vec3f* host_targets,
                              int n,
                              float stiffness) {
    pins_.set_stiffness(stiffness);
    pins_.set_pins(host_indices, host_targets, n);
    // Pins are diagonal-only — they don't add off-diag pairs to the
    // CSR structure — but the per-instance slot LUT still has to be
    // rebuilt against `H_` whenever the pin set changes.
    topology_dirty_ = true;
}

void ClothSimulator::clear_pins() noexcept {
    pins_.set_pins(nullptr, nullptr, 0);
    topology_dirty_ = true;
}

void ClothSimulator::set_mesh(const math::Vec3i* host_triangles,
                              int n_triangles) {
    if (n_triangles < 0) {
        throw std::invalid_argument(
            "ClothSimulator::set_mesh: negative triangle count");
    }
    buffers_.tris.resize(static_cast<std::size_t>(n_triangles));
    if (n_triangles > 0) {
        std::memcpy(buffers_.tris.cpu_data(), host_triangles,
                    n_triangles * sizeof(math::Vec3i));
        buffers_.tris.copy_to_device();
    }

    // Build the unique undirected edge list on the host (same
    // dedup-and-sort algorithm as TriangleMesh::build_edges; replicated
    // here to avoid a circular dependency between cloth/ and geometry/).
    std::vector<std::pair<int, int>> tmp;
    tmp.reserve(static_cast<std::size_t>(n_triangles) * 3);
    for (int t = 0; t < n_triangles; ++t) {
        const math::Vec3i tri = host_triangles[t];
        tmp.emplace_back(std::min(tri.x, tri.y), std::max(tri.x, tri.y));
        tmp.emplace_back(std::min(tri.y, tri.z), std::max(tri.y, tri.z));
        tmp.emplace_back(std::min(tri.z, tri.x), std::max(tri.z, tri.x));
    }
    std::sort(tmp.begin(), tmp.end());
    tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());

    buffers_.edges.resize(tmp.size());
    auto* out = buffers_.edges.cpu_data();
    for (std::size_t i = 0; i < tmp.size(); ++i) {
        out[i] = math::Vec2i(tmp[i].first, tmp[i].second);
    }
    if (!tmp.empty()) {
        buffers_.edges.copy_to_device();
    }
    // Mesh changed => topology must be rebuilt next step.
    topology_dirty_ = true;
}

void ClothSimulator::build_springs_from_current_positions(
    float stiffness, std::uintptr_t cuda_stream) {
    const int n_edges = static_cast<int>(buffers_.edges.cpu_size());
    if (n_edges == 0) {
        springs_.set_stiffness(stiffness);
        springs_.set_springs(nullptr, nullptr, 0);
        topology_dirty_ = true;
        return;
    }
    if (buffers_.pos.data() == nullptr) {
        throw std::runtime_error(
            "ClothSimulator::build_springs_from_current_positions: external "
            "positions must be set first via set_external_buffers().");
    }

    springs_.set_stiffness(stiffness);
    springs_.set_springs_from_positions(
        buffers_.edges.cpu_data(), n_edges, buffers_.pos, cuda_stream);
    topology_dirty_ = true;
}

void ClothSimulator::build_fem_stretch_from_current_positions(
    float stiffness, std::uintptr_t cuda_stream) {
    const int n_tris = static_cast<int>(buffers_.tris.cpu_size());
    if (n_tris == 0) {
        fem_stretch_.set_stiffness(stiffness);
        fem_stretch_.set_triangles_from_positions(nullptr, 0, buffers_.pos,
                                                  cuda_stream);
        topology_dirty_ = true;
        return;
    }
    if (buffers_.pos.data() == nullptr) {
        throw std::runtime_error(
            "ClothSimulator::build_fem_stretch_from_current_positions: "
            "external positions must be set first via set_external_buffers().");
    }

    fem_stretch_.set_stiffness(stiffness);
    fem_stretch_.set_triangles_from_positions(
        buffers_.tris.cpu_data(), n_tris, buffers_.pos, cuda_stream);
    topology_dirty_ = true;
}

void ClothSimulator::ensure_hessian_topology() {
    const int N = buffers_.particle_count();
    if (N <= 0) return;

    // Collect off-diagonal (i, j) pairs from spring + FEM constraints.
    // Pin contributes diagonal-only and is skipped here; the diagonal
    // entries are stored implicitly in `BlockCSR3::diag` and don't
    // need a CSR slot.
    std::vector<int> rows;
    std::vector<int> cols;

    const int n_sp = springs_.size();
    if (n_sp > 0) {
        const math::Vec2i* sp = springs_.indices().cpu_data();
        rows.reserve(rows.size() + 2u * static_cast<std::size_t>(n_sp));
        cols.reserve(cols.size() + 2u * static_cast<std::size_t>(n_sp));
        for (int e = 0; e < n_sp; ++e) {
            const int a = sp[e].x;
            const int b = sp[e].y;
            if (a == b) continue;  // degenerate spring — diag-only
            rows.push_back(a); cols.push_back(b);
            rows.push_back(b); cols.push_back(a);
        }
    }

    const int n_fm = fem_stretch_.size();
    if (n_fm > 0) {
        const math::Vec3i* tris = fem_stretch_.indices().cpu_data();
        rows.reserve(rows.size() + 6u * static_cast<std::size_t>(n_fm));
        cols.reserve(cols.size() + 6u * static_cast<std::size_t>(n_fm));
        for (int t = 0; t < n_fm; ++t) {
            const int v[3] = { tris[t].x, tris[t].y, tris[t].z };
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    if (a != b && v[a] != v[b]) {
                        rows.push_back(v[a]);
                        cols.push_back(v[b]);
                    }
                }
            }
        }
    }

    H_.build_topology(N, rows.data(), cols.data(),
                      static_cast<int>(rows.size()));

    // Bind every constraint's per-block slot LUT against the freshly
    // built topology.  Each call walks `indices_.cpu_data()` and runs
    // a binary search per block on the host — fine for cloth-scale
    // problems and amortised across many simulation steps.
    pins_.bind_hessian_layout(H_);
    springs_.bind_hessian_layout(H_);
    fem_stretch_.bind_hessian_layout(H_);

    H_num_block_rows_ = N;
    topology_dirty_ = false;
}

void ClothSimulator::resize_work_buffers(int n) {
    const auto sz = static_cast<std::size_t>(n);
    if (x_n_.gpu_size()     != sz) x_n_.resize(sz);
    if (x_tilde_.gpu_size() != sz) x_tilde_.resize(sz);
    if (dx_.gpu_size()      != sz) dx_.resize(sz);
    if (rhs_.gpu_size()     != sz) rhs_.resize(sz);
    if (mass_.gpu_size()    != sz) mass_.resize(sz);
}

void ClothSimulator::step(float dt, std::uintptr_t cuda_stream) {
    CHYSX_NVTX_RANGE_COLOUR("chysx::cloth::step", 0xff2980b9);

    const int n = buffers_.particle_count();
    if (n <= 0) {
        return;
    }
    if (buffers_.pos.data() == nullptr || buffers_.vel.data() == nullptr) {
        throw std::runtime_error(
            "chysx::cloth::ClothSimulator::step: pos / vel spans must be "
            "set via set_external_buffers() before stepping.");
    }
    if (dt <= 0.0f) {
        throw std::invalid_argument(
            "chysx::cloth::ClothSimulator::step: dt must be positive");
    }

    // Vec3f and float3 share layout (12 bytes, 4-byte aligned).
    auto* pos = reinterpret_cast<float3*>(buffers_.pos.data());
    auto* vel = reinterpret_cast<float3*>(buffers_.vel.data());
    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    const float3 g = make_float3(material_.gx, material_.gy, material_.gz);

    constexpr int block = 256;
    const int grid = (n + block - 1) / block;

    // ---- 1) lazy-resize work buffers ---------------------------------
    resize_work_buffers(n);

    // ---- 2) snapshot x_n and compute x_tilde ------------------------
    {
        CHYSX_NVTX_RANGE_COLOUR("step::predictor", 0xff16a085);
        check_cuda(cudaMemcpyAsync(x_n_.gpu_data(), pos, n * sizeof(float3),
                                   cudaMemcpyDeviceToDevice, stream),
                   "memcpy x_n <- pos");

        compute_x_tilde_kernel<<<grid, block, 0, stream>>>(
            pos, vel, reinterpret_cast<float3*>(x_tilde_.gpu_data()),
            n, g, dt);
        check_cuda(cudaGetLastError(), "compute_x_tilde kernel launch");

        mass_from_inv_mass_kernel<<<grid, block, 0, stream>>>(
            buffers_.inv_mass.data(), mass_.gpu_data(), n);
        check_cuda(cudaGetLastError(), "mass_from_inv_mass kernel launch");
    }

    // ---- 3) accumulate gradient at x_tilde --------------------------
    DeviceSpan<math::Vec3f> x_tilde_span(x_tilde_.gpu_data(), static_cast<std::size_t>(n));
    DeviceSpan<math::Vec3f> rhs_span(rhs_.gpu_data(), static_cast<std::size_t>(n));
    {
        CHYSX_NVTX_RANGE_COLOUR("step::gradient", 0xff27ae60);
        check_cuda(cudaMemsetAsync(rhs_.gpu_data(), 0, n * sizeof(float3),
                                   stream),
                   "memset rhs = 0");

        pins_.accumulate_gradient(x_tilde_span, rhs_span, cuda_stream);
        springs_.accumulate_gradient(x_tilde_span, rhs_span, cuda_stream);
        fem_stretch_.accumulate_gradient(x_tilde_span, rhs_span, cuda_stream);

        // RHS = -grad E(x_tilde).  (The inertial term M(x - x_tilde)/dt^2
        // is zero at x = x_tilde and contributes nothing to RHS here.)
        negate_vec3_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<float3*>(rhs_.gpu_data()), n);
        check_cuda(cudaGetLastError(), "negate rhs kernel launch");
    }

    // ---- 4) (re)build Hessian topology if anything changed ----------
    //
    // Off-diagonal CSR structure depends on the spring + FEM index
    // tables, which are constant from frame to frame for a fixed
    // mesh.  We only redo it on changes (set_mesh / build_springs /
    // build_fem_stretch / pin add-remove) — see `topology_dirty_`.
    if (topology_dirty_ || H_num_block_rows_ != n) {
        CHYSX_NVTX_RANGE_COLOUR("step::topology_rebuild", 0xfff39c12);
        ensure_hessian_topology();
    }

    // ---- 5) zero H_ then scatter Hessian contributions --------------
    //
    // Each constraint's accumulate_hessian uses its precomputed slot
    // LUT to atomicAdd 3x3 blocks directly into H_.diag (per-particle
    // diagonal) or H_.values (off-diagonal CSR).  No host triplets,
    // no per-step CSR build, no extract_diagonal pass.
    {
        CHYSX_NVTX_RANGE_COLOUR("step::hessian", 0xff8e44ad);
        H_.set_zero(cuda_stream);

        pins_.accumulate_hessian(x_tilde_span, H_, cuda_stream);
        springs_.accumulate_hessian(x_tilde_span, H_, cuda_stream);
        fem_stretch_.accumulate_hessian(x_tilde_span, H_, cuda_stream);

        const float inv_dt2 = 1.0f / (dt * dt);
        add_inertia_diag_kernel<<<grid, block, 0, stream>>>(
            H_.diag.gpu_data(), mass_.gpu_data(), n, inv_dt2);
        check_cuda(cudaGetLastError(), "add_inertia_diag kernel launch");
    }

    // ---- 6) PCG solve  (M/dt^2 + H_E) dx = -grad --------------------
    {
        CHYSX_NVTX_RANGE_COLOUR("step::pcg", 0xffe74c3c);
        pcg_.initialize(n);
        DeviceSpan<math::Vec3f> dx_span(
            dx_.gpu_data(), static_cast<std::size_t>(n));
        solver::PCGParams params;
        params.max_iterations = pcg_max_iterations_;
        pcg_.solve(H_, rhs_span, dx_span, params, cuda_stream);
    }

    // ---- 7) finalize positions / velocities -------------------------
    {
        CHYSX_NVTX_RANGE_COLOUR("step::finalize", 0xff7f8c8d);
        finalize_step_kernel<<<grid, block, 0, stream>>>(
            pos, vel,
            reinterpret_cast<float3*>(x_n_.gpu_data()),
            reinterpret_cast<float3*>(x_tilde_.gpu_data()),
            reinterpret_cast<float3*>(dx_.gpu_data()),
            n, dt, material_.damping);
        check_cuda(cudaGetLastError(), "finalize_step kernel launch");
    }
}

}  // namespace cloth
}  // namespace chysx
