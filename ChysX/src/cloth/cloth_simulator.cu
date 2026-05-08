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
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

// Assemble the full Newton-step RHS in place.
//
// We linearise the implicit-Euler residual r(x) = M*(x - x_tilde)/dt^2
// + grad E(x) at x_0 = x_n (the previous frame's position), so the
// RHS for the linear solve becomes
//
//     RHS = -r(x_n) = (M/dt^2) * (x_tilde - x_n) - grad E(x_n)
//                   = M*v/dt + M*g            - grad E(x_n).
//
// Linearising at x_n (instead of x_tilde) is a much better choice than
// the "RHS = -grad E(x_tilde)" trick: at rest, grad E(x_n) is tiny and
// the Hessian H_E(x_n) is evaluated on a physically valid
// configuration, whereas H_E(x_tilde) would see the cloth strained by
// dt^2*g of free-fall extrapolation.  This matches cuda-cloth's
// `KernelVBDDynamic` (Dynamic.cuh) and the standard implicit-Euler
// recipe used by Baraff–Witkin.
//
// On entry rhs[i] holds grad E(x_n) accumulated by the constraints;
// on exit rhs[i] holds the full Newton RHS above.
__global__ void assemble_rhs_kernel(const float3* __restrict__ x_n,
                                    const float3* __restrict__ x_tilde,
                                    const float* __restrict__ mass,
                                    float3* __restrict__ rhs,
                                    int n,
                                    float inv_dt2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float3 xn = x_n[i];
    const float3 xt = x_tilde[i];
    const float3 g  = rhs[i];  // = grad E(x_n)
    const float m_inv_dt2 = mass[i] * inv_dt2;
    rhs[i] = make_float3(m_inv_dt2 * (xt.x - xn.x) - g.x,
                         m_inv_dt2 * (xt.y - xn.y) - g.y,
                         m_inv_dt2 * (xt.z - xn.z) - g.z);
}

// Per-triangle scatter of `surface_density * area / 3` onto each of
// the triangle's three vertices.  Uses atomicAdd because a vertex is
// shared by ~6 triangles in a regular interior region, and we want
// every contribution summed without serialising the kernel.
//
// `density_over_3 = surface_density / 3` is folded in on the host so
// the inner kernel just does one multiply per triangle.
__global__ void scatter_triangle_mass_kernel(
    const math::Vec3i* __restrict__ tris,
    const math::Vec3f* __restrict__ pos,
    float density_over_3,
    float* __restrict__ mass_out,
    int n_tri) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tri) return;

    const math::Vec3i tri = tris[t];
    const math::Vec3f x0 = pos[tri.x];
    const math::Vec3f x1 = pos[tri.y];
    const math::Vec3f x2 = pos[tri.z];

    // Triangle area = 0.5 * |(x1 - x0) × (x2 - x0)|.  We use the
    // 3-D cross product because the cloth's rest configuration may
    // not lie in any one coordinate plane (chysx doesn't assume the
    // input is planar).
    const math::Vec3f e1 = x1 - x0;
    const math::Vec3f e2 = x2 - x0;
    const float area = 0.5f * math::length(math::cross(e1, e2));

    const float dm = density_over_3 * area;
    atomicAdd(&mass_out[tri.x], dm);
    atomicAdd(&mass_out[tri.y], dm);
    atomicAdd(&mass_out[tri.z], dm);
}

// Convert a per-vertex mass buffer into per-vertex inverse mass.
// Particles with zero accumulated mass (no incident triangles) are
// emitted as `inv_mass = 0` so the cloth solver treats them as
// kinematic instead of producing a NaN.
__global__ void mass_to_inv_mass_kernel(const float* __restrict__ mass,
                                        float* __restrict__ inv_mass,
                                        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float m = mass[i];
    inv_mass[i] = (m > 1.0e-12f) ? (1.0f / m) : 0.0f;
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
//     x_{n+1} = x_n + dx
//     v_{n+1} = (x_{n+1} - x_n) / dt = dx / dt    (with optional damping)
//
// We linearise the implicit-Euler residual at x_n (see
// assemble_rhs_kernel above), so dx is the displacement *from x_n*,
// not a correction to x_tilde.  `damping` follows the existing
// freefall convention: an exponential per-second decay applied as
// `exp(-damping * dt)`.
__global__ void finalize_step_kernel(float3* __restrict__ pos,
                                     float3* __restrict__ vel,
                                     const float3* __restrict__ x_n,
                                     const float3* __restrict__ dx,
                                     int n,
                                     float dt,
                                     float damping) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float3 d  = dx[i];
    const float3 xn = x_n[i];
    const float3 xnew = make_float3(xn.x + d.x, xn.y + d.y, xn.z + d.z);
    const float inv_dt = 1.0f / dt;
    float3 vnew = make_float3(d.x * inv_dt, d.y * inv_dt, d.z * inv_dt);
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
        buffers_.tris.cpu_data(), n_tris, buffers_.pos, cuda_stream,
        /*material_rotation_rad=*/0.0f);
    topology_dirty_ = true;
}

void ClothSimulator::build_fem_shear_from_current_positions(
    float stiffness, std::uintptr_t cuda_stream) {
    // 45-degree rotation of the material (u, v) axes turns the
    // stretch energy along the diagonals — i.e. shear.  See the
    // `material_rotation_rad` notes on TriangleStretchConstraint.
    constexpr float kQuarterPi = 0.7853981633974483f;  // pi / 4

    const int n_tris = static_cast<int>(buffers_.tris.cpu_size());
    if (n_tris == 0) {
        fem_shear_.set_stiffness(stiffness);
        fem_shear_.set_triangles_from_positions(nullptr, 0, buffers_.pos,
                                                cuda_stream, kQuarterPi);
        topology_dirty_ = true;
        return;
    }
    if (buffers_.pos.data() == nullptr) {
        throw std::runtime_error(
            "ClothSimulator::build_fem_shear_from_current_positions: "
            "external positions must be set first via set_external_buffers().");
    }

    fem_shear_.set_stiffness(stiffness);
    fem_shear_.set_triangles_from_positions(
        buffers_.tris.cpu_data(), n_tris, buffers_.pos, cuda_stream,
        kQuarterPi);
    topology_dirty_ = true;
}

void ClothSimulator::build_bending_from_current_positions(
    float stiffness, std::uintptr_t cuda_stream) {
    bending_.set_stiffness(stiffness);

    const int n_tris = static_cast<int>(buffers_.tris.cpu_size());
    if (n_tris == 0) {
        bending_.set_dihedrals_from_positions(nullptr, 0, buffers_.pos,
                                              cuda_stream);
        topology_dirty_ = true;
        return;
    }
    if (buffers_.pos.data() == nullptr) {
        throw std::runtime_error(
            "ClothSimulator::build_bending_from_current_positions: "
            "external positions must be set first via set_external_buffers().");
    }

    // Auto-detect dihedrals on the host: every directed edge of every
    // CCW-oriented triangle gets a "third vertex" entry; an interior
    // edge then has both directions populated and we emit one
    // dihedral.  Indexing trick: pack (from, to) into a 64-bit key.
    //
    // For triangles oriented consistently (e.g. cloth grids from
    // Newton, where every face is CCW from above), the directed edge
    // (u -> v) appears in *exactly* one triangle, and its reverse
    // (v -> u) in *exactly* the opposite triangle on the other side
    // of the edge — exactly the (v0, v1, v2, v3) labelling our
    // BendingConstraint expects.
    const math::Vec3i* tris = buffers_.tris.cpu_data();
    const auto pack = [](int from, int to) -> std::int64_t {
        return (static_cast<std::int64_t>(from) << 32) |
               static_cast<std::int64_t>(static_cast<std::uint32_t>(to));
    };

    std::unordered_map<std::int64_t, int> third_for_edge;
    third_for_edge.reserve(static_cast<std::size_t>(3 * n_tris));
    for (int t = 0; t < n_tris; ++t) {
        const int v[3] = { tris[t].x, tris[t].y, tris[t].z };
        third_for_edge[pack(v[0], v[1])] = v[2];
        third_for_edge[pack(v[1], v[2])] = v[0];
        third_for_edge[pack(v[2], v[0])] = v[1];
    }

    std::vector<math::Vec4i> dihedrals;
    dihedrals.reserve(static_cast<std::size_t>(2 * n_tris));  // worst-case
    for (const auto& kv : third_for_edge) {
        const int v0 = static_cast<int>(kv.first >> 32);
        const int v1 = static_cast<int>(kv.first & 0xffffffff);
        if (v0 >= v1) continue;  // process each undirected edge once

        auto rev = third_for_edge.find(pack(v1, v0));
        if (rev == third_for_edge.end()) continue;  // boundary edge

        const int v2 = kv.second;
        const int v3 = rev->second;
        dihedrals.push_back(math::Vec4i{ v0, v1, v2, v3 });
    }

    const int n_dih = static_cast<int>(dihedrals.size());
    bending_.set_dihedrals_from_positions(
        n_dih > 0 ? dihedrals.data() : nullptr,
        n_dih, buffers_.pos, cuda_stream);
    topology_dirty_ = true;
}

void ClothSimulator::redistribute_mass_area_weighted(
    float surface_density, std::uintptr_t inv_mass_ptr,
    int particle_count, std::uintptr_t cuda_stream) {
    if (particle_count <= 0) return;
    if (inv_mass_ptr == 0) {
        throw std::invalid_argument(
            "ClothSimulator::redistribute_mass_area_weighted: "
            "inv_mass_ptr must be non-null");
    }
    if (surface_density <= 0.0f) {
        throw std::invalid_argument(
            "ClothSimulator::redistribute_mass_area_weighted: "
            "surface_density must be positive");
    }

    const int n_tri = static_cast<int>(buffers_.tris.gpu_size());
    if (n_tri == 0) {
        throw std::runtime_error(
            "ClothSimulator::redistribute_mass_area_weighted: "
            "call set_mesh(...) before redistributing mass.");
    }
    if (buffers_.pos.data() == nullptr ||
        static_cast<int>(buffers_.pos.size()) < particle_count) {
        throw std::runtime_error(
            "ClothSimulator::redistribute_mass_area_weighted: "
            "external positions must be set first via "
            "set_external_buffers().");
    }

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    auto* inv_mass = reinterpret_cast<float*>(inv_mass_ptr);

    // Reuse `mass_` as scratch.  step() will repopulate it from
    // `inv_mass` next frame anyway, so there's no aliasing concern.
    if (static_cast<int>(mass_.gpu_size()) < particle_count) {
        mass_.resize(particle_count);
    }
    check_cuda(cudaMemsetAsync(mass_.gpu_data(), 0,
                               particle_count * sizeof(float), stream),
               "cudaMemsetAsync(mass scratch)");

    constexpr int block = 256;
    const int grid_tri = (n_tri + block - 1) / block;
    scatter_triangle_mass_kernel<<<grid_tri, block, 0, stream>>>(
        buffers_.tris.gpu_data(),
        buffers_.pos.data(),
        surface_density / 3.0f,
        mass_.gpu_data(), n_tri);
    check_cuda(cudaGetLastError(), "scatter_triangle_mass kernel launch");

    const int grid_v = (particle_count + block - 1) / block;
    mass_to_inv_mass_kernel<<<grid_v, block, 0, stream>>>(
        mass_.gpu_data(), inv_mass, particle_count);
    check_cuda(cudaGetLastError(), "mass_to_inv_mass kernel launch");

    if (cuda_stream == 0) {
        check_cuda(cudaStreamSynchronize(stream),
                   "cudaStreamSynchronize(redistribute_mass)");
    }
}

void ClothSimulator::ensure_hessian_topology() {
    const int N = buffers_.particle_count();
    if (N <= 0) return;

    // Collect off-diagonal (i, j) pairs from FEM stretch + FEM shear +
    // bending.  Pin contributes diagonal-only and is skipped here; the
    // diagonal entries are stored implicitly in `BlockCSR3::diag` and
    // don't need a CSR slot.
    //
    // SpringConstraint is intentionally excluded from the pipeline:
    // every edge spring has the FEM stretch element covering its
    // contribution already (and at higher fidelity), so running both
    // would double-count in-plane stiffness.  See the matching skips
    // in `step()` for accumulate_gradient / accumulate_hessian.
    std::vector<int> rows;
    std::vector<int> cols;

    // FEM stretch + FEM shear share the same per-triangle (i, j)
    // off-diagonal pattern; pushing both is redundant because
    // `BlockCSR3::build_topology` dedupes, but we keep the loop
    // tolerant of the case where only one of them is installed.
    auto append_triangle_pairs = [&](const constraint::TriangleStretchConstraint& c) {
        const int n = c.size();
        if (n <= 0) return;
        const math::Vec3i* tris = c.indices().cpu_data();
        rows.reserve(rows.size() + 6u * static_cast<std::size_t>(n));
        cols.reserve(cols.size() + 6u * static_cast<std::size_t>(n));
        for (int t = 0; t < n; ++t) {
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
    };
    append_triangle_pairs(fem_stretch_);
    append_triangle_pairs(fem_shear_);

    // Bending dihedrals: 4 verts per element → 12 directed (i, j)
    // pairs with i != j.  We just enumerate all of them and rely on
    // BlockCSR3's host-side dedup to merge entries shared with
    // stretch/shear edges.
    const int n_bend = bending_.size();
    if (n_bend > 0) {
        const math::Vec4i* dih = bending_.indices().cpu_data();
        rows.reserve(rows.size() + 12u * static_cast<std::size_t>(n_bend));
        cols.reserve(cols.size() + 12u * static_cast<std::size_t>(n_bend));
        for (int e = 0; e < n_bend; ++e) {
            const int v[4] = { dih[e].x, dih[e].y, dih[e].z, dih[e].w };
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
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
    // springs_.bind_hessian_layout(H_);  // disabled — see comment above
    fem_stretch_.bind_hessian_layout(H_);
    fem_shear_.bind_hessian_layout(H_);
    bending_.bind_hessian_layout(H_);

    H_num_block_rows_ = N;
    topology_dirty_ = false;
}

void ClothSimulator::resize_work_buffers(int n) {
    const auto sz = static_cast<std::size_t>(n);
    if (x_n_.gpu_size()     != sz) x_n_.resize(sz);
    if (x_tilde_.gpu_size() != sz) x_tilde_.resize(sz);
    if (rhs_.gpu_size()     != sz) rhs_.resize(sz);
    if (mass_.gpu_size()    != sz) mass_.resize(sz);

    // `dx_` is fed back into PCG every step as the warm-start initial
    // guess (see step()'s "PCG solve" section).  Zero it whenever we
    // grow the buffer so the first solve after a resize gets a clean
    // cold-start; subsequent solves consume whatever the previous
    // solve wrote.
    if (dx_.gpu_size() != sz) {
        dx_.resize(sz);
        check_cuda(cudaMemset(dx_.gpu_data(), 0, sz * sizeof(math::Vec3f)),
                   "cudaMemset(dx_ = 0)");
    }
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

    // ---- 3) accumulate gradient at x_n ------------------------------
    //
    // We linearise the implicit-Euler residual at x_n (the previous
    // frame's converged position), not at x_tilde.  See the comment on
    // `assemble_rhs_kernel` and cuda-cloth's `KernelVBDDynamic` for the
    // why; the short version is that x_n is a physically valid,
    // (near-)equilibrium configuration, so grad E(x_n) is small and
    // H_E(x_n) is well-conditioned, whereas H_E(x_tilde) sits in a
    // synthetic free-fall-strained state.
    DeviceSpan<math::Vec3f> x_n_span(x_n_.gpu_data(), static_cast<std::size_t>(n));
    DeviceSpan<math::Vec3f> x_tilde_span(x_tilde_.gpu_data(), static_cast<std::size_t>(n));
    DeviceSpan<math::Vec3f> rhs_span(rhs_.gpu_data(), static_cast<std::size_t>(n));
    {
        CHYSX_NVTX_RANGE_COLOUR("step::gradient", 0xff27ae60);
        check_cuda(cudaMemsetAsync(rhs_.gpu_data(), 0, n * sizeof(float3),
                                   stream),
                   "memset rhs = 0");

        pins_.accumulate_gradient(x_n_span, rhs_span, cuda_stream);
        // SpringConstraint is intentionally not part of the pipeline:
        // FEM stretch + shear already cover edge-stretch resistance,
        // and adding a Hookean spring on top would double-count
        // in-plane stiffness.  Kept around for diagnostics, but not
        // accumulated into the solve.
        // springs_.accumulate_gradient(x_n_span, rhs_span, cuda_stream);
        fem_stretch_.accumulate_gradient(x_n_span, rhs_span, cuda_stream);
        fem_shear_.accumulate_gradient(x_n_span, rhs_span, cuda_stream);
        bending_.accumulate_gradient(x_n_span, rhs_span, cuda_stream);

        // Now rhs[i] = grad E(x_n).  Fold in the inertial RHS to get
        //   rhs <- (M/dt^2)(x_tilde - x_n) - grad E(x_n)
        //        = M*v/dt + M*g            - grad E(x_n).
        const float inv_dt2 = 1.0f / (dt * dt);
        assemble_rhs_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float3*>(x_n_.gpu_data()),
            reinterpret_cast<const float3*>(x_tilde_.gpu_data()),
            mass_.gpu_data(),
            reinterpret_cast<float3*>(rhs_.gpu_data()),
            n, inv_dt2);
        check_cuda(cudaGetLastError(), "assemble_rhs kernel launch");
    }

    // ---- 4) (re)build Hessian topology if anything changed ----------
    //
    // Off-diagonal CSR structure depends on the FEM stretch / shear /
    // bending index tables, which are constant from frame to frame
    // for a fixed mesh.  We only redo it on changes (set_mesh /
    // build_fem_* / pin add-remove) — see `topology_dirty_`.
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

        // Hessian is also evaluated at x_n; same rationale as the
        // gradient.  M/dt^2 inertial diagonal is added afterwards.
        pins_.accumulate_hessian(x_n_span, H_, cuda_stream);
        // springs_ disabled — see comment in the gradient block above.
        // springs_.accumulate_hessian(x_n_span, H_, cuda_stream);
        fem_stretch_.accumulate_hessian(x_n_span, H_, cuda_stream);
        fem_shear_.accumulate_hessian(x_n_span, H_, cuda_stream);
        bending_.accumulate_hessian(x_n_span, H_, cuda_stream);

        const float inv_dt2 = 1.0f / (dt * dt);
        add_inertia_diag_kernel<<<grid, block, 0, stream>>>(
            H_.diag.gpu_data(), mass_.gpu_data(), n, inv_dt2);
        check_cuda(cudaGetLastError(), "add_inertia_diag kernel launch");
    }

    // ---- 6) PCG solve  (M/dt^2 + H_E) dx = rhs ----------------------
    //
    // Warm start: we deliberately do NOT zero `dx_` here.  The buffer
    // still holds the previous frame's solution, which for cloth (a
    // smoothly evolving system) is much closer to this frame's answer
    // than zero would be — typically ~2x fewer iterations to reach
    // the same residual.  PCGSolver::solve() reads `dx_` as the
    // initial guess and computes r_0 = b - A * dx_ accordingly.  For
    // the very first step after construction `dx_` is zero (see
    // resize_work_buffers).
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
            reinterpret_cast<float3*>(dx_.gpu_data()),
            n, dt, material_.damping);
        check_cuda(cudaGetLastError(), "finalize_step kernel launch");
    }
}

// ---------------------------------------------------------------------------
// Diagnostics — synchronous host-side dump of the last solve's state.
//
// All four call paths cudaDeviceSynchronize() to make sure any in-
// flight async work (capture-mode PCG, scatter kernels) is drained
// before we read GPU memory.  Don't call these from inside step() —
// they're for offline verification.
// ---------------------------------------------------------------------------

void ClothSimulator::debug_copy_hessian_diag(float* out) const {
    const int n = H_num_block_rows_;
    if (n == 0) return;
    if (out == nullptr) {
        throw std::invalid_argument(
            "ClothSimulator::debug_copy_hessian_diag: out must be non-null");
    }
    check_cuda(cudaDeviceSynchronize(),
               "cudaDeviceSynchronize(debug_copy_hessian_diag)");
    check_cuda(cudaMemcpy(out, H_.diag.gpu_data(),
                          n * sizeof(math::Mat3f),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(H.diag -> host)");
}

void ClothSimulator::debug_copy_hessian_csr(int* out_row_offsets,
                                            int* out_col_indices,
                                            float* out_values) const {
    const int n = H_num_block_rows_;
    if (n == 0) return;
    if (out_row_offsets == nullptr) {
        throw std::invalid_argument(
            "ClothSimulator::debug_copy_hessian_csr: out_row_offsets null");
    }
    check_cuda(cudaDeviceSynchronize(),
               "cudaDeviceSynchronize(debug_copy_hessian_csr)");
    check_cuda(cudaMemcpy(out_row_offsets, H_.row_offsets.gpu_data(),
                          (n + 1) * sizeof(int),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(row_offsets -> host)");
    const int nnz = H_.num_off_diag_blocks();
    if (nnz > 0) {
        if (out_col_indices == nullptr || out_values == nullptr) {
            throw std::invalid_argument(
                "ClothSimulator::debug_copy_hessian_csr: col_indices/values "
                "null but nnz_off > 0");
        }
        check_cuda(cudaMemcpy(out_col_indices, H_.col_indices.gpu_data(),
                              nnz * sizeof(int),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(col_indices -> host)");
        check_cuda(cudaMemcpy(out_values, H_.values.gpu_data(),
                              nnz * sizeof(math::Mat3f),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(values -> host)");
    }
}

void ClothSimulator::debug_copy_last_rhs(float* out) const {
    const int n = H_num_block_rows_;
    if (n == 0) return;
    if (out == nullptr) {
        throw std::invalid_argument(
            "ClothSimulator::debug_copy_last_rhs: out must be non-null");
    }
    check_cuda(cudaDeviceSynchronize(),
               "cudaDeviceSynchronize(debug_copy_last_rhs)");
    check_cuda(cudaMemcpy(out, rhs_.gpu_data(),
                          n * sizeof(math::Vec3f),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(rhs -> host)");
}

void ClothSimulator::debug_copy_last_dx(float* out) const {
    const int n = H_num_block_rows_;
    if (n == 0) return;
    if (out == nullptr) {
        throw std::invalid_argument(
            "ClothSimulator::debug_copy_last_dx: out must be non-null");
    }
    check_cuda(cudaDeviceSynchronize(),
               "cudaDeviceSynchronize(debug_copy_last_dx)");
    check_cuda(cudaMemcpy(out, dx_.gpu_data(),
                          n * sizeof(math::Vec3f),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(dx -> host)");
}

}  // namespace cloth
}  // namespace chysx
