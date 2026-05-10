// SPDX-License-Identifier: Apache-2.0
//
// chysx::cloth::ClothSimulator
//
// Glue class that pairs a ClothMaterial (parameters) with a
// ClothBuffers (memory) and exposes a single `step()` entry point.
//
// Lifecycle:
//
//   1. Construct.
//   2. Push parameters in via `set_material()`.  ChysX takes a copy;
//      the caller can throw the source ClothMaterial away.
//   3. Push raw device pointers in via `set_external_buffers()` each
//      step (or once if they don't change between steps).  ChysX does
//      not take ownership.
//   4. Call `step(dt)` to integrate; ChysX launches its own kernels
//      against the externally-owned pointers.
//
// The header deliberately does not pull in <cuda_runtime.h>; all CUDA
// runtime calls are confined to cloth_simulator.cu.

#pragma once

#include <cstdint>

#include "../collision/mesh_topology.h"
#include "../collision/self_collision.h"
#include "../collision/static_contact.h"
#include "../collision/untangle.h"
#include "../constraint/bending_constraint.h"
#include "../constraint/pin_constraint.h"
#include "../constraint/self_collision_constraint.h"
#include "../constraint/spring_constraint.h"
#include "../constraint/triangle_stretch_constraint.h"
#include "../constraint/untangle_constraint.h"
#include "../math/matrix.cuh"
#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "../solver/pcg_solver.h"
#include "../sparse/block_csr.h"
#include "cloth_buffers.h"
#include "cloth_material.h"

namespace chysx {
namespace cloth {

class ClothSimulator {
public:
    ClothSimulator() = default;

    // Move-only.  Two simulators with the same external pointers would
    // step the same particles twice, which is almost never desired.
    ClothSimulator(const ClothSimulator&) = delete;
    ClothSimulator& operator=(const ClothSimulator&) = delete;
    ClothSimulator(ClothSimulator&&) noexcept = default;
    ClothSimulator& operator=(ClothSimulator&&) noexcept = default;

    // ---- material -----------------------------------------------------

    // Copy `m` into this simulator.  After the call, mutating `m` no
    // longer affects the simulator state.
    void set_material(const ClothMaterial& m) noexcept { material_ = m; }

    ClothMaterial& material() noexcept { return material_; }
    const ClothMaterial& material() const noexcept { return material_; }

    // ---- buffers ------------------------------------------------------

    // Stash externally-owned CUDA device pointers.  The simulator does
    // not copy or free them; they must remain valid until step()
    // returns (or until they are replaced by a later call here).
    //
    // `pos_ptr` and `vel_ptr` are required (cast cudaMalloc'd
    // addresses to uintptr_t).  `inv_mass_ptr` is optional and
    // currently unused by the free-fall path.
    void set_external_buffers(std::uintptr_t pos_ptr,
                              std::uintptr_t vel_ptr,
                              int particle_count,
                              std::uintptr_t inv_mass_ptr = 0) noexcept;

    ClothBuffers& buffers() noexcept { return buffers_; }
    const ClothBuffers& buffers() const noexcept { return buffers_; }

    // ---- pinned particles --------------------------------------------
    //
    // While the linear-system solve isn't wired into the step yet, the
    // freefall integrator hard-clamps every pinned particle's position
    // back to its target and zeroes its velocity at the end of the
    // step.  That gives the same visual result as a penalty pin with
    // very large stiffness (which is what `PinConstraint` will
    // contribute once we run PCG inside step()).

    // Install `n` pins.  `host_indices[c]` is the global particle
    // index of pin c; `host_targets[c]` is its target world-space
    // position.  Replaces any previously installed pins.  `stiffness`
    // is stored on the constraint for future PCG-based solves; the
    // current freefall step ignores it (hard clamp).
    void set_pins(const int* host_indices,
                  const math::Vec3f* host_targets,
                  int n,
                  float stiffness = 1.0e6f);

    // Drop all pins.
    void clear_pins() noexcept;

    constraint::PinConstraint& pins() noexcept { return pins_; }
    const constraint::PinConstraint& pins() const noexcept { return pins_; }

    // ---- mesh topology + springs --------------------------------------
    //
    // `set_mesh` uploads a triangle index list (Vec3i triples) into
    // ChysX-owned memory and extracts the unique edge list on the host
    // (host-side dedup; cheap for the cloth sizes we target).  Call
    // `build_springs_from_current_positions` afterwards to install one
    // SpringConstraint instance per unique edge with rest length =
    // current edge length read from the externally-bound positions.
    void set_mesh(const math::Vec3i* host_triangles, int n_triangles);

    // Install one spring per mesh edge.  Rest length is taken from the
    // current `buffers_.pos` configuration (which must be set first
    // via `set_external_buffers`).  Replaces any previously installed
    // springs.
    void build_springs_from_current_positions(
        float stiffness,
        std::uintptr_t cuda_stream = 0);

    constraint::SpringConstraint& springs() noexcept { return springs_; }
    const constraint::SpringConstraint& springs() const noexcept {
        return springs_;
    }

    // ---- FEM triangle stretch (Baraff-Witkin) -------------------------
    //
    // Install one TriangleStretchConstraint instance per face in the
    // current `buffers_.tris` table.  The reference shape (Dm_inv,
    // rest area) is computed from the *current* externally-bound
    // positions, so call this once after `set_external_buffers` /
    // `set_mesh` and before stepping.
    void build_fem_stretch_from_current_positions(
        float stiffness,
        std::uintptr_t cuda_stream = 0);

    constraint::TriangleStretchConstraint& fem_stretch() noexcept {
        return fem_stretch_;
    }
    const constraint::TriangleStretchConstraint& fem_stretch() const noexcept {
        return fem_stretch_;
    }

    // ---- FEM triangle shear (Baraff-Witkin) ---------------------------
    //
    // Same triangle-FEM machinery as `fem_stretch_`, but the material
    // (u, v) axes are rotated by 45 degrees before storing `Dm_inv`.
    // This reuses the stretch energy / gradient / Hessian kernels to
    // resist *diagonal* stretching of the triangle, which is precisely
    // BW98 shear (cuda-cloth implements the two as separate kernels;
    // we share one, parameterised by the rotation).  Install after
    // `set_mesh(...)` and `set_external_buffers(...)`, just like the
    // stretch term.
    void build_fem_shear_from_current_positions(
        float stiffness,
        std::uintptr_t cuda_stream = 0);

    constraint::TriangleStretchConstraint& fem_shear() noexcept {
        return fem_shear_;
    }
    const constraint::TriangleStretchConstraint& fem_shear() const noexcept {
        return fem_shear_;
    }

    // ---- dihedral bending (Bridson / BW98) ----------------------------
    //
    // Auto-detect dihedrals from the currently installed mesh: every
    // edge that's shared by exactly two triangles becomes one bending
    // element with rest angle taken from the *current* externally-
    // bound positions.  Boundary edges (only one incident triangle)
    // and non-manifold edges (more than two) are skipped.
    //
    // Requires `set_mesh(...)` and `set_external_buffers(...)` to have
    // been called first.
    void build_bending_from_current_positions(
        float stiffness,
        std::uintptr_t cuda_stream = 0);

    constraint::BendingConstraint& bending() noexcept { return bending_; }
    const constraint::BendingConstraint& bending() const noexcept {
        return bending_;
    }

    // ---- self-collision (DCD) -----------------------------------------
    //
    // Brute-force VF (vertex-face) detection ahead of every Newton
    // iteration: each vertex is tested against every triangle that
    // does not contain it, and any pair within `thickness` becomes a
    // contact penalty constraint.
    //
    // Hessian contributions are *not* written into `H_`; they live in
    // a COO sidecar (`chysx::collision::ContactSpMVOp`) the PCG
    // solver applies during its `A * x` evaluation.  This keeps the
    // CSR topology of `H_` static across frames even though the
    // contact set churns every frame -- which in turn lets the PCG
    // CUDA-graph cache hit without re-capture as long as the contact
    // buffer pointers don't change.
    //
    // The detector's contact buffer is sized lazily on first call to
    // `set_self_collision_max_contacts(...)`; pass a generous cap
    // (e.g. 8 * particle_count for typical cloth).
    void set_self_collision_enabled(bool enabled) noexcept {
        self_collision_enabled_ = enabled;
    }
    bool self_collision_enabled() const noexcept {
        return self_collision_enabled_;
    }

    void set_self_collision_thickness(float t) noexcept {
        self_collision_thickness_ = t;
    }
    float self_collision_thickness() const noexcept {
        return self_collision_thickness_;
    }

    void set_self_collision_stiffness(float k) noexcept {
        self_collision_.set_stiffness(k);
    }
    float self_collision_stiffness() const noexcept {
        return self_collision_.stiffness();
    }

    // Allocate the detector's contact result buffers and the broadphase
    // EF-candidate list.  Idempotent; re-call only if you want to grow
    // the caps.  The default `max_ef_candidates = max_contacts` works
    // for typical cloth where each broadphase candidate yields ~1
    // narrow-phase contact on average; bump it if you observe many
    // candidates getting culled.
    void set_self_collision_max_contacts(int max_contacts,
                                         int max_ef_candidates = 0) {
        const int ef_cap = (max_ef_candidates > 0) ? max_ef_candidates
                                                   : max_contacts;
        self_collision_detector_.reserve(max_contacts, ef_cap);
        // Topology may need re-binding so the BVH picks up the new
        // ef-candidate cap; defer to next step via topology_dirty_.
        topology_dirty_ = true;
    }
    int self_collision_max_contacts() const noexcept {
        return self_collision_detector_.max_contacts();
    }

    collision::SelfCollisionDetector& self_collision_detector() noexcept {
        return self_collision_detector_;
    }
    const collision::SelfCollisionDetector& self_collision_detector() const noexcept {
        return self_collision_detector_;
    }

    // ---- untangle (5-vertex EF tangle) -------------------------------
    //
    // Penalty constraint that pushes apart edge-face pairs that have
    // already crossed (the proximity self-collision above only fires
    // for not-yet-crossed pairs at distance < thickness; once an edge
    // pokes all the way through a face, the proximity test sees zero
    // distance to anything and gives up).  Reuses the BVH the
    // proximity detector already built for its broadphase, then runs
    // a per-EF-candidate ray-triangle intersection inside its own
    // kernel; emit-on-pierce contacts have arity 5 (two edge endpoints
    // + three face vertices) and contribute only to the diagonal of
    // the implicit-Euler Hessian and to the RHS, never to the off-
    // diagonal SpMV sidecar.  Captured PCG graphs therefore stay
    // valid even as the tangle set churns frame-to-frame.
    //
    // Untangle requires the proximity detector to be enabled in the
    // same step (it consumes the BVH's EF-candidate stream).  When
    // `self_collision_enabled() == false` the untangle pass is
    // silently skipped.
    void set_untangle_enabled(bool enabled) noexcept {
        untangle_enabled_ = enabled;
    }
    bool untangle_enabled() const noexcept { return untangle_enabled_; }

    void set_untangle_thickness(float t) noexcept {
        untangle_thickness_ = t;
    }
    float untangle_thickness() const noexcept { return untangle_thickness_; }

    void set_untangle_stiffness(float k) noexcept {
        untangle_.set_stiffness(k);
    }
    float untangle_stiffness() const noexcept { return untangle_.stiffness(); }

    // Allocate the untangle detector's 5-vertex contact buffers.
    // Defaults to the same cap as `self_collision_max_contacts()`
    // when called with `max_contacts <= 0` (the typical worst case
    // is "every proximity contact is also tangled", which is loose
    // but safe).
    void set_untangle_max_contacts(int max_contacts) {
        if (max_contacts <= 0)
            max_contacts = self_collision_detector_.max_contacts();
        if (max_contacts > 0) untangle_detector_.reserve(max_contacts);
    }
    int untangle_max_contacts() const noexcept {
        return untangle_detector_.max_contacts();
    }

    collision::UntangleDetector& untangle_detector() noexcept {
        return untangle_detector_;
    }
    const collision::UntangleDetector& untangle_detector() const noexcept {
        return untangle_detector_;
    }

    // ---- static-shape contact (cloth ⇄ planes / boxes) ---------------
    //
    // Penalty contact between cloth particles and a small set of
    // *static* rigid primitives (oriented planes + oriented boxes,
    // think ground / table / wall).  Contributions land on the
    // diagonal of the implicit-Euler Hessian and on the RHS only —
    // see `chysx::collision::StaticContactSet` for the math.  No
    // off-diagonal SpMV sidecar (every contact is single-particle),
    // so adding / removing primitives never invalidates the captured
    // PCG graph.
    //
    // Usage:
    //
    //     sim.add_static_plane({{0, 0, 1}, 0.0f});  // ground at z=0
    //     sim.add_static_box({{0, 0, 0.5f}, {1, 1, 0.05f},
    //                        {1,0,0}, {0,1,0}, {0,0,1}});
    //     sim.set_static_contact_thickness(0.005f);
    //     sim.set_static_contact_stiffness(1.0e4f);
    //
    // Set thickness / stiffness once; call add_static_plane /
    // add_static_box at setup time (or between scenes).
    void add_static_plane(const collision::PlaneShape& p) {
        static_contacts_.add_plane(p);
    }
    void add_static_box(const collision::BoxShape& b) {
        static_contacts_.add_box(b);
    }
    void clear_static_shapes() { static_contacts_.clear(); }

    void set_static_contact_thickness(float t) noexcept {
        static_contacts_.set_thickness(t);
    }
    float static_contact_thickness() const noexcept {
        return static_contacts_.thickness();
    }
    void set_static_contact_stiffness(float k) noexcept {
        static_contacts_.set_stiffness(k);
    }
    float static_contact_stiffness() const noexcept {
        return static_contacts_.stiffness();
    }
    // Viscous tangential friction coefficient `μ_v` [N·s/m].  Adds
    // `(μ_v / dt) * (I - n n^T)` to the per-particle Hessian block of
    // every active static contact, producing implicit-Euler velocity-
    // proportional friction without needing to solve the full Coulomb
    // cone.  Zero (default) disables friction.
    void set_static_contact_friction(float mu_v) noexcept {
        static_contacts_.set_friction(mu_v);
    }
    float static_contact_friction() const noexcept {
        return static_contacts_.friction();
    }

    int static_plane_count() const noexcept {
        return static_contacts_.n_planes();
    }
    int static_box_count() const noexcept {
        return static_contacts_.n_boxes();
    }

    collision::StaticContactSet& static_contacts() noexcept {
        return static_contacts_;
    }
    const collision::StaticContactSet& static_contacts() const noexcept {
        return static_contacts_;
    }

    // ---- pin target update (no re-bind) -------------------------------
    //
    // Cheap per-frame update of the pinned particles' target world
    // positions, without touching their indices or the Hessian
    // topology.  `host_targets` is a packed (n_pins, 3) float buffer
    // whose row count must match the pin set most recently installed
    // by `set_pins(...)`.
    //
    // Use this to drive twist / dragging animations where pin
    // positions move every frame but the pin set stays the same:
    // changing indices via `set_pins(...)` would force a Hessian-
    // topology rebuild and flush the PCG CUDA graph cache.
    void update_pin_targets(const math::Vec3f* host_targets,
                            int n_pins,
                            std::uintptr_t cuda_stream = 0);

    // ---- area-weighted vertex mass ------------------------------------
    //
    // Recompute per-particle inverse mass from the cloth's mesh +
    // a uniform surface density [kg/m^2].  Each triangle contributes
    //
    //     m_t = surface_density * area(t)
    //
    // distributed equally across its three vertices, matching the
    // standard "lumped" finite-element mass model (cuda-cloth
    // `KernelComputeArea` / `KernelComputeAllDm`).  Vertices on the
    // boundary therefore end up lighter than interior vertices,
    // which is the physically correct behaviour and lets dense
    // meshes hang and drape naturally instead of pulling a heavy
    // ball of evenly-massed corner points.
    //
    // The per-vertex `inv_mass` is written into the externally-owned
    // buffer pointed to by `inv_mass_ptr` (cast cudaMalloc'd address
    // to uintptr_t).  Vertices touched by no triangle (e.g. isolated
    // particles) are written as `inv_mass = 0` (treated as
    // kinematic), so callers should pre-fill that buffer if they
    // want a different fallback.
    //
    // Requires `set_mesh(...)` and `set_external_buffers(...)` to
    // have been called first.
    void redistribute_mass_area_weighted(float surface_density,
                                         std::uintptr_t inv_mass_ptr,
                                         int particle_count,
                                         std::uintptr_t cuda_stream = 0);

    // ---- stepping -----------------------------------------------------

    // Advance the simulation by `dt` seconds using the currently set
    // material and external buffers.  For now this is a single
    // semi-implicit Euler free-fall update with optional velocity
    // damping; later we'll add elastic + bending contributions behind
    // the same entry point so callers don't need to re-wire anything.
    //
    // `cuda_stream`: cudaStream_t handle cast to uintptr_t. 0 = the
    // default stream (the call returns synchronously in that case).
    void step(float dt, std::uintptr_t cuda_stream = 0);

    // ---- solver tuning ------------------------------------------------

    void set_pcg_iterations(int max_iter) noexcept {
        pcg_max_iterations_ = (max_iter > 0) ? max_iter : 1;
    }
    int pcg_iterations() const noexcept { return pcg_max_iterations_; }

    // ---- diagnostics --------------------------------------------------
    //
    // Read-only accessors for the linear system that the *last* call
    // to `step(...)` solved.  Intended for offline verification
    // (validate symmetry / PSD / PCG residual from Python after a
    // step), so they cudaMemcpy through host buffers instead of
    // exposing device pointers — fine at cloth scales but don't
    // call them in a hot loop.

    // The Hessian (diagonal + CSR off-diagonal) used in the most
    // recent solve.  All four arrays are written into `diag_out`,
    // `row_offsets_out`, `col_indices_out`, `values_out` (caller-
    // sized: pass through ints with the expected lengths).
    int num_particles() const noexcept { return H_num_block_rows_; }
    int num_off_diag_blocks() const { return H_.num_off_diag_blocks(); }

    // Copy `H_.diag` (per-particle 3x3) onto `out`. Requires
    // `out` to point at `num_particles() * 9` floats.  Row-major
    // per block.
    void debug_copy_hessian_diag(float* out) const;

    // Copy CSR off-diag arrays.  `out_row_offsets` length =
    // num_particles() + 1; `out_col_indices`, `out_values` length =
    // num_off_diag_blocks() (* 9 for `out_values`).
    void debug_copy_hessian_csr(int* out_row_offsets,
                                int* out_col_indices,
                                float* out_values) const;

    // Copy the most recent right-hand side / solution vectors.
    // Length = num_particles() * 3 floats each (xyz per particle).
    void debug_copy_last_rhs(float* out) const;
    void debug_copy_last_dx(float* out) const;

private:
    // Lazy-resize all per-particle work buffers to length `n`.  No-op
    // if every buffer already matches.
    void resize_work_buffers(int n);

    // Rebuild `H_`'s topology from the currently installed
    // constraints (off-diagonal pairs only — pins are diagonal-only
    // and add nothing structurally) and bind every constraint's
    // Hessian-slot LUT against it.  Called lazily by `step()` when
    // `topology_dirty_` is set; flagged by anything that changes the
    // mesh, springs, or FEM stretch instance set.
    void ensure_hessian_topology();

    ClothMaterial material_;
    ClothBuffers buffers_;
    constraint::PinConstraint pins_;
    constraint::SpringConstraint springs_;
    constraint::TriangleStretchConstraint fem_stretch_;
    constraint::TriangleStretchConstraint fem_shear_;
    constraint::BendingConstraint         bending_;
    constraint::SelfCollisionConstraint   self_collision_;
    collision::MeshTopology               mesh_topology_;
    collision::SelfCollisionDetector      self_collision_detector_;
    bool                                  self_collision_enabled_ = false;
    float                                 self_collision_thickness_ = 0.0f;
    constraint::UntangleConstraint        untangle_;
    collision::UntangleDetector           untangle_detector_;
    bool                                  untangle_enabled_ = false;
    float                                 untangle_thickness_ = 0.0f;
    collision::StaticContactSet           static_contacts_;

    // ---- implicit-Euler PCG step working state ----------------------
    //
    // Per-particle workspace (length = particle_count):
    //
    //   x_n      : positions at the start of the step (snapshot)
    //   x_tilde  : inertial predicted positions (x_n + dt v + dt^2 g)
    //   dx       : PCG solution (displacement correction)
    //   rhs      : right-hand side b = -grad E(x_tilde)
    //   mass     : per-particle scalar mass (lazy from inv_mass)
    CudaArray<math::Vec3f> x_n_;
    CudaArray<math::Vec3f> x_tilde_;
    CudaArray<math::Vec3f> dx_;
    CudaArray<math::Vec3f> rhs_;
    CudaArray<float>       mass_;

    sparse::BlockCSR3 H_;
    int H_num_block_rows_ = 0;       // last N H_ was built for
    bool topology_dirty_ = true;     // mesh / springs / fem changed?
    solver::PCGSolver pcg_;
    int pcg_max_iterations_ = 50;
};

}  // namespace cloth
}  // namespace chysx
