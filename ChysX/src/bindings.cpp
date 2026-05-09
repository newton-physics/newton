// SPDX-License-Identifier: Apache-2.0
//
// pybind11 bindings for the _chysx_native module.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

#include "cloth/cloth_material.h"
#include "cloth/cloth_simulator.h"
#include "math/vec.cuh"

namespace py = pybind11;

PYBIND11_MODULE(_chysx_native, m) {
    m.doc() = "ChysX: minimal CUDA cloth physics simulator";

    // ---- ClothMaterial ----------------------------------------------------

    py::class_<chysx::cloth::ClothMaterial>(m, "ClothMaterial", R"pbdoc(
Plain-old-data material parameters for the cloth simulator.

Each field is a Python-mutable scalar; copy a fully populated instance
into a ClothSimulator with ``simulator.set_material(material)``.
)pbdoc")
        .def(py::init<>())
        .def_readwrite("lame_mu", &chysx::cloth::ClothMaterial::lame_mu,
                       "Lamé mu [Pa] (in-plane elasticity, unused by free-fall).")
        .def_readwrite("lame_lambda", &chysx::cloth::ClothMaterial::lame_lambda,
                       "Lamé lambda [Pa] (in-plane elasticity, unused by free-fall).")
        .def_readwrite("bending", &chysx::cloth::ClothMaterial::bending,
                       "Dihedral bending stiffness [N·m] (unused by free-fall).")
        .def_readwrite("density", &chysx::cloth::ClothMaterial::density,
                       "Surface mass density [kg/m^2].")
        .def_readwrite("damping", &chysx::cloth::ClothMaterial::damping,
                       "Velocity damping [1/s] (v *= exp(-damping * dt)).")
        .def_readwrite("gx", &chysx::cloth::ClothMaterial::gx,
                       "Gravity x-component [m/s^2].")
        .def_readwrite("gy", &chysx::cloth::ClothMaterial::gy,
                       "Gravity y-component [m/s^2].")
        .def_readwrite("gz", &chysx::cloth::ClothMaterial::gz,
                       "Gravity z-component [m/s^2].");

    // ---- ClothSimulator ---------------------------------------------------

    py::class_<chysx::cloth::ClothSimulator>(m, "ClothSimulator", R"pbdoc(
Cloth physics simulator.

Owns a copy of the material parameters and a set of buffer handles
(externally-owned device pointers + ChysX-owned working arrays).
Callers push parameters in once via ``set_material``, then push device
pointers in each step via ``set_external_buffers`` before calling
``step(dt)``.
)pbdoc")
        .def(py::init<>())
        .def("set_material", &chysx::cloth::ClothSimulator::set_material,
             py::arg("material"),
             "Copy `material` into the simulator.")
        .def("set_external_buffers",
             &chysx::cloth::ClothSimulator::set_external_buffers,
             py::arg("pos_ptr"),
             py::arg("vel_ptr"),
             py::arg("particle_count"),
             py::arg("inv_mass_ptr") = 0,
             R"pbdoc(
Stash externally-owned CUDA device pointers (cast to int) for the next
step().  ChysX never copies or frees these; the caller must keep them
alive until step() returns.
)pbdoc")
        .def("step", &chysx::cloth::ClothSimulator::step,
             py::arg("dt"),
             py::arg("cuda_stream") = 0,
             R"pbdoc(
Advance the simulation by `dt` seconds using the currently set material
and external buffers.  Throws if pos_ptr / vel_ptr were not set.
)pbdoc")
        .def(
            "set_pins",
            [](chysx::cloth::ClothSimulator& self,
               py::array_t<int, py::array::c_style | py::array::forcecast> indices,
               py::array_t<float, py::array::c_style | py::array::forcecast> targets,
               float stiffness) {
                if (indices.ndim() != 1) {
                    throw std::invalid_argument(
                        "ClothSimulator.set_pins: indices must be 1-D");
                }
                if (targets.ndim() != 2 || targets.shape(1) != 3) {
                    throw std::invalid_argument(
                        "ClothSimulator.set_pins: targets must have shape (N, 3)");
                }
                if (indices.shape(0) != targets.shape(0)) {
                    throw std::invalid_argument(
                        "ClothSimulator.set_pins: indices and targets must "
                        "have the same length");
                }
                const int n = static_cast<int>(indices.shape(0));
                // Vec3f and float[3] share the same 12-byte layout, so we
                // can hand the contiguous numpy buffer to set_pins via a
                // reinterpret_cast.
                const auto* targets_vec3 =
                    reinterpret_cast<const chysx::math::Vec3f*>(targets.data());
                self.set_pins(indices.data(), targets_vec3, n, stiffness);
            },
            py::arg("indices"),
            py::arg("targets"),
            py::arg("stiffness") = 1.0e6f,
            R"pbdoc(
Install pin constraints.

Parameters
----------
indices : numpy.ndarray, shape (N,), dtype int32
    Global particle index of each pin.
targets : numpy.ndarray, shape (N, 3), dtype float32
    World-space target position for each pin.
stiffness : float
    Penalty stiffness used by the future PCG step.  The current
    free-fall integrator hard-clamps pinned particles instead, so
    this value is stored but not consulted yet.
)pbdoc")
        .def("clear_pins", &chysx::cloth::ClothSimulator::clear_pins,
             "Remove every previously installed pin.")
        .def(
            "num_pins",
            [](const chysx::cloth::ClothSimulator& self) {
                return self.pins().size();
            },
            "Number of currently installed pins.")
        .def(
            "update_pin_targets",
            [](chysx::cloth::ClothSimulator& self,
               py::array_t<float, py::array::c_style | py::array::forcecast> targets,
               std::uintptr_t cuda_stream) {
                if (targets.ndim() != 2 || targets.shape(1) != 3) {
                    throw std::invalid_argument(
                        "ClothSimulator.update_pin_targets: targets must "
                        "have shape (N, 3) and dtype float32");
                }
                const int n = static_cast<int>(targets.shape(0));
                const auto* targets_vec3 =
                    reinterpret_cast<const chysx::math::Vec3f*>(targets.data());
                self.update_pin_targets(targets_vec3, n, cuda_stream);
            },
            py::arg("targets"),
            py::arg("cuda_stream") = 0,
            R"pbdoc(
Update the world-space target positions of the currently installed
pins without changing their indices.  Use this for animations where
pins move every frame (e.g. twisting a cloth around a moving boundary)
to avoid the Hessian-topology rebuild that ``set_pins(...)`` triggers.

Parameters
----------
targets : numpy.ndarray, shape (n_pins, 3), dtype float32
    New target positions; ``n_pins`` must equal ``num_pins()``.
cuda_stream : int, optional
    Stream to issue the host-to-device copy on.
)pbdoc")
        .def(
            "set_mesh",
            [](chysx::cloth::ClothSimulator& self,
               py::array_t<int, py::array::c_style | py::array::forcecast> tris) {
                if (tris.ndim() != 2 || tris.shape(1) != 3) {
                    throw std::invalid_argument(
                        "ClothSimulator.set_mesh: triangles must have shape "
                        "(M, 3) and dtype int32");
                }
                const int n = static_cast<int>(tris.shape(0));
                // Vec3i and int[3] share layout (12 bytes, native int).
                const auto* tris_vec3i =
                    reinterpret_cast<const chysx::math::Vec3i*>(tris.data());
                self.set_mesh(tris_vec3i, n);
            },
            py::arg("triangles"),
            R"pbdoc(
Upload the cloth's triangle topology into ChysX-owned device memory and
extract the unique edge list on the host.  Call this once at setup time;
edges are then available to ``build_springs_from_current_positions``.

Parameters
----------
triangles : numpy.ndarray, shape (M, 3), dtype int32
    Triangle vertex indices.
)pbdoc")
        .def("build_springs_from_current_positions",
             &chysx::cloth::ClothSimulator::build_springs_from_current_positions,
             py::arg("stiffness"),
             py::arg("cuda_stream") = 0,
             R"pbdoc(
Install one Hookean spring per unique mesh edge using the *current*
externally-bound positions as the rest configuration.  Requires
``set_mesh`` and ``set_external_buffers`` to have been called first.

Parameters
----------
stiffness : float
    Per-spring stiffness k [N/m] (shared by every spring).
)pbdoc")
        .def(
            "num_springs",
            [](const chysx::cloth::ClothSimulator& self) {
                return self.springs().size();
            },
            "Number of currently installed springs (unique mesh edges).")
        .def("redistribute_mass_area_weighted",
             &chysx::cloth::ClothSimulator::redistribute_mass_area_weighted,
             py::arg("surface_density"),
             py::arg("inv_mass_ptr"),
             py::arg("particle_count"),
             py::arg("cuda_stream") = 0,
             R"pbdoc(
Recompute per-particle inverse mass by distributing each triangle's
``surface_density * area`` equally across its three vertices, matching
cuda-cloth's lumped finite-element mass model.

Boundary vertices end up lighter than interior vertices (the
physically correct behaviour) so dense meshes drape naturally instead
of pulling a heavy uniform-mass corner down.

Parameters
----------
surface_density : float
    Material surface density in kg/m^2 (e.g. 0.3 for cotton).
inv_mass_ptr : int
    cudaMalloc'd address of the externally-owned inverse-mass buffer
    (typically Newton's ``model.particle_inv_mass.ptr``); the routine
    overwrites it.  Vertices with no incident triangle are written as
    ``inv_mass = 0`` (treated as kinematic).
particle_count : int
    Number of particles in the inverse-mass buffer.

Requires ``set_mesh(...)`` and ``set_external_buffers(...)`` to have
been called first so ChysX has access to the triangle topology and
the rest positions used for area computation.
)pbdoc")
        .def("build_fem_stretch_from_current_positions",
             &chysx::cloth::ClothSimulator::build_fem_stretch_from_current_positions,
             py::arg("stiffness"),
             py::arg("cuda_stream") = 0,
             R"pbdoc(
Install one Baraff-Witkin triangle stretch element per face of the
current mesh.  The reference shape (Dm_inv, rest area) is computed
from the *current* externally-bound positions, so call this once
after ``set_mesh`` and ``set_external_buffers``.

Parameters
----------
stiffness : float
    Per-area stretch stiffness ``k`` [N/m^2]; the per-element weight is
    ``area * k`` (cuda-cloth's ``k_stretch`` convention).
)pbdoc")
        .def(
            "num_fem_stretch_triangles",
            [](const chysx::cloth::ClothSimulator& self) {
                return self.fem_stretch().size();
            },
            "Number of currently installed FEM stretch triangles.")
        .def("build_fem_shear_from_current_positions",
             &chysx::cloth::ClothSimulator::build_fem_shear_from_current_positions,
             py::arg("stiffness"),
             py::arg("cuda_stream") = 0,
             R"pbdoc(
Install one Baraff-Witkin triangle *shear* element per face of the
current mesh.  Internally this uses exactly the same kernels as the
stretch element, just with the material (u, v) axes rotated 45 degrees
so the constraint pins the diagonal lengths instead of the U/V edges
— equivalent to cuda-cloth's
``KernelComputeStretchShearForceAndHessianFast``.

Parameters
----------
stiffness : float
    Per-area shear stiffness ``k`` [N/m^2]; per-element weight is
    ``area * k`` (cuda-cloth's ``k_stretch`` convention; the same
    constant is reused for both stretch and shear in cuda-cloth).
)pbdoc")
        .def(
            "num_fem_shear_triangles",
            [](const chysx::cloth::ClothSimulator& self) {
                return self.fem_shear().size();
            },
            "Number of currently installed FEM shear triangles.")
        .def("build_bending_from_current_positions",
             &chysx::cloth::ClothSimulator::build_bending_from_current_positions,
             py::arg("stiffness"),
             py::arg("cuda_stream") = 0,
             R"pbdoc(
Auto-detect dihedrals from the currently installed mesh and install
one Baraff-Witkin / Bridson bending element per interior edge (every
edge shared by exactly two triangles).  Rest angles are computed from
the *current* externally-bound positions, so call this once after
``set_mesh`` and ``set_external_buffers``.

Equivalent to cuda-cloth's
``KernelComputeDihedralForcesAndHessianFast`` with rest angles
populated by ``KernelComputeDihedralAngle``.

Parameters
----------
stiffness : float
    Bending stiffness ``k_bending`` shared by every dihedral.
    Cloth-like values are typically several orders of magnitude
    smaller than the in-plane stretch / shear stiffness.
)pbdoc")
        .def(
            "num_bending_dihedrals",
            [](const chysx::cloth::ClothSimulator& self) {
                return self.bending().size();
            },
            "Number of currently installed bending dihedrals.")
        // ---- self-collision (DCD, brute-force VF for v1) ------------
        .def("set_self_collision_enabled",
             &chysx::cloth::ClothSimulator::set_self_collision_enabled,
             py::arg("enabled"),
             "Toggle the brute-force VF self-collision pipeline.")
        .def("self_collision_enabled",
             &chysx::cloth::ClothSimulator::self_collision_enabled,
             "True if self-collision is currently enabled.")
        .def("set_self_collision_thickness",
             &chysx::cloth::ClothSimulator::set_self_collision_thickness,
             py::arg("thickness"),
             R"pbdoc(
Set the contact distance threshold (in world units, same as particle
positions).  A vertex within ``thickness`` of any non-incident
triangle becomes a contact.  cuda-cloth's twist case uses
``thickness ~ 0.2 * average_edge_length``.
)pbdoc")
        .def("self_collision_thickness",
             &chysx::cloth::ClothSimulator::self_collision_thickness,
             "Currently configured contact distance threshold.")
        .def("set_self_collision_stiffness",
             &chysx::cloth::ClothSimulator::set_self_collision_stiffness,
             py::arg("stiffness"),
             R"pbdoc(
Set the per-contact penalty stiffness ``k`` [N/m].  cuda-cloth's
twist case uses ``k = 1000`` for VF/EE (m_4_k); larger values produce
stiffer contact response at the cost of PCG conditioning.
)pbdoc")
        .def("self_collision_stiffness",
             &chysx::cloth::ClothSimulator::self_collision_stiffness,
             "Currently configured contact penalty stiffness.")
        .def("set_self_collision_max_contacts",
             &chysx::cloth::ClothSimulator::set_self_collision_max_contacts,
             py::arg("max_contacts"),
             py::arg("max_ef_candidates") = 0,
             R"pbdoc(
Allocate (or grow) the device-side contact buffer to hold up to
``max_contacts`` simultaneous contacts plus the LBVH broadphase
EF-candidate list (default cap = max_contacts).  Detector overflow
past these caps silently drops the newest pairs; size generously
(e.g. ``8 * particle_count``) for typical cloth.
)pbdoc")
        .def("self_collision_max_contacts",
             &chysx::cloth::ClothSimulator::self_collision_max_contacts,
             "Currently allocated contact buffer capacity.")
        .def(
            "self_collision_count",
            [](chysx::cloth::ClothSimulator& s,
               std::uintptr_t cuda_stream) {
                return s.self_collision_detector().count(cuda_stream);
            },
            py::arg("cuda_stream") = 0,
            "Number of contacts emitted by the most recent step (synchronous read).")
        .def("set_pcg_iterations",
             &chysx::cloth::ClothSimulator::set_pcg_iterations,
             py::arg("max_iter"),
             "Set the maximum number of PCG iterations per step.")
        .def("pcg_iterations", &chysx::cloth::ClothSimulator::pcg_iterations,
             "Currently configured maximum PCG iterations per step.")
        // ---- diagnostics: dump the last solve's linear system -------
        .def("debug_dump_last_solve",
             [](const chysx::cloth::ClothSimulator& s) {
                 const int n = s.num_particles();
                 const int nnz = s.num_off_diag_blocks();

                 py::array_t<float> diag({n, 3, 3});
                 py::array_t<int>   row_offsets({n + 1});
                 py::array_t<int>   col_indices({nnz});
                 py::array_t<float> values({nnz, 3, 3});
                 py::array_t<float> rhs({n, 3});
                 py::array_t<float> dx({n, 3});

                 if (n == 0) {
                     return py::make_tuple(diag, row_offsets, col_indices,
                                           values, rhs, dx);
                 }

                 s.debug_copy_hessian_diag(diag.mutable_data());
                 s.debug_copy_hessian_csr(row_offsets.mutable_data(),
                                          nnz > 0 ? col_indices.mutable_data() : nullptr,
                                          nnz > 0 ? values.mutable_data() : nullptr);
                 s.debug_copy_last_rhs(rhs.mutable_data());
                 s.debug_copy_last_dx(dx.mutable_data());

                 return py::make_tuple(diag, row_offsets, col_indices,
                                       values, rhs, dx);
             },
             R"pbdoc(
Return ``(diag, row_offsets, col_indices, values, rhs, dx)`` for the
linear system solved by the most recent ``step(...)``.

Shapes:
    diag         : (N, 3, 3)        per-particle 3x3 Hessian diagonal
    row_offsets  : (N + 1,)         CSR row pointer (off-diagonal)
    col_indices  : (nnz_off,)       block-column indices
    values       : (nnz_off, 3, 3)  off-diagonal 3x3 blocks
    rhs          : (N, 3)           right-hand side b
    dx           : (N, 3)           solution returned by PCG

This synchronises the device and copies through host buffers, so
don't call it inside the simulation loop.  The return is a snapshot
of whatever was in those buffers when ``step()`` finished — perfect
for verifying matrix symmetry, PSD-ness, and PCG residual offline.
)pbdoc")
        .def_property_readonly(
            "material",
            [](chysx::cloth::ClothSimulator& s) -> chysx::cloth::ClothMaterial& {
                return s.material();
            },
            py::return_value_policy::reference_internal,
            "In-place reference to the simulator's material (mutate freely).");
}
