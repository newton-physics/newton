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
        .def("set_pcg_iterations",
             &chysx::cloth::ClothSimulator::set_pcg_iterations,
             py::arg("max_iter"),
             "Set the maximum number of PCG iterations per step.")
        .def("pcg_iterations", &chysx::cloth::ClothSimulator::pcg_iterations,
             "Currently configured maximum PCG iterations per step.")
        .def("set_graph_enabled",
             &chysx::cloth::ClothSimulator::set_graph_enabled,
             py::arg("enabled"),
             R"pbdoc(
Toggle the PCG solver's CUDA Graph capture path.

When enabled (default), the entire 50-iteration PCG solve is captured
into a single `cudaGraphExec_t` the first time it runs and replayed
with one `cudaGraphLaunch` on every subsequent step, eliminating ~400
per-step kernel launch dispatches.  Disable for debugging or to see
every individual kernel launch in an Nsight Systems timeline.
)pbdoc")
        .def("graph_enabled", &chysx::cloth::ClothSimulator::graph_enabled,
             "True if the PCG solver is currently in CUDA Graph mode.")
        .def_property_readonly(
            "material",
            [](chysx::cloth::ClothSimulator& s) -> chysx::cloth::ClothMaterial& {
                return s.material();
            },
            py::return_value_policy::reference_internal,
            "In-place reference to the simulator's material (mutate freely).");
}
