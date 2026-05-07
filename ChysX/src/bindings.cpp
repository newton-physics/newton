// SPDX-License-Identifier: Apache-2.0
//
// pybind11 bindings for the _chysx_native module.

#include <pybind11/pybind11.h>

#include "cloth/cloth_material.h"
#include "cloth/cloth_simulator.h"

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
        .def_property_readonly(
            "material",
            [](chysx::cloth::ClothSimulator& s) -> chysx::cloth::ClothMaterial& {
                return s.material();
            },
            py::return_value_policy::reference_internal,
            "In-place reference to the simulator's material (mutate freely).");
}
