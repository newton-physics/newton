// SPDX-License-Identifier: Apache-2.0
//
// Material parameters for the chysx cloth simulator.
//
// Plain-old-data on purpose: every field is a scalar, so the struct can
// be (a) value-copied between host and device, (b) memcpy'd through a
// pybind11 binding, and (c) captured by a __global__ kernel without
// indirection.
//
// The toy free-fall integrator only reads `gx / gy / gz` (and
// optionally `damping`); the elastic / bending fields are placeholders
// that future kernels will consume once we layer real cloth dynamics
// on top of the same struct.

#pragma once

namespace chysx {
namespace cloth {

struct ClothMaterial {
    // Lamé parameters [Pa] for the in-plane elastic energy.  Ignored
    // by the free-fall path.
    float lame_mu     = 0.0f;
    float lame_lambda = 0.0f;

    // Dihedral bending stiffness [N·m].  Ignored by the free-fall path.
    float bending     = 0.0f;

    // Surface mass density [kg/m^2].  Used when allocating per-particle
    // masses from a triangle mesh; the free-fall path treats every
    // particle as having unit (cancelled-out) mass since gravity is the
    // only force.
    float density     = 0.05f;

    // Velocity damping [1/s], applied as v *= exp(-damping * dt).  Set
    // to 0 to disable.
    float damping     = 0.0f;

    // Gravity vector components [m/s^2].
    float gx          = 0.0f;
    float gy          = 0.0f;
    float gz          = -9.81f;
};

}  // namespace cloth
}  // namespace chysx
