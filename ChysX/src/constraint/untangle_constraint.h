// SPDX-License-Identifier: Apache-2.0
//
// chysx::constraint::UntangleConstraint
//
// Penalty constraint that consumes the 5-vertex tangle stream
// produced by `chysx::collision::UntangleDetector`.  Implements
// the linear-system contribution of Volino & Magnenat-Thalmann's
// ICM (Intersection Contour Minimization), SIGGRAPH 2006, with
// the same sign and magnitude conventions as
// `newton/_src/solvers/style3d/collision/kernels.py::
// solve_untangling_kernel`.
//
// Per-tangle contribution to the linearised system
// `(M/dt^2 + H_E + C) dx = b`:
//
//   For every contact c, with vertex ids id[5] = (e0, e1, f0, f1, f2),
//   weights w[5] = (u0, u1, w0, w1, w2), unit ICM gradient direction
//   G_c, and stored thickness t_c:
//
//       disp = 2 * t_c                            (Volino displacement target)
//       H_c  = stiffness * (G_c G_c^T)            (3x3 rank-1)
//       F_c  = stiffness * disp * G_c             (3-vector)
//
//   Diagonal-only Hessian distribution (NO off-diagonal terms):
//
//       diag[id_i] += w_i^2 * H_c                 for all i in 0..4
//
//   Right-hand-side / gradient distribution -- OPPOSITE SIGNS for
//   edge endpoints (i = 0, 1) vs face vertices (i = 2, 3, 4):
//
//       grad[edge_i] += -w_i * F_c                (force on edge: +F_c * w_i,
//                                                  pushes edge along +G)
//       grad[face_j] += +w_j * F_c                (force on face: -F_c * w_j,
//                                                  pushes face along -G)
//
//   The opposite signs are essential: equal-sign (cuda-cloth's
//   original `+w_i * F_c` for all 5 vertices) only translates the
//   whole 5-vertex group along G without separating the edge from
//   the face -- the contact never opens up.
//
// User-facing requirement: untangle is wired in EXACTLY like the
// 4-vertex contacts (VF / EE) on the diagonal, but NEVER appears
// in the off-diagonal SpMV sidecar -- callers do not need to
// invalidate the captured PCG graph when tangles come and go.

#pragma once

#include <cstdint>

#include "../collision/untangle.h"
#include "../math/matrix.cuh"
#include "../math/vec.cuh"
#include "../memory/device_span.h"

namespace chysx {
namespace constraint {

class UntangleConstraint {
public:
    UntangleConstraint() = default;

    UntangleConstraint(const UntangleConstraint&) = delete;
    UntangleConstraint& operator=(const UntangleConstraint&) = delete;
    UntangleConstraint(UntangleConstraint&&) noexcept = default;
    UntangleConstraint& operator=(UntangleConstraint&&) noexcept = default;

    void set_stiffness(float k) noexcept { stiffness_ = k; }
    float stiffness() const noexcept { return stiffness_; }

    // Scatter `+grad E_untangle` into `out_grad` (one Vec3f per
    // global particle).  Each contact contributes to all 5 of its
    // vertices via the (w_i * f_c) term.  Reads the device-side
    // contact counter so it captures cleanly into a CUDA graph.
    void accumulate_gradient(
        const collision::UntangleDetector& detector,
        DeviceSpan<math::Vec3f>            out_grad,
        std::uintptr_t                     cuda_stream = 0) const;

    // Bake the per-particle diagonal blocks into `diag_blocks[]`
    // (the BlockCSR3 `diag` array in chysx, length n_particles).
    // Each contact contributes `w_i^2 * H_c` to all 5 of its
    // vertex slots.  No off-diagonal blocks are touched -- this
    // is by design, see the class header docstring.
    void bake_diag(
        const collision::UntangleDetector& detector,
        math::Mat3f*                       diag_blocks,
        int                                n_particles,
        std::uintptr_t                     cuda_stream = 0) const;

private:
    float stiffness_ = 0.0f;
};

}  // namespace constraint
}  // namespace chysx
