# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Sim Cloth Stiff Material Stretch
#
# Five cloth sheets pinned at their +/- X edges and stretched along X by a
# fixed factor. Each sheet uses a different effective Poisson ratio (set
# via the ratio of `tri_ka` to `tri_ke`, which map to the Lamé parameters
# of the stable Neo-Hookean membrane). At equilibrium, the orthogonal
# direction contracts and the area ratio matches the closed-form
# prediction for the StableNH 2D membrane:
#
#     A(lambda) / A_0 = (lmbd_nh + mu) * lambda^2
#                       ---------------------------------
#                          mu + lmbd_nh * lambda^2
#
# where mu = tri_ke, lmbd_nh = tri_ka + tri_ke, and Poisson ratio
# nu = tri_ka / (tri_ka + 2 * tri_ke).
#
# Run on `main` (StVK) and on `vbd_stiff_material_stretch` (Neo-Hookean)
# to compare against the same theoretical curve.
#
# Command: python -m newton.examples cloth_stiff_material_stretch
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


def _theoretical_area_ratio(stretch: float, nu: float) -> float:
    """Closed-form area ratio for the StableNH 2D membrane under uniaxial stretch.

    Assumes mu = tri_ke and lmbd_nh = tri_ka + tri_ke = mu * (1 + nu) / (1 - nu).
    """
    if nu >= 1.0 - 1e-9:
        return 1.0  # incompressible limit
    # In normalized units (mu = 1):
    lmbd_over_mu = (1.0 + nu) / (1.0 - nu)
    return (lmbd_over_mu + 1.0) * stretch * stretch / (1.0 + lmbd_over_mu * stretch * stretch)


def _ka_from_nu(tri_ke: float, nu: float) -> float:
    """Solve tri_ka so the effective Poisson ratio is nu (for the NH membrane)."""
    if nu >= 1.0 - 1e-9:
        return tri_ke * 1.0e6  # ~incompressible
    return 2.0 * tri_ke * nu / (1.0 - nu)


class Example:
    POISSON_RATIOS = (0.10, 0.20, 0.30, 0.40, 0.49)
    STRETCH = 1.3
    RAMP_FRAMES = 60  # ease the right edge from rest -> stretched over this many frames
    DIM = 20
    CELL = 0.05  # 1 m square sheet at rest
    SHEET_SPACING = 1.6  # along Y, between sheets
    PARTICLE_MASS = 1.0  # kg — heavy enough that ramped stretch is stable
    TRI_KE = 1.0e3
    TRI_KD = 50.0
    EDGE_KE = 0.0

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 20
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        sheet_starts: list[int] = []
        sheet_right_indices: list[list[int]] = []
        for sheet_idx, nu in enumerate(self.POISSON_RATIOS):
            tri_ka = _ka_from_nu(self.TRI_KE, nu)
            start = len(builder.particle_q)
            builder.add_cloth_grid(
                pos=wp.vec3(0.0, sheet_idx * self.SHEET_SPACING, 1.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=self.DIM,
                dim_y=self.DIM,
                cell_x=self.CELL,
                cell_y=self.CELL,
                mass=self.PARTICLE_MASS,
                fix_left=True,
                fix_right=True,
                tri_ke=self.TRI_KE,
                tri_ka=tri_ka,
                tri_kd=self.TRI_KD,
                edge_ke=self.EDGE_KE,
                edge_kd=0.0,
            )
            stride = self.DIM + 1
            right = [start + y * stride + self.DIM for y in range(stride)]
            sheet_starts.append(start)
            sheet_right_indices.append(right)

        self.sheet_starts = sheet_starts
        self.sheet_right_indices = sheet_right_indices
        self.rest_side = self.DIM * self.CELL

        builder.color(include_bending=True)
        self.model = builder.finalize()

        # Disable gravity — pure stretch experiment.
        self.model.set_gravity((0.0, 0.0, 0.0))

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Right edge starts unstretched and is ramped to STRETCH * rest_side
        # over RAMP_FRAMES — easing avoids the instant force spike that breaks
        # VBD at low Poisson ratios.
        self._frame_index = 0

        # Triangle index ranges per sheet, for area measurement.
        tri_ind = self.model.tri_indices.numpy()
        self.sheet_tri_ranges: list[tuple[int, int]] = []
        sheet_starts_with_end = [*self.sheet_starts, self.model.particle_count]
        for s in range(len(self.POISSON_RATIOS)):
            lo = sheet_starts_with_end[s]
            hi = sheet_starts_with_end[s + 1]
            mask = (tri_ind[:, 0] >= lo) & (tri_ind[:, 0] < hi)
            indices = np.where(mask)[0]
            self.sheet_tri_ranges.append((int(indices.min()), int(indices.max()) + 1))

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _apply_stretch_ramp(self):
        if self._frame_index >= self.RAMP_FRAMES:
            target_stretch = self.STRETCH
        else:
            t = self._frame_index / self.RAMP_FRAMES
            target_stretch = 1.0 + t * (self.STRETCH - 1.0)
        target_x = target_stretch * self.rest_side
        q = self.state_0.particle_q.numpy()
        for right in self.sheet_right_indices:
            q[right, 0] = target_x
        self.state_0.particle_q.assign(q)

    def step(self):
        self._apply_stretch_ramp()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._frame_index += 1

    def measure_area_ratios(self) -> list[float]:
        """Bulk area ratio per sheet, lambda_1 * lambda_2_centerline.

        Measuring area straight from the triangle mesh undercounts because the
        pinned +/- X columns also fix Y, leaving a boundary layer of triangles
        that can't contract. Using lambda_2 from the centerline column gives
        the bulk constitutive response, which is what the closed-form theory
        predicts.
        """
        q = self.state_0.particle_q.numpy()
        stride = self.DIM + 1
        center_col = self.DIM // 2
        ratios: list[float] = []
        for start in self.sheet_starts:
            col_idx = [start + y * stride + center_col for y in range(stride)]
            col_q = q[col_idx]
            y_span = float(col_q[:, 1].max() - col_q[:, 1].min())
            lambda_2 = y_span / self.rest_side
            ratios.append(self.STRETCH * lambda_2)
        return ratios

    def test_final(self):
        # Non-explosion: velocities bounded.
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities do not explode",
            lambda q, qd: wp.length(qd) < 5.0,
        )

        # Sheets stay near their starting Y bands and above the ground.
        n_sheets = len(self.POISSON_RATIOS)
        y_max = (n_sheets - 1) * self.SHEET_SPACING + self.rest_side + 0.5
        p_lower = wp.vec3(-0.5, -0.5, 0.0)
        p_upper = wp.vec3(self.STRETCH * self.rest_side + 0.5, y_max, 2.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within the stretch volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

        # Area-preservation regression: each sheet's measured area should be
        # within 25% of the closed-form NH prediction.
        measured = self.measure_area_ratios()
        for nu, m in zip(self.POISSON_RATIOS, measured, strict=True):
            theory = _theoretical_area_ratio(self.STRETCH, nu)
            if not (abs(m - theory) / theory < 0.25):
                raise ValueError(f"area ratio for nu={nu:.2f}: measured {m:.3f}, theory {theory:.3f} (off by >25%)")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer=viewer, args=args)
    newton.examples.run(example, args)
