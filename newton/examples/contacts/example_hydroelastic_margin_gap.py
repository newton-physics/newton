# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Hydroelastic Margin and Gap
#
# Drops a sphere onto a box using exaggerated margins and gaps so the
# detected speculative band and the final surface separation are visible.
#
# Command: python -m newton.examples hydroelastic_margin_gap
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.geometry import HydroelasticSDF


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        self.box_half_height = 0.25
        self.sphere_radius = 0.3
        self.margin_sum = 0.2
        self.gap_sum = 0.16
        self.saw_no_contact = False
        self.saw_speculative_contact = False
        self.saw_penetrating_contact = False

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            is_hydroelastic=True,
            kh=1.0e9,
            margin=0.5 * self.margin_sum,
            gap=0.5 * self.gap_sum,
            sdf_max_resolution=64,
            sdf_narrow_band_range=(-0.25, 0.4),
            mu=0.5,
        )

        builder = newton.ModelBuilder()
        builder.add_shape_box(
            body=-1,
            hx=0.8,
            hy=0.8,
            hz=self.box_half_height,
            cfg=shape_cfg,
            label="hydroelastic_box",
        )
        self.sphere_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.8), wp.quat_identity()),
            label="falling_sphere",
        )
        builder.add_shape_sphere(
            body=self.sphere_body,
            radius=self.sphere_radius,
            cfg=shape_cfg,
            label="hydroelastic_sphere",
        )

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=20)
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            rigid_contact_max=2000,
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                output_contact_surface=True,
                reduce_contacts=True,
                buffer_fraction=1.0,
            ),
        )
        self.contacts = self.collision_pipeline.contacts()
        guide_half_width = 0.65
        guide_starts = []
        guide_ends = []
        guide_colors = []
        for height, color in (
            (self.box_half_height + self.margin_sum, (0.2, 0.9, 0.2)),
            (self.box_half_height + self.margin_sum + self.gap_sum, (0.9, 0.7, 0.1)),
        ):
            corners = (
                (-guide_half_width, -guide_half_width, height),
                (guide_half_width, -guide_half_width, height),
                (guide_half_width, guide_half_width, height),
                (-guide_half_width, guide_half_width, height),
            )
            for index in range(4):
                guide_starts.append(corners[index])
                guide_ends.append(corners[(index + 1) % 4])
                guide_colors.append(color)
        self.guide_starts = wp.array(guide_starts, dtype=wp.vec3)
        self.guide_ends = wp.array(guide_ends, dtype=wp.vec3)
        self.guide_colors = wp.array(guide_colors, dtype=wp.vec3)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(3.0, -4.0, 2.0), pitch=-12.0, yaw=145.0)
        self.viewer.show_hydro_contact_surface = True
        self.capture()

    def capture(self):
        self.graph = None
        if not self.model.device.is_cuda:
            return

        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _record_contact_bands(self):
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        if count == 0:
            self.saw_no_contact = True
            return

        point0 = self.contacts.rigid_contact_point0.numpy()[:count]
        point1 = self.contacts.rigid_contact_point1.numpy()[:count]
        normal = self.contacts.rigid_contact_normal.numpy()[:count]
        shape0 = self.contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:count]
        shape_body = self.model.shape_body.numpy()
        body_q = self.state_0.body_q.numpy()
        body0 = shape_body[shape0]
        body1 = shape_body[shape1]
        offset0 = np.where((body0 != -1)[:, None], body_q[np.maximum(body0, 0), :3], 0.0)
        offset1 = np.where((body1 != -1)[:, None], body_q[np.maximum(body1, 0), :3], 0.0)
        distances = np.einsum("ij,ij->i", point1 + offset1 - point0 - offset0, normal)

        self.saw_speculative_contact |= bool(np.any((distances >= 0.0) & (distances <= self.gap_sum)))
        self.saw_penetrating_contact |= bool(np.any(distances < 0.0))

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.collision_pipeline.collide(self.state_0, self.contacts)
        self._record_contact_bands()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.log_hydro_contact_surface(self.collision_pipeline.hydroelastic_sdf.get_contact_surface())
        self.viewer.log_lines(
            "/contact_band_guides",
            self.guide_starts,
            self.guide_ends,
            self.guide_colors,
            0.01,
        )
        self.viewer.end_frame()

    def test_final(self):
        """Verify the sphere traverses the speculative band and rests at the margin."""
        sphere_z = float(self.state_0.body_q.numpy()[self.sphere_body, 2])
        real_surface_separation = sphere_z - self.sphere_radius - self.box_half_height

        assert self.saw_no_contact, "Sphere never started outside the hydroelastic contact envelope."
        assert self.saw_speculative_contact, "Sphere never produced a speculative hydroelastic contact."
        assert self.saw_penetrating_contact, "Sphere never activated a hydroelastic contact."
        assert abs(real_surface_separation - self.margin_sum) < 0.06, (
            f"Expected visible surface separation near margin_sum={self.margin_sum:.3f}, "
            f"got {real_surface_separation:.3f}."
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
