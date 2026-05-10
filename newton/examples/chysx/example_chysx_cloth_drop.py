# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ChysX Cloth Drop
#
# A square cloth sheet is dropped from above onto a table sitting on top
# of an infinite ground plane, exercising chysx's *static-shape contact*
# pipeline (planes + boxes baked directly into the implicit-Euler
# linear system's diagonal block + RHS).
#
# Compared to the cloth-Franka example this scene is intentionally
# minimal: no rigid solver, no collision pipeline, no robot.  Just
#
#   - one square cloth grid (21x21)
#   - one ground plane                     -> chysx StaticPlaneShape
#   - one table box (half-extents 0.4 m)   -> chysx StaticBoxShape
#
# The :class:`SolverChysX` constructor scans the model for world-static
# (``shape_body == -1``) plane / box geometries, transcribes them into
# its :class:`chysx.collision.StaticContactSet`, and from then on every
# step does
#
#   1. per-particle DCD against every primitive (deepest hit wins);
#   2. accumulate ``-k * depth * n`` into the right-hand side; and
#   3. bake ``k * (n n^T)`` into the per-particle diagonal block of A.
#
# No off-diagonal sidecar is needed because each contact only couples
# one particle to a static shape, so the captured PCG graph stays
# valid frame-to-frame.
#
# Command: ``python -m newton.examples chysx_cloth_drop``
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        # ---- timing -----------------------------------------------------
        # 60 fps render with 4 substeps -> 240 Hz physics, the smallest
        # rate where stiff (k=1e4) ground / table contacts stay
        # well-conditioned for the default PCG iteration count.
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        # ---- world geometry --------------------------------------------
        # Z-up; a 1 m square cloth at 21x21 hangs slightly above the
        # table edge and drops under gravity.  All physical units are SI.
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

        # Pedestal-style "table": a small 0.6 m x 0.6 m top surface
        # sitting at z = 0.5 (we model the box itself as a slab of
        # half-extent 0.025 m so the top face is exactly at z=0.5).
        # Smaller than the cloth's footprint so the four edges
        # overhang and drape down — the classic tablecloth scene.
        self._table_half_ext = (0.3, 0.3, 0.025)
        self._table_top_z = 0.5
        table_center = (0.0, 0.0, self._table_top_z - self._table_half_ext[2])
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(table_center, wp.quat_identity()),
            hx=self._table_half_ext[0],
            hy=self._table_half_ext[1],
            hz=self._table_half_ext[2],
        )

        # Cloth: 21 x 21 grid, 1 m square, hovering 0.5 m above the
        # table.  Centred over the pedestal so the four overhanging
        # edges drape symmetrically.
        self._cloth_dim = 21
        self._cloth_size = 1.0
        cell = self._cloth_size / (self._cloth_dim - 1)
        self._edge_l = cell

        cloth_height = self._table_top_z + 0.5
        cloth_origin = wp.vec3(
            -0.5 * self._cloth_size,
            -0.5 * self._cloth_size,
            cloth_height,
        )
        builder.add_cloth_grid(
            pos=cloth_origin,
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self._cloth_dim - 1,
            dim_y=self._cloth_dim - 1,
            cell_x=cell,
            cell_y=cell,
            mass=0.0,
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=0.4 * cell,
        )

        # Infinite ground plane at z = 0.  We add this *after* the
        # table / cloth so the scene description reads top-down, but
        # the chysx static-contact registration order is irrelevant
        # (deepest-hit wins).
        builder.add_ground_plane()

        self.model = builder.finalize()

        # ---- solver -----------------------------------------------------
        #
        # `static_contact_enabled=True` is the only knob the example
        # really cares about: it tells the solver to scan the model
        # for static plane / box shapes and route their contact
        # response straight into the implicit-Euler linear system.
        #
        # Stiffness (1e4) is high enough to keep visible penetration
        # below `static_contact_thickness` (~5 mm) under cloth-on-cloth
        # piling; raise to 1e5 for an essentially-rigid response at
        # the cost of slower PCG convergence.  cuda-cloth's twist case
        # uses 1e3 for *self*-contact -- static contacts can afford to
        # be ~10x stiffer because each particle only sees one of them
        # per step (no cross-particle coupling to muddy the
        # preconditioner).
        thickness = 0.5 * cell  # ~25 mm at 21x21, 1 m square
        self.solver = newton.solvers.SolverChysX(
            self.model,
            damping=0.05,
            fem_stretch_stiffness=5.0e2,
            fem_shear_stiffness=5.0e2,
            bending_stiffness=1.0e-4,
            pcg_iterations=30,
            surface_density=0.3,
            self_collision_enabled=True,
            self_collision_thickness=thickness,
            self_collision_stiffness=1.0e3,
            self_collision_max_contacts_factor=8,
            self_collision_max_ef_candidates_factor=32,
            static_contact_enabled=True,
            static_contact_thickness=thickness,
            static_contact_stiffness=1.0e4,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self._initial_q = self.state_0.particle_q.numpy().reshape(-1, 3).copy()

        self.viewer.set_model(self.model)
        # Camera looking slightly down at the table; the cloth drape
        # over the edge is the most interesting view.
        self.viewer.set_camera(
            pos=wp.vec3(1.6, -1.6, 1.2),
            pitch=-15.0,
            yaw=45.0,
        )

        # ---- CUDA Graph capture ----------------------------------------
        #
        # Same idea as `example_chysx_twist`: record one frame's worth
        # of substeps once and replay every frame.  The captured graph
        # is valid for the lifetime of the example because:
        #
        #   * Static contacts only touch the diagonal block of A, so
        #     adding ground / table primitives never invalidates the
        #     PCG topology that was captured.
        #   * The cloth's mesh / FEM topology is fixed at construction.
        #   * Self-contact contributions ride along through
        #     `apply_contact_spmv`, which reads pair counts off a
        #     device-side counter (no host-side topology change).
        self._cuda_graph = None
        self._capture_graph()

    # ---- per-frame physics -------------------------------------------

    def _simulate_substeps(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0,
                self.state_0,
                self.control,
                self.contacts,
                self.sim_dt,
            )

    def _capture_graph(self) -> None:
        self._cuda_graph = None
        device = wp.get_device()
        if not device.is_cuda:
            return
        # Warm-up so any lazy buffer alloc / topology rebuild lands
        # *outside* the captured region.
        self._simulate_substeps()
        wp.synchronize_device(device)

        with wp.ScopedCapture() as capture:
            self._simulate_substeps()
        self._cuda_graph = capture.graph

    def step(self):
        if self._cuda_graph is not None:
            wp.capture_launch(self._cuda_graph)
        else:
            self._simulate_substeps()
        self.sim_time += self.sim_substeps * self.sim_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    # ---- regression check --------------------------------------------

    def test_final(self):
        """Sanity-check the final state.

        The cloth must remain finite, must not have escaped the
        scene's bounding box, and -- most importantly -- must not
        have fallen through the ground plane.  We tolerate
        ``-static_contact_thickness`` of penetration since the
        penalty contact only enforces the constraint up to its
        thickness band.
        """

        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        qd = self.state_0.particle_qd.numpy().reshape(-1, 3)

        if not (np.isfinite(q).all() and np.isfinite(qd).all()):
            raise ValueError("non-finite values in particle state")

        bound = 5.0  # m
        if (np.abs(q) > bound).any():
            raise ValueError(
                f"cloth particles escaped the {bound:.1f} m bounding box; "
                f"max |q| = {float(np.abs(q).max()):.3f}"
            )

        # Ground plane is at z = 0.  Allow a small slack of one
        # contact thickness (the penalty's natural error band).
        z_min = float(q[:, 2].min())
        slack = -1.5 * float(self.solver._sim.static_contact_thickness())
        if z_min < slack:
            raise ValueError(
                f"cloth fell through the ground: min z = {z_min:.4f} m "
                f"(allowed slack {slack:.4f} m)"
            )

        max_speed = float(np.linalg.norm(qd, axis=1).max())
        if max_speed > 25.0:
            raise ValueError(
                f"particle speed exploded: max |v| = {max_speed:.3f} m/s"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)  # 10 s at 60 fps

    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
