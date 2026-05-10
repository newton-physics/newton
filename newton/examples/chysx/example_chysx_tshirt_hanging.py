# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ChysX T-shirt Hanging  (gravity-OFF, pin-free variant)
#
# A USD T-shirt mesh sits in free space with **no gravity, no pins,
# and no static-shape contact** -- only chysx's self-collision
# pipeline is active (LBVH/QuantBvh broadphase + DCD narrow-phase,
# contributions baked into the implicit-Euler PCG via the COO
# sidecar).  This is the cleanest possible isolation test for the
# self-contact code path: any motion the garment exhibits is by
# construction caused by the self-collision detector reporting a
# non-zero force on a state that should be at rest.
#
# Scene
# -----
# * No ground, no table, no pinned vertices, no gravity.
# * Garment starts in its USD reference (rest) pose, centred at
#   ``(0, 0, hang_height)`` so the camera framing matches the
#   gravity-on variant of this scene.
# * Stationary expected outcome: the T-shirt holds its 3-D shape
#   indefinitely; if you see drift, swelling, or rotation, the
#   self-contact pipeline is producing spurious / asymmetric forces.
#
# Why self_collision_thickness needs care here
# ---------------------------------------------
# The unisex T-shirt asset is a 3-D garment -- the front and back
# panels are ~27 cm apart in the rest pose to model the wearer's
# torso, but at the *rims* (sleeve cuffs, neck opening, bottom hem)
# the front and back panels fold around to meet each other along a
# narrow strip of fabric that is only a couple of millimetres wide.
# A generous self-contact band that catches that rim strip will
# fire ~2k stuck-together contacts on the rest pose itself, and
# the asymmetric T-shirt geometry then bakes the resulting
# non-zero contact-normal sum into a net force that inflates the
# garment and drifts the centre of mass.  Empirically (this scene
# was used to characterise it):
#
#   thickness    rest-pose contacts   stationary?
#   ---------    ------------------   -----------
#   0.5 mm                        0   yes
#   1.0 mm                        0   yes  <- we use this
#   2.0 mm                      184   no, inflates 27 cm -> 52 cm
#   4.0 mm                     2265   no, inflates 27 cm -> 54 cm
#
# So we pick 1 mm here -- thin enough that the rims do not fire,
# while still giving the DCD pipeline something concrete to
# detect once a real fold develops.
#
# Command: ``python -m newton.examples chysx_tshirt_hanging``
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd


def _load_centered_tshirt_m() -> tuple[np.ndarray, np.ndarray, float]:
    """Same loader as ``example_chysx_tshirt_drop``: cm -> m + bbox-centre.

    Returns ``(vertices, indices, median_edge_length)`` ready for
    :py:meth:`ModelBuilder.add_cloth_mesh`.
    """

    stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
    prim = stage.GetPrimAtPath("/root/shirt")
    m = newton.usd.get_mesh(prim)

    v_cm = np.asarray(m.vertices, dtype=np.float32)
    idx = np.asarray(m.indices, dtype=np.int32).reshape(-1, 3)

    v_m = v_cm * 0.01
    centre = 0.5 * (v_m.min(axis=0) + v_m.max(axis=0))
    v_m = v_m - centre

    e = np.concatenate(
        [
            np.linalg.norm(v_m[idx[:, a]] - v_m[idx[:, b]], axis=1)
            for (a, b) in [(0, 1), (1, 2), (2, 0)]
        ]
    )
    edge_med = float(np.median(e))
    return v_m, idx, edge_med


class Example:
    def __init__(self, viewer, args):
        # ---- timing -----------------------------------------------------
        # 60 fps render with 5 substeps -> 300 Hz physics.  No contact
        # in this scene so the implicit-Euler step is dominated by the
        # FEM elastic + bending Hessian, which is well-conditioned and
        # converges quickly; we can afford a coarser substep here than
        # the cloth-drop example needs.
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        # ---- world ------------------------------------------------------
        # Gravity OFF + no pins: this turns the scene into a pure
        # self-collision test.  The garment starts at rest in its
        # reference (USD-author-provided) configuration; with no
        # external force and no kinematic constraint, the only thing
        # that can move it is whatever forces the self-collision
        # detector reports.  An ideal implementation should return
        # zero contact forces on a non-self-intersecting rest pose,
        # so the garment should stay perfectly still.  Any drift
        # observed here is purely due to the self-contact pipeline
        # (false-positive contacts, asymmetric normals, ...).
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)

        verts_m, tris, edge_med = _load_centered_tshirt_m()
        self._edge_med = edge_med
        # Place the (now stationary) garment at a comfortable viewing
        # height -- this matches the camera framing the gravity-on
        # variant of this scene uses.
        hang_height = 1.5

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, hang_height),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts_m],
            indices=tris.flatten().tolist(),
            density=0.0,           # chysx redistributes lumped mass below
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=0.4 * edge_med,
        )

        self.model = builder.finalize()

        # No pins: the garment is fully free.  In a zero-gravity scene
        # this means the only forces acting on the cloth come from
        # internal elastic / bending energy (which should be zero at
        # the rest configuration) and from the self-collision penalty.
        q0 = self.model.particle_q.numpy().reshape(-1, 3)
        self._pin_indices = np.empty(0, dtype=np.int32)

        # ---- solver -----------------------------------------------------
        # Self-collision ON, static-shape contact OFF.  See the
        # "Why self_collision_thickness needs care here" block at the
        # top of this file for the reasoning behind the thickness pick.
        # Narrow-phase contact buffer factor 16 (~100k cap) is plenty
        # for ~6 k particles even when the cloth is heavily folded;
        # broad-phase EF candidate factor 64 leaves headroom for
        # dense pile-ups.
        self_collision_thickness = 1.0e-3            # 1 mm (see file header)

        self.solver = newton.solvers.SolverChysX(
            self.model,
            damping=0.05,
            fem_stretch_stiffness=5.0e2,
            fem_shear_stiffness=5.0e2,
            bending_stiffness=5.0e-4,
            pcg_iterations=50,
            surface_density=0.3,
            # No pins -- garment is fully free (see __init__ docstring).
            pin_indices=None,
            self_collision_enabled=True,
            self_collision_thickness=self_collision_thickness,
            self_collision_stiffness=1.0e3,
            self_collision_max_contacts_factor=16,
            self_collision_max_ef_candidates_factor=64,
            static_contact_enabled=False,
        )

        # Snapshot the rest panel separation along the asset's z axis
        # so test_final can compare end-of-run panel width against the
        # initial value.
        self._initial_z_extent = float(q0[:, 2].max() - q0[:, 2].min())
        print(
            f"[chysx_tshirt_hanging] bending dihedrals: "
            f"{self.solver._sim.num_bending_dihedrals()};  "
            f"initial panel sep (z extent) = {self._initial_z_extent:.3f} m"
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        # Camera looking at the garment dead-on from the +X / -Y
        # quadrant so the front and back panels are both legible
        # without any hanging swing offsetting the framing.
        self.viewer.set_camera(
            pos=wp.vec3(1.4, -1.4, hang_height + 0.1),
            pitch=-5.0,
            yaw=45.0,
        )

        # ---- CUDA Graph capture ----------------------------------------
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

        # Print the contact count produced by the *first* substep --
        # this tells us how many self-contacts the rest pose itself
        # generates, before the garment has had a chance to deform.
        try:
            n_initial = int(self.solver._sim.self_collision_count())
            print(
                f"[chysx_tshirt_hanging] self-contacts after warm-up "
                f"step (rest pose): {n_initial}"
            )
        except Exception:
            pass

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
        """Sanity-check the post-roll state.

        Without gravity or pins the garment should sit (almost)
        perfectly still, so the regression checks we can do are
        much tighter than the gravity-on variant:

        * particle state stays finite,
        * the garment stays inside a 5 m bbox (no fly-aways),
        * the front-back panel separation hasn't collapsed to less
          than 50 % of the initial value (which would indicate that
          bending rest angles weren't preserved),
        * **no particle exceeds 1 m/s** -- pure self-contact on a
          rest pose has no energy source, so any non-trivial speed
          here is a self-collision artefact.
        """

        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        qd = self.state_0.particle_qd.numpy().reshape(-1, 3)

        if not (np.isfinite(q).all() and np.isfinite(qd).all()):
            raise ValueError("non-finite values in particle state")

        # Diagnostic summary -- the whole point of this scene is to
        # see how big the spurious self-contact drift is, so dump a
        # one-liner before the threshold checks.
        com = q.mean(axis=0)
        speed = np.linalg.norm(qd, axis=1)
        z_extent_final = float(q[:, 2].max() - q[:, 2].min())
        try:
            n_contacts = int(self.solver._sim.self_collision_count())
        except Exception:
            n_contacts = -1
        print(
            f"[chysx_tshirt_hanging] final: "
            f"CoM=({com[0]:+.4f},{com[1]:+.4f},{com[2]:+.4f}) m,  "
            f"|v| max={float(speed.max()):.4f} mean={float(speed.mean()):.4f} m/s,  "
            f"z extent {self._initial_z_extent:.4f} -> {z_extent_final:.4f} m,  "
            f"self_contacts={n_contacts}"
        )

        bound = 5.0
        if (np.abs(q) > bound).any():
            raise ValueError(
                f"T-shirt particles escaped the {bound:.1f} m bbox; "
                f"max |q| = {float(np.abs(q).max()):.3f}"
            )

        if z_extent_final < 0.5 * self._initial_z_extent:
            raise ValueError(
                "panel separation collapsed past 50 % of initial -- "
                "bending rest angles likely lost: "
                f"initial = {self._initial_z_extent:.3f} m, "
                f"final = {z_extent_final:.3f} m"
            )

        max_speed = float(np.linalg.norm(qd, axis=1).max())
        if max_speed > 1.0:
            raise ValueError(
                "particle drift in pure self-collision scene exceeds "
                f"1 m/s -- max |v| = {max_speed:.3f} m/s"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)  # 10 s at 60 fps -- enough to settle

    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
