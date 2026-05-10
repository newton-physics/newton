# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ChysX T-shirt Drop
#
# A USD T-shirt mesh free-falls onto a static table sitting on top of an
# infinite ground plane.  Same scene geometry as
# ``example_cloth_franka.py`` (table half-extents + T-shirt asset) but
# the Franka arm is removed: this is a pure cloth-on-static demonstration
# of chysx's static-shape contact pipeline driven by a real garment
# topology rather than a regular grid.
#
# Pipeline
# --------
#
# 1.  Load ``unisex_shirt.usd`` via newton.usd, convert from cm to m,
#     and centre the mesh at the origin so it can be re-positioned
#     anywhere with ``add_cloth_mesh(pos=...)``.
# 2.  Build a Newton model with one static box (the table top) and one
#     ``add_ground_plane()``.  ``SolverChysX(static_contact_enabled=True)``
#     scans the model at construction time and registers both as
#     :class:`chysx.collision.PlaneShape` / :class:`chysx.collision.BoxShape`
#     entries on the static contact set.
# 3.  Each step:
#       * static-shape DCD adds penalty contributions to A's diagonal
#         and to b (no off-diagonal sidecar);
#       * self-collision (LBVH/QuantBvh broadphase + DCD narrow-phase)
#         contributes through the existing COO sidecar;
#       * PCG solves a single implicit-Euler Newton step;
#       * positions and velocities are finalised in place.
# 4.  After ~3 s the cloth has settled into a draped pose on the table.
#
# Command: ``python -m newton.examples chysx_tshirt_drop``
#
###########################################################################

from __future__ import annotations

import os

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd


def _load_centered_tshirt_m() -> tuple[np.ndarray, np.ndarray, float]:
    """Load the unisex T-shirt USD mesh and return ``(vertices, indices,
    median_edge_length)`` in metre units, centred at the origin.

    The asset ships in centimetres so we apply a ``1e-2`` scale on
    load.  Centring lets ``add_cloth_mesh(pos=p)`` place the garment
    by its own bounding-box centre rather than by some arbitrary
    USD-author-chosen origin.
    """

    stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
    prim = stage.GetPrimAtPath("/root/shirt")
    m = newton.usd.get_mesh(prim)

    v_cm = np.asarray(m.vertices, dtype=np.float32)
    idx = np.asarray(m.indices, dtype=np.int32).reshape(-1, 3)

    v_m = v_cm * 0.01  # cm -> m
    centre = 0.5 * (v_m.min(axis=0) + v_m.max(axis=0))
    v_m = v_m - centre  # bbox-centre at origin

    # Median edge length drives our self-collision / static-contact
    # thickness picks below; cheap enough to compute on the host.
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
        # 60 fps render with 10 substeps -> 600 Hz physics.  Stiff
        # static-contact (k=1e4) plus the heavy self-contact load once
        # the cloth folds on itself needs a small time step to keep
        # the implicit-Euler step well-conditioned; with 4 substeps
        # the impact velocity (~2 m/s) is too coarsely resolved and
        # produces a noticeable bounce in the first 0.3 s.
        self.fps = 1000
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        # ---- world geometry --------------------------------------------
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

        # Table: same proportions as cloth_franka's 40 cm half-extents
        # but in metres.  Top surface at z = 0.5; thicker than chysx's
        # other static-contact demo so the GL viewer's box mesh is
        # legible from the default camera.
        self._table_half_ext = (0.4, 0.4, 0.05)
        self._table_top_z = 0.5
        table_centre = (0.0, 0.0, self._table_top_z - self._table_half_ext[2])
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(table_centre, wp.quat_identity()),
            hx=self._table_half_ext[0],
            hy=self._table_half_ext[1],
            hz=self._table_half_ext[2],
        )

        # Ground plane.  Registered second on purpose -- the chysx
        # static-contact set picks the deepest hit per particle, so
        # any registration order works, but reading the scene
        # top-down (table first, then ground) matches how a real
        # tabletop study is described.
        builder.add_ground_plane()

        # T-shirt mesh.  The asset is authored with X = width,
        # Y = body height, Z = front-back thickness (so a roughly
        # 65 cm x 65 cm x 27 cm bounding box once we scale to metres),
        # which already matches the "lying flat on a table" pose --
        # no rotation needed.  We just bbox-centre the vertices and
        # drop them in `add_cloth_mesh(pos=...)`.
        verts_m, tris, edge_med = _load_centered_tshirt_m()
        self._edge_med = edge_med
        self._tshirt_half_z = float(0.5 * (verts_m[:, 2].max() - verts_m[:, 2].min()))

        # Drop height: leave a clear ~0.2 m of free fall between the
        # garment's lowest vertex and the table top.  Higher drops
        # (e.g. 0.5 m) are visually more dramatic but the resulting
        # ~3 m/s impact velocity overdrives the penalty contact
        # (force = k * depth) and rings the cloth like a struck drum
        # for ~1 s before settling -- 0.2 m gives a clean
        # 2 m/s impact that the implicit-Euler step absorbs in a
        # single frame.
        drop_height = self._table_top_z + self._tshirt_half_z + 0.1

        # Snapshot the cloth particle range *before* we add the mesh
        # so the OBJ exporter can later slice particle_q to just the
        # cloth verts.  In this example there are no other particle
        # sources so the slice is the whole array, but doing it this
        # way means a future add_*_mesh() call won't silently break
        # the export.
        self._cloth_particle_start = int(builder.particle_count)
        self._cloth_particle_count = int(verts_m.shape[0])
        self._cloth_tris = np.ascontiguousarray(tris, dtype=np.int32)

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, drop_height),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts_m],
            indices=tris.flatten().tolist(),
            density=0.0,           # chysx recomputes lumped mass below
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=0.4 * edge_med,
        )

        self.model = builder.finalize()

        # ---- solver -----------------------------------------------------
        #
        # Self-collision thickness is *deliberately* tight (~3 mm).
        # The unisex T-shirt asset is authored as a draped 3-D
        # garment with the front and back panels separated by ~27 cm
        # to model the wearer's torso; under gravity those two
        # panels collapse onto each other.  A generous self-contact
        # band (e.g. 1.2x edge length, like the twist scene uses)
        # would treat every front-back vertex pair as a contact
        # the moment the cloth folds, and the asymmetric T-shirt
        # geometry then drags the whole sheet sideways under the
        # net penalty force.  3 mm is just thick enough to catch
        # genuine triangle inversion while letting the front and
        # back panels rest in contact at zero net force.
        #
        # Static-contact thickness is half the median edge length
        # so the table top reads as a real surface without
        # inflating the visible cloth thickness.
        self_thickness   = 1.2e-3            # 3 mm
        static_thickness = 0.5 * edge_med    # ~5 mm

        # Narrow-phase contact buffer.  ~6.4 k particles, expect
        # at most ~5 k VF/EE pairs at full collapse, so factor 8
        # (51 k cap) is plenty.  Broad-phase (LBVH-EF candidate)
        # traffic is ~4x narrow-phase, so factor 32.
        sc_narrow_factor = 32
        sc_broad_factor  = 128

        self.solver = newton.solvers.SolverChysX(
            self.model,
            # Moderate velocity damping bleeds off the impact energy
            # quickly enough that the cloth reaches a clearly
            # static drape within ~3 s -- without it the stiff
            # penalty contact rings the garment like a struck drum.
            damping=0.5,
            fem_stretch_stiffness=5.0e2,
            fem_shear_stiffness=5.0e2,
            bending_stiffness=2.0e-4,
            pcg_iterations=50,
            surface_density=0.1,             # cuda-cloth tablecloth-like
            self_collision_enabled=True,
            self_collision_thickness=self_thickness,
            self_collision_stiffness=5.0e3,
            self_collision_max_contacts_factor=sc_narrow_factor,
            self_collision_max_ef_candidates_factor=sc_broad_factor,
            static_contact_enabled=True,
            static_contact_thickness=static_thickness,
            static_contact_stiffness=1.0e4,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self._initial_q = self.state_0.particle_q.numpy().reshape(-1, 3).copy()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(1.6, -1.6, 1.4),
            pitch=-22.0,
            yaw=45.0,
        )

        # ---- OBJ export (optional, off by default) ---------------------
        # ``--obj-out DIR`` writes one Wavefront OBJ per frame holding
        # the cloth's deformed surface (verts + the original triangle
        # connectivity).  Frame 0 is the pristine drop pose written
        # before warm-up; frame N (N>=1) is the state after N replays
        # of the captured CUDA graph (i.e. N * frame_dt of sim time).
        # ``--obj-stride N`` thins the dump if disk space is tight.
        obj_out = getattr(args, "obj_out", None)
        self._obj_dir: str | None = None
        self._obj_stride = max(1, int(getattr(args, "obj_stride", 1)))
        self._frame_idx = 0
        if obj_out:
            self._obj_dir = os.path.abspath(str(obj_out))
            os.makedirs(self._obj_dir, exist_ok=True)
            print(
                f"[chysx_tshirt_drop] OBJ export enabled -> {self._obj_dir}  "
                f"(stride {self._obj_stride})"
            )
            self._export_obj_frame(0)

        # ---- CUDA Graph capture ----------------------------------------
        # Same single-graph-per-frame pattern as the other chysx
        # examples.  Static-contact contributions are diagonal-only,
        # so the captured PCG topology stays valid frame-to-frame.
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
        self._frame_idx += 1
        if self._obj_dir is not None and (self._frame_idx % self._obj_stride) == 0:
            self._export_obj_frame(self._frame_idx)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    # ---- OBJ export --------------------------------------------------

    def _export_obj_frame(self, frame_idx: int) -> None:
        """Write one Wavefront OBJ for the current cloth state.

        Triggers a synchronous device-to-host copy (``.numpy()``) on the
        particle buffer; this serialises the frame onto the calling
        thread, which is fine for offline data dumps but would torpedo
        FPS if you turned it on during interactive use.
        """

        assert self._obj_dir is not None
        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        s = self._cloth_particle_start
        n = self._cloth_particle_count
        verts = q[s : s + n]

        path = os.path.join(self._obj_dir, f"tshirt_{frame_idx:05d}.obj")
        with open(path, "w") as f:
            f.write(
                f"# chysx_tshirt_drop frame {frame_idx}, "
                f"t = {self.sim_time:.6f} s\n"
            )
            np.savetxt(f, verts, fmt="v %.6f %.6f %.6f")
            np.savetxt(f, self._cloth_tris + 1, fmt="f %d %d %d")

    # ---- regression check --------------------------------------------

    def test_final(self):
        """Sanity-check the post-roll state.

        The garment must remain finite, must not have escaped a
        generous bounding box, and must rest above the ground (modulo
        a one-thickness slack band that the penalty contact only
        enforces up to).
        """

        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        qd = self.state_0.particle_qd.numpy().reshape(-1, 3)

        if not (np.isfinite(q).all() and np.isfinite(qd).all()):
            raise ValueError("non-finite values in particle state")

        bound = 5.0  # m
        if (np.abs(q) > bound).any():
            raise ValueError(
                f"T-shirt particles escaped the {bound:.1f} m bbox; "
                f"max |q| = {float(np.abs(q).max()):.3f}"
            )

        z_min = float(q[:, 2].min())
        slack = -2.0 * float(self.solver._sim.static_contact_thickness())
        if z_min < slack:
            raise ValueError(
                f"T-shirt fell through the ground: min z = {z_min:.4f} m "
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
    parser.add_argument(
        "--obj-out",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "If set, dump one Wavefront OBJ per simulated frame into DIR "
            "(created if missing).  File names are tshirt_<frame:05d>.obj."
        ),
    )
    parser.add_argument(
        "--obj-stride",
        type=int,
        default=1,
        metavar="N",
        help="With --obj-out, write only every N-th frame.",
    )

    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
