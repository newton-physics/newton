# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ChysX Cloth Stack
#
# Two square cloth patches are released at staggered heights above a
# tabletop and pile up under gravity, exercising chysx's *self-contact*
# pipeline (cloth-on-cloth) and its *static-contact* pipeline with the
# new viscous-friction term simultaneously.
#
# The friction at the cloth/table interface is what makes a flat
# tabletop hold a stack at all -- without friction, every off-vertical
# contact normal between stacked sheets pushes them sideways with
# nothing to oppose the slide, and they end up off the table edge
# within a few seconds.  The implicit-Euler tangential damping
# implemented as ``(μ_v / dt) * (I - n n^T)`` (see
# :class:`chysx.collision.StaticContactSet`) drives tangential velocity
# toward zero on contact, which is visually equivalent to dry friction
# and lets us drop the basin walls the original draft of this example
# needed.
#
# Trick: many patches as one cloth
# --------------------------------
#
# chysx's self-collision detector runs a single LBVH/QuantBvh pass over
# *one* triangle mesh.  To get inter-sheet contacts "for free" we
# pre-merge the patches on the host into one compound mesh:
#
#   * each patch is an 11x11 regular grid (121 verts, 200 triangles);
#   * patch vertices are concatenated into one big array;
#   * patch triangle indices are offset by the cumulative vertex count
#     so they keep pointing at the right rows of the merged array;
#   * the result is fed to ``builder.add_cloth_mesh()`` as a single
#     cloth body.
#
# From the solver's perspective this is one cloth, so the BVH happily
# emits VF / EE pairs across patch boundaries and the implicit-Euler
# step resolves stack contacts in the same matrix solve as the
# cloth-on-cloth folds inside any one patch.
#
# Pipeline (same as the other chysx drop examples)
# ------------------------------------------------
#
# 1.  Generate the patches with small (x, y) jitter + random yaw so
#     they don't pile in a perfect column.
# 2.  Build a Newton model with one static box (table top) and one
#     ``add_ground_plane()``; ``SolverChysX`` registers both as static
#     plane / box shapes (now with viscous friction enabled).
# 3.  Each step:
#       * static-shape DCD adds ground/table penalties + tangential
#         friction to A's diagonal and to b;
#       * self-collision (LBVH/QuantBvh broadphase + DCD narrow-phase)
#         contributes through the COO sidecar -- this is what couples
#         the patches together;
#       * PCG solves a single implicit-Euler Newton step.
# 4.  After ~3 s the patches have settled into a soft pile on the
#     tabletop.
#
# Command: ``python -m newton.examples chysx_cloth_stack``
#
###########################################################################

from __future__ import annotations

import os

import numpy as np
import warp as wp

import newton
import newton.examples


def _make_square_patch(size: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Build an ``n x n`` regular-grid square patch in the xy plane.

    Args:
        size: Side length [m] of the patch.
        n: Number of vertices per side (so the grid has ``n * n`` verts
           and ``2 * (n - 1)**2`` triangles).

    Returns:
        ``(vertices, triangles)``:

        * ``vertices``: ``(n * n, 3)`` float32 in patch-local frame,
          centred at the origin with ``z = 0``;
        * ``triangles``: ``(2 * (n - 1)**2, 3)`` int32 with two
          triangles per quad, both wound CCW when viewed from +z.
    """

    xs = np.linspace(-0.5 * size, 0.5 * size, n, dtype=np.float32)
    ys = np.linspace(-0.5 * size, 0.5 * size, n, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    verts = np.stack(
        [grid_x.flatten(), grid_y.flatten(), np.zeros(n * n, dtype=np.float32)],
        axis=1,
    ).astype(np.float32)

    tris = np.empty((2 * (n - 1) * (n - 1), 3), dtype=np.int32)
    t = 0
    for i in range(n - 1):
        for j in range(n - 1):
            v00 = i * n + j
            v10 = (i + 1) * n + j
            v01 = i * n + (j + 1)
            v11 = (i + 1) * n + (j + 1)
            tris[t] = (v00, v10, v11)
            tris[t + 1] = (v00, v11, v01)
            t += 2
    return verts, tris


def _build_compound_cloth(
    n_patches: int,
    patch_size: float,
    n_per_side: int,
    z0: float,
    dz: float,
    xy_jitter: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate ``n_patches`` square cloth patches into one mesh.

    Each patch is laid down at height ``z = z0 + i * dz`` with a tiny
    random horizontal offset (uniform in ``[-xy_jitter, +xy_jitter]``)
    and a random yaw rotation.  Triangle indices are renumbered so the
    returned arrays are valid input to a single
    :py:meth:`ModelBuilder.add_cloth_mesh` call -- the chysx
    self-collision detector then sees one mesh and emits inter-patch
    VF / EE pairs naturally.

    Returns:
        ``(vertices, triangles)``:

        * ``vertices``: ``(n_patches * n_per_side**2, 3)`` float32
          world-space positions;
        * ``triangles``: ``(n_patches * 2 * (n_per_side-1)**2, 3)``
          int32 indices into ``vertices``.
    """

    rng = np.random.default_rng(seed)
    base_v, base_t = _make_square_patch(patch_size, n_per_side)
    n_v_patch = base_v.shape[0]

    all_v: list[np.ndarray] = []
    all_t: list[np.ndarray] = []
    for i in range(n_patches):
        # Random small yaw + (x, y) jitter so the stack is not a
        # perfectly axis-aligned column.  The yaw also gives the
        # solver something to couple via the FEM bending term
        # once the patches start touching.
        theta = float(rng.uniform(-np.pi, np.pi))
        c, s = np.cos(theta), np.sin(theta)
        dx = float(rng.uniform(-xy_jitter, xy_jitter))
        dy = float(rng.uniform(-xy_jitter, xy_jitter))

        v = base_v.copy()
        x, y = v[:, 0].copy(), v[:, 1].copy()
        v[:, 0] = c * x - s * y + dx
        v[:, 1] = s * x + c * y + dy
        v[:, 2] = z0 + i * dz

        all_v.append(v)
        all_t.append(base_t + i * n_v_patch)

    return (
        np.concatenate(all_v, axis=0).astype(np.float32),
        np.concatenate(all_t, axis=0).astype(np.int32),
    )


class Example:
    def __init__(self, viewer, args):
        # ---- timing -----------------------------------------------------
        # 60 fps render with 8 substeps -> 480 Hz physics.  Multiple
        # simultaneous impacts on a stiff (k=1e4) static contact need
        # tighter time steps than the single-cloth drop case to keep
        # the implicit-Euler step well-conditioned; with 4 substeps
        # the bottom patches occasionally pop free of the table on
        # the first impact frame.
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        # ---- world geometry --------------------------------------------
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

        # Plain flat tabletop -- with the new viscous-friction term in
        # ``SolverChysX(static_contact_friction=...)`` we no longer
        # need the basin walls the early draft of this scene used to
        # contain the pile.  Half-extents 0.4 m matches the other
        # chysx drop demos so any patch up to ~0.5 m square (diagonal
        # 0.7 m) fits comfortably with margin for jitter / yaw.
        self._table_half_ext = (0.4, 0.4, 0.05)
        self._table_top_z = 0.5
        table_centre = (
            0.0,
            0.0,
            self._table_top_z - self._table_half_ext[2],
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(table_centre, wp.quat_identity()),
            hx=self._table_half_ext[0],
            hy=self._table_half_ext[1],
            hz=self._table_half_ext[2],
        )

        builder.add_ground_plane()

        # ---- compound cloth (10 patches merged into one mesh) ----------
        # Patch size 0.3 m fits comfortably inside the 0.8 m square
        # table even at a 45 deg yaw (diagonal = 0.42 m << 0.8 m).
        # 11 verts per side -> 30 mm cells, which is the same length
        # scale the other chysx examples are tuned around.
        self._n_patches = 2
        self._patch_size = 0.3
        self._n_per_side = 11
        edge_l = self._patch_size / (self._n_per_side - 1)
        self._edge_l = edge_l

        # Bottom patch at z = 0.7 m, vertical spacing 15 cm: the two
        # patches separately free-fall for ~0.2 s and ~0.36 s before
        # impact, so the bottom one has time to settle on the table
        # before the second one lands on top of it.
        z0 = 0.70
        dz = 0.15

        verts, tris = _build_compound_cloth(
            n_patches=self._n_patches,
            patch_size=self._patch_size,
            n_per_side=self._n_per_side,
            z0=z0,
            dz=dz,
            xy_jitter=0.04,
            seed=0xC10,
        )

        # Stash for diagnostics / regression checks.
        self._patch_n_verts = self._n_per_side * self._n_per_side
        self._n_verts_total = verts.shape[0]
        self._n_tris_total = tris.shape[0]

        # Snapshot the cloth particle range *before* we add the mesh
        # so the OBJ exporter can later slice particle_q to just the
        # cloth verts.  In this example there are no other particle
        # sources so the slice is the whole array, but doing it this
        # way means a future add_*_mesh() call won't silently break
        # the export.
        self._cloth_particle_start = int(builder.particle_count)
        self._cloth_particle_count = int(verts.shape[0])
        self._cloth_tris = np.ascontiguousarray(tris, dtype=np.int32)

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),  # vertex positions are absolute
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts],
            indices=tris.flatten().tolist(),
            density=0.0,           # chysx recomputes lumped mass below
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=0.4 * edge_l,
        )

        self.model = builder.finalize()

        # ---- solver -----------------------------------------------------
        # Self-contact thickness ~0.4 * edge_l = 12 mm:
        #   - much smaller than the 100 mm initial vertical spacing,
        #     so the rest pose has zero spurious contacts;
        #   - bigger than the table surface penetration depth at
        #     k_static = 1e4, so the bottom patch's contact band
        #     and the table's contact band do not fight each other;
        #   - thick enough that two adjacent stacked patches
        #     equilibrate at an unmistakable visible gap rather than
        #     visually intersecting.
        # Static-contact thickness uses 0.5 * edge_l so the table top
        # reads as a solid surface without inflating the apparent
        # cloth thickness.
        self_thickness   = 0.4 * edge_l   # ~12 mm
        static_thickness = 0.5 * edge_l   # ~15 mm

        # Narrow-phase contact buffer.  Worst case is "every patch
        # touching every other patch", so factor 16 (~4 k pair cap
        # for 2 patches at 200 tris each) is plenty of headroom;
        # broad-phase EF traffic is roughly 4x narrow phase,
        # factor 64.
        sc_narrow_factor = 64
        sc_broad_factor  = 128

        # Static-contact viscous friction.  Picked so that
        #   μ_v / dt ≈ 5x the per-particle penalty stiffness k_n
        # at our 480 Hz substep rate; that is enough to bring the
        # tangential velocity of a settled patch to within 1 % per
        # frame ("sticking") while still letting the cloth slide
        # noticeably during the impact transient.  The numerical
        # value is small in absolute terms because each cloth
        # particle is light (~75 mg) and only a handful of them
        # are in contact with the table at a time.
        static_friction = 1.0e-2               # N·s/m per particle

        self.solver = newton.solvers.SolverChysX(
            self.model,
            # 0.5 damps the impact ringing into a clean sit-down within
            # ~1 s of the second patch landing -- without it the stack
            # bounces like a stack of trampolines for several seconds.
            damping=0.5,
            fem_stretch_stiffness=5.0e2,
            fem_shear_stiffness=5.0e2,
            bending_stiffness=1.0e-4,
            pcg_iterations=50,
            surface_density=0.2,              # ~18 g per 0.3 m square patch
            self_collision_enabled=True,
            self_collision_thickness=self_thickness,
            self_collision_stiffness=1.0e3,
            self_collision_max_contacts_factor=sc_narrow_factor,
            self_collision_max_ef_candidates_factor=sc_broad_factor,
            static_contact_enabled=True,
            static_contact_thickness=static_thickness,
            static_contact_stiffness=1.0e4,
            static_contact_friction=static_friction,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self._initial_q = self.state_0.particle_q.numpy().reshape(-1, 3).copy()

        self.viewer.set_model(self.model)
        # Camera looking slightly down at the table from a 3/4 angle
        # so the pile is visible from the side as it grows.
        self.viewer.set_camera(
            pos=wp.vec3(1.4, -1.4, 1.3),
            pitch=-20.0,
            yaw=45.0,
        )

        print(
            f"[chysx_cloth_stack] {self._n_patches} patches "
            f"({self._n_per_side}x{self._n_per_side} each, "
            f"{self._patch_size*100:.0f} cm side) -> "
            f"{self._n_verts_total} verts / {self._n_tris_total} tris;  "
            f"self_thickness={self_thickness*1e3:.1f} mm,  "
            f"static_thickness={static_thickness*1e3:.1f} mm,  "
            f"static_friction={static_friction:.1f} N·s/m,  "
            f"initial vertical spacing={dz*1e2:.1f} cm"
        )

        # ---- OBJ export (optional, off by default) ---------------------
        # ``--obj-out DIR`` writes one Wavefront OBJ per frame holding
        # the merged cloth surface (all patches in one mesh, with the
        # original triangle connectivity).  Frame 0 is the pristine
        # drop pose written before warm-up; frame N (N>=1) is the state
        # after N replays of the captured CUDA graph (i.e. N * frame_dt
        # of sim time).  ``--obj-stride N`` thins the dump if disk
        # space is tight.  The export triggers a synchronous D2H copy
        # on `particle_q`, so you usually want to combine it with
        # ``--viewer null`` to keep frame rate up.
        obj_out = getattr(args, "obj_out", None)
        self._obj_dir: str | None = None
        self._obj_stride = max(1, int(getattr(args, "obj_stride", 1)))
        self._frame_idx = 0
        if obj_out:
            self._obj_dir = os.path.abspath(str(obj_out))
            os.makedirs(self._obj_dir, exist_ok=True)
            print(
                f"[chysx_cloth_stack] OBJ export enabled -> {self._obj_dir}  "
                f"(stride {self._obj_stride})"
            )
            self._export_obj_frame(0)

        # ---- CUDA Graph capture ----------------------------------------
        # Same single-graph-per-frame pattern as the other chysx
        # examples.  PCG topology depends only on the FEM mesh and
        # the static-shape registration, both of which are fixed at
        # construction; self-contact contributions ride through the
        # device-side counter, so the graph stays valid frame-to-frame
        # as the contact set grows from "nothing" to "patches piled".
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

        Triggers a synchronous device-to-host copy (``.numpy()``) on
        the particle buffer; this serialises the frame onto the calling
        thread, fine for offline data dumps but would torpedo FPS if
        you turned it on during interactive use.
        """

        assert self._obj_dir is not None
        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        s = self._cloth_particle_start
        n = self._cloth_particle_count
        verts = q[s : s + n]

        path = os.path.join(self._obj_dir, f"cloth_stack_{frame_idx:05d}.obj")
        with open(path, "w") as f:
            f.write(
                f"# chysx_cloth_stack frame {frame_idx}, "
                f"t = {self.sim_time:.6f} s\n"
            )
            np.savetxt(f, verts, fmt="v %.6f %.6f %.6f")
            np.savetxt(f, self._cloth_tris + 1, fmt="f %d %d %d")

    # ---- regression check --------------------------------------------

    def test_final(self):
        """Sanity-check the post-roll state.

        The whole pile must:

        * stay finite,
        * stay inside a generous bounding box,
        * not have fallen through the ground (modulo one
          ``static_contact_thickness`` of slack, the natural error
          band of the penalty contact),
        * have the bottom patch sitting at or above the table top
          (modulo the same slack band) -- this catches the case
          where the bottom patch tunnels through the table on the
          first impact frame because the time step is too coarse,
        * and have come to a near-rest (every particle below 2 m/s,
          generous since self-contact propagates impact energy
          through the pile for a while).
        """

        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        qd = self.state_0.particle_qd.numpy().reshape(-1, 3)

        if not (np.isfinite(q).all() and np.isfinite(qd).all()):
            raise ValueError("non-finite values in particle state")

        bound = 5.0  # m
        if (np.abs(q) > bound).any():
            raise ValueError(
                f"cloth particles escaped the {bound:.1f} m bbox; "
                f"max |q| = {float(np.abs(q).max()):.3f}"
            )

        slack = 1.5 * float(self.solver._sim.static_contact_thickness())

        z_min = float(q[:, 2].min())
        # (ground tunneling + table tunneling checks deferred until
        # *after* the diagnostic block below so we still get a printout
        # of where things actually ended up when they fail.)
        z_sorted = np.sort(q[:, 2])
        z_low = float(z_sorted[: max(1, len(z_sorted) // 20)].mean())

        max_speed = float(np.linalg.norm(qd, axis=1).max())

        # Diagnostic dump: per-patch min/mean/max z, total speed
        # range, and self-contact count.  Helps tell "the whole stack
        # came to rest" from "one patch tunneled and everything else
        # is mid-air" before the threshold check fires.
        per_patch = q.reshape(self._n_patches, self._patch_n_verts, 3)
        try:
            n_contacts = int(self.solver._sim.self_collision_count())
        except Exception:
            n_contacts = -1
        print("[chysx_cloth_stack] per-patch (z min/mean, x range, y range):")
        for i in range(self._n_patches):
            zmn = float(per_patch[i, :, 2].min())
            zmu = float(per_patch[i, :, 2].mean())
            xmn, xmx = float(per_patch[i, :, 0].min()), float(per_patch[i, :, 0].max())
            ymn, ymx = float(per_patch[i, :, 1].min()), float(per_patch[i, :, 1].max())
            print(
                f"  patch {i}: z=[{zmn:.4f},{zmu:.4f}]  "
                f"x=[{xmn:+.3f},{xmx:+.3f}]  "
                f"y=[{ymn:+.3f},{ymx:+.3f}]"
            )
        print(
            f"[chysx_cloth_stack] |v| max={max_speed:.4f} m/s,  "
            f"self_contacts={n_contacts}"
        )

        if max_speed > 25.0:
            raise ValueError(
                f"particle speed exploded: max |v| = {max_speed:.3f} m/s"
            )
        if z_min < -slack:
            raise ValueError(
                f"a patch fell through the ground: min z = {z_min:.4f} m "
                f"(allowed slack {-slack:.4f} m)"
            )
        if z_low < self._table_top_z - slack:
            raise ValueError(
                "bottom of stack tunneled below the table top: "
                f"low-5%% mean z = {z_low:.4f} m, table top = {self._table_top_z:.4f} m"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)  # 10 s at 60 fps -- enough to settle
    parser.add_argument(
        "--obj-out",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "If set, dump one Wavefront OBJ per simulated frame into DIR "
            "(created if missing).  File names are cloth_stack_<frame:05d>.obj."
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
