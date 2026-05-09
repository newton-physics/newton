# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ChysX Twist
#
# 1:1 reproduction of cuda-cloth's `TwistCase`
# (see d:/physics/cuda-cloth/src/Simulator/TwistCase.{h,cpp}).
#
# The reference asset is `assets/quad/twist100.obj`, a 100x200 quad-tri
# mesh sitting in the YZ plane at x=0.  Vertex layout (row-major in Y,
# then column-major in Z) gives:
#
#     v[i]              for i in [0, 99]      -> z = -0.5  (right strip)
#     v[m_offset + i]   for i in [0, 99]      -> z = +0.5  (left strip)
#     m_offset = (cloth_size_x - 1) * cloth_size_y = 19900
#
# The two pinned strips run along +Y at the cloth's two ends; the
# *longitudinal* axis of the cloth (the line connecting the two strip
# centres) is +Z.  cuda-cloth twists each strip in the XY plane around
# that axis, with opposite sign on the two ends, exactly the way you'd
# wring out a towel:
#
#     left_target  (z=+0.5) = (sin(-w*t)*(i-mid), cos(-w*t)*(i-mid), 0) * edge_l
#     right_target (z=-0.5) = (sin(+w*t)*(i-mid), cos(+w*t)*(i-mid), 0) * edge_l
#
# with mid = (cloth_size_y - 1) / 2 = 49.5 and w = m_radius_velocity
# (= 0.01 rad per loop iteration; cuda-cloth uses dt=0.01 s, so the
# time-domain angular speed is 1.0 rad/s).
#
# Material parameters (cuda-cloth `TwistCase::Initialize`):
#   m_k          = 500     - BW98 stretch + shear stiffness
#   m_bending_k  = 5e-4    - dihedral bending
#   m_thickness  = edge_l * 0.2          (cloth thickness)
#   m_4_k        = 1000                  (collision stiffness)
#   m_4_thickness= edge_l * 0.2 * 1.2    (slightly larger so contacts
#                                         engage *before* the surfaces
#                                         actually overlap)
#   surface density (lumped FEM area-weighted): cuda-cloth scatters
#   `0.1 * area` to each of a triangle's 3 vertices (NOT divided by 3).
#   ChysX's `redistribute_mass_area_weighted(d)` scatters
#   `(d / 3) * area` per vertex, so to match cuda-cloth we use d = 0.3.
#
# Loop bound: cuda-cloth runs for 1000 steps at dt=0.01 -> 10 s.  We
# expose num_frames=1000 by default; rendering happens every step.
#
# Command: python -m newton.examples chysx_twist
#
###########################################################################

from __future__ import annotations

import os

import numpy as np
import warp as wp

import newton
import newton.examples


# Default cuda-cloth twist asset.  We ship a copy under
# `newton/examples/assets/chysx/` so the example runs out of the box;
# fall back to the original cuda-cloth asset folder for users running
# this side-by-side with the cuda-cloth checkout.
_DEFAULT_TWIST_OBJ = newton.examples.get_asset("chysx/twist100.obj")
_FALLBACK_TWIST_OBJ = r"d:\physics\cuda-cloth\assets\quad\twist100.obj"


def _resolve_twist_obj() -> str:
    if os.path.isfile(_DEFAULT_TWIST_OBJ):
        return _DEFAULT_TWIST_OBJ
    if os.path.isfile(_FALLBACK_TWIST_OBJ):
        return _FALLBACK_TWIST_OBJ
    raise FileNotFoundError(
        f"twist100.obj not found at either {_DEFAULT_TWIST_OBJ!r} or "
        f"{_FALLBACK_TWIST_OBJ!r}; copy the file from cuda-cloth's "
        f"assets/quad/ folder."
    )


def _load_twist_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a triangulated wavefront OBJ.

    Returns
    -------
    vertices : (N, 3) float32 array of vertex positions.
    indices  : (M, 3) int32 array of triangle vertex indices (0-based).

    Vertex order is preserved exactly as in the file so cuda-cloth's
    integer pin indices stay valid.  Only `v ...` and `f a b c` lines
    are recognised; UV / normal indices and quads are not handled
    (twist100.obj is plain triangles).
    """
    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            head, _, rest = line.partition(" ")
            if head == "v":
                xyz = rest.split()
                verts.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
            elif head == "f":
                parts = rest.split()
                # Strip "v/vt/vn" decorations; we only need the vertex idx.
                idx = [int(p.split("/")[0]) - 1 for p in parts]
                if len(idx) == 3:
                    tris.append((idx[0], idx[1], idx[2]))
                else:
                    # Fan-triangulate any non-triangle face (defensive;
                    # twist100.obj is already triangulated).
                    for k in range(1, len(idx) - 1):
                        tris.append((idx[0], idx[k], idx[k + 1]))
    return (
        np.asarray(verts, dtype=np.float32),
        np.asarray(tris, dtype=np.int32),
    )


class Example:
    def __init__(self, viewer, args):
        # cuda-cloth's TwistCase: dt = 0.01 s, no substepping, 1000
        # loop iterations.  PCG default `m_maxIter = 100`.
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        # ---- mesh (1:1 from cuda-cloth `TwistCase::Initialize`) ----------

        twist_obj = _resolve_twist_obj()
        verts_np, tris_np = _load_twist_obj(twist_obj)

        # Layout assumptions baked into TwistCase.cpp.  `cloth_size_y`
        # is the inner Y dimension (one column = `cloth_size_y` verts);
        # `cloth_size_x` is the number of columns.  `m_offset` is the
        # global index of the first vertex in the last column.
        self._cloth_size_y = 100
        self._cloth_size_x = 200
        self._offset = (self._cloth_size_x - 1) * self._cloth_size_y  # 19900
        if verts_np.shape[0] != self._cloth_size_x * self._cloth_size_y:
            raise ValueError(
                f"twist100.obj should hold {self._cloth_size_x * self._cloth_size_y} "
                f"vertices, got {verts_np.shape[0]}"
            )

        # cuda-cloth: m_edge_l = |v[0] - v[1]| (consecutive verts in
        # the first column = vertical edge of the right strip).
        edge_l = float(np.linalg.norm(verts_np[0] - verts_np[1]))
        self._edge_l = edge_l

        # ---- Newton model -----------------------------------------------

        # Z-up matches Newton's `cloth_twist` example and lets the
        # built-in ground / camera helpers behave sensibly even though
        # this scene has no ground plane.  cuda-cloth's gravity is
        # along world -Y (`force.y += m * gravity` in `KernelDynamic`),
        # so we keep gravity in world -Y on the chysx side too.
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)

        # `add_cloth_mesh` preserves vertex order, so cuda-cloth's
        # integer pin indices (0..99 and 19900..19999) stay valid.
        # Pass scale=1, rot=identity, pos=0 so positions go through
        # untouched.
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts_np],
            indices=tris_np.flatten().tolist(),
            density=0.0,  # ChysX recomputes per-particle mass below.
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=0.5 * edge_l * 0.4,
        )
        self.model = builder.finalize()

        # ---- pin set (right column 0..99 then left column 19900..19999) -
        #
        # cuda-cloth's TwistCase pushes both columns into `m_pin`:
        #     m_cloth.m_params.m_pin.push_back(i);
        #     m_cloth.m_params.m_pin.push_back(m_offset + i);
        # then writes:
        #     pinList[i]            = right_target  (rotates +omega)
        #     pinList[m_offset + i] = left_target   (rotates -omega)
        # We mirror that ordering here so the per-frame target update
        # below can write the two halves of `_pin_targets` independently.
        right = np.arange(self._cloth_size_y, dtype=np.int32)
        left = right + self._offset
        self._right_indices = right
        self._left_indices = left
        self._n_right = right.size
        self._n_left = left.size
        self._pin_indices = np.concatenate([right, left])

        # cuda-cloth: m_left_center  = avg(v[m_offset], v[m_offset + cy - 1])
        #             m_right_center = avg(v[0],         v[cy - 1])
        self._right_center = 0.5 * (verts_np[0] + verts_np[self._cloth_size_y - 1])
        self._left_center = 0.5 * (
            verts_np[self._offset] + verts_np[self._offset + self._cloth_size_y - 1]
        )
        # mid = (cloth_size_y - 1) / 2  (matches `float mid = (m_clothSize_y - 1) / 2.0f`)
        self._mid = 0.5 * (self._cloth_size_y - 1)

        # cuda-cloth twist parameters
        #   m_radius_velocity = 0.01 rad / loop iter, dt = 0.01 s
        #   -> 1.0 rad/s in the time domain.
        self.rot_angular_velocity = 1.0
        # cuda-cloth's loop bound: 1000 steps -> 10 s of sim time.
        self.rot_end_time = 20.0

        # ---- self-collision parameters ----------------------------------
        #
        # cuda-cloth's TwistCase:
        #   m_cloth.m_buffer.m_thickness        = edge_l * 0.2
        #   m_selfCollision.m_params.m_4_thickness = edge_l * 0.2 * 1.2
        #   m_selfCollision.m_params.m_4_k         = 1000
        # (m_5_k / m_5_thickness exist but `ResolveNarrowPhase` is
        # commented out in TwistCase::Run so they never fire.)
        thickness = 0.2 * 1.2 * edge_l
        self._self_collision_thickness = thickness

        # ---- solver -----------------------------------------------------

        # cuda-cloth scatters `0.1 * area` to each of a triangle's 3
        # vertices (NOT divided by 3 — see `KernelComputeAllDm` in
        # cuda-cloth/src/Physics/Cloth/BW98.cuh).  ChysX's
        # `redistribute_mass_area_weighted(surface_density)` scatters
        # `(surface_density / 3) * area` per vertex, so we use 0.3 to
        # reproduce the same lumped vertex masses.
        surface_density = 0.1

        # Narrow-phase contact buffer (actual VF / EE pairs after
        # geometric filtering).  diagnose_twist shows ~10k contacts at
        # the final wrung pose for this 20000-vert scene; factor=64
        # gives 1.28M cap which is ample headroom.  Bump up further
        # only if `solver._sim.self_collision_count()` actually
        # saturates this cap.
        self_collision_max_contacts_factor = 64
        # Broad-phase EF candidate buffer (LBVH `(edge_id, face_id)`
        # output before any geometric test).  Wrung-up dense cloth
        # produces O(10x) more candidates than surviving contacts, so
        # we feed broad-phase a separately-tuned (and larger) factor.
        # Truncating *this* buffer is the usual "cloth pierces itself
        # even though narrow-phase has plenty of room" symptom because
        # narrow-phase only ever runs on whatever pairs broad-phase
        # managed to write.
        self_collision_max_ef_candidates_factor = 256

        self.solver = newton.solvers.SolverChysX(
            self.model,
            # cuda-cloth's `m_gravity = -9.8` is applied as
            # `force.y += m * gravity` in the dynamic kernel, i.e.
            # gravity points in world -Y.  Keep that even though this
            # scene barely uses gravity (the cloth is fully constrained
            # at both ends).
            gravity=(0.0, -9.8, 0.0),
            damping=0.0,
            fem_stretch_stiffness=5.0e2,   # m_k = 500
            fem_shear_stiffness=5.0e2,     # m_k = 500 (cuda-cloth shares one constant)
            bending_stiffness=5.0e-4,      # m_bending_k = 5e-4
            pin_indices=self._pin_indices.tolist(),
            pin_stiffness=1.0e9,           # m_control_mag = 1e9
            pcg_iterations=50,            # PCGSolver m_maxIter = 100
            surface_density=surface_density,
            self_collision_enabled=True,
            self_collision_thickness=thickness,
            self_collision_stiffness=1.0e3,  # m_4_k = 1000
            self_collision_max_contacts_factor=self_collision_max_contacts_factor,
            self_collision_max_ef_candidates_factor=self_collision_max_ef_candidates_factor,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Pre-allocated scratch the per-frame pin update writes into.
        self._pin_targets = np.empty((self._pin_indices.size, 3), dtype=np.float32)

        # Snapshot the initial particle positions for `test_final`.
        self._initial_q = verts_np.copy()

        self.viewer.set_model(self.model)
        # Camera sits on +Z looking back along -Z, i.e. straight down
        # the longitudinal twist axis.  The wrung cross-section is
        # most legible from this angle.
        self.viewer.set_camera(
            pos=wp.vec3(0.0, 0.0, 1.5),
            pitch=0.0,
            yaw=0.0,
        )
        if hasattr(self.viewer, "_paused"):
            self.viewer._paused = True

        # ---- CUDA Graph capture (mirror of cloth_twist's pattern) -------
        #
        # Everything inside `_simulate_substeps()` (the per-frame
        # `clear_forces -> solver.step` loop) is recorded once into a
        # CUDA graph and replayed every frame, eliminating the
        # per-kernel-launch host overhead.  A 1000-frame run on a
        # 5090 drops from ~2.2 s (already QuantBvh-fast) to well under
        # a second of GPU wall time, with the host loop reduced to a
        # single `cudaGraphLaunch` per frame.
        #
        # Two design points keep the capture simple:
        #
        # 1. **No state_0 <-> state_1 swap inside the captured body.**
        #    SolverChysX detects `state_in is state_out` and skips the
        #    inertial copy, stepping in place on `self.state_0`.
        #    Skipping the swap means the captured graph keeps using
        #    the same buffer pointers across replays, so a single
        #    capture replays correctly for any number of frames
        #    (no even-substeps requirement like cloth_twist's VBD
        #    setup needs).
        #
        # 2. **Pin targets are updated *outside* the graph.**
        #    `_update_pin_targets()` does host-side numpy math + an
        #    H2D memcpy of the new target buffer (NOT capturable by
        #    wp.ScopedCapture).  Since the pin index set itself
        #    never changes, only the targets do, and chysx's
        #    `update_pin_targets` is exactly the cheap "rewrite the
        #    targets buffer" path we want.
        self._cuda_graph = None
        self._capture_graph()

    def _simulate_substeps(self) -> None:
        """One frame of physics, in-place on `self.state_0`.

        Captured into `self._cuda_graph` once at construction and
        replayed by `step()` every frame.  Pin-target writes are NOT
        in here -- they're host-side and run from `step()` before the
        capture launch.
        """
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0, self.state_0, self.control, self.contacts, self.sim_dt
            )

    def _capture_graph(self) -> None:
        """Record `_simulate_substeps()` into a CUDA graph if possible.

        Falls back to no-graph (eager replay in `step()`) on CPU
        devices or when graph capture is unavailable.  We do one
        warm-up call before capture so any lazy buffer allocation,
        kernel JIT, or topology rebuild lands *outside* the captured
        region (otherwise the first replay would re-run those one-time
        side effects -- harmless but slow).
        """
        self._cuda_graph = None
        device = wp.get_device()
        if not device.is_cuda:
            return
        # Warm-up to flush any lazy initialisation BEFORE capture.
        self._simulate_substeps()
        wp.synchronize_device(device)

        with wp.ScopedCapture() as capture:
            self._simulate_substeps()
        self._cuda_graph = capture.graph

    # ---- pin animation ------------------------------------------------

    def _update_pin_targets(self) -> None:
        """1:1 transcription of cuda-cloth `TwistCase::UpdateBoundary`.

        ::

            for i in 0 .. m_clothSize_y - 1:
                left  = m_left_center
                      + (sin(-w * t) * (i - mid),
                         cos(-w * t) * (i - mid),
                         0) * edge_l
                right = m_right_center
                      + (sin(+w * t) * (i - mid),
                         cos(+w * t) * (i - mid),
                         0) * edge_l
                pinList[i]            = right     # right strip (z=-0.5)
                pinList[m_offset + i] = left      # left  strip (z=+0.5)

        The rotation axis is world +Z (the line connecting the two
        strip centres).  At t=0 the offset is purely along +Y, so each
        pin sits exactly on its rest position.  After `rot_end_time`
        seconds we freeze the targets to give the cloth time to
        relax under self-contact without further forced wringing.
        """
        t = min(self.sim_time, self.rot_end_time)
        theta = self.rot_angular_velocity * t  # absolute twist angle [rad]
        c_pos = np.cos(theta)
        s_pos = np.sin(theta)
        # cuda-cloth: left uses (sin(-w*t), cos(-w*t)) = (-sin, cos)
        c_neg = c_pos
        s_neg = -s_pos

        i = np.arange(self._cloth_size_y, dtype=np.float32)
        radial = (i - self._mid) * self._edge_l            # (cloth_size_y,)

        # Right strip targets: rotates by +theta in XY plane around +Z.
        rx = s_pos * radial
        ry = c_pos * radial
        self._pin_targets[: self._n_right, 0] = self._right_center[0] + rx
        self._pin_targets[: self._n_right, 1] = self._right_center[1] + ry
        self._pin_targets[: self._n_right, 2] = self._right_center[2]

        # Left strip targets: rotates by -theta in XY plane around +Z.
        lx = s_neg * radial
        ly = c_neg * radial
        self._pin_targets[self._n_right :, 0] = self._left_center[0] + lx
        self._pin_targets[self._n_right :, 1] = self._left_center[1] + ly
        self._pin_targets[self._n_right :, 2] = self._left_center[2]

        self.solver.update_pin_targets(self._pin_targets)

    # ---- simulation loop ----------------------------------------------

    def step(self):
        # Pin targets are time-dependent and computed on the host;
        # update them *before* launching the captured graph (the H2D
        # copy is not part of the recorded sequence).  Once chysx's
        # `update_pin_targets` returns, the targets buffer the captured
        # kernels read from has been overwritten with this frame's
        # values, so the replay sees the new targets even though the
        # graph itself is unchanged.
        self._update_pin_targets()

        if self._cuda_graph is not None:
            wp.capture_launch(self._cuda_graph)
            # self._simulate_substeps()
        else:
            self._simulate_substeps()

        self.sim_time += self.sim_substeps * self.sim_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        qd = self.state_0.particle_qd.numpy().reshape(-1, 3)

        if not (np.isfinite(q).all() and np.isfinite(qd).all()):
            raise ValueError("non-finite values in particle state")

        # Cloth fits inside a ~2 m bbox even when fully wrung.
        bound = 2.0
        if (np.abs(q) > bound).any():
            raise ValueError(
                f"cloth particles escaped the {bound:.1f} m bounding box; "
                f"max |q| = {float(np.abs(q).max()):.3f}"
            )

        max_speed = float(np.linalg.norm(qd, axis=1).max())
        if max_speed > 10.0:
            raise ValueError(
                f"particle speed exploded: max |v| = {max_speed:.3f}"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    # cuda-cloth's TwistCase loops 1000 times.
    parser.set_defaults(num_frames=1000)

    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
