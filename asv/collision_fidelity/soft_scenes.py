# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small VBD soft-contact scenes with original procedural geometry.

These are smoke scenes, not numerical qualification runs. All deformable
interiors evolve through VBD. Prescribed boundary vertices and rigid fixtures
are identified in each scene description and its metrics.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton

from .public_vbd import SceneInputs, create_solver, set_command_time

SCENE_IDS = [
    "s1_cloth_2",
    "s1_cloth_5",
    "s1_cloth_10",
    "s1_pads_2",
    "s1_pads_5",
    "s2_lower",
    "s2_impact",
    "s3_cloth",
    "s3_pads",
    "s4_pads",
    "s5_pack_4",
    "s5_pack_8",
]

_COLORS = [(0.18, 0.60, 0.83), (0.95, 0.53, 0.22), (0.35, 0.72, 0.48), (0.72, 0.40, 0.77), (0.91, 0.75, 0.25)]


@wp.func
def _ramp(t: float, start: float, end: float):
    u = wp.clamp((t - start) / (end - start), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


@wp.func
def _motion(t: float, mode: int):
    p = wp.vec3(0.0)
    if mode == 1:  # Lower a pinned top face; release after a steady interval.
        p = wp.vec3(0.0, 0.0, -0.40 * _ramp(t, 0.3, 2.3))
    elif mode == 2:  # Compress, hold, shear, hold, unload, and separate pads.
        z = -0.045 * _ramp(t, 0.5, 1.8) + 0.045 * _ramp(t, 5.3, 6.3) + 0.12 * _ramp(t, 6.3, 7.5)
        p = wp.vec3(0.16 * _ramp(t, 3.0, 4.4), 0.0, z)
    elif mode == 3:  # Rigid platen compression of packed solids.
        z = -0.15 * _ramp(t, 0.7, 2.5) + 0.15 * _ramp(t, 4.5, 6.5)
        p = wp.vec3(0.0, 0.0, z)
    return p


@wp.kernel
def _drive_boundary(
    clock: wp.array[float],
    indices: wp.array[int],
    rest: wp.array[wp.vec3],
    mode: int,
    dt: float,
    flags: wp.array[int],
    q0: wp.array[wp.vec3],
    q1: wp.array[wp.vec3],
    v0: wp.array[wp.vec3],
    v1: wp.array[wp.vec3],
):
    i = wp.tid()
    vi = indices[i]
    t = clock[0]
    if mode == 1 and t >= 2.8:
        flags[vi] = flags[vi] | 1
    else:
        flags[vi] = flags[vi] & ~1
        d = _motion(t, mode)
        velocity = (_motion(t + dt, mode) - d) / dt
        q0[vi] = rest[i] + d
        q1[vi] = rest[i] + d
        v0[vi] = velocity
        v1[vi] = velocity


@wp.kernel
def _drive_plate(
    clock: wp.array[float],
    body: int,
    origin: wp.vec3,
    mode: int,
    dt: float,
    q0: wp.array[wp.transform],
    q1: wp.array[wp.transform],
    v0: wp.array[wp.spatial_vector],
    v1: wp.array[wp.spatial_vector],
):
    t = clock[0]
    d = _motion(t, mode)
    vel = (_motion(t + dt, mode) - d) / dt
    q = wp.transform(origin + d, wp.quat_identity())
    q0[body] = q
    q1[body] = q
    v0[body] = wp.spatial_vector(vel, wp.vec3(0.0))
    v1[body] = wp.spatial_vector(vel, wp.vec3(0.0))


@wp.kernel
def _advance_clock(clock: wp.array[float], dt: float):
    clock[0] += dt


@wp.kernel
def _mesh_lines(edges: wp.array[wp.vec2i], q: wp.array[wp.vec3], starts: wp.array[wp.vec3], ends: wp.array[wp.vec3]):
    i = wp.tid()
    starts[i] = q[edges[i][0]] + wp.vec3(0.0, 0.0, 0.0002)
    ends[i] = q[edges[i][1]] + wp.vec3(0.0, 0.0, 0.0002)


def _cloth(builder, pos, *, size=0.9, resolution=14, color=0, fixed=False):
    start = builder.particle_count
    builder.add_cloth_grid(
        pos=wp.vec3(*pos),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=resolution,
        dim_y=resolution,
        cell_x=size / resolution,
        cell_y=size / resolution,
        mass=0.002,
        tri_ke=2000.0,
        tri_ka=2000.0,
        tri_kd=0.05,
        edge_ke=0.02,
        edge_kd=0.00005,
        particle_radius=0.004,
        fix_left=fixed,
        fix_right=fixed,
        fix_top=fixed,
        fix_bottom=fixed,
        color=_COLORS[color % len(_COLORS)],
    )
    return (start, builder.particle_count)


def _solid(builder, pos, *, size=(0.6, 0.45, 0.12), dims=(6, 5, 2), color=0, density=180.0):
    start = builder.particle_count
    builder.add_soft_grid(
        pos=wp.vec3(*pos),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=dims[0],
        dim_y=dims[1],
        dim_z=dims[2],
        cell_x=size[0] / dims[0],
        cell_y=size[1] / dims[1],
        cell_z=size[2] / dims[2],
        density=density,
        k_mu=8000.0,
        k_lambda=8000.0,
        k_damp=0.5,
        particle_radius=0.004,
        color=_COLORS[color % len(_COLORS)],
    )
    return (start, builder.particle_count)


def _plate(builder, origin, half, *, driven=False, label="fixed rigid fixture", opacity=1.0):
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=30000.0, kd=0.1, mu=0.5)
    body = -1
    if driven:
        body = builder.add_body(
            xform=wp.transform(wp.vec3(*origin), wp.quat_identity()), is_kinematic=True, label=label
        )
        origin = (0.0, 0.0, 0.0)
    builder.add_shape_box(
        body,
        xform=wp.transform(wp.vec3(*origin), wp.quat_identity()),
        hx=half[0],
        hy=half[1],
        hz=half[2],
        cfg=cfg,
        color=(0.57, 0.61, 0.65),
        label=label,
        opacity=opacity,
    )
    return body


class SoftScene(SceneInputs):
    """A reproducible scene with one explicit physics step per call."""

    def __init__(
        self,
        scene_id: str,
        *,
        device=None,
        friction=0.5,
        substeps=None,
        iterations=None,
        **_kwargs,
    ):
        if scene_id not in SCENE_IDS:
            raise ValueError(f"Unknown soft scene {scene_id!r}")
        self.scene_id = scene_id
        self.time = 0.0
        self.frame_dt = 1.0 / 30.0
        self.substeps = substeps if substeps is not None else (10 if scene_id.startswith(("s3_", "s1_")) else 8)
        if (
            isinstance(self.substeps, bool)
            or not isinstance(self.substeps, (int, np.integer))
            or self.substeps < 2
            or self.substeps % 2
        ):
            raise ValueError("substeps must be an even integer >= 2 for the declared scene protocol")
        self.iterations = iterations if iterations is not None else (20 if scene_id.startswith("s1_") else 10)
        self.dt = self.frame_dt / self.substeps
        self.duration = 7.0
        self.title = scene_id
        self.description = ""
        self.asset_ids = []
        self.objects = []
        self.boundary_notes = []
        self.camera_pos = (1.6, -2.1, 1.35)
        self.camera_target = (0.0, 0.0, 0.30)
        self.mode = 0
        self._late_motion_samples = []
        self.driven_indices = []
        self.plate_body = -1
        self.plate_origin = (0.0, 0.0, 0.0)
        self.phase_windows = [(2.0, "settle"), (self.duration, "steady observation")]
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))

        if scene_id.startswith("s1_cloth"):
            count = int(scene_id.rsplit("_", 1)[1])
            self.title = f"S1 · {count} cloth layers"
            self.description = "Free cloth layers settle under gravity on a rigid plane. Successive layers have an 8 mm horizontal offset and 16 mm initial surface gap. No pinned cloth vertices."
            self.asset_ids = ["F0", "fixture_plane"]
            for i in range(count):
                self.objects.append(
                    _cloth(builder, (-0.45 + 0.008 * i, -0.45, 0.025 + 0.016 * i), color=i, resolution=12)
                )
            builder.add_ground_plane()
            self.camera_pos = (1.4, -1.8, 0.65)
            self.camera_target = (0.0, 0.0, 0.08)
        elif scene_id.startswith("s1_pads"):
            count = int(scene_id.rsplit("_", 1)[1])
            self.title = f"S1 · {count} soft pads"
            self.description = (
                "Free tetrahedral pads settle under gravity on a rigid plane. Initial inter-pad surface gap: 16 mm."
            )
            self.asset_ids = ["V1", "fixture_plane"]
            for i in range(count):
                self.objects.append(_solid(builder, (-0.3, -0.225, 0.01 + 0.136 * i), color=i))
            builder.add_ground_plane()
            if count == 5:
                self.camera_pos = (2.2, -2.8, 1.8)
                self.camera_target = (0.0, -0.1, 0.18)
        elif scene_id.startswith("s2_"):
            impact = scene_id == "s2_impact"
            self.title = "S2 · block impact on cloth" if impact else "S2 · lower, rest, and release a block on cloth"
            self.asset_ids = ["F0", "V0", "fixture_support_frame"]
            self.objects.append(_cloth(builder, (-0.5, -0.5, 0.32), size=1.0, resolution=20, fixed=True))
            z = 0.82 if impact else 0.72
            self.objects.append(
                _solid(builder, (-0.14, -0.14, z), size=(0.28, 0.28, 0.22), dims=(5, 5, 4), color=1, density=120.0)
            )
            self.boundary_notes.append("All four cloth boundary edges are fixed supports at z=0.32 m.")
            for x in (-0.52, 0.52):
                _plate(builder, (x, 0.0, 0.29), (0.018, 0.54, 0.025))
            for y in (-0.52, 0.52):
                _plate(builder, (0.0, y, 0.29), (0.54, 0.018, 0.025))
            builder.add_ground_plane()
            if impact:
                self.description = "Free soft block drops 0.50 m above the initial cloth plane. Cloth perimeter fixed; block fully dynamic."
                self.phase_windows = [(0.32, "free fall"), (1.2, "impact response"), (self.duration, "rest")]
            else:
                self.mode = 1
                start, end = self.objects[-1]
                q = np.asarray(builder.particle_q)
                self.driven_indices = [i for i in range(start, end) if abs(q[i, 2] - (z + 0.22)) < 1e-5]
                self.boundary_notes.append(
                    "Block top face is prescribed during lowering and rest, then becomes dynamic at t=2.8 s."
                )
                self.description = "Prescribed block top face lowers 0.40 m over 2 s. It rests briefly, releases at 2.8 s, and settles on supported cloth."
                self.phase_windows = [
                    (0.3, "initial hold"),
                    (2.3, "lower"),
                    (2.8, "supported rest"),
                    (self.duration, "release and settle"),
                ]
        elif scene_id.startswith("s3_"):
            cloth = scene_id == "s3_cloth"
            self.title = (
                "S3 · free-fall cloth proximity and control" if cloth else "S3 · free-fall pad proximity and control"
            )
            self.duration = 0.6
            self.frame_dt = 1.0 / 60.0
            self.dt = self.frame_dt / self.substeps
            self.phase_windows = [(self.duration, "gravity-only free fall")]
            self.asset_ids = ["F0" if cloth else "V1"]
            self.description = "Two pairs fall with identical initial velocity and only gravity. Left gap 10 mm; right control gap 80 mm. No plane, support, or preload."
            for pair, x in enumerate((-0.70, 0.15)):
                gap = 0.01 if pair == 0 else 0.08
                for layer in range(2):
                    if cloth:
                        self.objects.append(
                            _cloth(
                                builder, (x, -0.25, 2.2 + layer * gap), size=0.55, resolution=10, color=pair * 2 + layer
                            )
                        )
                    else:
                        self.objects.append(
                            _solid(
                                builder,
                                (x, -0.225, 2.2 + layer * (0.12 + gap)),
                                size=(0.55, 0.45, 0.12),
                                color=pair * 2 + layer,
                            )
                        )
            self.camera_pos = (2.8, -4.0, 2.8)
            self.camera_target = (0.0, 0.0, 1.3)
        elif scene_id == "s4_pads":
            self.duration = 9.0
            self.mode = 2
            self.title = "S4 · compress, hold, slide, unload, separate"
            self.description = "Two soft pads. Bottom face of lower pad fixed. Top face of upper pad prescribed: 45 mm compression, 160 mm slide, then 120 mm separation."
            self.asset_ids = ["V1", "fixture_boundary_clamps"]
            for i in range(2):
                self.objects.append(_solid(builder, (-0.3, -0.225, 0.01 + i * 0.136), color=i))
            q = np.asarray(builder.particle_q)
            for i in range(*self.objects[0]):
                if abs(q[i, 2] - 0.01) < 1e-5:
                    builder.particle_flags[i] = int(builder.particle_flags[i]) & ~int(newton.ParticleFlags.ACTIVE)
            self.driven_indices = [i for i in range(*self.objects[1]) if abs(q[i, 2] - 0.266) < 1e-5]
            self.boundary_notes = [
                "Lower pad bottom face fixed; upper pad top face follows the declared displacement. All other vertices are dynamic."
            ]
            builder.add_ground_plane()
            self.phase_windows = [
                (0.5, "settle"),
                (1.8, "compress"),
                (3.0, "hold"),
                (4.4, "slide"),
                (5.3, "hold after slide"),
                (6.3, "unload"),
                (7.5, "separate"),
                (9.0, "separated hold"),
            ]
            self.camera_target = (0.05, 0.0, 0.22)
        elif scene_id.startswith("s5_pack_"):
            count = int(scene_id.rsplit("_", 1)[1])
            self.mode = 3
            self.duration = 8.0
            self.title = f"S5 · {count} packed soft blocks under a driven plate"
            self.description = "Free soft blocks in a rigid tray. Initial lateral gaps 20 mm, vertical gaps 16 mm. Kinematic plate compresses 150 mm, holds, and unloads."
            self.asset_ids = ["V0", "fixture_tray", "fixture_compression_plate"]
            layers = count // 4
            for layer in range(layers):
                for iy in range(2):
                    for ix in range(2):
                        self.objects.append(
                            _solid(
                                builder,
                                (-0.25 + ix * 0.26, -0.25 + iy * 0.26, 0.01 + layer * 0.296),
                                size=(0.24, 0.24, 0.28),
                                dims=(4, 4, 5),
                                color=len(self.objects),
                                density=120.0,
                            )
                        )
            builder.add_ground_plane()
            wall_height = 0.3 * layers
            for x in (-0.29, 0.29):
                _plate(builder, (x, 0.0, wall_height / 2), (0.025, 0.315, wall_height / 2), opacity=0.17)
            for y in (-0.29, 0.29):
                _plate(builder, (0.0, y, wall_height / 2), (0.315, 0.025, wall_height / 2), opacity=0.17)
            self.plate_origin = (0.0, 0.0, 0.28 * layers + 0.016 * (layers - 1) + 0.09)
            self.plate_body = _plate(
                builder,
                self.plate_origin,
                (0.255, 0.255, 0.025),
                driven=True,
                label="prescribed compression plate",
                opacity=0.25,
            )
            self.boundary_notes = ["Rigid top plate follows a prescribed position; all soft vertices remain dynamic."]
            self.phase_windows = [(0.7, "settle"), (2.5, "compress"), (4.5, "hold"), (6.5, "unload"), (8.0, "recovery")]
            self.camera_target = (0.0, 0.0, 0.25 * layers)

        for i in self.driven_indices:
            builder.particle_flags[i] = int(builder.particle_flags[i]) & ~int(newton.ParticleFlags.ACTIVE)
        self.initial_positions = np.asarray(builder.particle_q, dtype=np.float32).copy()
        builder.color(include_bending=True)
        self.model = builder.finalize(device=device)
        triangles = self.model.tri_indices.numpy()
        surface_edges = np.unique(
            np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1), axis=0
        )
        self._surface_edges = wp.array(surface_edges, dtype=wp.vec2i, device=self.model.device)
        self._line_starts = wp.empty(len(surface_edges), dtype=wp.vec3, device=self.model.device)
        self._line_ends = wp.empty(len(surface_edges), dtype=wp.vec3, device=self.model.device)
        self._tets = self.model.tet_indices.numpy() if self.model.tet_count else np.empty((0, 4), dtype=np.int32)
        if self._tets.size:
            rest_tets = self.initial_positions[self._tets]
            self._rest_tet_determinants = np.linalg.det(rest_tets[:, 1:] - rest_tets[:, :1])
        self.model.soft_contact_ke = 30000.0
        self.model.soft_contact_kd = 0.1
        self.model.soft_contact_mu = friction
        self.solver, self.pipeline = create_solver(
            self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.008,
            particle_self_contact_gap=0.008,
            particle_vertex_contact_buffer_size=64,
            particle_edge_contact_buffer_size=128,
            rigid_body_particle_contact_buffer_size=4096,
            rigid_body_contact_buffer_size=256,
            soft_contact_gap=0.008,
            full_surface_contact=False,
        )
        self.contacts = self.solver.contacts
        self._state0 = self.model.state()
        self._state1 = self.model.state()
        self.control = self.model.control()
        self._clock = wp.zeros(1, dtype=float, device=self.model.device)
        self._indices = wp.array(self.driven_indices, dtype=int, device=self.model.device)
        self._rest = wp.array(self.initial_positions[self.driven_indices], dtype=wp.vec3, device=self.model.device)

    @property
    def state(self):
        """Return the current state after double-buffer swaps."""
        return self._state0

    @property
    def phase(self):
        for end, phase in self.phase_windows:
            if self.time < end - 1e-6:
                return phase
        return self.phase_windows[-1][1]

    def step_substep(self, command_time_s):
        """Advance one physical substep with prescribed endpoint commands."""
        set_command_time(self._clock, command_time_s + self.dt)
        if self.driven_indices:
            wp.launch(
                _drive_boundary,
                dim=len(self.driven_indices),
                inputs=[
                    self._clock,
                    self._indices,
                    self._rest,
                    self.mode,
                    self.dt,
                    self.model.particle_flags,
                    self._state0.particle_q,
                    self._state1.particle_q,
                    self._state0.particle_qd,
                    self._state1.particle_qd,
                ],
                device=self.model.device,
            )
        if self.plate_body >= 0:
            wp.launch(
                _drive_plate,
                dim=1,
                inputs=[
                    self._clock,
                    self.plate_body,
                    wp.vec3(*self.plate_origin),
                    self.mode,
                    self.dt,
                    self._state0.body_q,
                    self._state1.body_q,
                    self._state0.body_qd,
                    self._state1.body_qd,
                ],
                device=self.model.device,
            )
        self._state0.clear_forces()
        self.solver.step(self._state0, self._state1, self.control, None, self.dt)
        self._state0, self._state1 = self._state1, self._state0
        self.time = float(command_time_s + self.dt)
        return self.state


def build_scene(scene_id: str, **kwargs):
    """Construct one soft-contact smoke scene on the requested Warp device."""
    return SoftScene(scene_id, **kwargs)
