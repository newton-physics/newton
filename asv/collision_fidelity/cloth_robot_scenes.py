# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Preserved R1 cloth operations with original robot hands, without cloth pins."""

from __future__ import annotations

import hashlib

import numpy as np
import warp as wp

import newton

from .allegro_gripper import OPEN_FINGERS, AllegroHandDriver, add_allegro_hand
from .assets import CLOTH_NAMES, cloth_mesh
from .jacket_diagnostics import ClothStrain
from .public_vbd import SceneInputs, create_solver, set_command_time
from .robot_gripper import FrankaHandDriver, add_franka_hand

__all__ = ["ClothRobotScene", "build_scene"]

_ALLEGRO_INITIAL = (
    0.24080684739156585,
    0.7486478334019966,
    0.9856783590117453,
    0.9037601401083666,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0.9852496912189342,
    0.6315409427690029,
    0.990712751531764,
    0.5479841803283853,
)
_ALLEGRO_CLOSED = (
    0.24032326824430275,
    0.7532263416499715,
    0.9876414192090274,
    0.9035966293957304,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0.9928445545936402,
    0.6221628298063311,
    0.9926207634863553,
    0.5348133650316339,
)


def _refine(vertices, triangles, grips):
    """Conforming edge subdivision; every new point lies on the original surface."""
    v, t = np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int32)
    for _ in range(8):
        edges = np.unique(np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1), axis=0)
        centers = v[edges].mean(axis=1)
        near = np.min(np.linalg.norm(centers[:, None] - np.asarray(grips)[None], axis=2), axis=1) < 0.055
        limit = np.where(near, 0.006, 0.02)
        selected = edges[np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1) > limit]
        if not len(selected):
            break
        mids = {tuple(edge): len(v) + i for i, edge in enumerate(selected)}
        v = np.concatenate([v, v[selected].mean(axis=1)])
        out = []
        for row in t:
            a, b, c = row
            tri = [int(a), int(b), int(c)]
            mid = [mids.get(tuple(sorted(edge))) for edge in [(a, b), (b, c), (c, a)]]
            count = sum(m is not None for m in mid)
            if count == 0:
                out.append(tri)
            elif count == 3:
                ab, bc, ca = mid
                out.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
            elif count == 1:
                k = next(i for i, m in enumerate(mid) if m is not None)
                a, b, c = tri[k:] + tri[:k]
                m = mid[k]
                out.extend([(a, m, c), (m, b, c)])
            else:
                k = next(i for i in range(3) if mid[i] is not None and mid[(i + 1) % 3] is not None)
                a, b, c = tri[k:] + tri[:k]
                ab, bc = mid[k], mid[(k + 1) % 3]
                out.extend([(b, bc, ab), (a, ab, bc), (a, bc, c)])
        t = np.asarray(out, dtype=np.int32)
    return v.astype(np.float32), t


def _surface_audit(v, t):
    edges = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    area = np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1) / 2
    return {
        "vertices": len(v),
        "triangles": len(t),
        "bounds_m": [v.min(0).tolist(), v.max(0).tolist()],
        "area_m2": float(area.sum()),
        "minimum_area_m2": float(area.min()),
        "boundary_edges": int(sum(counts == 1)),
        "nonmanifold_edges": int(sum(counts > 2)),
        "euler_number": int(len(v) - len(unique) + len(t)),
        "maximum_edge_m": float(np.linalg.norm(v[unique[:, 0]] - v[unique[:, 1]], axis=1).max()),
    }


def _box_separation(center, axes, half, other_center, other_half):
    """Maximum separating-axis gap for an OBB and world-axis-aligned box."""
    directions = [*np.eye(3), *axes.T]
    directions.extend(np.cross(a, b) for a in axes.T for b in np.eye(3))
    gap = -np.inf
    for direction in directions:
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            continue
        axis = direction / norm
        radius = abs(axes.T @ axis) @ half + abs(axis) @ other_half
        gap = max(gap, float(abs((center - other_center) @ axis) - radius))
    return gap


class ClothRobotScene(SceneInputs):
    """Original-size cloth with prescribed original-hand FK and dynamic cloth."""

    def __init__(self, scene_id, *, device=None, **_kwargs):
        self.scene_id = scene_id
        self.allegro = scene_id == "r5_allegro_towel"
        name = "towel" if self.allegro else scene_id.removeprefix("r5_franka_").removesuffix("_contact_v2")
        if name not in CLOTH_NAMES:
            raise ValueError(scene_id)
        self.name = name
        self.contact_revision = (
            2 if scene_id.endswith("_contact_v2") or name in ("trousers", "dress", "gusset_bag") else 1
        )
        self.corrected = self.contact_revision == 2 and not self.allegro
        self.upper_layer = self.corrected and name == "skirt"
        self.closed_opening = (
            (0.00804 if name in ("tshirt", "trousers", "dress", "pouch") else 0.00604) if self.corrected else 0.004
        )
        self.title = f"R5 | {'Allegro' if self.allegro else 'Franka'}: original-size {name}"
        self.description = (
            "Preserves the original R1 cloth shape, scale, table and close/lift/carry/hold/place/release operation. "
            "The placement hand height increases by 20-25 mm to clear the finite table with the original hand. "
            "Original robot hand geometry replaces the cube jaws. Robot roots and joints are prescribed; "
            "all cloth vertices remain dynamic, with no pins or attachments. "
            + (
                "Allegro starts in declared shallow prepared contact at the original towel edge; no approach acquisition claim."
                if self.allegro
                else "Open Franka fingers close around the accessible original cloth edge or bag handle."
            )
        )
        self.asset_ids = [name, "G4_Allegro_hand" if self.allegro else "G1_franka_original_hand"]
        self.frame_dt, self.substeps, self.iterations = 1 / 30, 10, 30
        self.dt, self.time, self.duration = self.frame_dt / self.substeps, 0.0, 11.5
        self.radius, self.table_top = 0.003, 0.25
        self.placement_adjustment = 0.025 if self.allegro else 0.02
        self.table_center, self.table_half = np.array([0.0, 0.39, 0.12]), np.array([0.65, 0.31, 0.13])
        self.camera_pos, self.camera_target, self.camera_fov = (1.25, -1.65, 1.35), (0.10, 0.28, 0.51), 48.0
        mesh = cloth_mesh(name, resolution="coarse")
        self.mesh = mesh
        lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
        self.scale = 0.6 / max(hi - lo)
        q = mesh.vertices[:, [0, 2, 1]].astype(float) * self.scale
        q[:, 0] -= (lo[0] + hi[0]) * 0.5 * self.scale
        q[:, 1] -= lo[2] * self.scale
        q[:, 2] = 0.258 + (hi[1] - mesh.vertices[:, 1]) * self.scale
        self.source_audit = _surface_audit(q, mesh.triangles)
        rear = name in ("tote", "gusset_bag")
        if rear:
            candidates = q[(q[:, 1] > q[:, 1].max() - 0.02) & (q[:, 2] > q[:, 2].max() - 0.01)]
            anchor = candidates[np.argmin(abs(candidates[:, 0]))].copy()
            anchor[1] -= 0.004
            grips = [anchor]
            self.operation_change = "Same pick/carry/place/release; grip moves from closed bag bottom to its upper original handle, within the real80mm aperture."
        else:
            front = q[q[:, 1] < q[:, 1].min() + 0.02]
            xs = [float(np.median(front[:, 0]))] if self.allegro else np.quantile(front[:, 0], [0.25, 0.75])
            grips = []
            for x in xs:
                near = front[np.abs(front[:, 0] - x) < 0.02]
                if not len(near):
                    near = front[np.argsort(abs(front[:, 0] - x))[:2]]
                grips.append(
                    np.array([x, float(q[:, 1].min() + 0.012), float((near[:, 2].min() + near[:, 2].max()) / 2)])
                )
                if self.upper_layer:
                    grips[-1][2] = float(near[:, 2].max())
            self.operation_change = (
                "Same original towel and motion; prepared original Allegro thumb/index edge contact replaces cube jaws."
                if self.allegro
                else "Same original edge and motion; two original Franka hands replace the broad cube-jaw pair."
            )
        if self.corrected:
            self.operation_change += " Contact revision2 keeps material and geometry; layer-aware physical finger gap replaces4mm compression. Open hands retract before rising to avoid hooking the released cloth."
            if self.upper_layer:
                self.operation_change += (
                    " Grip the upper original hem layer because both skirt layers plus shells exceed80mm aperture."
                )
            if rear:
                self.operation_change += (
                    " Carry the rear handle to the original table centerY=.39m, so release stays over the table."
                )
        vertices, triangles = _refine(q, mesh.triangles, grips)
        self.refined_audit = _surface_audit(vertices, triangles)
        if self.refined_audit["nonmanifold_edges"] or self.refined_audit["minimum_area_m2"] <= 0:
            raise ValueError("Refinement produced invalid cloth topology")
        if self.refined_audit["euler_number"] != self.source_audit["euler_number"]:
            raise ValueError("Refinement changed source topology")
        if not np.allclose(self.refined_audit["bounds_m"], self.source_audit["bounds_m"], atol=1e-6):
            raise ValueError("Refinement changed source bounds")
        if not np.isclose(self.refined_audit["area_m2"], self.source_audit["area_m2"], rtol=1e-5):
            raise ValueError("Refinement changed source area")
        self.geometry_hash = hashlib.sha256(vertices.tobytes() + triangles.tobytes()).hexdigest()
        if self.allegro:
            # Put the prepared spherical-pad contact at a real retained mesh vertex.
            delta = vertices - grips[0]
            grips[0] = vertices[np.argmin(np.linalg.norm(delta, axis=1))].copy()
        self.grips = np.asarray(grips)
        b = newton.ModelBuilder(gravity=(0, 0, -9.81))
        b.default_shape_cfg.mu, b.default_shape_cfg.ke, b.default_shape_cfg.kd = 1.0, 50000.0, 0.1
        b.default_shape_cfg.margin, b.default_shape_cfg.gap = 0.001, 0.005
        b.default_shape_cfg.configure_sdf(force_sdf=True, max_resolution=64)
        b.add_cloth_mesh(
            pos=(0, 0, 0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0, 0, 0),
            vertices=vertices.tolist(),
            indices=triangles.ravel().tolist(),
            density=0.22,
            particle_radius=self.radius,
            tri_ke=1800.0,
            tri_ka=1800.0,
            tri_kd=0.025,
            edge_ke=0.08,
            edge_kd=0.0005,
            color=tuple(mesh.metadata["color_rgb"]),
        )
        b.add_shape_box(
            -1,
            xform=wp.transform(self.table_center, wp.quat_identity()),
            hx=self.table_half[0],
            hy=self.table_half[1],
            hz=self.table_half[2],
            color=(0.48, 0.51, 0.55),
            label="unchanged original start and placement table",
        )
        self.specs = []
        self.rotations, self.openings = [], []
        if self.allegro:
            rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(np.pi / 2))
            root = self.grips[0] - np.asarray(wp.quat_rotate(rotation, wp.vec3(0.085, -0.025, 0.011)))
            self.specs.append(add_allegro_hand(b, root_pos=root, rotation=rotation, finger_positions=_ALLEGRO_INITIAL))
            self.rotations.append(tuple(rotation))
            self.driver_bases = [root]
        else:
            self.driver_bases = self.grips.copy()
            for i, grip in enumerate(self.grips):
                rotation = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), float(np.pi / 2 if rear else -np.pi / 2))
                local = vertices[(abs(vertices[:, 0] - grip[0]) < 0.01) & (abs(vertices[:, 1] - grip[1]) < 0.012)]
                layer = local[abs(local[:, 2] - grip[2]) < 0.038]
                span = float(np.ptp(layer[:, 2])) if len(layer) else 0.0
                opening = min(0.08, max(0.04, span + 0.020))
                self.openings.append(opening)
                self.specs.append(
                    add_franka_hand(b, pinch_pos=grip, rotation=rotation, opening=opening, label=f"original_franka_{i}")
                )
                self.rotations.append(tuple(rotation))
        b.add_ground_plane()
        b.color(include_bending=True)
        self.model = b.finalize(device=device)
        self.model.soft_contact_ke, self.model.soft_contact_kd, self.model.soft_contact_mu = 50000.0, 0.1, 1.0
        self.solver, self.pipeline = create_solver(
            self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.002,
            particle_self_contact_gap=0.001,
            particle_rest_shape_contact_exclusion_radius=0.005,
            particle_topological_contact_filter_threshold=1,
            particle_vertex_contact_buffer_size=_kwargs.get("self_contact_capacity", 64),
            particle_edge_contact_buffer_size=_kwargs.get("self_contact_capacity", 64),
            friction_epsilon=0.001,
            soft_contact_gap=0.005,
            full_surface_contact=True,
        )
        self.contacts = self.solver.contacts
        self.state0, self.state1, self.control = self.model.state(), self.model.state(), self.model.control()
        self.clock = wp.zeros(1, dtype=float, device=self.model.device)
        self.times = [0, 0.25, 1.25, 1.5, 3.5, 4.0, 5.5, 7.0, 8.5, 9.5, 11.0, 11.5]
        shift = np.array(
            [[0, 0, 0]] * 4
            + [[0, 0, 0.65]] * 2
            + [[0.24, 0.28, 0.65]] * 2
            + [[0.24, 0.28, 0.08 + self.placement_adjustment]] * 2
            + [[0.24, 0.28, 0.33]] * 2
        )
        if self.corrected:
            if rear:
                shift[6:, 1] = self.table_center[1] - self.grips[0, 1]
            # The original palm/knuckles need axial extraction before upward withdrawal.
            self.times.insert(10, 10.25)
            extraction = shift[9].copy()
            extraction[1] += 0.18 if rear else -0.18
            shift = np.insert(shift, 10, extraction, axis=0)
            shift[11:, 1] = extraction[1]
        self.drivers = []
        for i, spec in enumerate(self.specs):
            positions = np.asarray(self.driver_bases[i]) + shift
            if self.allegro:
                fingers = [_ALLEGRO_INITIAL] * 2 + [_ALLEGRO_CLOSED] * 7 + [OPEN_FINGERS] * 3
                driver = AllegroHandDriver(
                    self.model,
                    spec,
                    times=self.times,
                    positions=positions,
                    rotations=self.rotations[i],
                    finger_positions=fingers,
                )
            else:
                openings = (
                    [self.openings[i]] * 2 + [self.closed_opening] * 7 + [self.openings[i]] * (len(self.times) - 9)
                )
                driver = FrankaHandDriver(
                    self.model,
                    spec,
                    times=self.times,
                    positions=positions,
                    rotations=self.rotations[i],
                    openings=openings,
                )
            self.drivers.append(driver)
        self.strain = ClothStrain(self.model)
        self.initial = self.strain.sample(self.state.particle_q.numpy())
        self._geometry_records = self._make_geometry_records()
        self._update(0.0)
        self.initial_audits = self._hand_audits(vertices)
        for audit in self.initial_audits:
            key = "vertices_inside_original_hand_solids" if self.allegro else "vertices_inside_robot_solids"
            if audit[key]:
                raise ValueError(f"Initial original hand intersects {name}: {audit[key]} vertices")
            if self.corrected and audit["minimum_shell_to_robot_distance_m"] < 0.001:
                raise ValueError(f"Initial cloth shell enters original hand margin: {audit}")
        self.swept_table_audit = self._audit_table_trajectory()
        if self.swept_table_audit["minimum_conservative_separation_m"] < 0:
            raise ValueError(f"Original hand/table overlap: {self.swept_table_audit}")
        self._update(0.0)
        self.history, self.audits = [], []

    def _make_geometry_records(self):
        records = []
        for spec in self.specs:
            for shape in spec.shape_ids:
                if not self.allegro:
                    v = spec.collision_vertices[shape]
                    lo, hi = v.min(0), v.max(0)
                else:
                    half = self.model.shape_scale.numpy()[shape].copy()
                    if self.model.shape_type.numpy()[shape] == newton.GeoType.SPHERE:
                        half[:] = half[0]
                    lo, hi = -half, half
                records.append((shape, (lo + hi) / 2, (hi - lo) / 2))
        return records

    def _update(self, time):
        self.clock.assign(np.array([time], dtype=np.float32))
        for driver in self.drivers:
            driver.update(self.clock, self.state0, self.state1)

    def _table_separation(self):
        bodies = self.state.body_q.numpy()
        shape_body = self.model.shape_body.numpy()
        transforms = self.model.shape_transform.numpy()
        margins = self.model.shape_margin.numpy()
        worst = (np.inf, None)
        for shape, center, half in self._geometry_records:
            tf = wp.transform(*bodies[shape_body[shape]]) * wp.transform(*transforms[shape])
            axes = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(tf))).reshape(3, 3)
            world = np.asarray(wp.transform_point(tf, wp.vec3(*center)))
            gap = _box_separation(world, axes, half + margins[shape], self.table_center, self.table_half + 0.001)
            if gap < worst[0]:
                worst = (gap, int(shape))
        return worst

    def _audit_table_trajectory(self):
        worst = (np.inf, None, None)
        interhand_gap = np.inf
        for time in np.linspace(0, self.duration, 116):
            self._update(float(time))
            gap, shape = self._table_separation()
            if gap < worst[0]:
                worst = (gap, float(time), shape)
            if len(self.specs) == 2:
                bodies, transforms = self.state.body_q.numpy(), self.model.shape_transform.numpy()
                shape_bodies, margins = self.model.shape_body.numpy(), self.model.shape_margin.numpy()
                bounds = []
                for spec in self.specs:
                    xs = []
                    for sid in spec.shape_ids:
                        tf = wp.transform(*bodies[shape_bodies[sid]]) * wp.transform(*transforms[sid])
                        points = np.array([wp.transform_point(tf, wp.vec3(*p)) for p in spec.collision_vertices[sid]])
                        xs.extend([points[:, 0].min() - margins[sid], points[:, 0].max() + margins[sid]])
                    bounds.append((min(xs), max(xs)))
                interhand_gap = min(interhand_gap, bounds[1][0] - bounds[0][1])
        if interhand_gap < 0:
            raise ValueError(f"Original hand collision envelopes overlap along X: {interhand_gap}")
        return {
            "minimum_conservative_separation_m": worst[0],
            "time_s": worst[1],
            "shape": worst[2],
            "samples": 116,
            "minimum_interhand_x_separation_m": float(interhand_gap) if len(self.specs) == 2 else None,
            "method": "Separating-axis test of every original collision-shape OBB against finite original table; palm mesh bounded by its own local vertices; both shape margins included.",
        }

    def _hand_audits(self, points):
        if self.allegro:
            return [self.drivers[0].geometry_audit(self.state, points=points, radii=self.radius)]
        result = []
        for driver in self.drivers:
            a = driver.geometry_audit(self.state, points=points, radii=self.radius, full_surface_contact=True)
            result.append(
                {
                    k: a[k]
                    for k in [
                        "vertices_inside_robot_solids",
                        "minimum_vertex_to_robot_distance_m",
                        "minimum_shell_to_robot_distance_m",
                        "minimum_robot_collision_z_m",
                        "physical_tip_aperture_m",
                    ]
                }
            )
        return result

    @property
    def state(self):
        return self.state0

    @property
    def phase(self):
        for end, name in [
            (0.25, "prepared contact" if self.allegro else "open supported edge"),
            (1.25, "close"),
            (1.5, "acquire"),
            (3.5, "lift"),
            (4, "raised pause"),
            (5.5, "carry"),
            (7, "hold"),
            (8.5, "place"),
            (9.5, "open"),
            (11, "withdraw"),
            (12, "released"),
        ]:
            if self.time < end - 1e-6:
                return name
        return "done"

    def step_substep(self, command_time_s):
        """Advance one physical substep with prescribed endpoint commands."""
        set_command_time(self.clock, command_time_s + self.dt)
        for driver in self.drivers:
            driver.update(self.clock, self.state0)
        self.state0.clear_forces()
        self.state1.clear_forces()
        self.solver.step(self.state0, self.state1, self.control, None, self.dt)
        self.state0, self.state1 = self.state1, self.state0
        self.time = float(command_time_s + self.dt)
        return self.state


def build_scene(scene_id, **kwargs):
    if scene_id != "r5_allegro_towel" and not scene_id.startswith("r5_franka_"):
        raise ValueError(scene_id)
    return ClothRobotScene(scene_id, **kwargs)
