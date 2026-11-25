# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""H1 scenes initialized through public model/state APIs before solver creation.

The fixtures are prescribed initial conditions, not reference rollouts. They
contain no solver history. All cloth particles remain free. Original hand
geometry and coupled joint commands act through ordinary rigid-soft contact.
"""

import numpy as np
import warp as wp

import newton

from .catalog import get_case
from .h1_gripper import H1HandDriver, add_h1_hand
from .h1_inputs import load_input
from .public_vbd import create_solver, set_command_time


def _restore_public_arrays(target, prefix, arrays):
    """Reject incompatible public layouts instead of silently changing inputs."""
    for key, source in arrays.items():
        if not key.startswith(prefix + "."):
            continue
        name = key.removeprefix(prefix + ".")
        destination = getattr(target, name, None)
        if not isinstance(destination, wp.array) or destination.numpy().shape != source.shape:
            raise ValueError(f"H1 initialized public layout is incompatible: {key}")
        if destination.numpy().dtype != source.dtype:
            raise ValueError(f"H1 initialized public dtype is incompatible: {key}")
        destination.assign(source)


class H1Scene:
    """Build one preserved H1 workload with a fresh standard VBD solver."""

    def __init__(self, case_id, *, device=None):
        case = get_case(case_id)
        metadata, arrays = load_input(case_id)
        self.scene_id = case_id
        self.input_metadata = metadata
        self.frame_dt = case["frame_dt_s"]
        self.substeps = case["substeps"]
        self.dt = self.frame_dt / self.substeps
        self.iterations = case["iterations"]
        self.duration = metadata["duration_s"]
        self.time = 0.0
        self.title = case["title"]
        self.description = (
            "Original H1 hand assets, free cloth, and explicit saved initial conditions. "
            "Only hand roots and source-coupled joints are prescribed; no attachments or particle pins. "
            + (
                "Prepared upright body-rim contact; no acquisition claim."
                if metadata["initial_hold_prepared"]
                else "Starts with audited open hands after the declared free preparation."
            )
        )
        self.camera_pos = tuple(metadata["camera"]["position"])
        self.camera_target = tuple(metadata["camera"]["target"])
        self.camera_fov = metadata["camera"]["fov"]
        material, table = metadata["material"], metadata["table"]
        self.radius = material["radius"]
        self.table_center, self.table_half = np.asarray(table["center"]), np.asarray(table["half"])
        builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
        builder.default_shape_cfg.ke = 50000.0
        builder.default_shape_cfg.kd = 0.1
        builder.default_shape_cfg.mu = material["hand_friction"]
        builder.default_shape_cfg.gap = 0.005
        builder.default_shape_cfg.margin = 0.001
        builder.default_shape_cfg.configure_sdf(force_sdf=True, max_resolution=material["sdf_resolution"])
        transform = metadata["cloth_transform"]
        rotation = wp.quat_identity()
        if transform["jacket_rotation"]:
            rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi) * wp.quat_from_axis_angle(
                wp.vec3(1, 0, 0), -np.pi / 2
            )
        builder.add_cloth_mesh(
            vertices=arrays["rest.vertices"].tolist(),
            indices=arrays["rest.triangles"].ravel().tolist(),
            pos=transform["position"],
            rot=rotation,
            scale=transform["scale"],
            vel=(0, 0, 0),
            density=0.22,
            particle_radius=self.radius,
            tri_ke=1800.0,
            tri_ka=1800.0,
            tri_kd=0.025,
            edge_ke=0.08,
            edge_kd=0.0005,
            color=(0.68, 0.27, 0.18) if case_id == "j3_h1_jacket" else (0.3, 0.62, 0.8),
        )
        table_cfg = newton.ModelBuilder.ShapeConfig(
            density=0, mu=table["friction"], ke=50000, kd=0.1, gap=0.005, margin=table["margin"]
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(self.table_center, wp.quat_identity()),
            hx=self.table_half[0],
            hy=self.table_half[1],
            hz=self.table_half[2],
            cfg=table_cfg,
            color=(0.48, 0.51, 0.55),
            label="fixed original support table",
        )
        self.specs = [
            add_h1_hand(
                builder,
                side=hand["side"],
                root_pos=hand["root_pos"],
                rotation=hand["rotation"],
                controls=hand["controls"],
                label=hand["label"],
            )
            for hand in metadata["hands"]
        ]
        self.hands = self.specs
        builder.add_ground_plane()
        builder.color(include_bending=case_id != "j3_h1_jacket")
        self.model = builder.finalize(device=device)
        self.model.soft_contact_ke = 50000.0
        self.model.soft_contact_kd = 0.1
        self.model.soft_contact_mu = material["friction"]
        if not np.all(self.model.particle_mass.numpy() > 0):
            raise ValueError("H1 workload must retain all free cloth particles")
        np.testing.assert_array_equal(self.model.tri_indices.numpy(), arrays["rest.triangles"])
        self.state0, self.state1, self.control = self.model.state(), self.model.state(), self.model.control()
        for prefix in ("state0", "state1", "control"):
            _restore_public_arrays(getattr(self, prefix), prefix, arrays)
        self.clock = wp.zeros(1, dtype=float, device=self.model.device)
        self.drivers = []
        for index, spec in enumerate(self.specs):
            prefix = f"driver{index}."
            fingers = arrays[prefix + "fingers"]
            driver = H1HandDriver(
                self.model,
                spec,
                times=arrays[prefix + "times"],
                positions=arrays[prefix + "positions"],
                rotations=arrays[prefix + "rotations"],
                controls=fingers[:, [0, 1, 4, 6, 8, 10]],
            )
            # Preserve source float32 coupling and quaternion bits exactly.
            for name in ("times", "positions", "rotations", "fingers"):
                getattr(driver, name).assign(arrays[prefix + name])
            self.drivers.append(driver)
        # Native VBD sees the complete initial state on its first ordinary step.
        # No previously-created solver exists, and no history reset is required.
        capacity = case.get("self_contact_capacity", 64)
        self.solver, self.pipeline = create_solver(
            self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.002,
            particle_self_contact_gap=0.001,
            particle_rest_shape_contact_exclusion_radius=0.005,
            particle_topological_contact_filter_threshold=1,
            particle_vertex_contact_buffer_size=capacity,
            particle_edge_contact_buffer_size=capacity,
            friction_epsilon=0.001,
            soft_contact_gap=0.005,
            full_surface_contact=True,
        )
        self.contacts = self.solver.contacts

    @property
    def state(self):
        """Return the latest physical state."""
        return self.state0

    @property
    def phase(self):
        """Report the command schedule, without claiming task success."""
        return "prescribed H1 manipulation"

    def step_substep(self, command_time_s):
        """Prescribe the hand endpoint and take one ordinary physical step."""
        self.state0.clear_forces()
        set_command_time(self.clock, command_time_s + self.dt)
        for driver in self.drivers:
            driver.update(self.clock, self.state0)
        self.solver.step(self.state0, self.state1, self.control, None, self.dt)
        self.state0, self.state1 = self.state1, self.state0
        self.time += self.dt
        return self.state0

    def metrics(self):
        """Return workload inputs, without treating initialization as success."""
        return {
            "initialization": self.input_metadata,
            "particle_count": self.model.particle_count,
            "triangle_count": self.model.tri_count,
            "all_particles_free": bool(np.all(self.model.particle_mass.numpy() > 0)),
            "physics_parameters": {
                "frame_dt_s": self.frame_dt,
                "substeps": self.substeps,
                "iterations": self.iterations,
                "soft_contact_ke": self.model.soft_contact_ke,
                "soft_contact_mu": self.model.soft_contact_mu,
            },
        }


def build_scene(case_id, *, device=None):
    """Build one of the four preserved H1 recipes through supported APIs."""
    return H1Scene(case_id, device=device)
