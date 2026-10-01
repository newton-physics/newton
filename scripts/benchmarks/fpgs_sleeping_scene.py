# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run controlled experimental FPGS sleeping scenes with GL or null viewers.

Use --viewer null --test for finite-state and scenario checks. Sleeping checks
require the experimental solver.sleeping body_awake/body_island interface.
This correctness runner includes diagnostic readbacks; its FPS is not a solver benchmark.
The ants scene reuses the public MuJoCo sleeping example's tilted stacks, with
drive gains and actuation cleared in both arms; passive damping and limits remain.
Both arms use friction_anchor_beta=0 and zero torsion: persistent friction patches
and torsion are outside this sleeping prototype's qualification.
"""

import argparse
import json
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFeatherPGS

ISLAND_PALETTE = ((0.27, 0.47, 0.67), (0.40, 0.80, 0.93), (0.13, 0.53, 0.20), (0.93, 0.40, 0.47))
SLEEP_COLOR = (0.32, 0.32, 0.32)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        if args.objects < 1 or args.num_worlds < 1:
            raise ValueError("objects and num-worlds must be positive")
        if args.scene == "pusher" and args.objects < 2:
            raise ValueError("pusher requires at least two objects per world")
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0
        self.kick_frame = 180
        self.mode = {"separated": 0, "pile": 0, "pusher": 1, "all-active": 2, "ants": 0}[args.scene]
        self.sleep_seen = np.zeros(args.num_worlds, dtype=bool)
        self.wake_seen = np.zeros(args.num_worlds, dtype=bool)
        self.neighbor_sleep_seen = np.zeros(args.num_worlds, dtype=bool)
        self.neighbor_wake_seen = np.zeros(args.num_worlds, dtype=bool)
        self.push_displacement = np.zeros(args.num_worlds)
        self.push_start = None

        builder = newton.ModelBuilder()
        if args.scene == "ants":
            scene, spacing = _build_ant_stacks(builder, args.stack_count)
        else:
            scene = newton.ModelBuilder()
            scene.default_shape_cfg.mu = 0.6
            columns = math.ceil(math.sqrt(args.objects))
            for index in range(args.objects):
                if args.scene == "pile":
                    x = (index % 2) * 0.205
                    y = ((index // 2) % 2) * 0.205
                    z = 0.11 + (index // 4) * 0.205
                elif args.scene == "pusher":
                    x, y, z = index * 0.22, 0.0, 0.105
                else:
                    x, y, z = (index % columns) * 0.55, (index // columns) * 0.55, 0.105
                body = scene.add_body(xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()), label=f"box_{index}")
                scene.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            spacing = max(args.objects * 0.22, columns * 0.55) + 2.0
        self.bodies_per_world = len(scene.body_mass)
        builder.add_ground_plane()
        world_columns = math.ceil(math.sqrt(args.num_worlds))
        for world in range(args.num_worlds):
            offset = wp.vec3((world % world_columns) * spacing, (world // world_columns) * spacing, 0.0)
            builder.add_world(scene, xform=wp.transform(offset, wp.quat_identity()))
        self.model = builder.finalize()
        if args.scene == "ants":
            for gains in (self.model.joint_target_ke, self.model.joint_target_kd):
                if np.any(gains.numpy() != 0.0):
                    raise RuntimeError("Passive ants require zero model drive gains")
            print(
                json.dumps(
                    {
                        "scene": "ants",
                        "passive": True,
                        "joint_target_ke": 0.0,
                        "joint_target_kd": 0.0,
                        "actuation": 0.0,
                        "passive_damping": "authored",
                        "friction_anchor_beta": 0.0,
                        "torsion_radius": 0.0,
                    }
                ),
                flush=True,
            )
        if args.graph and not self.model.device.is_cuda:
            raise ValueError("CUDA graphs require a CUDA device; use --no-graph on CPU")
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=64 * self.model.body_count)
        self.contacts = self.pipeline.contacts()
        sleeping_options = {}
        if args.sleeping:
            sleeping_options = {
                "enable_sleeping": True,
                "sleep_linear_threshold": 0.05,
                "sleep_angular_threshold": 0.15,
                "sleep_quiet_time": 0.5,
            }
        self.solver = SolverFeatherPGS(
            self.model,
            pgs_mode="matrix_free" if self.model.device.is_cuda else "split",
            articulated_contact_response="immediate",
            pgs_iterations=8,
            dense_max_constraints=1024,
            mf_max_constraints=1024,
            row_watermark=True,
            enable_joint_limits=args.scene == "ants",
            friction_anchor_beta=0.0,
            contact_torsion_radius=0.0,
            **sleeping_options,
        )
        self.sleeping = self.solver.sleeping if args.sleeping else None
        if args.sleeping and self.sleeping is None:
            raise RuntimeError("Sleeping was requested but the solver has no sleeping state")
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        if args.scene == "ants":
            for forces in (self.control.joint_f, self.control.joint_act):
                if forces is not None and np.any(forces.numpy() != 0.0):
                    raise RuntimeError("Passive ants require zero initial joint controls")
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.device_frame = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        self.default_shape_color = wp.clone(self.model.shape_color)
        self.island_palette = wp.array(ISLAND_PALETTE, dtype=wp.vec3, device=self.model.device)
        self.viewer.set_model(self.model)
        if args.scene == "ants":
            self.viewer.set_camera(pos=wp.vec3(8.0, -10.0, 6.5), pitch=-27.0, yaw=128.0)
        else:
            self.viewer.set_camera(pos=wp.vec3(5.0, -6.0, 5.0), pitch=-30.0, yaw=130.0)
        self.graphs = []
        if args.graph:
            # Odd substep counts alternate the captured input and output buffers.
            for _ in range(2 if self.sim_substeps % 2 else 1):
                with wp.ScopedCapture(device=self.model.device) as capture:
                    self.solver.seed_double_buffer_events()
                    self.simulate()
                self.graphs.append(capture.graph)

    def simulate(self):
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            if substep == 0:
                wp.launch(
                    _apply_impulses,
                    dim=self.model.body_count,
                    inputs=[
                        self.device_frame,
                        self.model.body_mass,
                        self.args.objects,
                        self.mode,
                        self.kick_frame,
                        self.sim_dt,
                    ],
                    outputs=[self.state_0.body_f],
                    device=self.model.device,
                )
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(_advance_frame, dim=1, inputs=[self.device_frame], device=self.model.device)

    def step(self):
        if self.frame == self.kick_frame and self.mode == 1:
            self.push_start = self.state_0.body_q.numpy()[1 :: self.args.objects, :3].copy()
        if not self.graphs:
            self.simulate()
        else:
            wp.capture_launch(self.graphs[self.frame % len(self.graphs)])
            if self.sim_substeps % 2:
                self.state_0, self.state_1 = self.state_1, self.state_0
        self.frame += 1
        self.sim_time = self.frame * self.frame_dt
        if self.args.sleeping:
            awake = self.sleeping.body_awake.numpy().reshape(self.args.num_worlds, self.bodies_per_world)
            if self.frame <= self.kick_frame:
                self.sleep_seen |= ~awake[:, 0].astype(bool)
                if self.mode == 1:
                    self.neighbor_sleep_seen |= ~awake[:, 1].astype(bool)
            if self.mode == 1 and self.kick_frame < self.frame <= self.kick_frame + 60:
                self.wake_seen |= awake[:, 0].astype(bool)
                self.neighbor_wake_seen |= awake[:, 1].astype(bool)
        if self.mode == 1 and self.kick_frame < self.frame <= self.kick_frame + 60:
            position = self.state_0.body_q.numpy()[1 :: self.args.objects, :3]
            self.push_displacement = np.maximum(
                self.push_displacement, np.linalg.norm(position - self.push_start, axis=1)
            )
        if self.frame % self.fps == 0:
            self.solver.check_constraint_capacity()
            if self.args.sleeping:
                print(
                    json.dumps(
                        {"frame": self.frame, "awake": int(np.count_nonzero(awake)), "bodies": self.model.body_count}
                    ),
                    flush=True,
                )

    def test_post_step(self):
        """Reject nonfinite state and any latched capacity failure."""
        for array in (self.state_0.body_q, self.state_0.body_qd, self.state_0.joint_q, self.state_0.joint_qd):
            if not np.isfinite(array.numpy()).all():
                raise AssertionError("Nonfinite simulation state")
        self.solver.check_constraint_capacity()

    def test_final(self):
        """Check sleep, wake interaction, or sustained excitation for the selected scene."""
        self.test_post_step()
        if self.frame < 360:
            raise AssertionError("Scenario checks require at least 360 simulated frames")
        positions = self.state_0.body_q.numpy()
        if self.args.scene == "ants":
            roots = self.model.joint_child.numpy()[self.model.articulation_start.numpy()[:-1]]
            if np.any(positions[roots, 2] < 0.0):
                raise AssertionError("An ant root fell through the ground")
        elif np.any(positions[:, 2] < 0.05):
            raise AssertionError("A box fell through the ground")
        if self.mode == 1 and not np.all(self.push_displacement > 0.01):
            raise AssertionError("The impulse did not move the neighboring box in every world")
        result = {"scene": self.args.scene, "frames": self.frame, "sleeping": self.args.sleeping}
        if self.args.sleeping:
            awake = self.sleeping.body_awake.numpy().reshape(self.args.num_worlds, self.bodies_per_world)
            islands = self.sleeping.body_island.numpy()
            if islands.shape != (self.model.body_count,) or np.any(islands < 0):
                raise AssertionError("Invalid dynamic-body island mapping")
            if self.mode == 0 and not np.all(np.any(awake == 0, axis=1)):
                raise AssertionError("No sleeping body in at least one settled world")
            if self.mode == 1 and not np.all(self.sleep_seen & self.wake_seen):
                raise AssertionError("Expected sleep followed by impulse wake in every world")
            if self.mode == 1 and not np.all(self.neighbor_sleep_seen & self.neighbor_wake_seen):
                raise AssertionError("Expected the sleeping neighbor to wake on impact in every world")
            if self.mode == 2 and not np.all(awake):
                raise AssertionError("A periodically excited body remained asleep")
            result["awake"] = int(np.count_nonzero(awake))
            self._update_shape_colors()
            bodies = self.model.shape_body.numpy()
            colors = self.model.shape_color.numpy()
            expected = self.default_shape_color.numpy()
            dynamic = bodies >= 0
            flat_awake = awake.ravel()
            expected[dynamic] = np.asarray(ISLAND_PALETTE)[islands[bodies[dynamic]] % len(ISLAND_PALETTE)]
            expected[dynamic & (flat_awake[np.maximum(bodies, 0)] == 0)] = SLEEP_COLOR
            np.testing.assert_allclose(colors, expected, rtol=0.0, atol=1e-6)
        if self.mode == 1:
            result["neighbor_displacement_m"] = self.push_displacement.tolist()
            result["neighbor_sleep_seen"] = self.neighbor_sleep_seen.tolist()
            result["neighbor_wake_seen"] = self.neighbor_wake_seen.tolist()
        result["capacity"] = self.solver.constraint_row_watermarks()
        print(json.dumps(result), flush=True)

    def render(self):
        if self.sleeping is not None and self.args.viewer != "null":
            self._update_shape_colors()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def _update_shape_colors(self):
        wp.launch(
            _color_islands,
            dim=self.model.shape_count,
            inputs=[
                self.model.shape_body,
                self.sleeping.body_awake,
                self.sleeping.body_island,
                self.island_palette,
                wp.vec3(*SLEEP_COLOR),
                self.default_shape_color,
            ],
            outputs=[self.model.shape_color],
            device=self.model.device,
        )


def _build_ant_stacks(builder, stack_count):
    """Reuse the public ant stack with passive joints in every comparison arm."""
    # Defer the MuJoCo example's backend imports to the optional ant scene.
    from newton.examples.mujoco import example_mujoco_sleeping as ants
    from newton.solvers import SolverMuJoCo

    if stack_count < 1:
        raise ValueError("stack-count must be positive")
    stack = ants.build_stack()
    stack.joint_target_ke[:] = [0.0] * len(stack.joint_target_ke)
    stack.joint_target_kd[:] = [0.0] * len(stack.joint_target_kd)
    stack.joint_act[:] = [0.0] * len(stack.joint_act)
    if any(value != 0.0 for value in stack.joint_spring_stiffness):
        raise ValueError("Ant sleeping scene requires spring-free passive joints")
    scene = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(scene)
    SolverMuJoCo.register_custom_attributes(builder)
    columns = math.ceil(math.sqrt(stack_count))
    rows = math.ceil(stack_count / columns)
    for index in range(stack_count):
        x = (index % columns - 0.5 * (columns - 1)) * ants.STACK_SPACING
        y = (index // columns - 0.5 * (rows - 1)) * ants.STACK_SPACING
        yaw = math.atan2(-y, -x) if x != 0.0 or y != 0.0 else 0.0
        scene.add_builder(
            stack,
            xform=wp.transform(wp.vec3(x, y, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)),
            label_prefix=f"stack_{index}_",
        )
    return scene, max(columns, rows) * ants.STACK_SPACING + 5.0


@wp.kernel
def _color_islands(
    shape_body: wp.array[int],
    body_awake: wp.array[int],
    body_island: wp.array[int],
    palette: wp.array[wp.vec3],
    sleep_color: wp.vec3,
    default_color: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
):
    shape = wp.tid()
    body = shape_body[shape]
    if body < 0:
        colors[shape] = default_color[shape]
    elif body_awake[body] == 0:
        colors[shape] = sleep_color
    elif body_island[body] >= 0:
        colors[shape] = palette[body_island[body] % palette.shape[0]]
    else:
        colors[shape] = default_color[shape]


@wp.kernel
def _apply_impulses(
    frame: wp.array[wp.int32],
    mass: wp.array[float],
    objects: int,
    mode: int,
    kick_frame: int,
    dt: float,
    forces: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    velocity_change = float(0.0)
    if mode == 1 and frame[0] == kick_frame and body % objects == 0:
        velocity_change = 4.0
    elif mode == 2 and frame[0] % 6 == 0:
        velocity_change = 0.5
        if (frame[0] // 6) % 2 != 0:
            velocity_change = -velocity_change
    forces[body] = wp.spatial_vector(wp.vec3(mass[body] * velocity_change / dt, 0.0, 0.0), wp.vec3(0.0))


@wp.kernel
def _advance_frame(frame: wp.array[wp.int32]):
    frame[0] += 1


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.description = __doc__
    parser.set_defaults(num_frames=600)
    parser.add_argument("--scene", choices=("separated", "pile", "pusher", "all-active", "ants"), default="separated")
    parser.add_argument("--sleeping", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-worlds", type=int, default=1)
    parser.add_argument("--objects", type=int, default=16, help="Boxes per world for box scenes.")
    parser.add_argument("--stack-count", type=int, default=4, help="Five-ant stacks per world for the ants scene.")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
