# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example Robot Allegro Hand

Shows how to set up a simulation of a Allegro hand articulation
from a USD file using newton.ModelBuilder.add_usd().
We also apply a sinusoidal trajectory to the joint targets and
apply a continuous rotation to the fixed root joint in the form
of the joint parent transform. The solver is updated
about this change in the joint parent transform by calling
self.solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES).

Success criteria:
- Floating Allegro hands keep a cube on each palm throughout the run.
- The hands are manipulating the cube to make it turn.
- Finger targets and positions change over time, and all hands have the same movement.

Runs:
- `python -m newton.examples robot_allegro_hand --world-count 18`
  - 18 floating Allegro hands appear.
- `python -m newton.examples robot_allegro_hand --world-count 16`
  - 16 floating Allegro hands appear.
- `python -m newton.examples robot_allegro_hand --solver kamino`
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode, ModelFlags


@wp.kernel
def move_hand(
    joint_world_start: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    hand_dof_count: int,
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    sim_time: wp.array[wp.float32],
    sim_dt: float,
    hand_rotation: wp.quat,
    # outputs
    joint_target_q: wp.array[wp.float32],
    joint_parent_xform: wp.array[wp.transform],
):
    """Animate finger targets and root transforms without actuating the floating cubes."""
    world_id = wp.tid()
    root_joint_id = joint_world_start[world_id]
    t = sim_time[world_id]

    # The free cube has seven coordinates but six DOFs, so these offsets differ.
    root_coord_start = joint_q_start[root_joint_id]
    root_dof_start = joint_qd_start[root_joint_id]

    # animate the finger joints
    for i in range(hand_dof_count):
        qi = root_coord_start + i
        di = root_dof_start + i
        target = wp.sin(t + float(i * 6) * 0.1) * 0.08 + 0.3
        joint_target_q[qi] = wp.clamp(target, joint_limit_lower[di], joint_limit_upper[di])

    # animate the root joint transform
    q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.sin(t) * 0.1)
    root_xform = joint_parent_xform[root_joint_id]
    joint_parent_xform[root_joint_id] = wp.transform(root_xform.p, q * hand_rotation)

    # update the sim time
    sim_time[world_id] += sim_dt


class Example:
    def __init__(self, viewer, args):
        """Build replicated Allegro hands and configure the selected solver."""
        self.fps = 50
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.solver_type = args.solver
        self.world_count = args.world_count
        if self.world_count is None:
            self.world_count = 100

        self.viewer = viewer

        self.device = wp.get_device()

        self.hand_rotation = wp.normalize(wp.quat(0.21643, 0.706218, -0.648166, 0.185191))
        max_contacts_per_world = 300

        allegro_hand = newton.ModelBuilder()
        if self.solver_type == "kamino":
            newton.solvers.SolverKamino.register_custom_attributes(allegro_hand)
        else:
            newton.solvers.SolverMuJoCo.register_custom_attributes(allegro_hand)
        allegro_hand.default_shape_cfg.ke = 1.0e3
        allegro_hand.default_shape_cfg.kd = 1.0e2
        allegro_hand.default_shape_cfg.margin = 0.005
        allegro_hand.default_shape_cfg.gap = 0.015

        asset_path = newton.utils.download_asset("wonik_allegro")
        asset_file = str(asset_path / "usd" / "allegro_left_hand_with_cube.usda")
        allegro_hand.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0, 0, 0.5)),
            enable_self_collisions=False,
            ignore_paths=[".*Dummy", ".*CollisionPlane"],
            hide_collision_shapes=True,
            load_static_visual_shapes=False,
        )

        # set joint targets and joint drive gains (only on hand, not the floating-body cube)
        self.hand_dof_count = allegro_hand.joint_dof_count - 6
        for joint_id, joint_type in enumerate(allegro_hand.joint_type):
            if joint_type != newton.JointType.REVOLUTE:
                continue
            coord_start = allegro_hand.joint_q_start[joint_id]
            dof_start = allegro_hand.joint_qd_start[joint_id]
            initial_q = 0.6 if allegro_hand.joint_label[joint_id].endswith("_0") else 0.3
            allegro_hand.joint_q[coord_start] = initial_q
            allegro_hand.joint_target_q[coord_start] = initial_q
            allegro_hand.joint_target_ke[dof_start] = 150
            allegro_hand.joint_target_kd[dof_start] = 5
            allegro_hand.joint_target_mode[dof_start] = int(JointTargetMode.POSITION)
            allegro_hand.joint_armature[dof_start] = 1e-2

        # Update root pose of the cube (free joint)
        q = np.array(allegro_hand.joint_q)
        q[-7:-4] += np.array([0.0, 0.0, 0.05])
        q[-4:] = wp.quat_rpy(0.3, 0.5, 0.1)
        allegro_hand.joint_q = q.tolist()

        builder = newton.ModelBuilder()
        builder.replicate(allegro_hand, self.world_count)

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.add_ground_plane()

        self.model = builder.finalize()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.initial_world_positions = self.model.body_q.numpy()[:: allegro_hand.body_count, :3].copy()

        # Find the cube body index (it's the last body in each world)
        self.cube_body_offset = allegro_hand.body_count - 1

        self.world_time = wp.zeros(self.world_count, dtype=wp.float32)
        self._test_metrics = None

        if self.solver_type == "kamino":
            solver_config = newton.solvers.SolverKamino.Config.from_model(
                self.model, dynamics_solver="dvi", sparse_dynamics=True, sparse_jacobian=True
            )
            solver_config.dvi.max_alternating_iterations = 8
            solver_config.dvi.bilateral_solve_interval = 8
            solver_config.dvi.bilateral_solver_type = "LLTBRCM"
            solver_config.dvi.omega = 1.2
            solver_config.dvi.contact_warmstart_method = (
                "key_and_position_with_net_force_backup_and_tangential_net_force"
            )
            solver_config.dvi.use_schur_complement = True
            self.solver = newton.solvers.SolverKamino(self.model, config=solver_config)
        else:
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                solver="newton",
                integrator="implicitfast",
                njmax=200,
                nconmax=max_contacts_per_world,
                impratio=20.0,
                # Preserve the example's solref-inherited grasp friction; its
                # purpose is articulation control rather than kf mapping.
                cone="pyramidal",
                iterations=100,
                ls_iterations=50,
                use_mujoco_contacts=False,
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        self.graph = None
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        """Advance one frame while updating finger targets and hand root transforms."""
        if self.solver_type != "kamino":
            self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            wp.launch(
                move_hand,
                dim=self.world_count,
                inputs=[
                    self.model.joint_world_start,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    self.hand_dof_count,
                    self.model.joint_limit_lower,
                    self.model.joint_limit_upper,
                    self.world_time,
                    self.sim_dt,
                    self.hand_rotation,
                ],
                outputs=[self.control.joint_target_q, self.model.joint_X_p],
            )

            # Update the solver since we have updated the joint parent transforms.
            self.solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)
            if self.solver_type == "kamino":
                self.collision_pipeline.collide(self.state_0, self.contacts)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.add_argument("--solver", choices=["mujoco", "kamino"], default="mujoco")
        parser.set_defaults(world_count=None)
        return parser

    def test_post_step(self):
        """Accumulate finger and cube checks on device throughout the trajectory."""
        if self._test_metrics is None:
            self._test_metrics = wp.zeros(self.world_count, dtype=_TestMetrics, device=self.device)
            self._test_first_finger_targets = wp.clone(self.control.joint_target_q)
        wp.launch(
            _check_hand_state,
            dim=self.world_count,
            inputs=[
                self.model.joint_world_start,
                self.model.joint_q_start,
                self.model.body_world_start,
                self.hand_dof_count,
                self.cube_body_offset,
                self.control.joint_target_q,
                self.state_0.joint_q,
                self.state_0.body_q,
                self.model.joint_target_q,
                self.model.joint_q,
                self.model.body_q,
                self._test_first_finger_targets,
            ],
            outputs=[self._test_metrics],
            device=self.device,
        )

    def test_final(self):
        """Check accumulated motion results, initial configuration, and viewer layout."""
        self.test_post_step()
        metrics = self._test_metrics.numpy()
        for field, message in (
            ("target_mismatch", "Replicated hands must receive identical finger targets"),
            ("cube_target_mismatch", "Hand animation must not modify the floating cube's targets"),
            ("finger_mismatch", "Replicated hands must have the same finger motion"),
            ("cube_lost", "Cubes left the hands"),
        ):
            failed_worlds = np.flatnonzero(metrics[field]).tolist()
            assert not failed_worlds, f"{message}; failed worlds: {failed_worlds}"

        # Track the whole trajectory so runs ending at the initial sinusoidal phase still pass.
        # model.joint_q retains the physical initial pose from before graph capture.
        if self.sim_time >= 1.0:
            for field, label in (("max_target_change", "Finger targets"), ("max_finger_change", "Finger positions")):
                failed_worlds = np.flatnonzero(metrics[field] <= 0.02).tolist()
                assert not failed_worlds, f"{label} must move more than 0.02 rad; failed worlds: {failed_worlds}"

        hand_joints = np.flatnonzero(self.model.joint_type.numpy() == newton.JointType.REVOLUTE)
        hand_coords = self.model.joint_q_start.numpy()[hand_joints]
        hand_dofs = self.model.joint_qd_start.numpy()[hand_joints]
        expected_initial_q = [0.6 if self.model.joint_label[j].endswith("_0") else 0.3 for j in hand_joints]
        np.testing.assert_allclose(
            self.model.joint_q.numpy()[hand_coords],
            expected_initial_q,
            atol=1.0e-6,
            err_msg="Initial finger poses must follow joint labels",
        )
        np.testing.assert_allclose(
            self.model.joint_armature.numpy()[hand_dofs],
            1.0e-2,
            err_msg="Every finger joint must have the configured armature",
        )
        # Scale the bound with the grid size so larger world counts remain valid.
        offset_limit = np.ceil(np.sqrt(self.world_count))
        max_world_offset = float(np.abs(self.viewer.world_offsets.numpy()).max())
        assert max_world_offset < offset_limit, (
            f"World offsets reach {max_world_offset:g} m, expected less than {offset_limit:g} m for a compact hand layout"
        )

        num_bodies_per_world = self.model.body_count // self.world_count

        for i in range(self.world_count):
            world_offset = i * num_bodies_per_world
            world_pos = wp.vec3(*self.initial_world_positions[i])

            # Test hand bodies - must stay near initial position
            hand_lower = world_pos - wp.vec3(0.5, 0.5, 0.5)
            hand_upper = world_pos + wp.vec3(0.5, 0.5, 0.5)
            hand_body_indices = np.arange(num_bodies_per_world - 1, dtype=np.int32) + world_offset
            newton.examples.test_body_state(
                self.model,
                self.state_0,
                f"hand bodies from world {i} are close to the initial position",
                lambda q, qd: newton.math.vec_inside_limits(q.p, hand_lower, hand_upper),  # noqa: B023
                indices=hand_body_indices,
            )


@wp.struct
class _TestMetrics:
    """Persistent per-world failures and motion maxima for the example checks."""

    target_mismatch: int
    cube_target_mismatch: int
    finger_mismatch: int
    cube_lost: int
    max_target_change: float
    max_finger_change: float


@wp.kernel
def _check_hand_state(
    joint_world_start: wp.array[int],
    joint_q_start: wp.array[int],
    body_world_start: wp.array[int],
    hand_dof_count: int,
    cube_body_offset: int,
    joint_target_q: wp.array[float],
    joint_q: wp.array[float],
    body_q: wp.array[wp.transform],
    initial_joint_target_q: wp.array[float],
    initial_joint_q: wp.array[float],
    initial_body_q: wp.array[wp.transform],
    first_finger_targets: wp.array[float],
    # outputs
    metrics: wp.array[_TestMetrics],
):
    """Accumulate trajectory checks on device without synchronizing each frame."""
    world = wp.tid()
    coord_start = joint_q_start[joint_world_start[world]]
    coord_end = joint_q_start[joint_world_start[world + 1]]
    reference_start = joint_q_start[joint_world_start[0]]
    result = metrics[world]

    for i in range(hand_dof_count):
        qi = coord_start + i
        if joint_target_q[qi] != joint_target_q[reference_start + i]:
            result.target_mismatch = 1
        # Allow small solver/contact differences between otherwise identical worlds.
        if not wp.abs(joint_q[qi] - joint_q[reference_start + i]) <= 0.01:
            result.finger_mismatch = 1
        result.max_target_change = wp.max(
            result.max_target_change, wp.abs(joint_target_q[qi] - first_finger_targets[qi])
        )
        result.max_finger_change = wp.max(result.max_finger_change, wp.abs(joint_q[qi] - initial_joint_q[qi]))

    for qi in range(coord_start + hand_dof_count, coord_end):
        if joint_target_q[qi] != initial_joint_target_q[qi]:
            result.cube_target_mismatch = 1

    body_start = body_world_start[world]
    world_pos = initial_body_q[body_start].p
    cube_lower = world_pos - wp.vec3(0.5, 0.5, 0.5)
    cube_lower[2] = 0.9
    cube_upper = world_pos + wp.vec3(0.5, 0.5, 0.5)
    if not newton.math.vec_inside_limits(body_q[body_start + cube_body_offset].p, cube_lower, cube_upper):
        result.cube_lost = 1

    # Keep failures sticky so a later recovery cannot hide a bad frame.
    metrics[world] = result


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
