# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Controllers — Differential IK driving four Franka Panda arms
#
# A ControlLawDifferentialIK manages four Franka Panda (fr3 + hand) robots
# arranged in a row. Each robot has its own 6DOF draggable gizmo in the
# viewer; the controller drives every arm's TCP toward its gizmo every
# step. The MuJoCo solver applies the controller's commanded q-targets
# via its built-in joint-position PD.
#
# Command: python -m newton.examples diff_ik_panda
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.solvers
import newton.utils
from newton import JointTargetMode
from newton.controllers import ControllerDifferentialKinematics
from newton.utils import compute_world_offsets

ROBOT_COUNT = 4
ROBOT_SPACING_Y = 1.2
ARM_DOF_COUNT = 7
FINGER_DOF_COUNT = 2
DOFS_PER_ROBOT = ARM_DOF_COUNT + FINGER_DOF_COUNT
# fr3_link7 → fr3_link8 (z=+0.107) → fr3_hand (id) → fr3_hand_tcp (z=+0.1034).
# These fixed joints are collapsed into fr3_link7 at load time, so the TCP
# expressed in fr3_link7's body-local frame is the sum.
TCP_OFFSET = wp.vec3(0.0, 0.0, 0.2104)


@wp.kernel
def _scatter_kernel(
    src: wp.array[float],
    indices: wp.array[int],
    dst: wp.array[float],
):
    """Copy ``src[indices[i]] → dst[indices[i]]`` — used to push only the arm
    DOFs of the controller's q-buffer into ``control.joint_target_q``,
    leaving finger entries (initialized to home once) undisturbed.
    """
    i = wp.tid()
    idx = indices[i]
    dst[idx] = src[idx]


class Example:
    def __init__(self, viewer, args):
        # joint_target_q / joint_target_qd are the canonical coord-layout
        # arrays MuJoCo's joint-target PD reads from.
        newton.use_coord_layout_targets = True

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        # ------------------------------------------------------------------
        # Template — one Franka with MuJoCo PD attrs + a TCP site
        # ------------------------------------------------------------------
        template = newton.ModelBuilder()

        urdf_path = newton.utils.download_asset("franka_emika_panda") / "urdf" / "fr3_franka_hand.urdf"
        template.add_urdf(
            str(urdf_path),
            floating=False,
            collapse_fixed_joints=True,
        )

        # After collapse_fixed_joints, the hand/TCP fixed-joint chain folds
        # into fr3_link7 — that's the body the IK site rides on.
        ee_body_in_template = template.body_label.index("fr3/fr3_link7")
        self._ee_body_in_template = ee_body_in_template

        template.add_site(
            ee_body_in_template,
            label="ee",
            xform=wp.transform(p=TCP_OFFSET, q=wp.quat_identity()),
            visible=True,
            scale=(0.02, 0.02, 0.02),
        )

        # Reasonable home pose: arm folded forward-up, gripper fingers open.
        home_q = np.array(
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
            dtype=np.float32,
        )
        template.joint_q = home_q.tolist()
        self.home_q_per_robot = home_q

        # ------------------------------------------------------------------
        # Scene — replicate 4 robots along +Y, add ground plane.
        # ------------------------------------------------------------------
        scene = newton.ModelBuilder()
        scene.replicate(template, ROBOT_COUNT, spacing=(0.0, ROBOT_SPACING_Y, 0.0))
        scene.add_ground_plane()
        self.model = scene.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # disable_contacts: the arms shouldn't be colliding with anything in
        # this demo, and turning contacts off keeps the simulation fast and
        # focused on tracking behavior.
        self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)

        # ------------------------------------------------------------------
        # DiffIK setup
        # ------------------------------------------------------------------
        total_dofs = ROBOT_COUNT * DOFS_PER_ROBOT
        dof_indices = wp.array(np.arange(total_dofs, dtype=np.uint32), device=self.device)

        # DiffIK's internal model is built from ``diffik_template`` — all
        # robots at world origin. The scene's replicate spaces robots out
        # *and centers the grid* (for 4 robots along Y at spacing 1.2 the
        # bases sit at y = -1.8, -0.6, 0.6, 1.8). Mirror that offset
        # computation here so a scene-world gizmo translates into the
        # right DiffIK-internal target by subtracting the base offset.
        self._base_offsets_np = compute_world_offsets(
            ROBOT_COUNT, (0.0, ROBOT_SPACING_Y, 0.0), up_axis=self.model.up_axis
        )

        bodies_per_robot = self.model.body_count // ROBOT_COUNT
        self._bodies_per_robot = bodies_per_robot
        body_q_np = self.state_0.body_q.numpy()

        # Initialize gizmos at each robot's home TCP pose in world coords.
        self.gizmo_tfs: list[wp.transform] = []
        target_pos_init = np.zeros((ROBOT_COUNT, 3), dtype=np.float32)
        target_quat_init = np.zeros((ROBOT_COUNT, 4), dtype=np.float32)
        for r in range(ROBOT_COUNT):
            ee_scene_idx = r * bodies_per_robot + ee_body_in_template
            ee_world = wp.transform(*body_q_np[ee_scene_idx])
            site_world = ee_world * wp.transform(p=TCP_OFFSET, q=wp.quat_identity())
            site_pos = wp.transform_get_translation(site_world)
            site_quat = wp.transform_get_rotation(site_world)
            self.gizmo_tfs.append(wp.transform(p=site_pos, q=site_quat))
            offset = self._base_offsets_np[r]

            # Initialize the targets to exactly where the target-site starts:
            target_pos_init[r] = [
                site_pos[0] - offset[0],
                site_pos[1] - offset[1],
                site_pos[2] - offset[2],
            ]
            target_quat_init[r] = [site_quat[0], site_quat[1], site_quat[2], site_quat[3]]

        # ------------------------------------------------------------------
        # DiffIK template — a separate N-articulation builder with no
        # ground plane and no physical spacing. The controller finalizes
        # this independently and runs eval_fk / eval_jacobian on it; the
        # all-at-origin layout means the user's per-robot target_pos lives
        # in the same frame as the scene minus each robot's base offset.
        # ------------------------------------------------------------------
        diffik_template = newton.ModelBuilder()
        diffik_template.replicate(template, ROBOT_COUNT)

        self.controller = ControllerDifferentialKinematics(
            model_builder=diffik_template,
            controlled_site_label="ee",
            default_dof_indices=dof_indices,
            solver_damping=wp.full(ROBOT_COUNT, 0.05, dtype=wp.float32, device=self.device),
            bandwidth=wp.full(ROBOT_COUNT, 20.0, dtype=wp.float32, device=self.device),
            device=self.device,
        )

        # input_struct() / output_struct() return fresh dataclasses with one
        # wp.zeros field per live port. We seed the per-robot target fields
        # with the home TCPs, then reassign joint_q / joint_qd each frame to
        # point at the current sim state.
        self._input = self.controller.input_struct()
        self._input.joint_q = self.state_0.joint_q
        self._input.joint_qd = self.state_0.joint_qd
        self._input.site_target_position.assign(target_pos_init)
        self._input.site_target_quaternion.assign(target_quat_init)
        self._output = self.controller.output_struct()

        # Seed control.joint_target_q with the home pose for every robot.
        # The arm slots get overwritten every substep by the scatter; the
        # finger slots persist (PD holds them open).
        home_tile = np.tile(self.home_q_per_robot, ROBOT_COUNT).astype(np.float32)
        wp.copy(
            self.control.joint_target_q,
            wp.array(home_tile, dtype=wp.float32, device=self.device),
        )

        # Arm DOF indices in the scene's per-robot layout. The DiffIK's
        # dof_indices is arange(total_dofs), so q_buffer flat index equals
        # the scene's joint_target_q index — we just pick out the 7 arm
        # slots per robot for the scatter.
        arm_indices = []
        for r in range(ROBOT_COUNT):
            base = r * DOFS_PER_ROBOT
            arm_indices.extend(range(base, base + ARM_DOF_COUNT))
        self._arm_indices = wp.array(arm_indices, dtype=wp.int32, device=self.device)

        self.viewer.set_model(self.model)
        # The robots are already spaced apart in physics; without this the
        # viewer would auto-add a *visual* world offset on top of that (its
        # default for multi-world scenes), which leaves the gizmos — logged
        # in world coords — drifting off the rendered TCPs.
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        self.viewer.set_camera(
            pos=wp.vec3(2.5, 0.0, 1.3),
            pitch=-15.0,
            yaw=180.0,
        )

        # CUDA graph capture skipped: _push_gizmos mutates wp.arrays from the
        # Python side each frame to follow gizmo drags, which isn't capturable.
        self.graph = None

    def _push_gizmos(self) -> None:
        """Read the four gizmos' world transforms, translate each into the
        corresponding robot's DiffIK frame (base offset removed), and update
        the controller's per-robot target arrays.
        """
        pos = np.zeros((ROBOT_COUNT, 3), dtype=np.float32)
        quat = np.zeros((ROBOT_COUNT, 4), dtype=np.float32)
        for r, tf in enumerate(self.gizmo_tfs):
            p = wp.transform_get_translation(tf)
            q = wp.transform_get_rotation(tf)
            offset = self._base_offsets_np[r]
            pos[r] = [p[0] - offset[0], p[1] - offset[1], p[2] - offset[2]]
            quat[r] = [q[0], q[1], q[2], q[3]]
        self._input.site_target_position.assign(pos)
        self._input.site_target_quaternion.assign(quat)

    def _scatter_arm_targets(self) -> None:
        wp.launch(
            _scatter_kernel,
            dim=len(self._arm_indices),
            inputs=[self._output.joint_target_q, self._arm_indices],
            outputs=[self.control.joint_target_q],
            device=self.device,
        )

    def step(self) -> None:
        self._push_gizmos()
        # Rebind the live joint_q / joint_qd to whichever State buffer the
        # substep swap left at ``self.state_0`` after the previous frame.
        # SimpleNamespace's attributes are just Python refs, so this is
        # cheap; we do it every frame for robustness regardless of the
        # substep parity.
        self._input.joint_q = self.state_0.joint_q
        self._input.joint_qd = self.state_0.joint_qd
        # One controller step per frame. DiffIK's output_q is current_q +
        # q_dot * frame_dt — a position target one frame ahead. The MuJoCo PD
        # then tracks that fixed target for all ``sim_substeps`` physics
        # substeps; running DiffIK inside the substep loop would refresh the
        # target against the drifted current_q on every substep, leaving PD
        # with no restoring signal against gravity.
        self.controller.compute(self._input, self._output, None, None, time_step=self.frame_dt)
        self._scatter_arm_targets()
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)

        # Log each gizmo with snap_to set to the current TCP pose so a
        # released gizmo jumps back to the arm's actual tip.
        body_q_np = self.state_0.body_q.numpy()
        for r, tf in enumerate(self.gizmo_tfs):
            ee_scene_idx = r * self._bodies_per_robot + self._ee_body_in_template
            ee_world = wp.transform(*body_q_np[ee_scene_idx])
            site_world = ee_world * wp.transform(p=TCP_OFFSET, q=wp.quat_identity())
            self.viewer.log_gizmo(f"target_{r}", tf, snap_to=site_world)

        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()
        wp.synchronize()

    def test_final(self) -> None:
        # Gizmos aren't touched in headless test mode, so every arm should
        # remain near its home pose under MuJoCo's joint-position PD.
        joint_q = self.state_0.joint_q.numpy()
        for r in range(ROBOT_COUNT):
            base = r * DOFS_PER_ROBOT
            arm_q = joint_q[base : base + ARM_DOF_COUNT]
            assert np.all(np.isfinite(arm_q)), f"Robot {r} joint_q has NaN/Inf: {arm_q}"
            assert np.allclose(arm_q, self.home_q_per_robot[:ARM_DOF_COUNT], atol=0.1), (
                f"Robot {r} drifted from home pose: {arm_q}"
            )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
