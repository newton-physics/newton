# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from enum import IntEnum

import numpy as np
import warp as wp

import newton
from newton._src.utils import is_graph_capture_allocation_enabled
from newton.solvers import SolverMuJoCo
from newton.solvers import mujoco as mjc_author
from newton.tests.unittest_utils import MUJOCO_MODULES, requires


@requires(*MUJOCO_MODULES)
class TestMujocoFixedTendon(unittest.TestCase):
    class LimitBreachType(IntEnum):
        UPPER_LIMIT_FROM_ABOVE = 0
        UPPER_LIMIT_FROM_BELOW = 1
        LOWER_LIMIT_FROM_BELOW = 2
        LOWER_LIMIT_FROM_ABOVE = 3

    def test_single_mujoco_fixed_tendon_length_behaviour(self):
        """Test that tendon length works as expected"""
        mjcf = """<?xml version="1.0" ?>
<mujoco model="two_prismatic_links">
  <compiler angle="degree"/>

  <option timestep="0.002" gravity="0 0 0"/>

  <worldbody>
    <!-- Root body (fixed to world) -->
    <body name="root" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>

      <!-- First child link with prismatic joint along x -->
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom solmix="1.0" type="cylinder" size="0.05 0.025" rgba="1 0 0 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>

      <!-- Second child link with prismatic joint along x -->
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom type="cylinder" size="0.05 0.025" rgba="0 0 1 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>

  <tendon>
    <!-- Fixed tendon coupling joint1 and joint2 -->
	<fixed
		name="coupling_tendon"
		stiffness="2"
		damping="1"
		springlength="0.0">
      <joint joint="joint1" coef="1"/>
      <joint joint="joint2" coef="1"/>
    </fixed>
  </tendon>

</mujoco>

"""

        individual_builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        # Use geometry-based inertia since MJCF-defined values are unrealistic (20x too high)
        individual_builder.add_mjcf(mjcf, ignore_inertial_definitions=True)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for _i in range(0, 2):
            builder.add_world(individual_builder)
        model = builder.finalize()
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(state_in, contacts)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        solver = SolverMuJoCo(model, iterations=10, ls_iterations=10)

        dt = 0.02

        coeff0 = 1.0  # from mjcf above
        coeff1 = 1.0  # from mjcf above
        expected_tendon_length = 0.0  # from mjcf above

        # Length of tendon at start is: pos*coef0 + pos1*coef1 = 1*0.5 + 1*0.0 = 0.5
        # Target length is 0.0 (see mjcf above)
        joint_start_positions = [0.5, 0.0, 0.5, 0.0]
        state_in.joint_q.assign(joint_start_positions)

        device = model.device
        use_graph = is_graph_capture_allocation_enabled(device)
        if use_graph:
            # warmup (2 steps for full ping-pong cycle)
            solver.step(state_in=state_in, state_out=state_out, contacts=contacts, control=control, dt=dt)
            solver.step(state_in=state_out, state_out=state_in, contacts=contacts, control=control, dt=dt)
            with wp.ScopedCapture(device) as capture:
                solver.step(state_in=state_in, state_out=state_out, contacts=contacts, control=control, dt=dt)
                solver.step(state_in=state_out, state_out=state_in, contacts=contacts, control=control, dt=dt)
            graph = capture.graph

        remaining = 200 - (4 if use_graph else 0)
        for _i in range(remaining // 2 if use_graph else remaining):
            if use_graph:
                wp.capture_launch(graph)
            else:
                solver.step(state_in=state_in, state_out=state_out, contacts=contacts, control=control, dt=dt)
                state_in, state_out = state_out, state_in
        if use_graph and remaining % 2 == 1:
            solver.step(state_in=state_in, state_out=state_out, contacts=contacts, control=control, dt=dt)
            state_in, state_out = state_out, state_in

        # World 0 should have achieved the rest length of the tendon.
        joint_q = state_in.joint_q.numpy()
        q0 = joint_q[0]
        q1 = joint_q[1]
        measured_tendon_length = coeff0 * q0 + coeff1 * q1
        self.assertAlmostEqual(
            expected_tendon_length,
            measured_tendon_length,
            places=3,
            msg=f"Expected tendon length: {expected_tendon_length}, Measured tendon length: {measured_tendon_length}",
        )

        # World 1 and world 0 should have identical state.
        q2 = joint_q[2]
        self.assertAlmostEqual(
            q2,
            q0,
            places=3,
            msg=f"Expected joint_q[2]: {q0}, Measured q2: {q2}",
        )
        q3 = joint_q[3]
        self.assertAlmostEqual(
            q3,
            q1,
            places=3,
            msg=f"Expected joint_q[3]: {q1}, Measured q3: {q3}",
        )

    def run_test_mujoco_fixed_tendon_limit_behavior(self, mode: LimitBreachType):
        """Test that tendons limits are respected"""
        mjcf = """<?xml version="1.0" ?>
<mujoco model="two_prismatic_links">
  <compiler angle="degree"/>

  <option timestep="0.002" gravity="0 0 0"/>

  <worldbody>
    <!-- Root body (fixed to world) -->
    <body name="root" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>

      <!-- First child link with prismatic joint along x -->
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom solmix="1.0" type="cylinder" size="0.05 0.025" rgba="1 0 0 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>

      <!-- Second child link with prismatic joint along x -->
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom type="cylinder" size="0.05 0.025" rgba="0 0 1 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>

  <tendon>
    <!-- Fixed tendon coupling joint1 and joint2 -->
	<fixed
		name="coupling_tendon"
    range = "-1.0 1.0"
		stiffness="0.0"
		damping="0.0"
        solreflimit="0.004 1"
        solimplimit="0.95 0.99 0.001"
		springlength="0.0">
      <joint joint="joint1" coef="1"/>
      <joint joint="joint2" coef="1"/>
    </fixed>
  </tendon>

</mujoco>

"""
        coeff0 = 1.0  # from mjcf above
        coeff1 = 1.0  # from mjcf above
        lower_limit = -1.0  # from mjcf above
        upper_limit = 1.0  # from mjcf above

        # Configure the start state of world 0
        joint_start_positions = [0.0, 0.0, 0.0, 0.0]
        joint_start_velocities = [0.0, 0.0, 0.0, 0.0]
        if mode is self.LimitBreachType.UPPER_LIMIT_FROM_ABOVE:
            joint_start_positions[0] = upper_limit + 0.1
        elif mode is self.LimitBreachType.UPPER_LIMIT_FROM_BELOW:
            joint_start_positions[0] = upper_limit - 0.1
            joint_start_velocities[0] = 1.0
        elif mode is self.LimitBreachType.LOWER_LIMIT_FROM_BELOW:
            joint_start_positions[0] = lower_limit - 0.1
        elif mode is self.LimitBreachType.LOWER_LIMIT_FROM_ABOVE:
            joint_start_positions[0] = lower_limit + 0.1
            joint_start_velocities[0] = -1.0

        # Configure the start state of world 1 to be identical
        # to that of world 0.
        joint_start_positions[2] = joint_start_positions[0]
        joint_start_velocities[2] = joint_start_velocities[0]

        individual_builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        # Use geometry-based inertia since MJCF-defined values are unrealistic (20x too high)
        individual_builder.add_mjcf(mjcf, ignore_inertial_definitions=True)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for _i in range(0, 2):
            builder.add_world(individual_builder)
        model = builder.finalize()
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(state_in, contacts)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        solver = SolverMuJoCo(model, iterations=10, ls_iterations=10)

        dt = 0.02

        state_in.joint_q.assign(joint_start_positions)
        state_in.joint_qd.assign(joint_start_velocities)

        for _i in range(0, 20):
            solver.step(state_in=state_in, state_out=state_out, contacts=contacts, control=control, dt=dt)
            state_in, state_out = state_out, state_in

        joint_q = state_in.joint_q.numpy()

        # Test that the limits are observed in world 0.
        q0 = joint_q[0]
        q1 = joint_q[1]
        measured_tendon_length = coeff0 * q0 + coeff1 * q1
        has_legal_length = measured_tendon_length > lower_limit and measured_tendon_length < upper_limit
        self.assertTrue(
            has_legal_length,
            f"Allowed range is {lower_limit} to {upper_limit}. measured length is {measured_tendon_length}",
        )

        # World 1 and world 0 should have identical state.
        q2 = joint_q[2]
        self.assertAlmostEqual(
            q2,
            q0,
            places=3,
            msg=f"Expected joint_q[2]: {q0}, Measured q2: {q2}",
        )
        q3 = joint_q[3]
        self.assertAlmostEqual(
            q3,
            q1,
            places=3,
            msg=f"Expected joint_q[3]: {q1}, Measured q3: {q3}",
        )

    def test_upper_tendon_limit_breach_from_above(self):
        self.run_test_mujoco_fixed_tendon_limit_behavior(self.LimitBreachType.UPPER_LIMIT_FROM_ABOVE)

    def test_upper_tendon_limit_breach_from_below(self):
        self.run_test_mujoco_fixed_tendon_limit_behavior(self.LimitBreachType.UPPER_LIMIT_FROM_BELOW)

    def test_lower_tendon_limit_breach_from_below(self):
        self.run_test_mujoco_fixed_tendon_limit_behavior(self.LimitBreachType.LOWER_LIMIT_FROM_BELOW)

    def test_lower_tendon_limit_breach_from_above(self):
        self.run_test_mujoco_fixed_tendon_limit_behavior(self.LimitBreachType.LOWER_LIMIT_FROM_ABOVE)

    def test_mjcf_fixed_tendon_hyphenated_and_combined_joints(self):
        """Fixed tendons resolve hyphenated MJCF joint names and individual DOFs of merged joints."""
        import mujoco

        mjcf = """<mujoco model="combined_fixed_tendon">
          <option timestep="0.002" gravity="0 0 -9.81" integrator="Euler"/>
          <worldbody>
            <body name="finger" pos="0 0 0.5">
              <joint name="mcp-flex" type="hinge" axis="0 1 0" damping="0.1"/>
              <joint name="mcp_abduction" type="hinge" axis="1 0 0" damping="0.1"/>
              <geom type="capsule" size="0.03 0.1" mass="0.5"/>
              <body name="distal" pos="0 0 -0.2">
                <joint name="dip_slide" type="slide" axis="0 0 1" damping="0.2"/>
                <joint name="dip-hinge" type="hinge" axis="0 1 0" damping="0.1"/>
                <geom type="capsule" size="0.025 0.08" mass="0.3"/>
              </body>
            </body>
          </worldbody>
          <tendon>
            <fixed name="synergy-tendon" stiffness="2.0" damping="0.5" springlength="0.0">
              <joint joint="mcp_abduction" coef="-0.5"/>
              <joint joint="mcp-flex" coef="1.2"/>
              <joint joint="dip-hinge" coef="-0.8"/>
              <joint joint="dip_slide" coef="2.0"/>
            </fixed>
          </tendon>
          <actuator>
            <motor name="synergy-drive" tendon="synergy-tendon" gear="1.5"/>
          </actuator>
        </mujoco>"""

        ref_model = mujoco.MjModel.from_xml_string(mjcf)
        ref_data = mujoco.MjData(ref_model)
        q0 = [0.25, -0.15, 0.04, -0.35]
        ref_data.qpos[:] = q0
        ref_data.ctrl[:] = [0.6]
        for _ in range(20):
            mujoco.mj_step(ref_model, ref_data)

        for use_mujoco_cpu in (True, False):
            world_count = 1 if use_mujoco_cpu else 2
            with self.subTest(use_mujoco_cpu=use_mujoco_cpu):
                blueprint = newton.ModelBuilder()
                blueprint.add_mjcf(mjcf, ctrl_direct=True, verbose=use_mujoco_cpu)
                builder = newton.ModelBuilder()
                builder.replicate(blueprint, world_count)
                model = builder.finalize(device="cpu" if use_mujoco_cpu else None)

                solver = SolverMuJoCo(
                    model,
                    use_mujoco_cpu=use_mujoco_cpu,
                    separate_worlds=True,
                    disable_contacts=True,
                    kinematic_switchable_roots=False,
                )
                self.assertEqual(solver.mj_model.ntendon, 1)
                self.assertEqual(solver.mj_model.nwrap, 4)

                state_in = model.state()
                state_out = model.state()
                control = model.control()
                state_in.joint_q.assign(q0 * world_count)
                newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
                control.mujoco.ctrl.assign([0.6] * world_count)

                dt = float(ref_model.opt.timestep)
                for _ in range(20):
                    solver.step(state_in=state_in, state_out=state_out, control=control, contacts=None, dt=dt)
                    state_in, state_out = state_out, state_in

                q_np = state_in.joint_q.numpy().reshape(world_count, 4)
                for w in range(world_count):
                    np.testing.assert_allclose(q_np[w], ref_data.qpos, atol=1e-4, rtol=1e-4)

        # Distinct MJCF names that sanitize identically ('a-b' vs 'a_b') resolve to distinct DOFs.
        collision_mjcf = """<mujoco>
          <worldbody>
            <body name="b">
              <joint name="a-b" type="hinge" axis="1 0 0"/>
              <joint name="a_b" type="hinge" axis="0 1 0"/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
          <tendon>
            <fixed name="t">
              <joint joint="a-b" coef="1.0"/>
              <joint joint="a_b" coef="-2.0"/>
            </fixed>
          </tendon>
        </mujoco>"""
        collision_builder = newton.ModelBuilder()
        collision_builder.add_mjcf(collision_mjcf)
        collision_solver = SolverMuJoCo(
            collision_builder.finalize(device="cpu"), use_mujoco_cpu=True, kinematic_switchable_roots=False
        )
        np.testing.assert_array_equal(collision_solver.mj_model.wrap_objid, [0, 1])
        np.testing.assert_allclose(collision_solver.mj_model.wrap_prm, [1.0, -2.0])

        # Equivalent 1-DOF targets from MJCF (explicit dof=0) and add_tendon_fixed (dof=-1) pass validation,
        # whereas differing coefficients across worlds raise ValueError.
        one_hinge_mjcf = """<mujoco>
          <worldbody>
            <body name="b"><joint name="h" type="hinge" axis="0 1 0"/><geom type="sphere" size="0.1"/></body>
          </worldbody>
          <tendon><fixed name="t"><joint joint="h" coef="1.5"/></fixed></tendon>
        </mujoco>"""
        mixed_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(mixed_builder)
        mixed_builder.begin_world()
        mixed_builder.add_mjcf(one_hinge_mjcf)
        mixed_builder.end_world()
        mixed_builder.begin_world()
        link = mixed_builder.add_link(mass=1.0)
        mixed_builder.add_shape_sphere(link, radius=0.1)
        rev = mixed_builder.add_joint_revolute(parent=-1, child=link, axis=wp.vec3(0.0, 1.0, 0.0))
        mixed_builder.add_articulation([rev])
        mjc_author.add_tendon_fixed(mixed_builder, [(rev, 1.5)], label="t")
        mixed_builder.end_world()
        mixed_solver = SolverMuJoCo(
            mixed_builder.finalize(device="cpu"),
            use_mujoco_cpu=False,
            separate_worlds=True,
            disable_contacts=True,
            kinematic_switchable_roots=False,
        )
        self.assertEqual(mixed_solver.mj_model.ntendon, 1)

        mixed_builder.custom_attributes["mujoco:tendon_coef"].values[1] = -1.5
        with self.assertRaisesRegex(ValueError, "homogeneous worlds"):
            SolverMuJoCo(
                mixed_builder.finalize(device="cpu"),
                use_mujoco_cpu=False,
                separate_worlds=True,
                disable_contacts=True,
                kinematic_switchable_roots=False,
            )

        # Corrupted DOF selector pointing outside its joint raises ValueError.
        bad_builder = newton.ModelBuilder()
        bad_builder.add_mjcf(mjcf, ctrl_direct=True)
        bad_builder.custom_attributes["mujoco:tendon_joint_dof"].values[0] = 3
        bad_model = bad_builder.finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "does not belong to joint"):
            SolverMuJoCo(bad_model, use_mujoco_cpu=True, kinematic_switchable_roots=False)

        # Heterogeneous per-world DOF selectors or shifted tendon boundaries raise ValueError.
        two_tendon_mjcf = """<mujoco>
          <worldbody>
            <body name="b">
              <joint name="rx" type="hinge" axis="1 0 0"/>
              <joint name="ry" type="hinge" axis="0 1 0"/>
              <joint name="rz" type="hinge" axis="0 0 1"/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
          <tendon>
            <fixed name="t0"><joint joint="rx" coef="1.0"/><joint joint="ry" coef="1.0"/></fixed>
            <fixed name="t1"><joint joint="rz" coef="1.0"/></fixed>
          </tendon>
        </mujoco>"""
        boundary_blueprint = newton.ModelBuilder()
        boundary_blueprint.add_mjcf(two_tendon_mjcf)
        boundary_builder = newton.ModelBuilder()
        boundary_builder.replicate(boundary_blueprint, 2)
        # Keep the flattened wrap sequence identical in world 1 ([3, 4, 5]), but change boundaries to [1, 2].
        boundary_builder.custom_attributes["mujoco:tendon_joint_num"].values[2] = 1
        boundary_builder.custom_attributes["mujoco:tendon_joint_adr"].values[3] = 4
        boundary_builder.custom_attributes["mujoco:tendon_joint_num"].values[3] = 2
        with self.assertRaisesRegex(ValueError, "homogeneous worlds"):
            SolverMuJoCo(
                boundary_builder.finalize(device="cpu"),
                use_mujoco_cpu=False,
                separate_worlds=True,
                disable_contacts=True,
                kinematic_switchable_roots=False,
            )

        # Converted 3-hinge ball joints raise ValueError rather than silently exporting a truncated tendon.
        ball_mjcf = """<mujoco>
          <worldbody>
            <body name="b">
              <joint name="rx" type="hinge" axis="1 0 0"/>
              <joint name="ry" type="hinge" axis="0 1 0"/>
              <joint name="rz" type="hinge" axis="0 0 1"/>
              <geom type="sphere" size="0.1"/>
              <body name="c" pos="0 0 0.2">
                <joint name="child_h" type="hinge" axis="0 1 0"/>
                <geom type="sphere" size="0.05"/>
              </body>
            </body>
          </worldbody>
          <tendon>
            <fixed name="t">
              <joint joint="ry" coef="1.0"/>
              <joint joint="child_h" coef="0.5"/>
            </fixed>
          </tendon>
        </mujoco>"""
        ball_builder = newton.ModelBuilder()
        ball_builder.add_mjcf(ball_mjcf, convert_3d_hinge_to_ball_joints=True)
        ball_model = ball_builder.finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "convert_3d_hinge_to_ball_joints"):
            SolverMuJoCo(ball_model, use_mujoco_cpu=True, kinematic_switchable_roots=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
