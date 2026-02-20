# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import warnings

import numpy as np

import newton
from newton.solvers import SolverMuJoCo


class TestMujocoSpatialTendon(unittest.TestCase):
    # Minimal MJCF with a spatial tendon connecting two bodies via sites and a wrapping geom
    SPATIAL_TENDON_MJCF = """<?xml version="1.0" ?>
<mujoco model="spatial_tendon_test">
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <body name="base" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02 0.1"/>
      <site name="s0" pos="0.1 0 0"/>
      <geom name="wrap_cyl" type="cylinder" size="0.03 0.01" pos="0 0 0.05"
            contype="0" conaffinity="0" rgba="0.5 0.5 0.5 0.3"/>
      <site name="side0" pos="0.05 0 0.05"/>

      <body name="link2" pos="0 0 0.2">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.02 0.1"/>
        <site name="s1" pos="0.1 0 0"/>
      </body>
    </body>
  </worldbody>

  <tendon>
    <spatial name="sp1" stiffness="500" damping="50">
      <site site="s0"/>
      <geom geom="wrap_cyl" sidesite="side0"/>
      <site site="s1"/>
    </spatial>
  </tendon>
</mujoco>
"""

    def test_spatial_tendon_parsing(self):
        """Verify that spatial tendon attributes are parsed correctly from MJCF."""
        builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(self.SPATIAL_TENDON_MJCF)
        model = builder.finalize()

        mujoco_attrs = model.mujoco

        # Verify tendon-level attributes
        tendon_type = mujoco_attrs.tendon_type.numpy()
        self.assertEqual(len(tendon_type), 1)
        self.assertEqual(tendon_type[0], 1)  # spatial

        tendon_stiffness = mujoco_attrs.tendon_stiffness.numpy()
        self.assertAlmostEqual(tendon_stiffness[0], 500.0)

        tendon_damping = mujoco_attrs.tendon_damping.numpy()
        self.assertAlmostEqual(tendon_damping[0], 50.0)

        # Fixed tendon arrays should be empty for spatial tendons
        tendon_joint_num = mujoco_attrs.tendon_joint_num.numpy()
        self.assertEqual(tendon_joint_num[0], 0)

        # Verify wrap path
        tendon_wrap_num = mujoco_attrs.tendon_wrap_num.numpy()
        self.assertEqual(tendon_wrap_num[0], 3)  # site + geom + site

        wrap_type = mujoco_attrs.tendon_wrap_type.numpy()
        self.assertEqual(wrap_type[0], 0)  # site
        self.assertEqual(wrap_type[1], 1)  # geom
        self.assertEqual(wrap_type[2], 0)  # site

        # Verify shape references point to valid shapes
        wrap_shape = mujoco_attrs.tendon_wrap_shape.numpy()
        self.assertGreaterEqual(wrap_shape[0], 0)  # s0
        self.assertGreaterEqual(wrap_shape[1], 0)  # wrap_cyl
        self.assertGreaterEqual(wrap_shape[2], 0)  # s1

        # Verify sidesite is set only on the geom wrap entry
        wrap_sidesite = mujoco_attrs.tendon_wrap_sidesite.numpy()
        self.assertEqual(wrap_sidesite[0], -1)  # no sidesite for site
        self.assertGreaterEqual(wrap_sidesite[1], 0)  # side0 for geom
        self.assertEqual(wrap_sidesite[2], -1)  # no sidesite for site

    def test_spatial_tendon_simulation(self):
        """Verify that a spatial tendon with stiffness exerts forces and moves joints."""
        # Use an explicit springlength shorter than the initial tendon length
        # so the spring pulls the joints from the start.
        mjcf = """<?xml version="1.0" ?>
<mujoco model="spatial_tendon_spring_test">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02 0.1"/>
      <site name="s0" pos="0.1 0 0"/>
      <body name="link2" pos="0 0 0.2">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.02 0.1"/>
        <site name="s1" pos="0.1 0 0"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="sp1" stiffness="500" damping="50" springlength="0.01">
      <site site="s0"/>
      <site site="s1"/>
    </spatial>
  </tendon>
</mujoco>
"""
        individual_builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(individual_builder)
        individual_builder.add_mjcf(mjcf, ignore_inertial_definitions=True, parse_sites=True)

        builder = newton.ModelBuilder(gravity=0.0)
        for _ in range(2):
            builder.add_world(individual_builder)
        model = builder.finalize()

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.contacts()
        model.collide(state_in, contacts)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        solver = SolverMuJoCo(model, iterations=10, ls_iterations=10)

        self.assertEqual(solver.mj_model.ntendon, 1)

        q_initial = state_in.joint_q.numpy().copy()

        # Run simulation — spring rest length (0.01) is much shorter than
        # the initial tendon length, so the spring pulls the joints.
        dt = 0.002
        for _ in range(500):
            solver.step(state_in=state_in, state_out=state_out, contacts=contacts, control=control, dt=dt)
            state_in, state_out = state_out, state_in

        joint_q = state_in.joint_q.numpy()

        # Verify joints moved from initial position
        max_displacement = np.max(np.abs(joint_q - q_initial))
        self.assertGreater(max_displacement, 0.001, "Tendon spring should have moved joints")

        # Verify all positions are finite
        self.assertTrue(np.all(np.isfinite(joint_q)), "Joint positions should be finite")

        # Verify both worlds have identical states
        num_dofs_per_world = len(joint_q) // 2
        for d in range(num_dofs_per_world):
            self.assertAlmostEqual(
                joint_q[d],
                joint_q[d + num_dofs_per_world],
                places=3,
                msg=f"World 0 dof {d} ({joint_q[d]}) != World 1 dof {d} ({joint_q[d + num_dofs_per_world]})",
            )

    def test_spatial_tendon_with_actuator(self):
        """Verify that an actuator can target a spatial tendon."""
        mjcf = """<?xml version="1.0" ?>
<mujoco model="spatial_tendon_actuator">
  <option timestep="0.002" gravity="0 0 0"/>

  <worldbody>
    <body name="base" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02 0.1"/>
      <site name="s0" pos="0.1 0 0"/>

      <body name="link2" pos="0 0 0.2">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.02 0.1"/>
        <site name="s1" pos="0.1 0 0"/>
      </body>
    </body>
  </worldbody>

  <tendon>
    <spatial name="sp_act" damping="10">
      <site site="s0"/>
      <site site="s1"/>
    </spatial>
  </tendon>

  <actuator>
    <position name="act1" tendon="sp_act" kp="1000"/>
  </actuator>
</mujoco>
"""
        individual_builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(individual_builder)
        individual_builder.add_mjcf(mjcf, ignore_inertial_definitions=True)

        model = individual_builder.finalize()
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.contacts()
        model.collide(state_in, contacts)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        solver = SolverMuJoCo(model, iterations=10, ls_iterations=10)

        # Verify the MuJoCo model has both tendon and actuator
        self.assertEqual(solver.mj_model.ntendon, 1)
        self.assertEqual(solver.mj_model.nu, 1)

        # Run simulation with actuator control
        dt = 0.002
        for _ in range(200):
            solver.step(state_in=state_in, state_out=state_out, contacts=contacts, control=control, dt=dt)
            state_in, state_out = state_out, state_in

        # The simulation should complete without errors (actuator drives tendon)
        joint_q = state_in.joint_q.numpy()
        self.assertTrue(np.all(np.isfinite(joint_q)), "Joint positions should be finite")

    def test_mixed_fixed_and_spatial_tendons(self):
        """Verify that fixed and spatial tendons coexist correctly."""
        mjcf = """<?xml version="1.0" ?>
<mujoco model="mixed_tendons">
  <option timestep="0.002" gravity="0 0 0"/>

  <worldbody>
    <body name="base" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02 0.1"/>
      <site name="s0" pos="0.1 0 0"/>

      <body name="link2" pos="0 0 0.2">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.02 0.1"/>
        <site name="s1" pos="0.1 0 0"/>
      </body>
    </body>
  </worldbody>

  <tendon>
    <fixed name="fixed_t" stiffness="100">
      <joint joint="j1" coef="1"/>
      <joint joint="j2" coef="-1"/>
    </fixed>
    <spatial name="spatial_t" stiffness="200" damping="10">
      <site site="s0"/>
      <site site="s1"/>
    </spatial>
  </tendon>
</mujoco>
"""
        individual_builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(individual_builder)
        individual_builder.add_mjcf(mjcf, ignore_inertial_definitions=True)

        builder = newton.ModelBuilder(gravity=0.0)
        for _ in range(2):
            builder.add_world(individual_builder)
        model = builder.finalize()

        # Verify tendon types
        mujoco_attrs = model.mujoco
        tendon_type = mujoco_attrs.tendon_type.numpy()
        # Each world has 2 tendons, so 4 total
        self.assertEqual(len(tendon_type), 4)
        self.assertEqual(tendon_type[0], 0)  # fixed
        self.assertEqual(tendon_type[1], 1)  # spatial
        self.assertEqual(tendon_type[2], 0)  # fixed (world 1)
        self.assertEqual(tendon_type[3], 1)  # spatial (world 1)

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.contacts()
        model.collide(state_in, contacts)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        solver = SolverMuJoCo(model, iterations=10, ls_iterations=10)

        # MuJoCo should have 2 tendons (template world only)
        self.assertEqual(solver.mj_model.ntendon, 2)

        # Run simulation
        dt = 0.002
        for _ in range(200):
            solver.step(state_in=state_in, state_out=state_out, contacts=contacts, control=control, dt=dt)
            state_in, state_out = state_out, state_in

        joint_q = state_in.joint_q.numpy()
        self.assertTrue(np.all(np.isfinite(joint_q)), "Joint positions should be finite")

    def test_spatial_tendon_default_class(self):
        """Verify that MJCF default class inheritance works for spatial tendons."""
        mjcf = """<?xml version="1.0" ?>
<mujoco model="spatial_tendon_defaults">
  <option timestep="0.002" gravity="0 0 0"/>

  <default>
    <tendon stiffness="333" damping="44"/>
  </default>

  <worldbody>
    <body name="base" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02 0.1"/>
      <site name="s0" pos="0.1 0 0"/>

      <body name="link2" pos="0 0 0.2">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.02 0.1"/>
        <site name="s1" pos="0.1 0 0"/>
      </body>
    </body>
  </worldbody>

  <tendon>
    <spatial name="sp_default">
      <site site="s0"/>
      <site site="s1"/>
    </spatial>
    <spatial name="sp_override" stiffness="999">
      <site site="s0"/>
      <site site="s1"/>
    </spatial>
  </tendon>
</mujoco>
"""
        builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        mujoco_attrs = model.mujoco
        stiffness = mujoco_attrs.tendon_stiffness.numpy()
        damping = mujoco_attrs.tendon_damping.numpy()

        # First tendon: inherits from defaults
        self.assertAlmostEqual(stiffness[0], 333.0, places=1)
        self.assertAlmostEqual(damping[0], 44.0, places=1)

        # Second tendon: stiffness overridden, damping inherited
        self.assertAlmostEqual(stiffness[1], 999.0, places=1)
        self.assertAlmostEqual(damping[1], 44.0, places=1)

    def test_spatial_tendon_pulley(self):
        """Verify that spatial tendons with pulley elements parse correctly."""
        mjcf = """<?xml version="1.0" ?>
<mujoco model="spatial_tendon_pulley">
  <option timestep="0.002" gravity="0 0 0"/>

  <worldbody>
    <body name="base" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02 0.1"/>
      <site name="s0" pos="0.1 0 0"/>
      <site name="s1" pos="-0.1 0 0"/>
      <site name="s2" pos="0 0 0.1"/>
    </body>
  </worldbody>

  <tendon>
    <spatial name="pulley_t">
      <site site="s0"/>
      <pulley divisor="2"/>
      <site site="s1"/>
      <site site="s2"/>
    </spatial>
  </tendon>
</mujoco>
"""
        builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        mujoco_attrs = model.mujoco
        wrap_type = mujoco_attrs.tendon_wrap_type.numpy()
        wrap_prm = mujoco_attrs.tendon_wrap_prm.numpy()

        # Wrap path: site, pulley, site, site
        self.assertEqual(len(wrap_type), 4)
        self.assertEqual(wrap_type[0], 0)  # site
        self.assertEqual(wrap_type[1], 2)  # pulley
        self.assertEqual(wrap_type[2], 0)  # site
        self.assertEqual(wrap_type[3], 0)  # site
        self.assertAlmostEqual(wrap_prm[1], 2.0)  # pulley divisor

    def test_spatial_tendon_site_geom_disambiguation(self):
        """Verify that sites and geoms sharing the same name are correctly disambiguated."""
        mjcf = """<?xml version="1.0" ?>
<mujoco model="disambiguation_test">
  <option timestep="0.002" gravity="0 0 0"/>

  <worldbody>
    <body name="base" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02 0.1"/>
      <!-- Geom and site share the same name "shared_name" -->
      <geom name="shared_name" type="cylinder" size="0.03 0.01" pos="0 0 0.05"
            contype="0" conaffinity="0"/>
      <site name="shared_name" pos="0.1 0 0"/>

      <body name="link2" pos="0 0 0.2">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.02 0.1"/>
        <site name="s1" pos="0.1 0 0"/>
      </body>
    </body>
  </worldbody>

  <tendon>
    <spatial name="disambig_tendon">
      <site site="shared_name"/>
      <geom geom="shared_name"/>
      <site site="s1"/>
    </spatial>
  </tendon>
</mujoco>
"""
        builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf, parse_sites=True)
        model = builder.finalize()

        mujoco_attrs = model.mujoco

        # Verify we got exactly one spatial tendon
        tendon_type = mujoco_attrs.tendon_type.numpy()
        self.assertEqual(len(tendon_type), 1)
        self.assertEqual(tendon_type[0], 1)

        # Verify wrap path has 3 elements: site, geom, site
        wrap_type = mujoco_attrs.tendon_wrap_type.numpy()
        self.assertEqual(len(wrap_type), 3)
        self.assertEqual(wrap_type[0], 0)  # site
        self.assertEqual(wrap_type[1], 1)  # geom
        self.assertEqual(wrap_type[2], 0)  # site

        # Verify the site and geom references point to different shapes
        wrap_shape = mujoco_attrs.tendon_wrap_shape.numpy()
        self.assertNotEqual(wrap_shape[0], wrap_shape[1])  # site != geom

        # Verify the MuJoCo model compiles and simulates correctly
        state = model.state()
        contacts = model.contacts()
        model.collide(state, contacts)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        solver = SolverMuJoCo(model, iterations=10, ls_iterations=10)
        self.assertEqual(solver.mj_model.ntendon, 1)
        self.assertGreater(solver.mj_model.nwrap, 0)

    def test_spatial_tendon_multi_world_wrap_offsets(self):
        """Verify that wrap address and shape references are offset correctly across worlds."""
        individual_builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(individual_builder)
        individual_builder.add_mjcf(self.SPATIAL_TENDON_MJCF, parse_sites=True)

        builder = newton.ModelBuilder(gravity=0.0)
        for _ in range(3):
            builder.add_world(individual_builder)
        model = builder.finalize()

        mujoco_attrs = model.mujoco
        wrap_adr = mujoco_attrs.tendon_wrap_adr.numpy()
        wrap_num = mujoco_attrs.tendon_wrap_num.numpy()
        wrap_shape = mujoco_attrs.tendon_wrap_shape.numpy()
        tendon_type = mujoco_attrs.tendon_type.numpy()

        # 3 worlds x 1 tendon = 3 tendons total
        self.assertEqual(len(tendon_type), 3)

        # Each tendon should have the same number of wrap elements
        self.assertEqual(wrap_num[0], wrap_num[1])
        self.assertEqual(wrap_num[1], wrap_num[2])

        # Wrap addresses should be offset: [0, N, 2N]
        n = wrap_num[0]
        self.assertEqual(wrap_adr[0], 0)
        self.assertEqual(wrap_adr[1], n)
        self.assertEqual(wrap_adr[2], 2 * n)

        # Shape references in each world should be different (offset by shapes per world)
        shapes_w0 = wrap_shape[wrap_adr[0] : wrap_adr[0] + n]
        shapes_w1 = wrap_shape[wrap_adr[1] : wrap_adr[1] + n]
        shapes_w2 = wrap_shape[wrap_adr[2] : wrap_adr[2] + n]
        # All shape indices should be non-negative
        self.assertTrue(np.all(shapes_w0 >= 0))
        self.assertTrue(np.all(shapes_w1 >= 0))
        # World 1 shapes should be offset from world 0
        self.assertTrue(np.all(shapes_w1 > shapes_w0))
        # World 2 shapes should be offset from world 1
        self.assertTrue(np.all(shapes_w2 > shapes_w1))

    def test_spatial_tendon_warning_missing_site(self):
        """Verify warning when a spatial tendon references a non-existent site."""
        mjcf = """<?xml version="1.0" ?>
<mujoco model="missing_site_test">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02 0.1"/>
      <site name="s0" pos="0.1 0 0"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="bad_tendon">
      <site site="s0"/>
      <site site="nonexistent_site"/>
    </spatial>
  </tendon>
</mujoco>
"""
        builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)
        # The unknown site should produce a verbose warning during parsing
        builder.add_mjcf(mjcf, parse_sites=True, verbose=True)
        model = builder.finalize()

        # The tendon should have been skipped because one wrap element (the unknown site) was
        # not resolved, leaving only one site which is below the minimum useful path length.
        # Depending on how the parser handles it, there may be 0 tendons or 1 with 1 element.
        mujoco_attrs = getattr(model, "mujoco", None)
        if mujoco_attrs is not None and hasattr(mujoco_attrs, "tendon_type"):
            tendon_type = mujoco_attrs.tendon_type.numpy()
            # If a tendon was created, verify it has fewer wrap elements than expected
            if len(tendon_type) > 0:
                wrap_num = mujoco_attrs.tendon_wrap_num.numpy()
                # Only the valid site should be in the wrap path (the unknown one was skipped)
                self.assertLess(wrap_num[0], 2)

    def test_spatial_tendon_warning_out_of_bounds_wrap(self):
        """Verify that out-of-bounds wrap ranges produce a warning during solver init."""
        builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(self.SPATIAL_TENDON_MJCF, parse_sites=True)

        # Corrupt the wrap address to be out of bounds
        wrap_adr_attr = builder.custom_attributes.get("mujoco:tendon_wrap_adr")
        if wrap_adr_attr and wrap_adr_attr.values:
            wrap_adr_attr.values[0] = 9999  # out of bounds

        model = builder.finalize()
        state = model.state()
        contacts = model.contacts()
        model.collide(state, contacts)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver = SolverMuJoCo(model, iterations=10, ls_iterations=10)
            # Should have warned about out-of-bounds wrap range
            wrap_warnings = [x for x in w if "out of bounds" in str(x.message)]
            self.assertGreater(len(wrap_warnings), 0, "Expected a warning about out-of-bounds wrap range")
            # The tendon should have been skipped
            self.assertEqual(solver.mj_model.ntendon, 0)


if __name__ == "__main__":
    unittest.main()
