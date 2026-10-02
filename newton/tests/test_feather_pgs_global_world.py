# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Global (world ``-1``) bodies in multi-world SolverFeatherPGS models."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

_DT = 0.01


def _floor_model(device, floor: str, kinematic_world0=False):
    """Two worlds with one unit box each on a floor box whose top is at z = 0.5.

    ``floor`` is ``"static"`` (world geometry), ``"kinematic"`` (a global kinematic body)
    or ``"dynamic"`` (a global dynamic body).
    """
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for x in (-2.0, 2.0):
        world = newton.ModelBuilder(gravity=wp.vec3(0.0))
        kinematic = kinematic_world0 and builder.world_count == 0
        body = world.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, 0.999), wp.quat_identity()), mass=1.0, is_kinematic=kinematic
        )
        world.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5, cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
        builder.add_world(world)
    floor_body = -1 if floor == "static" else builder.add_body(is_kinematic=floor == "kinematic", mass=1.0)
    builder.add_shape_box(floor_body, hx=5.0, hy=5.0, hz=0.5, cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
    return builder.finalize(device=device)


def _step_once(model, solver, box_velocity=(-1.0, -1.0), floor_velocity=0.0):
    state, output = model.state(), model.state()
    joint_qd = state.joint_qd.numpy()
    joint_qd[2], joint_qd[8] = box_velocity
    if joint_qd.size > 12:
        joint_qd[14] = floor_velocity
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    solver.step(state, output, model.control(), contacts, _DT)
    return output.body_qd.numpy()[:, 2], contacts


def _construct(test, model, expect_warning, **kwargs):
    """Construct the solver and check whether it warns about unsolvable global contacts."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = SolverFeatherPGS(model, **kwargs)
    messages = [str(w.message) for w in caught if "cannot be solved" in str(w.message)]
    test.assertEqual(len(messages), 1 if expect_warning else 0, msg=messages)
    if expect_warning:
        test.assertIn("global (world -1) articulations", messages[0])
    return solver


def _modes(device):
    return ["matrix_free", "split"] if device.is_cuda else ["split"]


class TestFeatherPGSGlobalWorld(unittest.TestCase):
    def test_global_kinematic_floor_supports_every_world(self):
        """A global kinematic floor stops the boxes of every world exactly like world geometry does."""
        device = wp.get_device()
        for mode in ["matrix_free"] if device.is_cuda else []:
            with self.subTest(mode=mode):
                static_model = _floor_model(device, "static")
                static_v = _step_once(static_model, SolverFeatherPGS(static_model, pgs_mode=mode))[0][:2]
                self.assertLess(np.max(np.abs(static_v)), 0.05)

                model = _floor_model(device, "kinematic")
                solver = SolverFeatherPGS(model, pgs_mode=mode, warn_constraint_overflow=False)
                v, contacts = _step_once(model, solver)
                count = int(contacts.rigid_contact_count.numpy()[0])
                self.assertGreater(count, 0)
                np.testing.assert_array_equal(solver.contact_path.numpy()[:count] >= 0, True)
                np.testing.assert_allclose(v[:2], static_v, atol=1.0e-5)
                np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [False, False])

                moving_v = _step_once(model, SolverFeatherPGS(model, pgs_mode=mode), floor_velocity=0.5)[0][:2]
                np.testing.assert_allclose(moving_v, static_v + 0.5, atol=1.0e-4)

                # The floor is elided even when world 0, where it is stored, has no dynamic body.
                model = _floor_model(device, "kinematic", kinematic_world0=True)
                solver = SolverFeatherPGS(model, pgs_mode=mode, warn_constraint_overflow=False)
                v, _ = _step_once(model, solver, box_velocity=(0.0, -1.0))
                np.testing.assert_allclose(v[1], static_v[1], atol=1.0e-5)
                np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [False, False])

    def test_split_global_kinematic_floor_flags_other_worlds(self):
        """Without prescribed-response elision, other worlds' contacts with a global kinematic body are flagged."""
        device = wp.get_device()
        model = _floor_model(device, "kinematic")
        solver = SolverFeatherPGS(model, pgs_mode="split", warn_constraint_overflow=False)
        _, contacts = _step_once(model, solver)
        count = int(contacts.rigid_contact_count.numpy()[0])
        paths = solver.contact_path.numpy()[:count]
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        world1 = (shape0 == 1) | (shape1 == 1)
        self.assertTrue(np.any(world1))
        np.testing.assert_array_equal(paths[~world1] >= 0, True)
        np.testing.assert_array_equal(paths[world1], -1)
        np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [True, True])

    def test_dynamic_global_body_flags_other_world_contacts(self):
        """Contacts between a dynamic global body and another world are dropped and flagged, not ignored."""
        device = wp.get_device()
        for mode in _modes(device):
            with self.subTest(mode=mode):
                model = _floor_model(device, "dynamic")
                joint_q = model.joint_q.numpy()
                joint_q[2] += 3.0  # lift world 0's box out of contact
                model.joint_q.assign(joint_q)
                solver = SolverFeatherPGS(model, pgs_mode=mode, warn_constraint_overflow=False)
                _, contacts = _step_once(model, solver)
                count = int(contacts.rigid_contact_count.numpy()[0])
                self.assertGreater(count, 0)
                np.testing.assert_array_equal(solver.contact_path.numpy()[:count], -1)
                # The global body is solved in world 0, so both worlds are flagged.
                np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [True, True])

    def test_construction_warns_about_unsolvable_global_contacts(self):
        """The constructor warns once when shapes allow contacts that couple a global articulation with another world."""
        device = wp.get_device()
        for mode in _modes(device):
            with self.subTest(mode=mode):
                # A dynamic global body always keeps response DOFs in world 0.
                _construct(self, _floor_model(device, "dynamic"), True, pgs_mode=mode)
                # A global kinematic free body is prescribed only on the matrix_free/immediate path.
                _construct(self, _floor_model(device, "kinematic"), mode != "matrix_free", pgs_mode=mode)
                # World geometry is solvable in every world.
                _construct(self, _floor_model(device, "static"), False, pgs_mode=mode)

                # One world, or global shapes that cannot collide with the worlds' bodies, never warn.
                builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
                body = builder.add_body(mass=1.0)
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
                builder.add_ground_plane()
                _construct(self, builder.finalize(device=device), False, pgs_mode=mode)
                model = _floor_model(device, "dynamic")
                groups = model.shape_collision_group.numpy()
                groups[model.shape_body.numpy() == 2] = 7
                model.shape_collision_group.assign(groups)
                _construct(self, model, False, pgs_mode=mode)

        if device.is_cuda:
            # A kinematic global articulation with joints keeps response DOFs on every path.
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
            for x in (-2.0, 2.0):
                world = newton.ModelBuilder(gravity=wp.vec3(0.0))
                body = world.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.999), wp.quat_identity()), mass=1.0)
                world.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
                builder.add_world(world)
            floor = builder.add_link(mass=1.0, is_kinematic=True)
            builder.add_articulation([builder.add_joint_prismatic(-1, floor, axis=newton.Axis.Z)])
            builder.add_shape_box(floor, hx=5.0, hy=5.0, hz=0.5)
            _construct(self, builder.finalize(device=device), True, pgs_mode="matrix_free")


if __name__ == "__main__":
    unittest.main(verbosity=2)
