# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exercise contact-torsion lifecycle and row-storage safety across steps."""

import itertools
import unittest

import numpy as np
import warp as wp

import newton
from newton import GeoType
from newton._src.solvers.feather_pgs.contact_torsion import _contact_groups
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_JOINT_TARGET, PGS_CONSTRAINT_TYPE_TORSION
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_contact_torsion import fixture


@unittest.skipUnless(wp.is_cuda_available(), "Contact torsion requires CUDA")
class TestContactTorsionRegressions(unittest.TestCase):
    """Reject invalid storage/lifecycle paths rather than silently welding spin."""

    def test_radius_is_construction_only(self):
        """Reject radius mutation for both compiled feature variants."""
        for radius in (0.0, 0.01):
            _, solver, *_ = fixture(radius, center_only=True)
            for replacement in (0.0, 0.02, -1.0, float("nan")):
                with self.subTest(radius=radius, replacement=replacement), self.assertRaises(AttributeError):
                    solver.contact_torsion_radius = replacement

    def test_previous_friction_rows_are_not_current_rows(self):
        """Reject stale tangent metadata when a later step admits only a normal."""
        _, solver, model, initial, contacts = fixture(0.01, center_only=True)
        self.assertEqual(solver._torsion_stats["rows"], 1)
        solver.contact_friction_gap_threshold = -1.0
        solver.step(initial, model.state(), model.control(), contacts, 0.0025)
        self.assertEqual(solver._torsion_stats["rows"], 0)

    def test_both_tangent_rows_must_belong_to_current_normal(self):
        """Reject a missing or mismatched second tangent, not just the first."""
        for invalid in ("count", "type", "parent"):
            with self.subTest(invalid=invalid):
                _, solver, _model, initial, contacts = fixture(0.01, center_only=True)
                row = solver._torsion_stats["groups"][0]["normal_rows"][0]
                if invalid == "count":
                    counts = solver.constraint_count.numpy()
                    counts[0] = row + 2
                    solver.constraint_count.assign(counts)
                else:
                    array = solver.row_type if invalid == "type" else solver.row_parent
                    values = array.numpy()
                    values[0, row + 2] = -1
                    array.assign(values)
                self.assertEqual(_contact_groups(solver, initial, contacts), [])

    def test_selectors_are_construction_only(self):
        """Reject selector mutation rather than leaving a stale resolved selection."""
        _, solver, *_ = fixture(0.01, center_only=True)
        for name in ("contact_torsion_shape_indices", "contact_torsion_shape_patterns"):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(solver, name, ())

    def test_normal_parent_metadata_is_preserved(self):
        """Keep CONTACT parents available for pooled friction-patch load rings."""
        result, _solver, *_ = fixture(0.01, center_only=True)
        normal = result["row_type"][0, : result["count"][0]] == 0
        self.assertGreater(np.count_nonzero(normal), 0)
        np.testing.assert_array_equal(result["row_parent"][0, : result["count"][0]][normal], -1)

    def test_torsion_uses_cfm_floor(self):
        """Apply the same configured diagonal floor as other dense rows."""
        result, solver, *_ = fixture(0.01, center_only=True, pgs_cfm=0.0007)
        active = result["row_type"] == 7
        self.assertGreater(np.count_nonzero(active), 0)
        np.testing.assert_allclose(solver.row_cfm.numpy()[active], 0.0007)

    def test_torsion_respects_relaxation(self):
        """Apply half the unconstrained spin correction at omega one half."""
        result, solver, *_ = fixture(100.0, center_only=True, pgs_iterations=1, pgs_omega=0.5)
        self.assertEqual(solver._torsion_stats["rows"], 1)
        spin = np.abs(result["v_out"].reshape(2, 6)[:, 5])
        np.testing.assert_allclose(spin, 0.5, atol=1e-4)

    def test_torsion_keeps_joint_velocity_limits(self):
        """Let joint velocity limits have the last word over the spin rows of each sweep."""
        for (iterations, velocity_iterations), preparation, radius in _VELOCITY_LIMIT_CASES:
            with self.subTest(
                iterations=iterations, velocity_iterations=velocity_iterations, preparation=preparation, radius=radius
            ):
                _, solver, model, initial, contacts = fixture(
                    radius,
                    center_only=True,
                    spin=0.0,
                    pgs_iterations=iterations,
                    pgs_velocity_iterations=velocity_iterations,
                    contact_torsion_device=preparation != "host",
                )
                limits = np.full(model.joint_dof_count, np.inf, dtype=np.float32)
                limits[5] = _VELOCITY_LIMIT
                model.joint_velocity_limit.assign(limits)
                qd = _step_spinning_pad(solver, model, initial, contacts, preparation)
                self.assertEqual(_row_count(solver, PGS_CONSTRAINT_TYPE_TORSION), int(radius > 0.0))
                self.assertLessEqual(abs(float(qd[5])), _VELOCITY_LIMIT * (1.0 + 1e-5))
                if radius > 0.0:
                    # The spin row still resists the relative spin.
                    self.assertLess(float(qd[11]), 9.0)

    def test_torsion_keeps_fused_drive_velocity_limits(self):
        """Let the fused drive-row velocity clamp have the last word over the spin rows of each sweep."""
        dofs = 12
        limits = np.full(dofs, np.inf, dtype=np.float32)
        limits[5] = _VELOCITY_LIMIT
        damping = np.zeros(dofs, dtype=np.float32)
        damping[5] = 1.0e-3
        for (iterations, velocity_iterations), preparation, radius in _VELOCITY_LIMIT_CASES:
            with self.subTest(
                iterations=iterations, velocity_iterations=velocity_iterations, preparation=preparation, radius=radius
            ):
                _, solver, model, initial, contacts = fixture(
                    radius,
                    center_only=True,
                    spin=0.0,
                    model_arrays={"joint_velocity_limit": limits, "joint_target_kd": damping},
                    drive_mode="physx_pgs",
                    fuse_joint_velocity_limits=True,
                    pgs_iterations=iterations,
                    pgs_velocity_iterations=velocity_iterations,
                    contact_torsion_device=preparation != "host",
                )
                self.assertTrue(solver.fuse_joint_velocity_limits)
                qd = _step_spinning_pad(solver, model, initial, contacts, preparation)
                self.assertEqual(_row_count(solver, PGS_CONSTRAINT_TYPE_JOINT_TARGET), 1)
                self.assertEqual(_row_count(solver, PGS_CONSTRAINT_TYPE_TORSION), int(radius > 0.0))
                self.assertLessEqual(abs(float(qd[5])), _VELOCITY_LIMIT * (1.0 + 1e-5))
                if radius > 0.0:
                    self.assertLess(float(qd[11]), 9.0)

    def test_torsion_keeps_free_body_velocity_limits(self):
        """Let the free-body velocity-limit rows have the last word over a mixed contact's spin row."""
        for (iterations, velocity_iterations), preparation, radius in _VELOCITY_LIMIT_CASES:
            with self.subTest(
                iterations=iterations, velocity_iterations=velocity_iterations, preparation=preparation, radius=radius
            ):
                # A spinning articulated pad presses on a resting free body whose spin is limited.
                builder = newton.ModelBuilder(gravity=(0, 0, 0))
                SolverFeatherPGS.register_custom_attributes(builder)
                inertia = wp.mat33(np.diag([0.00012, 0.00012, 0.00008]).astype(np.float32))
                shape = newton.ModelBuilder.ShapeConfig(density=0, mu=0.5, restitution=0.0)
                pose = wp.transform(wp.vec3(0, 0, -0.025), wp.quat_identity())
                pad = builder.add_link(xform=pose, mass=0.3, inertia=inertia)
                axes = [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(*a)) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
                builder.add_articulation(
                    [builder.add_joint_d6(-1, pad, parent_xform=pose, linear_axes=axes, angular_axes=axes)]
                )
                builder.add_shape_box(pad, hx=0.02, hy=0.015, hz=0.025, cfg=shape)
                free = builder.add_body(
                    xform=wp.transform(wp.vec3(0, 0, 0.025), wp.quat_identity()),
                    mass=0.3,
                    inertia=inertia,
                    custom_attributes={"rigid_body_max_angular_velocity": _VELOCITY_LIMIT},
                )
                builder.add_shape_box(free, hx=0.02, hy=0.015, hz=0.025, cfg=shape)
                model = builder.finalize(device="cuda:0")
                initial = model.state()
                initial.joint_qd.assign(np.array([0, 0, 0.1, 0, 0, 10.0, 0, 0, -0.1, 0, 0, 0], np.float32))
                newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
                pipeline = newton.CollisionPipeline(
                    model, contact_matching="latest", reduce_contacts=False, broad_phase="nxn", rigid_contact_max=64
                )
                contacts = pipeline.contacts()
                pipeline.collide(initial, contacts)
                # One central witness: any free-body spin comes from the spin row.
                poses = initial.body_q.numpy()
                for side in (0, 1):
                    name = f"rigid_contact_point{side}"
                    body = model.shape_body.numpy()[getattr(contacts, f"rigid_contact_shape{side}").numpy()[0]]
                    points = getattr(contacts, name).numpy()
                    points[0] = -poses[body, :3]
                    getattr(contacts, name).assign(points)
                contacts.rigid_contact_count.assign(np.array([1], np.int32))
                solver = SolverFeatherPGS(
                    model,
                    pgs_mode="matrix_free",
                    articulated_contact_response="immediate",
                    friction_anchor_beta=0.0,
                    pgs_iterations=iterations,
                    pgs_velocity_iterations=velocity_iterations,
                    pgs_beta=0.05,
                    pgs_cfm=0,
                    pgs_contact_regularization=0,
                    pgs_warmstart=False,
                    angular_damping=0,
                    dense_max_constraints=64,
                    mf_max_constraints=16,
                    contact_torsion_radius=radius,
                    contact_torsion_device=preparation != "host",
                )
                qd = _step(solver, model, initial, contacts, preparation)
                self.assertEqual(_row_count(solver, PGS_CONSTRAINT_TYPE_TORSION), int(radius > 0.0))
                self.assertGreater(int(solver.mf_constraint_count.numpy()[0]), 0)
                self.assertLessEqual(abs(float(qd[11])), _VELOCITY_LIMIT * (1.0 + 1e-5))
                if radius > 0.0:
                    self.assertLess(float(qd[5]), 9.0)

    def test_contact_row_loss_fails_without_diagnostics(self):
        """Detect rolled-back dropped contacts even with optional telemetry off."""
        baseline, *_ = fixture(0.0, center_only=True)
        limit = int(baseline["count"][0]) - 1
        with self.assertRaisesRegex(RuntimeError, "[Oo]verflow|[Dd]ropped|capacity"):
            fixture(0.01, center_only=True, row_limit=limit, row_watermark=False, warn_constraint_overflow=False)

    def test_selected_unsupported_shape_fails_at_construction(self):
        """Reject explicitly selected unsupported geometry instead of ignoring it."""
        _, _solver, model, *_ = fixture(0.0, center_only=True)
        types = model.shape_type.numpy()
        types[0] = int(GeoType.MESH)
        model.shape_type.assign(types)
        with self.assertRaisesRegex(ValueError, "[Uu]nsupported.*shape|shape.*[Uu]nsupported"):
            SolverFeatherPGS(
                model, pgs_mode="matrix_free", contact_torsion_radius=0.01, contact_torsion_shape_indices=(0,)
            )

    def test_hydro_contact_input_is_rejected(self):
        """Reject actual positive hydro contact stiffness, not a synthetic solver flag."""
        _, solver, model, initial, contacts = fixture(0.01, center_only=True)
        contacts.rigid_contact_stiffness = wp.ones(contacts.rigid_contact_max, device=model.device)
        with self.assertRaisesRegex(ValueError, "hydroelastic"):
            solver.step(initial, model.state(), model.control(), contacts, 0.0025)


_VELOCITY_LIMIT = 0.1
_VELOCITY_LIMIT_CASES = tuple(itertools.product(((1, 0), (4, 0), (1, 2)), ("host", "device", "graph"), (0.0, 1.0)))


def _step_spinning_pad(solver, model, initial, contacts, preparation):
    """Spin only the second pad, so any DOF 5 motion is transferred through the contact."""
    velocity = initial.joint_qd.numpy()
    velocity[5], velocity[11] = 0.0, 10.0
    initial.joint_qd.assign(velocity)
    newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
    return _step(solver, model, initial, contacts, preparation)


def _step(solver, model, initial, contacts, preparation):
    """Step once with host, device or graph-captured torsion preparation and return joint_qd."""
    output = model.state()
    if preparation == "graph":
        solver.prepare_contact_torsion_capture(initial, output)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(initial, output, model.control(), contacts, 0.0025)
        wp.capture_launch(capture.graph)
        solver.validate_contact_torsion()
    else:
        solver.step(initial, output, model.control(), contacts, 0.0025)
    return output.joint_qd.numpy()


def _row_count(solver, row_type):
    count = int(solver.constraint_count.numpy()[0])
    return int(np.count_nonzero(solver.row_type.numpy()[0, :count] == row_type))


if __name__ == "__main__":
    unittest.main()
