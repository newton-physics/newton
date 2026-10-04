# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the parallel tree schedule of SolverFeatherPGS and its execution against serial traversal."""

import inspect
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.solver_feather_pgs import _FeatherPGSTreePlan
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_leaf_publication import (
    _build_branched_model,
    _build_contact_model,
    _build_model,
    _dynamics_fields,
    _poison,
    _run_stage1,
    _solver,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def _build_fingers(device="cpu", *, articulations=3, include_chain=False, ragged=False):
    """Build four four-joint fingers per moving palm, plus an optional serial tree."""
    builder = newton.ModelBuilder(gravity=(0.7, -1.2, -9.1))
    inertia = wp.mat33(0.3, 0.02, 0.01, 0.02, 0.4, 0.03, 0.01, 0.03, 0.5)
    for articulation in range(articulations + int(include_chain)):
        root = builder.add_link(mass=1.0, inertia=inertia, com=wp.vec3(0.03, -0.02, 0.01))
        joints = [
            builder.add_joint_revolute(
                parent=-1,
                child=root,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(2.0 * articulation, -0.3, 0.8), wp.quat_rpy(0.2, -0.1, 0.3)),
            )
        ]
        fingers = 1 if articulation == articulations else 4
        for finger in range(fingers):
            parent = root
            length = finger + 1 if ragged and articulation == 1 else 4
            for level in range(length):
                child = builder.add_link(mass=0.5, inertia=inertia, com=wp.vec3(0.02, 0.01, -0.03))
                joints.append(
                    builder.add_joint_revolute(
                        parent=parent,
                        child=child,
                        axis=newton.Axis.Y if level % 2 else newton.Axis.X,
                        parent_xform=wp.transform(wp.vec3(0.05, 0.04 * finger, 0.08), wp.quat_rpy(0.1, -0.2, 0.05)),
                        child_xform=wp.transform(wp.vec3(-0.01, 0.02, 0.03), wp.quat_identity()),
                    )
                )
                parent = child
            if ragged and articulation == 1 and finger == 1:
                for branch in range(2):
                    child = builder.add_link(mass=0.6, inertia=inertia)
                    joints.append(
                        builder.add_joint_revolute(
                            parent=parent,
                            child=child,
                            axis=newton.Axis.X,
                            parent_xform=wp.transform(wp.vec3(0.1, 0.1 * branch, 0.2), wp.quat_identity()),
                        )
                    )
        if ragged and articulation == 2:
            parent = -1
            for _level in range(3):
                child = builder.add_link(mass=0.8, inertia=inertia, com=wp.vec3(0.02, -0.03, 0.04))
                joints.append(
                    builder.add_joint_revolute(
                        parent=parent,
                        child=child,
                        axis=newton.Axis.Y,
                        parent_xform=wp.transform(wp.vec3(0.2, -0.1, 0.3), wp.quat_identity()),
                    )
                )
                parent = child
        builder.add_articulation(joints)
    model = builder.finalize(device=device)
    model.joint_q.assign(np.linspace(-0.3, 0.4, model.joint_coord_count, dtype=np.float32))
    model.joint_qd.assign(np.linspace(-0.7, 0.8, model.joint_dof_count, dtype=np.float32))
    return model


def _tree_ends(model):
    return model.articulation_start.numpy()[1:]


class TestFeatherPGSTreePlan(unittest.TestCase):
    def test_parallel_tree_is_keyword_only_and_defaults_to_serial(self):
        """Require an explicit keyword-only opt-in."""
        parameters = inspect.signature(SolverFeatherPGS).parameters
        self.assertIn("parallel_tree", parameters)
        self.assertIs(parameters["parallel_tree"].default, False)
        self.assertEqual(parameters["parallel_tree"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_complete_finger_levels_and_serial_group(self):
        """Schedule every finger segment, parents before children, and keep serial trees in one-lane groups."""
        model = _build_fingers(include_chain=True)
        plan = _FeatherPGSTreePlan.build(model, _tree_ends(model))
        self.assertIsNotNone(plan)
        self.assertTrue(any(group.lanes > 1 for group in plan.groups))
        self.assertTrue(any(group.lanes == 1 for group in plan.groups))
        seen = []
        parents = model.joint_parent.numpy()
        children = model.joint_child.numpy()
        starts = model.articulation_start.numpy()
        for group in plan.groups:
            offsets = group.level_offsets.numpy()
            segment_offsets = group.segment_offsets.numpy()
            joints = group.segment_joints.numpy()
            child_offsets = group.child_offsets.numpy()
            child_segments = group.child_segments.numpy()
            for row, articulation in enumerate(group.articulations.numpy()):
                visited = set()
                for level in range(group.max_levels):
                    previous_levels = visited.copy()
                    for segment in range(offsets[row, level], offsets[row, level + 1]):
                        segment_joints = joints[segment_offsets[segment] : segment_offsets[segment + 1]]
                        first_parent = int(parents[segment_joints[0]])
                        if first_parent >= 0:
                            self.assertIn(first_parent, previous_levels)
                        for joint in segment_joints:
                            if parents[joint] >= 0:
                                self.assertIn(int(parents[joint]), visited)
                            visited.add(int(children[joint]))
                            seen.append(int(joint))
                        child_ids = child_segments[child_offsets[segment] : child_offsets[segment + 1]]
                        child_starts = [int(joints[segment_offsets[child]]) for child in child_ids]
                        self.assertEqual(child_starts, sorted(child_starts, reverse=True))
                        for child in child_starts:
                            self.assertEqual(int(parents[child]), int(children[segment_joints[-1]]))
                if articulation < 3:
                    self.assertEqual([int(offsets[row, level + 1] - offsets[row, level]) for level in range(2)], [1, 4])
                    first, last = offsets[row, 0], offsets[row, 2]
                    np.testing.assert_array_equal(np.diff(segment_offsets[first : last + 1]), [1, 4, 4, 4, 4])
                    self.assertEqual(len(visited), int(starts[articulation + 1] - starts[articulation]))
        self.assertEqual(sorted(seen), list(range(model.joint_count)))

    def test_comb_retains_joint_levels_when_chain_compression_adds_rounds(self):
        """Keep comb teeth parallel when compressing unary chains would lengthen the schedule."""
        builder = newton.ModelBuilder()
        joints = []
        parent = -1
        for _ in range(4):
            spine = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)))
            joints.append(builder.add_joint_revolute(parent=parent, child=spine, axis=newton.Axis.Z))
            tooth_parent = spine
            for _ in range(4):
                tooth = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)))
                joints.append(builder.add_joint_revolute(parent=tooth_parent, child=tooth, axis=newton.Axis.Y))
                tooth_parent = tooth
            parent = spine
        builder.add_articulation(joints)
        model = builder.finalize(device="cpu")

        # Compression delays the next fork until the whole preceding tooth ends.
        compressed_lengths = [[1], [4, 1], [4, 1], [4, 5]]
        joint_widths = [1, 2, 3, 4, 4, 3, 2, 1]
        compressed_rounds = sum(max(level) for level in compressed_lengths)
        joint_rounds = sum((width + 3) // 4 for width in joint_widths)
        self.assertEqual((compressed_rounds, joint_rounds), (14, 8))

        plan = _FeatherPGSTreePlan.build(model, _tree_ends(model))
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.groups), 1)
        group = plan.groups[0]
        self.assertEqual(group.lanes, 4)
        self.assertEqual(group.max_levels, len(joint_widths))
        np.testing.assert_array_equal(np.diff(group.level_offsets.numpy()[0]), joint_widths)
        np.testing.assert_array_equal(
            np.diff(group.segment_offsets.numpy()), np.ones(model.joint_count, dtype=np.int32)
        )
        self.assertEqual(sorted(group.segment_joints.numpy().tolist()), list(range(model.joint_count)))

    def test_serial_and_unsafe_topologies_fall_back(self):
        """Return no plan for wholly serial, aliased, cross-articulation and loop-trimmed topologies."""
        model, _, _ = _build_model(chain=True)
        self.assertIsNone(_FeatherPGSTreePlan.build(model, _tree_ends(model)))
        for unsafe in ("alias", "foreign_parent", "loop"):
            with self.subTest(unsafe=unsafe):
                model = _build_fingers(articulations=2)
                ends = _tree_ends(model).copy()
                if unsafe == "alias":
                    children = model.joint_child.numpy().copy()
                    children[2] = children[1]
                    model.joint_child.assign(children)
                elif unsafe == "foreign_parent":
                    parents = model.joint_parent.numpy().copy()
                    parents[1] = model.joint_child.numpy()[model.articulation_start.numpy()[1]]
                    model.joint_parent.assign(parents)
                else:
                    ends[0] -= 1
                self.assertIsNone(_FeatherPGSTreePlan.build(model, ends))


def test_serial_selection_skips_tree_setup(test, device):
    """Default and explicit serial execution build no tree plan and no branch scratch."""
    model = _build_fingers(device, articulations=1)
    states = []
    for options in ({}, {"parallel_tree": False}):
        with test.subTest(options=options):
            with patch.object(_FeatherPGSTreePlan, "build", wraps=_FeatherPGSTreePlan.build) as build:
                solver = SolverFeatherPGS(model, **options)
            build.assert_not_called()
            test.assertIsNone(solver._tree_plan)
            test.assertEqual(solver._tree_net_wrenches, ())
            state = model.state()
            solver._prepare_augmented_state(state)
            solver._stage7_update_kinematics(state)
            states.append(state)
    for name in ("body_q", "body_qd"):
        np.testing.assert_array_equal(getattr(states[0], name).numpy(), getattr(states[1], name).numpy())


def test_explicit_parallel_preserves_eligibility_fallback(test, device):
    """Build a plan for branched trees and fall back to serial traversal for an unbranched chain."""
    model = _build_fingers(device, articulations=1)
    with patch.object(_FeatherPGSTreePlan, "build", wraps=_FeatherPGSTreePlan.build) as build:
        solver = SolverFeatherPGS(model, parallel_tree=True)
    build.assert_called_once()
    test.assertIsNotNone(solver._tree_plan)
    test.assertTrue(any(group.lanes > 1 for group in solver._tree_plan.groups))
    test.assertEqual(len(solver._tree_net_wrenches), len(solver._tree_plan.groups))
    model = _build_model(device, chain=True)[0]
    solver = SolverFeatherPGS(model, parallel_tree=True)
    test.assertIsNone(solver._tree_plan)
    test.assertEqual(solver._tree_net_wrenches, ())


def test_backward_torques_with_live_forces_and_passive_terms(test, device):
    """Match serial joint torques and subtree wrenches with external forces, joint forces, drives and damping."""
    for velocity_limits in (False, True):
        with test.subTest(velocity_limits=velocity_limits):
            model = _build_fingers(device, ragged=True)
            model.joint_velocity_limit.fill_(0.05)
            damping = np.linspace(0.1, 0.4, model.joint_dof_count, dtype=np.float32)
            model.joint_damping.assign(damping)
            model.joint_target_ke.assign(np.linspace(0.0, 4.0, model.joint_dof_count, dtype=np.float32))
            model.joint_target_kd.assign(np.linspace(0.5, 0.0, model.joint_dof_count, dtype=np.float32))
            # Hold every finite limit row: these checks cover traversal, not row capacity.
            options = {"velocity_limits": velocity_limits, "dense_max_constraints": 2 * model.joint_dof_count}
            solvers = [_solver(model, enabled=value, **options) for value in (False, True)]
            states = [model.state() for _ in solvers]
            following = [model.state() for _ in solvers]
            controls = [model.control() for _ in solvers]
            seed = np.linspace(-0.4, 0.3, model.joint_dof_count, dtype=np.float32)
            previous_tau = None
            for force_phase in range(2):
                for index, solver in enumerate(solvers):
                    state = states[index]
                    solver._stage1_fk_id(state, solver, following[index])
                    wrench = np.arange(model.body_count * 6, dtype=np.float32).reshape(-1, 6)
                    wrench = (wrench % 11 - 5) * (0.03 + force_phase * 0.04)
                    state.body_f.assign(wrench)
                    controls[index].joint_f.assign(seed * (1.0 + force_phase))
                    solver.joint_tau.fill_(-123.25)
                    for scratch in solver._tree_net_wrenches:
                        scratch.fill_(-123.25)
                    solver._stage1_joint_tau(state, solver, following[index], controls[index], 1.0 / 240.0)
                for name in ("body_ft_s", "joint_tau", "aug_row_counts", "aug_row_dof_index", "aug_row_K"):
                    np.testing.assert_allclose(
                        getattr(solvers[1], name).numpy(),
                        getattr(solvers[0], name).numpy(),
                        rtol=3e-6,
                        atol=3e-6,
                        err_msg=f"forces {force_phase}: {name}",
                    )
                tau = solvers[1].joint_tau.numpy().copy()
                if previous_tau is not None:
                    test.assertGreater(float(np.max(np.abs(tau - previous_tau))), 0.01)
                previous_tau = tau


def test_stage1_matches_serial_with_velocity_prescale(test, device):
    """Match every forward-pass field across partial warps, a state edit and active velocity prescaling."""
    for velocity_limits in (False, True):
        with test.subTest(velocity_limits=velocity_limits):
            model = _build_fingers(device, ragged=True)
            model.joint_velocity_limit.fill_(0.05)
            # Hold every finite limit row: these checks cover traversal, not row capacity.
            options = {"velocity_limits": velocity_limits, "dense_max_constraints": 2 * model.joint_dof_count}
            solvers = [_solver(model, enabled=value, **options) for value in (False, True)]
            states = [model.state() for _ in solvers]
            outputs = [model.state() for _ in solvers]
            for solver, state in zip(solvers, states, strict=True):
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                _poison(_dynamics_fields(solver, state), keep=("body_q",))
            original_qd = model.joint_qd.numpy().copy()
            for phase in ("first", "edited"):
                for solver, state, following in zip(solvers, states, outputs, strict=True):
                    if phase == "edited":
                        q = state.joint_q.numpy().copy()
                        start, end = model.articulation_start.numpy()[1:3]
                        q0, q1 = model.joint_q_start.numpy()[[start, end]]
                        q[q0:q1] += 0.1
                        state.joint_q.assign(q)
                    predictor_qd = _run_stage1(solver, state, following)
                    np.testing.assert_array_equal(state.joint_qd.numpy(), original_qd)
                    if velocity_limits:
                        test.assertGreater(float(np.max(np.abs(predictor_qd.numpy() - original_qd))), 0.01)
                fields = [_dynamics_fields(solver, state) for solver, state in zip(solvers, states, strict=True)]
                for name, actual in fields[1].items():
                    np.testing.assert_allclose(
                        actual.numpy(), fields[0][name].numpy(), rtol=3e-6, atol=3e-6, err_msg=f"{phase}: {name}"
                    )
                if velocity_limits:
                    np.testing.assert_array_equal(solvers[1].qd_work.numpy(), solvers[0].qd_work.numpy())


def test_publication_matches_public_fk(test, device):
    """Publish FREE and mixed D6 states with the public forward-kinematics convention."""
    model, _, _ = _build_branched_model(device, floating_root=True)
    solver = _solver(model, enabled=True)
    state = model.state()
    state.body_q.fill_(-123.25)
    state.body_qd.fill_(-123.25)
    solver._stage7_update_kinematics(state)
    reference = model.state()
    newton.eval_fk(model, reference.joint_q, reference.joint_qd, reference)
    for name in ("body_q", "body_qd"):
        np.testing.assert_allclose(
            getattr(state, name).numpy(), getattr(reference, name).numpy(), rtol=3e-6, atol=3e-6, err_msg=name
        )


def test_contact_trajectory_matches_serial(test, device):
    """Match serial traversal over a contact-loaded trajectory with a branched articulation."""
    model = _build_contact_model(device)
    model.rigid_contact_max = 32
    results = []
    for parallel in (False, True):
        solver = _solver(model, enabled=parallel)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
        contacts = pipeline.contacts()
        state, following = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        for _ in range(40):
            pipeline.collide(state, contacts)
            solver.step(state, following, control, contacts, 1.0 / 240.0)
            state, following = following, state
        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        results.append(state)
    for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
        np.testing.assert_allclose(
            getattr(results[1], name).numpy(), getattr(results[0], name).numpy(), rtol=1e-5, atol=2e-6, err_msg=name
        )


class TestFeatherPGSTreeExecution(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _name in (
    "test_serial_selection_skips_tree_setup",
    "test_explicit_parallel_preserves_eligibility_fallback",
    "test_backward_torques_with_live_forces_and_passive_terms",
    "test_stage1_matches_serial_with_velocity_prescale",
    "test_publication_matches_public_fk",
    "test_contact_trajectory_matches_serial",
):
    add_function_test(TestFeatherPGSTreeExecution, _name, globals()[_name], devices=devices)


if __name__ == "__main__":
    unittest.main()
