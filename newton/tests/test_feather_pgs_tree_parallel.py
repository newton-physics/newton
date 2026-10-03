# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check complete level-synchronous FPGS trees against serial dynamics."""

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
    _fields,
    _solver,
)


def _build_fingers(device="cpu", *, articulations=3, include_chain=False, ragged=False, requires_grad=False):
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
    model = builder.finalize(device=device, requires_grad=requires_grad)
    model.joint_q.assign(np.linspace(-0.3, 0.4, model.joint_coord_count, dtype=np.float32))
    model.joint_qd.assign(np.linspace(-0.7, 0.8, model.joint_dof_count, dtype=np.float32))
    return model


class TestFeatherPGSTreePlan(unittest.TestCase):
    def test_parallel_tree_is_keyword_only_and_defaults_to_serial(self):
        """Require an explicit construction-only opt-in without shifting positional arguments."""
        parameters = inspect.signature(SolverFeatherPGS).parameters
        self.assertIn("parallel_tree", parameters)
        self.assertIs(parameters["parallel_tree"].default, False)
        self.assertEqual(parameters["parallel_tree"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_serial_selection_skips_tree_setup(self):
        """Keep default and explicit serial execution free of tree plans and scratch."""
        for device in (wp.get_device("cpu"), *wp.get_cuda_devices()):
            model = _build_fingers(device, articulations=1)
            states = []
            for options in ({}, {"parallel_tree": False}):
                with self.subTest(device=str(device), options=options):
                    with patch.object(_FeatherPGSTreePlan, "build", wraps=_FeatherPGSTreePlan.build) as build:
                        solver = SolverFeatherPGS(model, use_parallel_streams=False, **options)
                    build.assert_not_called()
                    self.assertIsNone(solver._tree_plan)
                    self.assertEqual(solver._tree_net_wrenches, ())
                    state = model.state()
                    solver._prepare_augmented_state(state, model.state(), model.control())
                    solver._stage7_update_kinematics(state, solver)
                    states.append(state)
            for name in ("body_q", "body_qd"):
                np.testing.assert_array_equal(getattr(states[0], name).numpy(), getattr(states[1], name).numpy())

    def test_explicit_parallel_preserves_eligibility_fallback(self):
        """Admit CUDA branches while retaining CPU, differentiable and serial-tree fallback."""
        for device in (wp.get_device("cpu"), *wp.get_cuda_devices()):
            with self.subTest(device=str(device), topology="branched"):
                model = _build_fingers(device, articulations=1)
                with patch.object(_FeatherPGSTreePlan, "build", wraps=_FeatherPGSTreePlan.build) as build:
                    solver = SolverFeatherPGS(model, parallel_tree=True, use_parallel_streams=False)
                if device.is_cuda:
                    build.assert_called_once()
                    self.assertIsNotNone(solver._tree_plan)
                    self.assertTrue(any(group.lanes > 1 for group in solver._tree_plan.groups))
                    self.assertEqual(len(solver._tree_net_wrenches), len(solver._tree_plan.groups))
                else:
                    build.assert_not_called()
                    self.assertIsNone(solver._tree_plan)
                    self.assertEqual(solver._tree_net_wrenches, ())
            if device.is_cuda:
                with self.subTest(device=str(device), requires_grad=True):
                    model = _build_fingers(device, articulations=1, requires_grad=True)
                    with patch.object(_FeatherPGSTreePlan, "build", wraps=_FeatherPGSTreePlan.build) as build:
                        solver = SolverFeatherPGS(model, parallel_tree=True, use_parallel_streams=False)
                    build.assert_not_called()
                    self.assertIsNone(solver._tree_plan)
                    self.assertEqual(solver._tree_net_wrenches, ())
            with self.subTest(device=str(device), topology="chain"):
                model = _build_model(device, chain=True)[0]
                solver = SolverFeatherPGS(model, parallel_tree=True, use_parallel_streams=False)
                self.assertIsNone(solver._tree_plan)
                self.assertEqual(solver._tree_net_wrenches, ())

    def test_complete_finger_levels_and_serial_group(self):
        """Schedule every finger segment and retain serial trees as one-lane groups."""
        model = _build_fingers(include_chain=True)
        solver = SolverFeatherPGS(model, pgs_mode="split", use_parallel_streams=False)
        plan = _FeatherPGSTreePlan.build(model, solver.articulation_joint_end)
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
        """Keep overlapping comb teeth parallel when compressed chains increase span."""
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
        ends = wp.array(model.articulation_start.numpy()[1:], dtype=wp.int32, device=model.device)

        # Compression delays the next fork until the whole preceding tooth ends.
        compressed_lengths = [[1], [4, 1], [4, 1], [4, 5]]
        joint_widths = [1, 2, 3, 4, 4, 3, 2, 1]
        compressed_rounds = sum(max(level) for level in compressed_lengths)
        joint_rounds = sum((width + 3) // 4 for width in joint_widths)
        self.assertEqual((compressed_rounds, joint_rounds), (14, 8))

        plan = _FeatherPGSTreePlan.build(model, ends)
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
        """Retain wholly serial, aliased, cross-articulation and loop topologies."""
        model, _, _ = _build_model(chain=True)
        solver = SolverFeatherPGS(model, pgs_mode="split", use_parallel_streams=False)
        self.assertIsNone(_FeatherPGSTreePlan.build(model, solver.articulation_joint_end))
        for unsafe in ("alias", "foreign_parent", "loop"):
            with self.subTest(unsafe=unsafe):
                model = _build_fingers(articulations=2)
                solver = SolverFeatherPGS(model, pgs_mode="split", use_parallel_streams=False)
                ends = solver.articulation_joint_end.numpy().copy()
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
                plan = _FeatherPGSTreePlan.build(model, wp.array(ends, dtype=wp.int32, device=model.device))
                self.assertIsNone(plan)


class TestFeatherPGSTreeExecution(unittest.TestCase):
    def test_cuda_tree_and_device_torsion_initialize_together(self):
        """Retain both opt-in initializers and torsion capture preparation after merging."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Device torsion and parallel trees require CUDA")
        for device in devices:
            with self.subTest(device=str(device)):
                model = _build_contact_model(device)
                solver = SolverFeatherPGS(
                    model,
                    pgs_mode="matrix_free",
                    parallel_tree=True,
                    contact_torsion_device=True,
                    contact_torsion_radius=0.01,
                    friction_anchor_beta=0.0,
                    use_parallel_streams=False,
                )
                self.assertIsNotNone(solver._tree_plan)
                self.assertEqual(len(solver._tree_net_wrenches), len(solver._tree_plan.groups))
                self.assertIsNotNone(getattr(solver, "_device_torsion", None))
                solver.prepare_contact_torsion_capture(model.state(), model.state())
                solver.validate_contact_torsion()

    def test_cuda_backward_torques_with_live_forces_and_passive_terms(self):
        """Preserve child reductions, additive torque and original-velocity damping."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Parallel backward tree execution requires CUDA")
        for device in devices:
            for velocity_limits in (False, True):
                with self.subTest(device=str(device), velocity_limits=velocity_limits):
                    model = _build_fingers(device, ragged=True)
                    model.joint_velocity_limit.fill_(0.05)
                    damping = np.linspace(0.1, 0.4, model.joint_dof_count, dtype=np.float32)
                    model.joint_damping.assign(damping)
                    model.joint_spring_stiffness.assign(np.linspace(0.3, 0.8, model.joint_dof_count, dtype=np.float32))
                    model.joint_spring_ref.fill_(0.1)
                    solvers = [
                        _solver(model, enabled=value, mode="matrix_free", velocity_limits=velocity_limits)
                        for value in (False, True)
                    ]
                    states = [model.state() for _ in solvers]
                    following = [model.state() for _ in solvers]
                    controls = [model.control() for _ in solvers]
                    seed = np.linspace(-0.4, 0.3, model.joint_dof_count, dtype=np.float32)
                    previous_tau = None
                    for force_phase in range(2):
                        overwritten = None
                        for additive in (False, True):
                            for index, solver in enumerate(solvers):
                                state = states[index]
                                if force_phase == 0 and not additive:
                                    solver._stage1_fk_id(state, solver, following[index])
                                elif force_phase == 1 and not additive:
                                    solver._fk_id_cache_valid.fill_(1)
                                    solver._stage1_fk_id(state, solver, following[index])
                                wrench = np.arange(model.body_count * 6, dtype=np.float32).reshape(-1, 6)
                                wrench = (wrench % 11 - 5) * (0.03 + force_phase * 0.04)
                                state.body_f.assign(wrench)
                                controls[index].joint_f.assign(seed * (1.0 + force_phase))
                                solver.joint_tau.assign(seed)
                                solver.body_ft_s.fill_(-123.25)
                                for scratch in solver._tree_net_wrenches:
                                    scratch.fill_(-123.25)
                                solver._launch_rigid_tau(state, solver, controls[index], add_to_existing=additive)
                            for name in ("body_ft_s", "joint_tau"):
                                np.testing.assert_allclose(
                                    getattr(solvers[1], name).numpy(),
                                    getattr(solvers[0], name).numpy(),
                                    rtol=3e-6,
                                    atol=3e-6,
                                    err_msg=f"forces {force_phase}, additive {additive}: {name}",
                                )
                            tau = solvers[1].joint_tau.numpy().copy()
                            if additive:
                                np.testing.assert_allclose(tau - overwritten, seed, rtol=3e-6, atol=3e-6)
                            else:
                                overwritten = tau
                        if previous_tau is not None:
                            self.assertGreater(float(np.max(np.abs(overwritten - previous_tau))), 0.01)
                        previous_tau = overwritten
                    model.joint_damping.zero_()
                    for index, solver in enumerate(solvers):
                        solver._launch_rigid_tau(states[index], solver, controls[index])
                        np.testing.assert_allclose(
                            previous_tau - solver.joint_tau.numpy(),
                            -damping * states[index].joint_qd.numpy(),
                            rtol=3e-5,
                            atol=3e-6,
                        )

    def test_cuda_stage1_cold_mixed_cache_and_velocity_prescale(self):
        """Match every canonical dynamics field across partial warps and active prescaling."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Parallel tree execution requires CUDA")
        for device in devices:
            for velocity_limits in (False, True):
                with self.subTest(device=str(device), velocity_limits=velocity_limits):
                    model = _build_fingers(device, ragged=True)
                    model.joint_velocity_limit.fill_(0.05)
                    solvers = []
                    states = []
                    outputs = []
                    for parallel in (False, True):
                        solver = SolverFeatherPGS(
                            model,
                            pgs_mode="matrix_free",
                            pgs_iterations=8,
                            use_parallel_streams=False,
                            enable_joint_velocity_limits=velocity_limits,
                            parallel_tree=parallel,
                        )
                        if parallel:
                            self.assertIsNotNone(solver._tree_plan)
                        else:
                            self.assertIsNone(solver._tree_plan)
                        state, following = model.state(), model.state()
                        solver._prepare_augmented_state(state, following, model.control())
                        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                        for name, array in _fields(solver, state).items():
                            if name not in ("body_q", "body_qd", "valid"):
                                array.fill_(-123.25)
                        solver._fk_id_cache_valid.zero_()
                        solvers.append(solver)
                        states.append(state)
                        outputs.append(following)
                    original_qd = model.joint_qd.numpy().copy()
                    for phase in ("cold", "mixed", "warm"):
                        for solver, state, following in zip(solvers, states, outputs, strict=True):
                            if phase == "mixed":
                                q = state.joint_q.numpy().copy()
                                start, end = model.articulation_start.numpy()[1:3]
                                q0, q1 = model.joint_q_start.numpy()[[start, end]]
                                q[q0:q1] += 0.1
                                state.joint_q.assign(q)
                                solver._fk_id_cache_valid.fill_(1)
                                solver.notify_state_changed(wp.array([1], dtype=wp.int32, device=device))
                                np.testing.assert_array_equal(
                                    solver._fk_id_cache_valid.numpy(),
                                    [1, 0, 1] if solver._fk_id_cache_enabled else [1, 1, 1],
                                )
                            elif phase == "warm":
                                solver._fk_id_cache_valid.fill_(1)
                            _, predictor_qd = solver._stage1_fk_id(state, solver, following)
                            np.testing.assert_array_equal(state.joint_qd.numpy(), original_qd)
                            if velocity_limits:
                                self.assertGreater(float(np.max(np.abs(predictor_qd.numpy() - original_qd))), 0.01)
                        for name, actual in _fields(solvers[1], states[1]).items():
                            np.testing.assert_allclose(
                                actual.numpy(),
                                _fields(solvers[0], states[0])[name].numpy(),
                                rtol=3e-6,
                                atol=3e-6,
                                err_msg=f"{phase}: {name}",
                            )
                        if velocity_limits:
                            np.testing.assert_array_equal(solvers[1].qd_work.numpy(), solvers[0].qd_work.numpy())

    def test_cuda_uncached_publication_matches_public_fk(self):
        """Publish uncached FREE and mixed D6 states using the public FK convention."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Parallel tree publication requires CUDA")
        for device in devices:
            model, _, _ = _build_branched_model(device, floating_root=True)
            solver = _solver(model, enabled=True, mode="matrix_free")
            solver._fk_id_cache_enabled = False
            state = model.state()
            state.body_q.fill_(-123.25)
            state.body_qd.fill_(-123.25)
            solver._stage7_update_kinematics(state, solver)
            reference = model.state()
            newton.eval_fk(model, reference.joint_q, reference.joint_qd, reference)
            for name in ("body_q", "body_qd"):
                np.testing.assert_allclose(
                    getattr(state, name).numpy(), getattr(reference, name).numpy(), rtol=3e-6, atol=3e-6, err_msg=name
                )


if __name__ == "__main__":
    unittest.main()
