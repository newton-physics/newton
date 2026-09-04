# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.sim.modal import _estimate_sample_psi
from newton._src.solvers.vbd.reduced_elastic_kernels import create_solve_elastic_body_tiled
from newton.examples.basic._reduced_elastic import (
    beam_render_sample_points,
    beam_torsion_linear_modal_properties,
    box_surface_mesh,
    finite_torsion_displacement,
    joint_endpoint_world,
    mesh_volume,
    quat_rotate,
)
from newton.examples.basic._reduced_elastic_contact import (
    apply_kinematic_targets,
)
from newton.examples.basic.example_basic_reduced_elastic_angular_frame_coupling import (
    Example as AngularFrameCouplingExample,
)
from newton.examples.basic.example_basic_reduced_elastic_base_excitation import Example as BaseExcitationExample
from newton.examples.basic.example_basic_reduced_elastic_base_rotation import Example as BaseRotationExample
from newton.examples.basic.example_basic_reduced_elastic_centrifugal import Example as CentrifugalExample
from newton.examples.basic.example_basic_reduced_elastic_chair_stick_slip import (
    BIN_NAME as CHAIR_BIN_NAME,
)
from newton.examples.basic.example_basic_reduced_elastic_chair_stick_slip import (
    GLTF_NAME as CHAIR_GLTF_NAME,
)
from newton.examples.basic.example_basic_reduced_elastic_chair_stick_slip import Example as ChairStickSlipExample
from newton.examples.basic.example_basic_reduced_elastic_chair_stick_slip import (
    _asset_cache_dir as chair_asset_cache_dir,
)
from newton.examples.basic.example_basic_reduced_elastic_clamp_moment import Example as ClampMomentExample
from newton.examples.basic.example_basic_reduced_elastic_coriolis import Example as CoriolisExample
from newton.examples.basic.example_basic_reduced_elastic_dipper import Example as DipperExample
from newton.examples.basic.example_basic_reduced_elastic_frame_coupling import Example as FrameCouplingExample
from newton.examples.basic.example_basic_reduced_elastic_gravity_coupling import Example as GravityCouplingExample
from newton.examples.basic.example_basic_reduced_elastic_gripper_contact import Example as GripperContactExample
from newton.examples.basic.example_basic_reduced_elastic_matrix_rom import Example as MatrixROMExample
from newton.examples.basic.example_basic_reduced_elastic_scraper_contact import Example as ScraperContactExample
from newton.examples.basic.example_basic_reduced_elastic_wall_contact import Example as WallContactExample
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()


class _ElasticMeshColorProbe(newton.viewer.ViewerNull):
    def __init__(self):
        super().__init__(num_frames=1)
        self.mesh_colors = {}
        self.mesh_points = {}
        self.mesh_hidden = {}

    def log_mesh(
        self,
        name,
        points,
        indices,
        normals=None,
        uvs=None,
        texture=None,
        hidden=False,
        backface_culling=True,
        color=None,
        roughness=None,
        metallic=None,
        dynamic=False,
        opacity=None,
        colors=None,
    ):
        if name.startswith("/model/elastic_shapes/"):
            self.mesh_colors[name] = None if colors is None else colors.numpy()
            self.mesh_points[name] = points.numpy()
            self.mesh_hidden[name] = hidden


def _identity_inertia():
    return wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _quat_from_angle_z(theta: float):
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), theta)


def _quat_angle_error(q_a: np.ndarray, q_b: np.ndarray) -> float:
    dot = abs(float(np.dot(q_a, q_b)))
    dot = min(max(dot, -1.0), 1.0)
    return 2.0 * math.acos(dot)


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    qv = np.array([x, y, z], dtype=float)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + w * v)


def _transform_point_np(xform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return xform[:3] + _quat_rotate_np(xform[3:7], point)


def _add_modal_projection_joint(builder, joint_kind: str, parent: int, child: int, parent_xform, child_xform):
    if joint_kind == "ball":
        return builder.add_joint_ball(parent=parent, child=child, parent_xform=parent_xform, child_xform=child_xform)
    if joint_kind == "fixed":
        return builder.add_joint_fixed(parent=parent, child=child, parent_xform=parent_xform, child_xform=child_xform)
    if joint_kind == "revolute":
        return builder.add_joint_revolute(
            parent=parent,
            child=child,
            axis=(1.0, 0.0, 0.0),
            parent_xform=parent_xform,
            child_xform=child_xform,
        )
    if joint_kind == "prismatic":
        return builder.add_joint_prismatic(
            parent=parent,
            child=child,
            axis=(1.0, 0.0, 0.0),
            parent_xform=parent_xform,
            child_xform=child_xform,
        )
    raise ValueError(f"Unsupported joint kind: {joint_kind}")


def _assert_elastic_modal_projection_matches_joint_force(test, device, joint_kind: str, elastic_side: str = "child"):
    phi_local = np.array([0.35, -0.45, 0.25], dtype=np.float32)
    elastic_anchor = np.array([0.12, -0.03, 0.04], dtype=float)
    elastic_pos = np.array([0.02, 0.01, -0.015], dtype=float)
    elastic_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.37)
    elastic_xform = wp.transform(wp.vec3(*elastic_pos), elastic_quat)
    elastic_quat_np = np.array([elastic_quat[0], elastic_quat[1], elastic_quat[2], elastic_quat[3]], dtype=float)
    k = 2500.0

    def shape_fn(_x):
        return np.array([phi_local], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if elastic_side == "child":
        body = builder.add_body_elastic(
            xform=elastic_xform,
            mass=1.0,
            inertia=_identity_inertia(),
            mode_count=1,
            mode_mass=[0.0],
            mode_stiffness=[0.0],
            mode_damping=[0.0],
            mode_q=[0.0],
            mode_shape_fn=shape_fn,
            is_kinematic=True,
        )
        parent_anchor = np.array([-0.08, 0.07, -0.02], dtype=float)
        child_anchor = elastic_anchor
        _add_modal_projection_joint(
            builder,
            joint_kind,
            parent=-1,
            child=body,
            parent_xform=wp.transform(wp.vec3(*parent_anchor), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(*child_anchor), wp.quat_identity()),
        )
        x_p = parent_anchor
        x_c = elastic_pos + _quat_rotate_np(elastic_quat_np, child_anchor)
        dC_dq = _quat_rotate_np(elastic_quat_np, phi_local.astype(float))
        side_force_sign = -1.0
    elif elastic_side == "parent":
        body = builder.add_body_elastic(
            xform=elastic_xform,
            mass=1.0,
            inertia=_identity_inertia(),
            mode_count=1,
            mode_mass=[0.0],
            mode_stiffness=[0.0],
            mode_damping=[0.0],
            mode_q=[0.0],
            mode_shape_fn=shape_fn,
            is_kinematic=True,
        )
        child_pos = np.array([-0.06, 0.09, 0.035], dtype=float)
        child = builder.add_body(
            xform=wp.transform(wp.vec3(*child_pos), wp.quat_identity()),
            mass=1.0,
            inertia=_identity_inertia(),
            is_kinematic=True,
        )
        parent_anchor = elastic_anchor
        child_anchor = np.array([0.03, -0.015, 0.02], dtype=float)
        _add_modal_projection_joint(
            builder,
            joint_kind,
            parent=body,
            child=child,
            parent_xform=wp.transform(wp.vec3(*parent_anchor), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(*child_anchor), wp.quat_identity()),
        )
        x_p = elastic_pos + _quat_rotate_np(elastic_quat_np, parent_anchor)
        x_c = child_pos + child_anchor
        endpoint_phi_world = _quat_rotate_np(elastic_quat_np, phi_local.astype(float))
        dC_dq = -endpoint_phi_world
        side_force_sign = 1.0
    else:
        raise ValueError(f"Unsupported elastic side: {elastic_side}")

    builder.color()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    P = np.eye(3)
    if joint_kind == "prismatic":
        axis = np.array([1.0, 0.0, 0.0])
        P -= np.outer(axis, axis)

    C = x_c - x_p
    PC = P @ C
    PdC = P @ dC_dq
    h = k * float(np.dot(PdC, PdC))
    grad = k * float(np.dot(PC, PdC))
    test.assertGreater(h, 0.0)

    endpoint_phi_world = _quat_rotate_np(elastic_quat_np, phi_local.astype(float))
    rigid_side_force = side_force_sign * k * PC
    modal_force_from_joint = float(np.dot(endpoint_phi_world, rigid_side_force))
    modal_force_from_energy = -grad
    np.testing.assert_allclose(modal_force_from_energy, modal_force_from_joint, rtol=1.0e-6, atol=1.0e-7)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_joint_linear_k_start=k,
        rigid_joint_linear_ke=k,
        rigid_joint_angular_ke=0.0,
        rigid_joint_linear_kd=0.0,
        rigid_joint_angular_kd=0.0,
        rigid_joint_adaptive_stiffness=False,
    )
    solver.step(state_0, state_1, control, None, 0.01)

    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    q_expected = modal_force_from_joint / h
    np.testing.assert_allclose(state_1.joint_q.numpy()[q_start + 7], q_expected, rtol=1.0e-5, atol=1.0e-6)


def test_modal_basis_add_sample(test, device):
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]]],
        mode_mass=[2.0, 3.0],
        mode_stiffness=[4.0, 5.0],
        mode_damping=[0.1, 0.2],
    )

    test.assertEqual(basis.mode_count, 2)
    test.assertEqual(basis.sample_count, 1)
    test.assertEqual(basis.add_sample([0.0, 0.0, 0.0]), 0)
    sample = basis.add_sample([1.0, 0.0, 0.0], phi=[[0.5, 0.0, 0.0], [0.0, 0.0, 0.25]])
    test.assertEqual(sample, 1)
    np.testing.assert_allclose(basis.sample_phi[sample], [[0.5, 0.0, 0.0], [0.0, 0.0, 0.25]], atol=1.0e-7)


def test_modal_basis_add_samples_preserves_mass_invariant(test, device):
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[1.0, 0.0, 0.0]]],
        sample_mass=[2.0],
        mode_stiffness=[1.0],
    )
    indices = basis.add_samples(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        phi=[[[9.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[7.0, 0.0, 0.0]]],
    )

    np.testing.assert_array_equal(indices, [0, 1, 1])
    np.testing.assert_allclose(basis.sample_phi[:, 0], [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    np.testing.assert_allclose(basis.sample_mass, [2.0, 0.0])
    clone = basis.copy()
    np.testing.assert_allclose(clone.sample_mass, basis.sample_mass)


def test_modal_generator_beam_samples(test, device):
    length = 1.0
    basis = newton.ModalGeneratorBeam(
        length=length,
        half_width_y=0.1,
        half_width_z=0.05,
        mode_specs=[
            {"type": newton.ModalGeneratorBeam.Mode.AXIAL},
            {
                "type": newton.ModalGeneratorBeam.Mode.BENDING_Y,
                "boundary": newton.ModalGeneratorBeam.Boundary.PINNED_PINNED,
            },
            {
                "type": newton.ModalGeneratorBeam.Mode.TORSION,
                "boundary": newton.ModalGeneratorBeam.Boundary.LINEAR,
            },
        ],
        sample_count=3,
    ).build()

    sample = basis.add_sample([0.0, 0.1, 0.05])
    phi = basis.sample_phi[sample]
    np.testing.assert_allclose(phi[0], [0.0, 0.0, 0.0], atol=1.0e-7)
    np.testing.assert_allclose(phi[1], [0.0, 1.0, 0.0], atol=1.0e-7)
    np.testing.assert_allclose(phi[2], [0.0, -0.025, 0.05], atol=1.0e-7)
    test.assertGreater(float(basis.mode_stiffness[0]), 0.0)


def test_modal_generator_pod_rank_one(test, device):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    displacements = np.array([[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=np.float32)

    basis = newton.ModalGeneratorPOD(
        sample_points=points,
        displacements=displacements,
        mode_count=1,
        total_mass=2.0,
        stiffness_scale=3.0,
    ).build()

    test.assertEqual(basis.mode_count, 1)
    np.testing.assert_allclose(np.abs(basis.sample_phi[1][0]), [0.0, 1.0, 0.0], atol=1.0e-7)
    np.testing.assert_allclose(basis.mode_mass, [1.0], atol=1.0e-7)
    np.testing.assert_allclose(basis.mode_stiffness, [3.0], atol=1.0e-7)


def test_modal_generator_pod_explicit_zero_modes(test, device):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    displacements = np.array([[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=np.float32)
    basis = newton.ModalGeneratorPOD(points, displacements=displacements, mode_count=0).build()

    test.assertEqual(basis.mode_count, 0)
    test.assertEqual(basis.sample_phi.shape, (2, 0, 3))


def test_modal_generator_beam_linear_torsion_properties(test, device):
    length = 2.0
    shear_modulus = 7.0
    polar_moment = 0.25
    order = 2
    basis = newton.ModalGeneratorBeam(
        length=length,
        half_width_y=0.1,
        half_width_z=0.1,
        shear_modulus=shear_modulus,
        polar_moment=polar_moment,
        mode_specs=[
            {
                "type": newton.ModalGeneratorBeam.Mode.TORSION,
                "boundary": newton.ModalGeneratorBeam.Boundary.LINEAR,
                "order": order,
            }
        ],
    ).build()

    expected = shear_modulus * polar_moment * order**2 / (length * (2 * order - 1))
    np.testing.assert_allclose(basis.mode_stiffness, [expected], rtol=1.0e-6)
    with test.assertRaisesRegex(ValueError, "at least 1"):
        newton.ModalGeneratorBeam(
            length=length,
            mode_specs=[{"type": newton.ModalGeneratorBeam.Mode.TORSION, "order": 0}],
        )


def test_modal_generator_fem_matrix_rom(test, device):
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    mass = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    stiffness = np.zeros((6, 6), dtype=np.float64)
    stiffness[3, 3] = 8.0
    stiffness[4, 4] = 18.0
    stiffness[5, 5] = 32.0
    damping = np.zeros((6, 6), dtype=np.float64)
    damping[3, 3] = 0.4
    damping[4, 4] = 0.6
    damping[5, 5] = 0.8
    sample_points = np.array([[1.0, 0.0, 0.2], [0.0, 0.0, -0.1]], dtype=np.float32)

    generator = newton.ModalGeneratorFEM(
        node_positions=nodes,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        damping_matrix=damping,
        sample_points=sample_points,
        sample_node_indices=[1, 0],
        fixed_node_indices=[0],
        mode_count=3,
    )
    basis = generator.build()

    test.assertEqual(basis.mode_count, 3)
    np.testing.assert_allclose(basis.mode_mass, [1.0, 1.0, 1.0], atol=1.0e-6)
    np.testing.assert_allclose(basis.mode_stiffness, [4.0, 9.0, 16.0], atol=1.0e-6)
    np.testing.assert_allclose(basis.mode_damping, [0.2, 0.3, 0.4], atol=1.0e-6)
    np.testing.assert_allclose(
        generator.frequencies, [1.0 / math.pi, 3.0 / (2.0 * math.pi), 2.0 / math.pi], atol=1.0e-6
    )
    np.testing.assert_allclose(np.abs(basis.sample_phi[0]), np.eye(3) / math.sqrt(2.0), atol=1.0e-6)
    np.testing.assert_allclose(basis.sample_phi[1], np.zeros((3, 3)), atol=1.0e-7)


def _build_craig_bampton_test_data():
    interface_positions = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)

    def skew(value):
        x, y, z = value
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)

    rigid_map = np.zeros((12, 6), dtype=np.float64)
    for interface, position in enumerate(interface_positions):
        start = 6 * interface
        rigid_map[start : start + 3, :3] = np.eye(3)
        rigid_map[start : start + 3, 3:] = -skew(position)
        rigid_map[start + 3 : start + 6, 3:] = np.eye(3)

    coordinate_map = np.zeros((14, 14), dtype=np.float64)
    coordinate_map[:12, :6] = rigid_map
    coordinate_map[6:12, 6:12] = np.eye(6)
    coordinate_map[12:, 12:] = np.eye(2)

    spatial_mass = np.diag([4.0, 4.0, 4.0, 0.2, 0.3, 0.4])
    elastic_mass = np.diag([2.0, 2.5, 3.0, 0.5, 0.6, 0.7, 1.0, 1.1])
    elastic_mass[0, 1] = elastic_mass[1, 0] = 0.2
    elastic_mass[0, 6] = elastic_mass[6, 0] = 0.15
    elastic_mass[1, 7] = elastic_mass[7, 1] = -0.12
    elastic_stiffness = np.diag([20.0, 30.0, 40.0, 5.0, 6.0, 7.0, 100.0, 120.0])
    elastic_stiffness[1, 2] = elastic_stiffness[2, 1] = 1.5
    rigid_elastic_mass = np.zeros((6, 8), dtype=np.float64)
    rigid_elastic_mass[:, :6] = np.diag([0.10, -0.08, 0.06, 0.02, -0.015, 0.01])
    rigid_elastic_mass[0, 6] = 0.03
    rigid_elastic_mass[1, 7] = -0.025

    transformed_mass = np.zeros((14, 14), dtype=np.float64)
    transformed_mass[:6, :6] = spatial_mass
    transformed_mass[:6, 6:] = rigid_elastic_mass
    transformed_mass[6:, :6] = rigid_elastic_mass.T
    transformed_mass[6:, 6:] = elastic_mass
    transformed_stiffness = np.zeros((14, 14), dtype=np.float64)
    transformed_stiffness[6:, 6:] = elastic_stiffness
    transformed_damping = 0.01 * transformed_stiffness

    inverse_map = np.linalg.inv(coordinate_map)
    mass = inverse_map.T @ transformed_mass @ inverse_map
    stiffness = inverse_map.T @ transformed_stiffness @ inverse_map
    damping = inverse_map.T @ transformed_damping @ inverse_map

    sample_points = np.array(
        [
            [-0.5, -0.1, -0.1],
            [-0.5, 0.1, 0.1],
            [0.0, -0.1, 0.1],
            [0.0, 0.1, -0.1],
            [0.5, -0.1, -0.1],
            [0.5, 0.1, 0.1],
        ],
        dtype=np.float64,
    )
    sample_rigid_map = np.hstack(
        (
            np.tile(np.eye(3), (sample_points.shape[0], 1)),
            np.vstack([-skew(point) for point in sample_points]),
        )
    )
    sample_elastic_map = np.zeros((3 * sample_points.shape[0], 8), dtype=np.float64)
    x_weight = sample_points[:, 0] + 0.5
    for sample, weight in enumerate(x_weight):
        rows = slice(3 * sample, 3 * sample + 3)
        sample_elastic_map[rows, :3] = weight * np.eye(3)
        sample_elastic_map[rows, 3:6] = -weight * skew(sample_points[sample])
        internal_weight = 1.0 - 4.0 * sample_points[sample, 0] ** 2
        sample_elastic_map[rows, 6] = internal_weight * np.array([0.0, 0.0, 0.2])
        sample_elastic_map[rows, 7] = internal_weight * np.array([0.0, 0.15, 0.0])
    recovery = np.column_stack((sample_rigid_map, sample_elastic_map)) @ inverse_map

    return interface_positions, mass, stiffness, damping, sample_points, recovery, spatial_mass


def test_modal_generator_craig_bampton_interface_modes(test, device):
    interface_positions, mass, stiffness, damping, sample_points, recovery, spatial_mass = (
        _build_craig_bampton_test_data()
    )

    generator = newton.ModalGeneratorCraigBampton(
        interface_positions=interface_positions,
        interface_names=["left", "right"],
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        damping_matrix=damping,
        sample_points=sample_points,
        recovery_matrix=recovery,
    )
    basis = generator.build()

    test.assertEqual(generator.fixed_interface_mode_count, 2)
    test.assertEqual(generator.discarded_mode_count, 0)
    test.assertEqual(basis.mode_count, 8)
    test.assertEqual(set(generator.interface_sample_indices), {"left", "right"})
    test.assertAlmostEqual(generator.mass, 4.0, places=6)
    np.testing.assert_allclose(generator.com, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(generator.inertia, np.diag([0.2, 0.3, 0.4]), atol=1.0e-7)

    modes = generator.modal_matrix
    np.testing.assert_allclose(modes.T @ mass @ modes, np.eye(8), atol=1.0e-7)
    np.testing.assert_allclose(
        modes.T @ stiffness @ modes,
        np.diag(basis.mode_stiffness),
        atol=1.0e-5,
    )
    np.testing.assert_allclose(modes.T @ damping @ modes, np.diag(basis.mode_damping), atol=1.0e-7)
    np.testing.assert_allclose(basis.mode_coupling_linear, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(basis.mode_coupling_angular, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(generator.spatial_mass, spatial_mass, atol=1.0e-7)

    recovered_modes = recovery @ modes
    expected_phi = np.transpose(recovered_modes.reshape((sample_points.shape[0], 3, 8)), (0, 2, 1))
    np.testing.assert_allclose(basis.sample_phi[: sample_points.shape[0]], expected_phi, atol=1.0e-7)
    test.assertGreater(float(np.linalg.norm(modes[12:])), 0.0)

    left = generator.interface_sample_indices["left"]
    right = generator.interface_sample_indices["right"]
    np.testing.assert_allclose(basis.sample_phi[left], modes[:3].T, atol=1.0e-7)
    np.testing.assert_allclose(basis.sample_psi[left], modes[3:6].T, atol=1.0e-7)
    np.testing.assert_allclose(basis.sample_phi[right], modes[6:9].T, atol=1.0e-7)
    np.testing.assert_allclose(basis.sample_psi[right], modes[9:12].T, atol=1.0e-7)


def test_modal_generator_craig_bampton_single_interface(test, device):
    interface_positions = np.zeros((1, 3), dtype=np.float64)
    spatial_mass = np.diag([3.0, 3.0, 3.0, 0.2, 0.25, 0.3])
    mass = np.zeros((8, 8), dtype=np.float64)
    mass[:6, :6] = spatial_mass
    mass[6:, 6:] = np.diag([1.0, 1.2])
    mass[0, 6] = mass[6, 0] = 0.05
    mass[1, 7] = mass[7, 1] = -0.04
    stiffness = np.zeros((8, 8), dtype=np.float64)
    stiffness[6:, 6:] = np.diag([40.0, 75.0])
    damping = 0.02 * stiffness
    sample_points = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.2]], dtype=np.float64)
    sample_rigid = np.hstack(
        (
            np.tile(np.eye(3), (sample_points.shape[0], 1)),
            np.vstack(
                [
                    np.array([[0.0, point[2], -point[1]], [-point[2], 0.0, point[0]], [point[1], -point[0], 0.0]])
                    for point in sample_points
                ]
            ),
        )
    )
    sample_internal = np.zeros((3 * sample_points.shape[0], 2), dtype=np.float64)
    sample_internal[3:, 0] = np.tile([0.0, 0.0, 0.1], sample_points.shape[0] - 1)
    sample_internal[3:, 1] = np.tile([0.0, 0.08, 0.0], sample_points.shape[0] - 1)
    recovery = np.column_stack((sample_rigid, sample_internal))

    generator = newton.ModalGeneratorCraigBampton(
        interface_positions=interface_positions,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        damping_matrix=damping,
        sample_points=sample_points,
        recovery_matrix=recovery,
    )
    basis = generator.build()

    test.assertEqual(generator.fixed_interface_mode_count, 2)
    test.assertEqual(basis.mode_count, 2)
    np.testing.assert_allclose(generator.spatial_mass, spatial_mass, atol=1.0e-7)
    np.testing.assert_allclose(generator.modal_matrix.T @ mass @ generator.modal_matrix, np.eye(2), atol=1.0e-7)


def test_modal_generator_craig_bampton_nonclassical_damping(test, device):
    interface_positions, mass, stiffness, _, sample_points, recovery, _ = _build_craig_bampton_test_data()
    undamped_generator = newton.ModalGeneratorCraigBampton(
        interface_positions=interface_positions,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        sample_points=sample_points,
        recovery_matrix=recovery,
    )
    undamped_basis = undamped_generator.build()

    modes = undamped_generator.modal_matrix
    modal_damping = np.diag(0.01 * undamped_basis.mode_stiffness.astype(np.float64))
    modal_damping[0, 1] = modal_damping[1, 0] = 0.05 * float(np.max(np.diag(modal_damping)))
    damping = mass @ modes @ modal_damping @ modes.T @ mass
    generator = newton.ModalGeneratorCraigBampton(
        interface_positions=interface_positions,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        damping_matrix=damping,
        sample_points=sample_points,
        recovery_matrix=recovery,
    )

    with test.assertWarnsRegex(UserWarning, "damping is not diagonal"):
        basis = generator.build()
    test.assertGreater(generator.damping_off_diagonal_ratio, generator.damping_coupling_tolerance)
    np.testing.assert_allclose(basis.mode_damping, np.diag(modal_damping), rtol=1.0e-5)


def test_modal_basis_lumped_inertia_coupling(test, device):
    points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    sample_phi = np.array(
        [
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    basis = newton.ModalBasis(sample_points=points, sample_phi=sample_phi, sample_mass=[1.0, 1.0])

    np.testing.assert_allclose(basis.mode_coupling_linear, [[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]], atol=1.0e-6)
    np.testing.assert_allclose(basis.mode_coupling_angular, [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]], atol=1.0e-6)


def test_modal_basis_coupling_unavailable_without_mass(test, device):
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]],
    )
    test.assertIsNone(basis.sample_mass)
    test.assertIsNone(basis.mode_coupling_linear)
    test.assertIsNone(basis.mode_coupling_angular)
    test.assertIsNotNone(basis.mode_mass)
    np.testing.assert_allclose(basis.mode_mass, [1.0], atol=1.0e-7)


def test_modal_basis_coupling_explicit_override(test, device):
    explicit = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]],
        sample_mass=[1.0, 1.0],
        mode_coupling_linear=explicit,
    )
    np.testing.assert_allclose(basis.mode_coupling_linear, explicit, atol=1.0e-7)
    test.assertIsNotNone(basis.mode_coupling_angular)
    with test.assertRaises(ValueError):
        newton.ModalBasis(
            sample_points=[[0.0, 0.0, 0.0]],
            sample_phi=[[[0.0, 0.0, 1.0]]],
            mode_coupling_linear=[[1.0, 2.0]],
        )


def test_modal_generator_fem_inertia_coupling(test, device):
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    mass = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    stiffness = np.zeros((6, 6), dtype=np.float64)
    stiffness[3, 3] = 8.0
    stiffness[4, 4] = 18.0
    stiffness[5, 5] = 32.0
    basis = newton.ModalGeneratorFEM(
        node_positions=nodes,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        sample_points=nodes,
        sample_node_indices=[0, 1],
        fixed_node_indices=[0],
        mode_count=3,
    ).build()

    node_mass = np.array([1.0, 2.0])
    phi_nodes = np.stack([basis.sample_phi[0], basis.sample_phi[1]]).astype(np.float64)
    expected_linear = np.einsum("j,jmc->mc", node_mass, phi_nodes)
    expected_angular = np.einsum("j,jmc->mc", node_mass, np.cross(phi_nodes, nodes[:, None, :].astype(np.float64)))
    np.testing.assert_allclose(basis.mode_coupling_linear, expected_linear, atol=1.0e-5)
    np.testing.assert_allclose(basis.mode_coupling_angular, expected_angular, atol=1.0e-5)


def test_modal_generator_pod_inertia_coupling(test, device):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    displacements = np.array([[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=np.float32)
    basis = newton.ModalGeneratorPOD(
        sample_points=points,
        displacements=displacements,
        mode_count=1,
        total_mass=2.0,
    ).build()

    test.assertIsNotNone(basis.mode_coupling_linear)
    sample_mass = np.full(2, 1.0)
    expected_linear = np.einsum("s,smc->mc", sample_mass, basis.sample_phi.astype(np.float64))
    np.testing.assert_allclose(basis.mode_coupling_linear, expected_linear, atol=1.0e-6)


def test_modal_generator_pod_coupling_matches_lumped(test, device):
    n = 9
    xs = np.linspace(0.0, 1.0, n, dtype=np.float32)
    points = np.column_stack([xs, 0.3 * xs, -0.2 * xs]).astype(np.float32)
    displacements = np.zeros((2, n, 3), dtype=np.float32)
    displacements[0, :, 2] = xs
    displacements[1, :, 1] = xs * xs
    total_mass = 3.0
    basis = newton.ModalGeneratorPOD(
        sample_points=points,
        displacements=displacements,
        mode_count=2,
        total_mass=total_mass,
    ).build()

    sample_mass = np.full(n, total_mass / n, dtype=np.float64)
    phi = basis.sample_phi.astype(np.float64)
    points64 = points.astype(np.float64)

    expected_linear = np.einsum("s,smc->mc", sample_mass, phi)
    expected_angular = np.einsum("s,smc->mc", sample_mass, np.cross(phi, points64[:, None, :]))
    expected_centrifugal = np.einsum("s,smc,sd->mcd", sample_mass, phi, points64)
    cross_pairs = np.cross(phi[:, None, :, :], phi[:, :, None, :])
    expected_coriolis = np.einsum("s,sijc->ijc", sample_mass, cross_pairs)

    np.testing.assert_allclose(basis.mode_coupling_linear, expected_linear, atol=1.0e-6)
    np.testing.assert_allclose(basis.mode_coupling_angular, expected_angular, atol=1.0e-6)
    np.testing.assert_allclose(basis.mode_coupling_centrifugal, expected_centrifugal, atol=1.0e-6)
    np.testing.assert_allclose(basis.mode_coupling_coriolis, expected_coriolis, atol=1.0e-6)


def test_estimate_sample_psi_rotation_and_translation(test, device):
    rng = np.random.default_rng(0)
    points = rng.normal(size=(40, 3)).astype(np.float32)

    omega = np.array([0.3, -0.5, 0.2], dtype=np.float64)
    rotation = np.cross(np.tile(omega, (points.shape[0], 1)), points.astype(np.float64))
    psi = _estimate_sample_psi(points, rotation[:, None, :].astype(np.float32))
    np.testing.assert_allclose(psi[:, 0, :], np.tile(omega, (points.shape[0], 1)), atol=1.0e-5)

    translation = np.tile(np.array([1.0, -2.0, 3.0], dtype=np.float32), (points.shape[0], 1))
    psi_t = _estimate_sample_psi(points, translation[:, None, :])
    np.testing.assert_allclose(psi_t, 0.0, atol=1.0e-6)


def test_estimate_sample_psi_degenerate_warns(test, device):
    xs = np.linspace(-1.0, 1.0, 6, dtype=np.float32)
    points = np.array([[x, y, 0.0] for x in xs for y in xs], dtype=np.float32)
    phi = np.cross(np.tile([0.0, 0.0, 1.0], (points.shape[0], 1)), points)[:, None, :].astype(np.float32)
    with test.assertWarns(UserWarning):
        psi = _estimate_sample_psi(points, phi)
    test.assertTrue(np.isfinite(psi).all())
    np.testing.assert_allclose(psi[:, 0, 2], 1.0, atol=1.0e-4)


def test_modal_generator_sampled_psi(test, device):
    n = 5
    xs = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    points = np.column_stack([xs, np.zeros(n), np.zeros(n)]).astype(np.float32)
    phi = np.zeros((n, 1, 3), dtype=np.float32)
    phi[:, 0, 1] = xs + 0.5
    explicit_psi = np.tile(np.array([0.7, 0.0, 0.0], dtype=np.float32), (n, 1))[:, None, :]

    passthrough = newton.ModalGeneratorSampled(points, phi, sample_psi=explicit_psi).build()
    np.testing.assert_allclose(passthrough.sample_psi, explicit_psi, atol=1.0e-7)

    default = newton.ModalGeneratorSampled(points, phi).build()
    np.testing.assert_allclose(default.sample_psi, 0.0, atol=1.0e-7)


def test_modal_generator_pod_derives_psi(test, device):
    rng = np.random.default_rng(1)
    points = rng.normal(size=(48, 3)).astype(np.float32)
    omega = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    snapshot = np.cross(np.tile(omega, (points.shape[0], 1)), points.astype(np.float64))

    basis = newton.ModalGeneratorPOD(
        sample_points=points,
        displacements=snapshot[None].astype(np.float32),
        mode_count=1,
        derive_psi=True,
    ).build()
    psi = basis.sample_psi[:, 0, :].astype(np.float64)
    test.assertGreater(float(np.linalg.norm(psi, axis=1).min()), 1.0e-3)
    directions = psi / np.linalg.norm(psi, axis=1, keepdims=True)
    np.testing.assert_allclose(np.abs(directions @ omega), 1.0, atol=1.0e-3)

    uncoupled = newton.ModalGeneratorPOD(
        sample_points=points,
        displacements=snapshot[None].astype(np.float32),
        mode_count=1,
        derive_psi=False,
    ).build()
    np.testing.assert_allclose(uncoupled.sample_psi, 0.0, atol=1.0e-7)


def test_modal_generator_fem_derives_psi(test, device):
    nodes = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)], dtype=np.float32)
    dof = 3 * nodes.shape[0]
    mass = np.eye(dof)
    stiffness = np.diag(np.linspace(1.0, 5.0, dof))

    basis = newton.ModalGeneratorFEM(
        node_positions=nodes,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        fixed_node_indices=[0],
        mode_count=3,
        derive_psi=True,
    ).build()
    test.assertEqual(basis.sample_psi.shape, (nodes.shape[0], 3, 3))
    test.assertTrue(np.isfinite(basis.sample_psi).all())
    test.assertGreater(float(np.max(np.abs(basis.sample_psi))), 0.0)

    uncoupled = newton.ModalGeneratorFEM(
        node_positions=nodes,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        fixed_node_indices=[0],
        mode_count=3,
        derive_psi=False,
    ).build()
    np.testing.assert_allclose(uncoupled.sample_psi, 0.0, atol=1.0e-7)


def test_modal_basis_copy_preserves_coupling(test, device):
    n = 5
    xs = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    sample_points = np.column_stack([xs, np.zeros(n), np.zeros(n)]).astype(np.float32)
    sample_phi = np.zeros((n, 2, 3), dtype=np.float32)
    sample_phi[:, 0, 2] = xs + 0.5
    sample_phi[:, 1, 1] = (xs + 0.5) ** 2
    basis = newton.ModalBasis(
        sample_points=sample_points,
        sample_phi=sample_phi,
        sample_mass=np.full(n, 0.2, dtype=np.float32),
        mode_stiffness=[10.0, 20.0],
        mode_damping=[0.1, 0.2],
    )
    test.assertIsNotNone(basis.sample_mass)
    test.assertIsNotNone(basis.mode_coupling_linear)
    test.assertIsNotNone(basis.mode_coupling_angular)
    test.assertIsNotNone(basis.mode_coupling_centrifugal)
    test.assertIsNotNone(basis.mode_coupling_coriolis)

    clone = basis.copy()
    np.testing.assert_allclose(clone.sample_mass, basis.sample_mass, atol=0)
    np.testing.assert_allclose(clone.mode_coupling_linear, basis.mode_coupling_linear, atol=0)
    np.testing.assert_allclose(clone.mode_coupling_angular, basis.mode_coupling_angular, atol=0)
    np.testing.assert_allclose(clone.mode_coupling_centrifugal, basis.mode_coupling_centrifugal, atol=0)
    np.testing.assert_allclose(clone.mode_coupling_coriolis, basis.mode_coupling_coriolis, atol=0)


def test_modal_basis_mode_mass_from_sample_mass(test, device):
    basis = newton.ModalBasis(
        sample_points=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        sample_phi=[
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ],
        sample_mass=[1.0, 1.0],
    )
    np.testing.assert_allclose(basis.mode_mass, [2.0, 2.0], atol=1.0e-6)


def test_modal_basis_mode_mass_explicit_wins(test, device):
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]],
        sample_mass=[1.0, 1.0],
        mode_mass=[99.0],
    )
    np.testing.assert_allclose(basis.mode_mass, [99.0], atol=1.0e-6)


def test_elastic_mode_coupling_arrays(test, device):
    basis = newton.ModalBasis(
        sample_points=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]],
        sample_mass=[1.0, 1.0],
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), modal_basis=basis, label="coupled")
    builder.add_body_elastic(
        mass=1.0, inertia=_identity_inertia(), mode_count=1, mode_mass=[1.0], mode_stiffness=[1.0], label="plain"
    )
    builder.color()
    model = builder.finalize(device=device)

    np.testing.assert_allclose(
        model.elastic_mode_coupling_linear.numpy(), [[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]], atol=1.0e-6
    )
    np.testing.assert_allclose(
        model.elastic_mode_coupling_angular.numpy(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], atol=1.0e-6
    )
    np.testing.assert_allclose(model.elastic_mode_coupling_centrifugal.numpy(), np.zeros((2, 3, 3)), atol=1.0e-6)
    np.testing.assert_allclose(model.elastic_mode_coupling_coriolis.numpy(), np.zeros((2, 3)), atol=1.0e-6)


def test_elastic_coriolis_array_padding(test, device):
    basis_a = newton.ModalBasis(
        sample_points=[[1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]]],
        sample_mass=[1.0],
    )
    basis_b = newton.ModalBasis(
        sample_points=[[1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]],
        sample_mass=[2.0],
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), modal_basis=basis_a, label="a")
    builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), modal_basis=basis_b, label="b")
    builder.color()
    model = builder.finalize(device=device)

    expected = np.zeros((8, 3), dtype=np.float32)
    expected[5] = [2.0, 0.0, 0.0]
    expected[6] = [-2.0, 0.0, 0.0]
    np.testing.assert_allclose(model.elastic_mode_coupling_coriolis.numpy(), expected, atol=1.0e-6)


def test_elastic_mode_coupling_arrays_merge(test, device):
    basis = newton.ModalBasis(
        sample_points=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]],
        sample_mass=[1.0, 1.0],
    )
    sub = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    sub.add_body_elastic(mass=1.0, inertia=_identity_inertia(), modal_basis=basis, label="e")
    main = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    main.add_builder(sub)
    main.add_builder(sub)
    main.color()
    model = main.finalize(device=device)

    np.testing.assert_allclose(
        model.elastic_mode_coupling_linear.numpy(), [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]], atol=1.0e-6
    )


def test_elastic_coriolis_array_merge_different_mode_count(test, device):
    basis_a = newton.ModalBasis(
        sample_points=[[1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]]],
        sample_mass=[1.0],
    )
    basis_b = newton.ModalBasis(
        sample_points=[[1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]],
        sample_mass=[2.0],
    )
    sub_a = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    sub_a.add_body_elastic(mass=1.0, inertia=_identity_inertia(), modal_basis=basis_a, label="a")
    sub_b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    sub_b.add_body_elastic(mass=1.0, inertia=_identity_inertia(), modal_basis=basis_b, label="b")
    main = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    main.add_builder(sub_a)
    main.add_builder(sub_b)
    main.color()
    model = main.finalize(device=device)

    test.assertEqual(model.elastic_body_count, 2)
    test.assertEqual(model.elastic_max_mode_count, 2)
    max_modes = int(model.elastic_max_mode_count)
    block_a = 0 * max_modes * max_modes
    block_b = 1 * max_modes * max_modes
    test.assertEqual(block_b, 4)

    coriolis = model.elastic_mode_coupling_coriolis.numpy()
    test.assertEqual(coriolis.shape, (8, 3))
    np.testing.assert_allclose(coriolis[block_a : block_a + 4], np.zeros((4, 3)), atol=1.0e-6)
    expected_b = np.zeros((4, 3), dtype=np.float32)
    expected_b[1] = [2.0, 0.0, 0.0]
    expected_b[2] = [-2.0, 0.0, 0.0]
    np.testing.assert_allclose(coriolis[block_b : block_b + 4], expected_b, atol=1.0e-6)


def _deformed_endpoint_world(model, state, joint: int, side: str) -> np.ndarray:
    body = int(model.joint_parent.numpy()[joint]) if side == "parent" else int(model.joint_child.numpy()[joint])
    xforms = model.joint_X_p.numpy() if side == "parent" else model.joint_X_c.numpy()
    local = np.array(xforms[joint, :3], dtype=float)
    endpoint_ids = (
        model.joint_parent_elastic_endpoint.numpy() if side == "parent" else model.joint_child_elastic_endpoint.numpy()
    )
    endpoint = int(endpoint_ids[joint])
    if endpoint >= 0:
        elastic = int(model.body_elastic_index.numpy()[body])
        owner_joint = int(model.elastic_joint.numpy()[elastic])
        q_start = int(model.joint_q_start.numpy()[owner_joint])
        mode_count = int(model.elastic_mode_count.numpy()[elastic])
        phi = model.elastic_endpoint_phi.numpy().reshape((-1, 3))
        max_modes = int(model.elastic_max_mode_count)
        q = state.joint_q.numpy()
        for i in range(mode_count):
            local += phi[endpoint * max_modes + i] * q[q_start + 7 + i]
    if body < 0:
        return local
    return _transform_point_np(state.body_q.numpy()[body], local)


def _solve_fourbar(theta2: float, a: float, b: float, c: float, d: float) -> tuple[float, float]:
    """Return open-branch coupler and rocker angles for a planar four-bar."""
    k1 = d / a
    k2 = d / c
    k3 = (a * a - b * b + c * c + d * d) / (2.0 * a * c)
    A = k1 - math.cos(theta2)
    B = -math.sin(theta2)
    C = k2 * math.cos(theta2) - k3
    denom = math.sqrt(A * A + B * B)
    theta4 = math.atan2(B, A) + math.acos(np.clip(C / denom, -1.0, 1.0))
    cx = d + c * math.cos(theta4) - a * math.cos(theta2)
    cy = c * math.sin(theta4) - a * math.sin(theta2)
    theta3 = math.atan2(cy, cx)
    return theta3, theta4


def test_elastic_link_layout(test, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=2,
        mode_mass=[2.0, 3.0],
        mode_stiffness=[5.0, 7.0],
        mode_damping=[0.2, 0.3],
        mode_q=[0.1, -0.2],
        mode_qd=[0.4, -0.5],
        label="elastic",
    )
    builder.color()
    model = builder.finalize(device=device)

    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    qd_start = int(model.joint_qd_start.numpy()[owner_joint])

    test.assertEqual(body, 0)
    test.assertEqual(model.elastic_body_count, 1)
    test.assertEqual(model.elastic_max_mode_count, 2)
    test.assertEqual(int(model.body_elastic_index.numpy()[body]), 0)
    test.assertEqual(int(model.joint_type.numpy()[owner_joint]), int(newton.JointType.ELASTIC))
    test.assertEqual(int(model.joint_q_start.numpy()[owner_joint + 1] - model.joint_q_start.numpy()[owner_joint]), 9)
    test.assertEqual(int(model.joint_qd_start.numpy()[owner_joint + 1] - model.joint_qd_start.numpy()[owner_joint]), 8)
    np.testing.assert_allclose(model.joint_q.numpy()[q_start + 7 : q_start + 9], [0.1, -0.2], atol=1.0e-7)
    np.testing.assert_allclose(model.joint_qd.numpy()[qd_start + 6 : qd_start + 8], [0.4, -0.5], atol=1.0e-7)
    np.testing.assert_allclose(model.elastic_mode_mass.numpy(), [2.0, 3.0], atol=1.0e-7)


def test_add_joint_elastic_initializes_relative_frame_coordinates(test, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent_xform_world = wp.transform(wp.vec3(0.4, -0.2, 0.1), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.35))
    child_xform_world = wp.transform(wp.vec3(-0.3, 0.6, 0.7), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.25))
    parent = builder.add_link(xform=parent_xform_world, mass=1.0, inertia=_identity_inertia())
    child = builder.add_link(xform=child_xform_world, mass=1.0, inertia=_identity_inertia())
    parent_anchor = wp.transform(wp.vec3(0.1, 0.2, -0.1), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.2))
    child_anchor = wp.transform(wp.vec3(-0.2, 0.05, 0.15), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -0.15))
    joint = builder.add_joint_elastic(
        parent=parent,
        child=child,
        mode_count=1,
        parent_xform=parent_anchor,
        child_xform=child_anchor,
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)

    expected_frame = wp.transform_inverse(parent_xform_world * parent_anchor) * child_xform_world * child_anchor
    q_start = int(model.joint_q_start.numpy()[joint])
    np.testing.assert_allclose(model.joint_q.numpy()[q_start : q_start + 7], np.asarray(expected_frame), atol=1.0e-6)

    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    np.testing.assert_allclose(state.body_q.numpy()[child], np.asarray(child_xform_world), atol=1.0e-6)


def test_elastic_fk_sync(test, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), mode_count=1)
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    qd_start = int(model.joint_qd_start.numpy()[owner_joint])

    q = state.joint_q.numpy()
    q[q_start : q_start + 7] = [1.0, 2.0, 3.0, 0.0, 0.0, math.sin(0.25), math.cos(0.25)]
    q[q_start + 7] = 0.35
    state.joint_q.assign(q)

    qd = state.joint_qd.numpy()
    qd[qd_start : qd_start + 7] = [0.4, 0.5, 0.6, 0.1, 0.2, 0.3, -0.7]
    state.joint_qd.assign(qd)

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    np.testing.assert_allclose(state.body_q.numpy()[body], q[q_start : q_start + 7], atol=1.0e-6)
    np.testing.assert_allclose(state.body_qd.numpy()[body], qd[qd_start : qd_start + 6], atol=1.0e-6)
    np.testing.assert_allclose(state.joint_q.numpy()[q_start + 7], 0.35, atol=1.0e-7)


def test_elastic_fk_ik_com_velocity_round_trip(test, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        com=(0.2, -0.1, 0.3),
        inertia=_identity_inertia(),
        mode_count=1,
    )
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    qd_start = int(model.joint_qd_start.numpy()[owner_joint])

    q = state.joint_q.numpy()
    q[q_start : q_start + 7] = [0.3, -0.2, 0.4, 0.0, math.sin(0.2), 0.0, math.cos(0.2)]
    state.joint_q.assign(q)
    expected_qd = np.array([0.4, -0.5, 0.6, 0.7, -0.8, 0.9], dtype=np.float32)
    qd = state.joint_qd.numpy()
    qd[qd_start : qd_start + 6] = expected_qd
    state.joint_qd.assign(qd)

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    np.testing.assert_allclose(state.body_qd.numpy()[body], expected_qd, atol=1.0e-6)

    recovered_q = wp.zeros_like(state.joint_q)
    recovered_qd = wp.zeros_like(state.joint_qd)
    newton.eval_ik(model, state, recovered_q, recovered_qd)
    np.testing.assert_allclose(recovered_q.numpy()[q_start : q_start + 7], q[q_start : q_start + 7], atol=1.0e-6)
    np.testing.assert_allclose(recovered_qd.numpy()[qd_start : qd_start + 6], expected_qd, atol=1.0e-6)


def test_elastic_endpoint_shape_sampling(test, device):
    def shape_fn(x):
        return np.array([[2.0 * x[0], -x[1], 0.5]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_body(mass=1.0, inertia=_identity_inertia())
    child = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), mode_count=1, mode_shape_fn=shape_fn)
    joint = builder.add_joint_revolute(
        parent=parent,
        child=child,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.5, -0.25, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)

    test.assertEqual(int(model.joint_parent_elastic_endpoint.numpy()[joint]), -1)
    endpoint = int(model.joint_child_elastic_endpoint.numpy()[joint])
    test.assertGreaterEqual(endpoint, 0)
    phi = model.elastic_endpoint_phi.numpy().reshape((-1, 3))
    np.testing.assert_allclose(phi[endpoint], [1.0, 0.25, 0.5], atol=1.0e-7)


def test_elastic_endpoint_modal_basis_sampling(test, device):
    basis = newton.ModalBasis(
        sample_points=[[0.5, -0.25, 0.0]],
        sample_phi=[[[1.0, 0.25, 0.5]]],
        mode_mass=[1.0],
    )

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_body(mass=1.0, inertia=_identity_inertia())
    child = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), modal_basis=basis)
    joint = builder.add_joint_revolute(
        parent=parent,
        child=child,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.5, -0.25, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)

    endpoint = int(model.joint_child_elastic_endpoint.numpy()[joint])
    test.assertGreaterEqual(endpoint, 0)
    test.assertEqual(int(model.elastic_endpoint_sample.numpy()[endpoint]), 0)
    test.assertEqual(model.modal_basis_count, 1)
    np.testing.assert_allclose(model.elastic_basis.numpy(), [0], atol=0)
    phi = model.elastic_endpoint_phi.numpy().reshape((-1, 3))
    np.testing.assert_allclose(phi[endpoint], [1.0, 0.25, 0.5], atol=1.0e-7)


def test_elastic_endpoint_angular_sampling(test, device):
    length = 2.0
    attach_local = (-0.5 * length, 0.0, 0.0)
    gen = newton.ModalGeneratorBeam(
        length=length,
        half_width_y=0.1,
        half_width_z=0.1,
        mode_specs=[{"type": "bending_z", "boundary": "pinned-pinned"}],
        sample_count=9,
    )
    basis = gen.build()
    _phi, expected = basis.evaluate(np.array(attach_local, dtype=np.float32))
    test.assertGreater(float(np.linalg.norm(expected)), 1.0e-3)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_body(mass=1.0, inertia=_identity_inertia())
    child = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), modal_basis=basis)
    joint = builder.add_joint_fixed(
        parent=parent,
        child=child,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(*attach_local), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)

    endpoint = int(model.joint_child_elastic_endpoint.numpy()[joint])
    test.assertGreaterEqual(endpoint, 0)
    max_modes = int(model.elastic_max_mode_count)
    psi = model.elastic_endpoint_psi.numpy().reshape((-1, max_modes, 3))
    np.testing.assert_allclose(psi[endpoint, 0], expected[0], atol=1.0e-6)


def test_elastic_endpoint_angular_zero_without_beam(test, device):
    basis = newton.ModalBasis(
        sample_points=[[0.5, -0.25, 0.0]],
        sample_phi=[[[1.0, 0.25, 0.5]]],
        mode_mass=[1.0],
    )
    np.testing.assert_allclose(basis.sample_psi, np.zeros((1, 1, 3)), atol=0)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_body(mass=1.0, inertia=_identity_inertia())
    child = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), modal_basis=basis)
    joint = builder.add_joint_fixed(
        parent=parent,
        child=child,
        parent_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.5, -0.25, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)

    endpoint = int(model.joint_child_elastic_endpoint.numpy()[joint])
    test.assertGreaterEqual(endpoint, 0)
    max_modes = int(model.elastic_max_mode_count)
    psi = model.elastic_endpoint_psi.numpy().reshape((-1, max_modes, 3))
    np.testing.assert_allclose(psi[endpoint, 0], np.zeros(3), atol=0)


def _pure_twist_basis(
    clamp_local, has_psi: bool, mode_mass: float = 0.05, mode_stiffness: float = 2.0, mode_damping: float = 0.0
) -> "newton.ModalBasis":
    psi = [1.0, 0.0, 0.0] if has_psi else [0.0, 0.0, 0.0]
    return newton.ModalBasis(
        sample_points=[list(clamp_local)],
        sample_phi=[[[0.0, 0.0, 0.0]]],
        sample_psi=[[psi]],
        mode_mass=[mode_mass],
        mode_stiffness=[mode_stiffness],
        mode_damping=[mode_damping],
    )


def test_elastic_clamp_moment_reaction(test, device):
    clamp_local = (0.5, 0.0, 0.0)
    initial_twist = 0.3

    shape_cfg = newton.ModelBuilder.ShapeConfig()
    shape_cfg.density = 0.0
    shape_cfg.has_shape_collision = False
    shape_cfg.has_particle_collision = False

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    rigid_bodies = {}
    for name, has_psi, y in (("twist", True, 0.0), ("control", False, 2.0)):
        beam = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(0.0, y, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=_identity_inertia(),
            mode_q=[initial_twist],
            modal_basis=_pure_twist_basis(clamp_local, has_psi),
            label=f"twist_beam_{name}",
        )
        builder.add_shape_box(beam, hx=0.5, hy=0.05, hz=0.05, cfg=shape_cfg)
        rigid = builder.add_body(
            xform=wp.transform(wp.vec3(0.5, y, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
            label=f"twist_rigid_{name}",
        )
        builder.add_joint_fixed(
            parent=rigid,
            child=beam,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform(wp.vec3(*clamp_local), wp.quat_identity()),
        )
        rigid_bodies[name] = rigid

    builder.color()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    solver = newton.solvers.SolverVBD(model, iterations=24)

    identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    max_angle = dict.fromkeys(rigid_bodies, 0.0)
    dt = 1.0 / (60.0 * 8.0)
    for _ in range(120):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        body_q = state_0.body_q.numpy()
        if not np.isfinite(body_q).all():
            raise AssertionError("body transforms contain non-finite values")
        for name, body in rigid_bodies.items():
            max_angle[name] = max(max_angle[name], _quat_angle_error(body_q[body][3:7], identity_quat))

    test.assertGreater(
        max_angle["twist"],
        1.0e-2,
        f"clamp moment did not rotate the rigid body: max angle = {max_angle['twist']:.3e}",
    )
    test.assertLess(
        max_angle["control"],
        1.0e-3,
        f"rigid body rotated without angular mode shapes: max angle = {max_angle['control']:.3e}",
    )


def test_elastic_clamp_moment_reaction_parent_side(test, device):
    clamp_local = (0.5, 0.0, 0.0)
    initial_twist = 0.3

    shape_cfg = newton.ModelBuilder.ShapeConfig()
    shape_cfg.density = 0.0
    shape_cfg.has_shape_collision = False
    shape_cfg.has_particle_collision = False

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    rigid_bodies = {}
    twist_joint = -1
    for name, has_psi, y in (("twist", True, 0.0), ("control", False, 2.0)):
        beam = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(0.0, y, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=_identity_inertia(),
            mode_q=[initial_twist],
            modal_basis=_pure_twist_basis(clamp_local, has_psi),
            label=f"twist_beam_{name}",
        )
        builder.add_shape_box(beam, hx=0.5, hy=0.05, hz=0.05, cfg=shape_cfg)
        rigid = builder.add_body(
            xform=wp.transform(wp.vec3(0.5, y, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
            label=f"twist_rigid_{name}",
        )
        joint = builder.add_joint_fixed(
            parent=beam,
            child=rigid,
            parent_xform=wp.transform(wp.vec3(*clamp_local), wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
        if name == "twist":
            twist_joint = joint
        rigid_bodies[name] = rigid

    builder.color()
    model = builder.finalize(device=device)

    test.assertGreaterEqual(int(model.joint_parent_elastic_endpoint.numpy()[twist_joint]), 0)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    solver = newton.solvers.SolverVBD(model, iterations=24)

    identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    max_angle = dict.fromkeys(rigid_bodies, 0.0)
    dt = 1.0 / (60.0 * 8.0)
    for _ in range(120):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        body_q = state_0.body_q.numpy()
        if not np.isfinite(body_q).all():
            raise AssertionError("body transforms contain non-finite values")
        for name, body in rigid_bodies.items():
            max_angle[name] = max(max_angle[name], _quat_angle_error(body_q[body][3:7], identity_quat))

    test.assertGreater(
        max_angle["twist"],
        1.0e-2,
        f"clamp moment did not rotate the rigid body: max angle = {max_angle['twist']:.3e}",
    )
    test.assertLess(
        max_angle["control"],
        1.0e-3,
        f"rigid body rotated without angular mode shapes: max angle = {max_angle['control']:.3e}",
    )


def test_elastic_clamp_moment_conserves_angular_momentum(test, device):
    clamp_local = (0.5, 0.0, 0.0)
    initial_twist = 0.3
    inertia = wp.mat33(0.02, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05)

    shape_cfg = newton.ModelBuilder.ShapeConfig()
    shape_cfg.density = 0.0
    shape_cfg.has_shape_collision = False
    shape_cfg.has_particle_collision = False

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    beam = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=inertia,
        com=wp.vec3(0.0, 0.0, 0.0),
        mode_q=[initial_twist],
        modal_basis=_pure_twist_basis(clamp_local, True, mode_mass=1.0e6, mode_stiffness=0.0, mode_damping=0.0),
        label="momentum_beam",
    )
    builder.add_shape_box(beam, hx=0.5, hy=0.05, hz=0.05, cfg=shape_cfg)
    rigid = builder.add_body(
        xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=inertia,
        label="momentum_rigid",
    )
    builder.add_joint_fixed(
        parent=rigid,
        child=beam,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(*clamp_local), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    solver = newton.solvers.SolverVBD(model, iterations=128)

    max_spin = 0.0
    max_imbalance = 0.0
    max_off_axis = 0.0
    dt = 1.0 / (60.0 * 8.0)
    for _ in range(120):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        body_q = state_0.body_q.numpy()
        if not np.isfinite(body_q).all():
            raise AssertionError("body transforms contain non-finite values")
        spin_beam = float(body_q[beam][3])
        spin_rigid = float(body_q[rigid][3])
        max_spin = max(max_spin, abs(spin_beam), abs(spin_rigid))
        max_imbalance = max(max_imbalance, abs(spin_beam + spin_rigid))
        for body in (beam, rigid):
            max_off_axis = max(max_off_axis, abs(float(body_q[body][4])), abs(float(body_q[body][5])))

    test.assertGreater(max_spin, 1.0e-2, f"system did not rotate about the twist axis: max |qx| = {max_spin:.3e}")
    test.assertLess(max_off_axis, 1.0e-2, f"rotation left the twist axis: max off-axis |q| = {max_off_axis:.3e}")
    test.assertLess(
        max_imbalance,
        0.1 * max_spin,
        f"angular momentum not conserved at the joint: max |qx_beam + qx_rigid| = {max_imbalance:.3e} "
        f"vs max |qx| = {max_spin:.3e}",
    )


def test_modal_basis_shared_by_elastic_bodies(test, device):
    basis = newton.ModalGeneratorBeam(
        length=1.0,
        half_width_y=0.05,
        half_width_z=0.03,
        mode_specs=[{"type": newton.ModalGeneratorBeam.Mode.AXIAL}],
        sample_count=3,
    ).build()

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
        modal_basis=basis,
    )
    builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, 0.4, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
        modal_basis=basis,
    )
    builder.color()
    model = builder.finalize(device=device)

    test.assertEqual(model.elastic_body_count, 2)
    test.assertEqual(model.modal_basis_count, 1)
    np.testing.assert_allclose(model.elastic_basis.numpy(), [0, 0], atol=0)


def test_elastic_render_shape_sampling(test, device):
    def shape_fn(x):
        xi = x[0] / 0.5
        return np.array([[0.0, 1.0 - xi * xi, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_shape_fn=shape_fn,
    )
    builder.add_shape_box(body, hx=0.5, hy=0.05, hz=0.03)
    builder.color()
    model = builder.finalize(device=device)

    test.assertEqual(model.elastic_render_point_total_count, 33)
    test.assertEqual(int(model.elastic_render_point_start.numpy()[0]), 0)
    test.assertEqual(int(model.elastic_render_point_count.numpy()[0]), 33)

    local = model.elastic_render_point_local.numpy()
    np.testing.assert_allclose(local[0, 0], -0.5, atol=1.0e-7)
    np.testing.assert_allclose(local[-1, 0], 0.5, atol=1.0e-7)
    test.assertGreater(local[0, 2], 0.03)

    phi = model.elastic_render_point_phi.numpy().reshape((-1, 3))
    np.testing.assert_allclose(phi[16], [0.0, 1.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(phi[0], [0.0, 0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(phi[-1], [0.0, 0.0, 0.0], atol=1.0e-6)


def test_elastic_shape_mesh_sampling(test, device):
    length = 1.0
    hy = 0.05
    hz = 0.03

    def shape_fn(x):
        s = float(x[0] + 0.5 * length)
        xi = s / length
        phi = math.sin(math.pi * xi)
        slope = (math.pi / length) * math.cos(math.pi * xi)
        return np.array([[-float(x[1]) * slope, phi, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), mode_count=1, mode_shape_fn=shape_fn)
    builder.add_shape_box(body, hx=0.5 * length, hy=hy, hz=hz)
    builder.color()
    model = builder.finalize(device=device)

    test.assertEqual(model.elastic_shape_count, 1)
    test.assertGreater(model.elastic_shape_vertex_total_count, 8)
    test.assertGreater(model.elastic_shape_index_total_count, 36)
    test.assertEqual(int(model.elastic_shape_shape.numpy()[0]), 0)

    local = model.elastic_shape_vertex_local.numpy()
    phi = model.elastic_shape_vertex_phi.numpy().reshape((-1, 3))
    mid = int(np.argmin(np.abs(local[:, 0]) + np.abs(local[:, 1] - hy) + np.abs(local[:, 2] - hz)))
    np.testing.assert_allclose(phi[mid], [0.0, 1.0, 0.0], atol=1.0e-6)

    left = np.where(np.isclose(local[:, 0], -0.5 * length))[0]
    right = np.where(np.isclose(local[:, 0], 0.5 * length))[0]
    np.testing.assert_allclose(phi[left, 1], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(phi[right, 1], 0.0, atol=1.0e-6)


def test_elastic_shape_box_winding(test, device):
    hx = 0.5
    hy = 0.05
    hz = 0.03

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), mode_count=1)
    builder.add_shape_box(body, hx=hx, hy=hy, hz=hz)
    builder.color()
    model = builder.finalize(device=device)

    vertices = model.elastic_shape_vertex_local.numpy()
    indices = model.elastic_shape_indices.numpy()
    test.assertGreater(len(indices), 36)

    for tri in indices.reshape((-1, 3)):
        a, b, c = vertices[tri]
        normal = np.cross(b - a, c - a)
        center = (a + b + c) / 3.0
        axis = int(np.argmax(np.abs(center / np.array([hx, hy, hz], dtype=np.float32))))
        expected_sign = 1.0 if center[axis] > 0.0 else -1.0
        test.assertGreater(float(normal[axis] * expected_sign), 0.0)


def test_elastic_shape_box_exact_modal_samples(test, device):
    length = 0.8
    hy = 0.04
    hz = 0.03
    surface_points, _ = box_surface_mesh(length, hy, hz)
    for axis in range(3):
        axis_values = np.unique(np.round(surface_points[:, axis], decimals=8))
        test.assertLessEqual(float(np.max(np.diff(axis_values))), 0.010001)
    test.assertGreater(np.unique(surface_points[:, 1]).size, 2)
    test.assertGreater(np.unique(surface_points[:, 2]).size, 2)

    basis = newton.ModalGeneratorBeam(
        length=length,
        half_width_y=hy,
        half_width_z=hz,
        mode_specs=[
            {
                "type": newton.ModalGeneratorBeam.Mode.BENDING_Z,
                "boundary": newton.ModalGeneratorBeam.Boundary.CANTILEVER_TIP,
            }
        ],
    ).build(
        sample_points=beam_render_sample_points(
            length,
            hy,
            hz,
            extra_points=((-0.5 * length, 0.0, 0.0), (0.5 * length, 0.0, 0.0)),
        )
    )

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), modal_basis=basis)
    builder.add_shape_box(body, hx=0.5 * length, hy=hy, hz=hz)
    builder.color()
    model = builder.finalize(device=device)

    samples = model.elastic_shape_vertex_sample.numpy()
    np.testing.assert_array_equal(samples, np.arange(surface_points.shape[0], dtype=np.int32))
    np.testing.assert_allclose(model.elastic_shape_vertex_local.numpy(), surface_points, atol=1.0e-7)

    phi = model.elastic_shape_vertex_phi.numpy().reshape((-1, model.elastic_max_mode_count, 3))[:, 0]
    expected_phi = np.asarray([basis.sample_phi[i][0] for i in range(surface_points.shape[0])], dtype=np.float32)
    np.testing.assert_allclose(phi, expected_phi, atol=1.0e-7)


def _build_elastic_ground_contact_model(device, z: float, q0: float = 0.0, is_kinematic: bool = True):
    def downward_shape_fn(_x):
        return np.array([[0.0, 0.0, -1.0]], dtype=np.float32)

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.ke = 1000.0
    cfg.kd = 0.0
    cfg.mu = 0.0
    cfg.margin = 0.0
    cfg.gap = 0.0

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis="Z")
    builder.add_ground_plane(cfg=cfg)
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_mass=[0.0 if is_kinematic else 1.0],
        mode_stiffness=[0.0 if is_kinematic else 100.0],
        mode_damping=[0.0],
        mode_q=[q0],
        mode_shape_fn=downward_shape_fn,
        is_kinematic=is_kinematic,
    )
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    builder.color()
    return builder.finalize(device=device), body


def test_elastic_surface_contact_generation(test, device):
    model, body = _build_elastic_ground_contact_model(device, z=0.04)
    state = model.state()
    contacts = model.contacts()
    model.collide(state, contacts)

    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)

    samples0 = contacts.rigid_contact_elastic_sample0.numpy()[:count]
    samples1 = contacts.rigid_contact_elastic_sample1.numpy()[:count]
    shapes0 = contacts.rigid_contact_shape0.numpy()[:count]
    shapes1 = contacts.rigid_contact_shape1.numpy()[:count]
    shape_body = model.shape_body.numpy()
    active = np.logical_and(shapes0 >= 0, shapes1 >= 0)
    test.assertTrue(bool(np.all(active)))

    np.testing.assert_array_equal(samples0[active], -np.ones_like(samples0[active]))
    test.assertTrue(bool(np.any(samples1[active] >= 0)))
    test.assertTrue(bool(np.all(shape_body[shapes0[active]] == -1)))
    test.assertTrue(bool(np.all(shape_body[shapes1[active]] == body)))


def test_elastic_surface_contact_work_uses_pair_vertex_ranges(test, device):
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]]],
        mode_mass=[1.0],
        mode_stiffness=[1.0],
    )
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for x in (-0.2, 0.2):
        body = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(x, 0.0, 0.2), wp.quat_identity()),
            mass=1.0,
            inertia=_identity_inertia(),
            modal_basis=basis,
        )
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")

    shape_to_count = dict(
        zip(model.elastic_shape_shape.numpy().tolist(), model.elastic_shape_vertex_count.numpy().tolist(), strict=True)
    )
    shape_body = model.shape_body.numpy()
    body_elastic_index = model.body_elastic_index.numpy()
    expected_work = 0
    for shape_a, shape_b in pipeline.elastic_shape_pairs.numpy():
        body_a = int(shape_body[shape_a])
        elastic_shape = int(shape_a) if body_a >= 0 and body_elastic_index[body_a] >= 0 else int(shape_b)
        expected_work += int(shape_to_count[elastic_shape])

    test.assertEqual(pipeline.elastic_contact_work_count, expected_work)
    test.assertLess(
        pipeline.elastic_contact_work_count,
        pipeline.elastic_shape_pairs_max * model.elastic_shape_vertex_total_count,
    )


def test_elastic_surface_contact_deterministic_sort_preserves_samples(test, device):
    model, _body = _build_elastic_ground_contact_model(device, z=0.04)
    state = model.state()
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)

    count = min(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
    test.assertGreater(count, 1)
    samples = contacts.rigid_contact_elastic_sample1.numpy()[:count]
    points = contacts.rigid_contact_point1.numpy()[:count]
    rest_points = model.elastic_shape_vertex_local.numpy()
    elastic_rows = samples >= 0
    test.assertTrue(bool(np.any(elastic_rows)))
    np.testing.assert_allclose(points[elastic_rows], rest_points[samples[elastic_rows]], atol=0.0)


def test_elastic_surface_contact_reduction_preserves_modal_samples(test, device):
    model, _body = _build_elastic_ground_contact_model(device, z=0.04)
    state = model.state()

    reduced_pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=True)
    reduced_contacts = reduced_pipeline.contacts()
    reduced_pipeline.collide(state, reduced_contacts)
    reduced_count = min(int(reduced_contacts.rigid_contact_count.numpy()[0]), reduced_contacts.rigid_contact_max)

    full_pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
    full_contacts = full_pipeline.contacts()
    full_pipeline.collide(state, full_contacts)
    full_count = min(int(full_contacts.rigid_contact_count.numpy()[0]), full_contacts.rigid_contact_max)

    test.assertGreater(reduced_count, 0)
    test.assertLess(reduced_count, full_count)
    samples = reduced_contacts.rigid_contact_elastic_sample1.numpy()[:reduced_count]
    test.assertTrue(bool(np.all(samples >= 0)))
    test.assertTrue(bool(np.all(samples < model.elastic_shape_vertex_total_count)))


def test_elastic_surface_contact_uses_deformed_vertex(test, device):
    model, _body = _build_elastic_ground_contact_model(device, z=0.07)
    state = model.state()
    contacts = model.contacts()

    model.collide(state, contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)

    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    q = state.joint_q.numpy()
    q[q_start + 7] = 0.03
    state.joint_q.assign(q)

    model.collide(state, contacts)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_vbd_elastic_contact_solves_modal_penetration(test, device):
    q0 = 0.005
    model, _body = _build_elastic_ground_contact_model(device, z=0.05, q0=q0)
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    control = model.control()

    model.collide(state_0, contacts)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_contact_k_start=1000.0,
        rigid_avbd_contact_alpha=0.0,
        rigid_body_contact_buffer_size=128,
        elastic_body_relaxation=1.0,
        elastic_solver_metrics=True,
    )
    solver.step(state_0, state_1, control, contacts, 0.01)

    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    test.assertLess(abs(float(state_1.joint_q.numpy()[q_start + 7])), 1.0e-5)

    metrics = solver.elastic_mode_solve_metrics()
    for values in metrics.values():
        test.assertEqual(values.shape, (1,))
        test.assertTrue(bool(np.isfinite(values).all()))
    initial_residual = float(metrics["initial_residual_norm"][0])
    test.assertGreater(initial_residual, 0.0)
    test.assertLess(float(metrics["solve_residual_norm"][0]) / initial_residual, 1.0e-6)
    test.assertLess(float(metrics["applied_residual_norm"][0]) / initial_residual, 1.0e-6)
    test.assertGreater(float(metrics["update_norm"][0]), 1.0e-4)
    test.assertLess(float(metrics["update_norm"][0]), 1.0e-2)


def test_vbd_elastic_body_relaxation_is_general_and_validated(test, device):
    model, _body = _build_elastic_ground_contact_model(device, z=0.2)
    solver = newton.solvers.SolverVBD(model, elastic_body_relaxation=0.37)

    contact_max, _contact_count, relaxation = solver._elastic_solve_inputs(None)
    test.assertEqual(contact_max, 0)
    test.assertAlmostEqual(relaxation, 0.37)
    for values in solver.elastic_mode_solve_metrics().values():
        test.assertEqual(values.shape, (0,))
    for invalid in (-0.1, 1.1, np.nan, np.inf):
        with test.assertRaisesRegex(ValueError, "elastic_body_relaxation"):
            newton.solvers.SolverVBD(model, elastic_body_relaxation=invalid)


def test_vbd_elastic_compliant_contact_solves_modal_penetration(test, device):
    initial_modal_q = 0.005
    model, _body = _build_elastic_ground_contact_model(device, z=0.05, q0=initial_modal_q)
    model.elastic_mode_mass.fill_(1.0)
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    model.collide(state_0, contacts)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=4,
        rigid_compliant_alm=True,
        rigid_avbd_contact_alpha=0.0,
        rigid_body_contact_buffer_size=128,
        elastic_body_relaxation=1.0,
    )
    solver.step(state_0, state_1, model.control(), contacts, 0.01)

    count = min(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
    samples = contacts.rigid_contact_elastic_sample1.numpy()[:count]
    elastic_rows = samples >= 0
    normals = contacts.rigid_contact_normal.numpy()[:count][elastic_rows]
    C0 = solver.body_body_contact_C0.numpy()[:count][elastic_rows]
    normal_C0 = np.einsum("ij,ij->i", C0, normals)
    test.assertTrue(bool(np.any(elastic_rows)))
    np.testing.assert_allclose(normal_C0, 0.005, atol=1.0e-6)

    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    modal_q = abs(float(state_1.joint_q.numpy()[q_start + 7]))
    test.assertGreater(modal_q, 0.0)
    test.assertLess(modal_q, 0.4 * initial_modal_q)


def test_vbd_elastic_compliant_contact_uses_modal_mobility_for_rho_and_forces(test, device):
    """A kinematic frame with dynamic modes conditions and reports ALM contact consistently."""
    model, _body = _build_elastic_ground_contact_model(device, z=0.05, q0=0.005)
    model.elastic_mode_mass.fill_(1.0)
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    model.collide(state_0, contacts)
    body_q_prev = wp.clone(state_0.body_q)
    joint_q_prev = wp.clone(state_0.joint_q)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_avbd_contact_alpha=0.0,
        rigid_body_contact_buffer_size=128,
        elastic_body_relaxation=0.0,
    )
    solver.step(state_0, state_1, model.control(), contacts, 0.01)

    count = min(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
    elastic_rows = contacts.rigid_contact_elastic_sample1.numpy()[:count] >= 0
    test.assertTrue(bool(np.any(elastic_rows)))
    normal_rho = solver.body_body_contact_normal_rho.numpy()[:count][elastic_rows]
    np.testing.assert_allclose(normal_rho, 1.0e4, rtol=2.0e-5)

    _body0, _body1, _point0, _point1, force, _count = solver.collect_rigid_contact_forces(
        state_1.body_q,
        body_q_prev,
        contacts,
        0.01,
        joint_q=state_1.joint_q,
        joint_q_prev=joint_q_prev,
    )
    force_norm = np.linalg.norm(force.numpy()[:count][elastic_rows], axis=1)
    multiplier_norm = np.linalg.norm(solver.body_body_contact_lambda.numpy()[:count][elastic_rows], axis=1)
    test.assertGreater(float(np.max(multiplier_norm)), 0.0)
    test.assertGreater(float(np.max(force_norm)), 0.0)


def test_vbd_elastic_contact_overflow_assembles_consistent_block(test, device):
    """Overflowing the per-body contact list must not truncate only the rigid frame block."""

    def assemble_block(contact_buffer_size: int):
        model, _body = _build_elastic_ground_contact_model(device, z=0.04, is_kinematic=False)
        state_0 = model.state()
        state_1 = model.state()
        pipeline = newton.CollisionPipeline(model, broad_phase="explicit", deterministic=True)
        contacts = pipeline.contacts()
        pipeline.collide(state_0, contacts)
        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertGreater(contact_count, 1)

        solver = newton.solvers.SolverVBD(
            model,
            iterations=1,
            rigid_contact_k_start=1000.0,
            rigid_body_contact_buffer_size=contact_buffer_size,
            elastic_body_relaxation=1.0,
        )
        solver.step(state_0, state_1, model.control(), contacts, 0.01)

        matrix_upper = solver.elastic_body_block_matrix.numpy()[0]
        matrix = np.triu(matrix_upper) + np.triu(matrix_upper, 1).T
        return contact_count, matrix, solver.elastic_body_block_delta.numpy()[0]

    contact_count, overflow_matrix, overflow_delta = assemble_block(contact_buffer_size=1)
    _, complete_matrix, complete_delta = assemble_block(contact_buffer_size=contact_count)

    np.testing.assert_allclose(overflow_matrix, complete_matrix, rtol=2.0e-6, atol=1.0e-5)
    np.testing.assert_allclose(overflow_delta, complete_delta, rtol=2.0e-6, atol=1.0e-7)
    test.assertGreater(float(np.linalg.eigvalsh(overflow_matrix)[0]), 0.0)


def test_elastic_contact_local_mat33_projection_matches_world(test, device):
    del device

    quat = wp.quat_from_axis_angle(wp.vec3(0.3, -0.7, 0.2), 0.83)
    quat_np = np.array([quat[0], quat[1], quat[2], quat[3]], dtype=float)
    rot = np.column_stack(
        (
            _quat_rotate_np(quat_np, np.array([1.0, 0.0, 0.0])),
            _quat_rotate_np(quat_np, np.array([0.0, 1.0, 0.0])),
            _quat_rotate_np(quat_np, np.array([0.0, 0.0, 1.0])),
        )
    )
    phi_local = np.array(
        [
            [0.35, -0.12, 0.08],
            [-0.18, 0.24, 0.31],
            [0.07, 0.42, -0.16],
            [-0.29, -0.05, 0.23],
        ],
        dtype=float,
    )
    force_world = np.array([4.0, -7.0, 2.5], dtype=float)
    hessian_world = np.array(
        [
            [18.0, -2.0, 4.0],
            [-2.0, 11.0, 1.5],
            [4.0, 1.5, 7.0],
        ],
        dtype=float,
    )

    phi_world = phi_local @ rot.T
    grad_world = -(phi_world @ force_world)
    hessian_modal_world = phi_world @ hessian_world @ phi_world.T

    force_local = rot.T @ force_world
    hessian_local = rot.T @ hessian_world @ rot
    grad_local = -(phi_local @ force_local)
    hessian_modal_local = phi_local @ hessian_local @ phi_local.T

    np.testing.assert_allclose(grad_local, grad_world, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(hessian_modal_local, hessian_modal_world, rtol=1.0e-12, atol=1.0e-12)


def test_elastic_strain_visualization_colors(test, device):
    length = 1.0

    def axial_gradient_shape(x):
        s = float(x[0] + 0.5 * length)
        return np.array([[s / length, 0.0, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_shape_fn=axial_gradient_shape,
    )
    builder.add_shape_box(body, hx=0.5 * length, hy=0.04, hz=0.03)
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()

    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])

    viewer = _ElasticMeshColorProbe()
    viewer.set_model(model)
    viewer.show_elastic_strain = True

    q = state.joint_q.numpy()
    q[q_start + 7] = 0.04
    state.joint_q.assign(q)
    viewer.log_state(state)
    colors_small = viewer.mesh_colors["/model/elastic_shapes/shape_0"].copy()

    q[q_start + 7] = 0.08
    state.joint_q.assign(q)
    viewer.log_state(state)

    test.assertIn("/model/elastic_shapes/shape_0", viewer.mesh_colors)
    colors = viewer.mesh_colors["/model/elastic_shapes/shape_0"]
    test.assertIsNotNone(colors)
    test.assertEqual(colors.shape[1], 3)
    test.assertGreater(float(np.max(np.abs(colors - colors_small))), 0.2)
    test.assertGreater(float(np.ptp(colors[:, 0])), 0.1)
    test.assertGreater(float(np.ptp(colors[:, 2])), 0.1)
    test.assertTrue(bool(np.all(colors >= -1.0e-6)))
    test.assertTrue(bool(np.all(colors <= 1.0 + 1.0e-6)))


def test_elastic_rendering_respects_layer_transform_and_visibility(test, device):
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 0.0]]],
        mode_mass=[1.0],
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body_position = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(*body_position), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
        modal_basis=basis,
    )
    builder.add_shape_box(body, hx=0.05, hy=0.04, hz=0.03)
    builder.color()
    model = builder.finalize(device=device)

    viewer = _ElasticMeshColorProbe()
    viewer.activate("elastic")
    viewer.set_model(model)
    layer_offset = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    viewer.set_layer_transform("elastic", tuple(layer_offset))
    viewer.log_state(model.state())

    test.assertEqual(len(viewer.mesh_points), 1)
    points = next(iter(viewer.mesh_points.values()))
    expected = model.elastic_shape_vertex_local.numpy() + body_position + layer_offset
    np.testing.assert_allclose(points, expected, atol=1.0e-6)

    viewer.set_layer_visible("elastic", False)
    viewer.log_state(model.state())
    test.assertTrue(all(viewer.mesh_hidden.values()))


def test_torsion_render_shape_sampling(test, device):
    length = 1.0
    hy = 0.06
    hz = 0.025

    def torsion_shape_fn(x):
        s = float(x[0] + 0.5 * length)
        twist = s / length
        return np.array([[0.0, -float(x[2]) * twist, float(x[1]) * twist]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), mode_count=1, mode_shape_fn=torsion_shape_fn)
    builder.add_shape_box(body, hx=0.5 * length, hy=hy, hz=hz)
    builder.color()
    model = builder.finalize(device=device)

    np.testing.assert_allclose(torsion_shape_fn(np.array([0.0, 0.0, 0.0], dtype=np.float32))[0], 0.0, atol=1.0e-7)

    local = model.elastic_shape_vertex_local.numpy()
    phi = model.elastic_shape_vertex_phi.numpy().reshape((-1, 3))
    tip = int(np.argmin(np.abs(local[:, 0] - 0.5 * length) + np.abs(local[:, 1] - hy) + np.abs(local[:, 2] - hz)))
    np.testing.assert_allclose(phi[tip], [0.0, -hz, hy], atol=1.0e-6)


def test_finite_torsion_exemplar_preserves_volume(test, device):
    length = 1.0
    hy = 0.085
    hz = 0.05
    tip_twist = math.pi / 2.0
    vertices, indices = box_surface_mesh(length, hy, hz)
    rest_volume = mesh_volume(vertices, indices)

    finite_vertices = vertices + finite_torsion_displacement(vertices, length, tip_twist)
    finite_ratio = mesh_volume(finite_vertices, indices) / rest_volume
    np.testing.assert_allclose(finite_ratio, 1.0, atol=2.0e-3)

    s = np.clip(vertices[:, 0] + 0.5 * length, 0.0, length)
    theta = tip_twist * s / length
    linear_vertices = vertices.copy()
    linear_vertices[:, 1] -= vertices[:, 2] * theta
    linear_vertices[:, 2] += vertices[:, 1] * theta
    linear_ratio = mesh_volume(linear_vertices, indices) / rest_volume
    test.assertGreater(linear_ratio, 1.5)


def test_finite_torsion_pod_modes_project_exemplar(test, device):
    length = 1.0
    hy = 0.085
    hz = 0.05
    tip_twist = math.pi / 2.0
    mode_count = 8
    vertices, indices = box_surface_mesh(length, hy, hz)
    sample_points = beam_render_sample_points(
        length,
        hy,
        hz,
        extra_points=((-0.5 * length, 0.0, 0.0), (0.5 * length, 0.0, 0.0)),
    )
    snapshot_amplitudes = np.linspace(-1.0, 1.0, 17, dtype=np.float32)
    snapshot_amplitudes = snapshot_amplitudes[np.abs(snapshot_amplitudes) > 1.0e-6]
    snapshots = np.asarray(
        [
            finite_torsion_displacement(sample_points, length, tip_twist * float(amplitude))
            for amplitude in snapshot_amplitudes
        ],
        dtype=np.float32,
    )

    basis = newton.ModalGeneratorPOD(
        sample_points=sample_points,
        displacements=snapshots,
        mode_count=mode_count,
        total_mass=1.0,
        stiffness_scale=1.0,
    ).build()
    phi = basis.sample_phi.reshape((sample_points.shape[0], mode_count, 3))
    projection_matrix = np.transpose(phi, (0, 2, 1)).reshape((-1, mode_count))
    target = finite_torsion_displacement(sample_points, length, tip_twist)
    q, *_ = np.linalg.lstsq(projection_matrix, target.reshape(-1), rcond=None)
    projected = np.einsum("smc,m->sc", phi, q)

    polar_moment = (4.0 / 3.0) * hy * hz**3 + (4.0 / 3.0) * hz * hy**3
    modal_mass, modal_stiffness = beam_torsion_linear_modal_properties(
        length,
        hy,
        hz,
        density=500.0,
        shear_modulus=4.0e4,
    )
    np.testing.assert_allclose(modal_mass, 500.0 * polar_moment * length / 3.0, rtol=1.0e-6)
    np.testing.assert_allclose(modal_stiffness, 4.0e4 * polar_moment / length, rtol=1.0e-6)

    test.assertEqual(basis.mode_count, mode_count)
    test.assertLess(float(np.max(np.linalg.norm(projected - target, axis=1))), 1.0e-5)
    ratio = mesh_volume(vertices + projected[: vertices.shape[0]], indices) / mesh_volume(vertices, indices)
    np.testing.assert_allclose(ratio, 1.0, atol=2.0e-3)


def test_cantilever_tip_load_solution(test, device):
    length = 0.8
    ei = 0.32
    tip_load = 0.2
    stiffness = 3.0 * ei / (length**3)

    def cantilever_shape_fn(x):
        s = float(x[0] + 0.5 * length)
        s = min(max(s, 0.0), length)
        phi = (s * s * (3.0 * length - s)) / (2.0 * length**3)
        slope = (3.0 * s * (2.0 * length - s)) / (2.0 * length**3)
        return np.array([[-float(x[2]) * slope, 0.0, phi]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_mass=[0.0],
        mode_stiffness=[stiffness],
        mode_damping=[0.0],
        mode_shape_fn=cantilever_shape_fn,
    )
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    qd_start = int(model.joint_qd_start.numpy()[owner_joint])
    jf = control.joint_f.numpy()
    jf[qd_start + 6] = tip_load
    control.joint_f.assign(jf)

    solver = newton.solvers.SolverVBD(model, iterations=0)
    solver.step(state_0, state_1, control, None, 0.01)

    expected_tip_deflection = tip_load * length**3 / (3.0 * ei)
    np.testing.assert_allclose(state_1.joint_q.numpy()[q_start + 7], expected_tip_deflection, atol=1.0e-7)
    tip_phi = cantilever_shape_fn(np.array([0.5 * length, 0.0, 0.0], dtype=np.float32))[0]
    np.testing.assert_allclose(tip_phi, [0.0, 0.0, 1.0], atol=1.0e-7)


def test_cantilever_modal_vibration_solution(test, device):
    length = 0.9
    basis = newton.ModalGeneratorBeam(
        length=length,
        half_width_y=0.045,
        half_width_z=0.035,
        mode_specs=[
            {
                "type": newton.ModalGeneratorBeam.Mode.BENDING_Z,
                "boundary": newton.ModalGeneratorBeam.Boundary.CANTILEVER_TIP,
            }
        ],
        density=250.0,
        young_modulus=3.2e7,
        damping_ratio=0.001,
    ).build()

    mode_mass = float(basis.mode_mass[0])
    stiffness = float(basis.mode_stiffness[0])
    damping = float(basis.mode_damping[0])
    q_expected = 0.09
    qd_expected = 0.0
    dt = 1.0 / 480.0
    min_q = q_expected

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_q=[q_expected],
        mode_qd=[qd_expected],
        modal_basis=basis,
    )
    builder.add_shape_box(body, hx=0.5 * length, hy=0.045, hz=0.035)
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    qd_start = int(model.joint_qd_start.numpy()[owner_joint])

    solver = newton.solvers.SolverVBD(model, iterations=0)
    for _ in range(180):
        solver.step(state_0, state_1, control, None, dt)
        denom = mode_mass + dt * damping + dt * dt * stiffness
        qd_expected = (mode_mass * qd_expected - dt * stiffness * q_expected) / denom
        q_expected = q_expected + dt * qd_expected
        min_q = min(min_q, q_expected)
        state_0, state_1 = state_1, state_0

    np.testing.assert_allclose(state_0.joint_q.numpy()[q_start + 7], q_expected, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(state_0.joint_qd.numpy()[qd_start + 6], qd_expected, rtol=1.0e-6, atol=1.0e-5)
    test.assertLess(min_q, -0.01)


def test_elastic_modal_implicit_solution_vbd(test, device):
    mode_mass = 2.0
    stiffness = 18.0
    damping = 0.5
    q0 = 0.1
    v0 = -0.2
    force = 1.3
    dt = 0.01

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_mass=[mode_mass],
        mode_stiffness=[stiffness],
        mode_damping=[damping],
        mode_q=[q0],
        mode_qd=[v0],
    )
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    qd_start = int(model.joint_qd_start.numpy()[owner_joint])
    jf = control.joint_f.numpy()
    jf[qd_start + 6] = force
    control.joint_f.assign(jf)

    solver = newton.solvers.SolverVBD(model, iterations=0)
    solver.step(state_0, state_1, control, None, dt)

    v_expected = (mode_mass * v0 + dt * (force - stiffness * q0)) / (mode_mass + dt * damping + dt * dt * stiffness)
    q_expected = q0 + dt * v_expected
    np.testing.assert_allclose(state_0.joint_q.numpy()[q_start + 7], q0, rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(state_0.joint_qd.numpy()[qd_start + 6], v0, rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(state_1.joint_q.numpy()[q_start + 7], q_expected, rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(state_1.joint_qd.numpy()[qd_start + 6], v_expected, rtol=1.0e-6, atol=1.0e-7)


def test_elastic_gravity_modal_force(test, device):
    g = 9.81
    stiffness = 18.0
    dt = 0.01
    basis = newton.ModalBasis(
        sample_points=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]],
        sample_mass=[1.0, 1.0],
        mode_stiffness=[stiffness],
        mode_damping=[0.0],
    )
    builder = newton.ModelBuilder(gravity=-g)
    body = builder.add_body_elastic(
        mass=1.0, inertia=_identity_inertia(), modal_basis=basis, mode_q=[0.0], mode_qd=[0.0], is_kinematic=True
    )
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])

    solver = newton.solvers.SolverVBD(model, iterations=16)
    solver.step(state_0, state_1, control, None, dt)

    coupling_z = 2.0
    modal_mass = 2.0
    modal_force = coupling_z * (-g)
    q_expected = modal_force / (stiffness + modal_mass / dt**2)
    np.testing.assert_allclose(state_1.joint_q.numpy()[q_start + 7], q_expected, rtol=1.0e-4, atol=1.0e-9)


def test_elastic_free_fall_no_modal_force(test, device):
    g = 9.81
    dt = 0.005
    basis = newton.ModalBasis(
        sample_points=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]],
        sample_mass=[1.0, 1.0],
        mode_stiffness=[18.0],
        mode_damping=[0.0],
    )
    builder = newton.ModelBuilder(gravity=-g)
    body = builder.add_body_elastic(
        mass=1.0, inertia=_identity_inertia(), modal_basis=basis, mode_q=[0.0], mode_qd=[0.0]
    )
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])

    solver = newton.solvers.SolverVBD(model, iterations=16)
    max_abs_q = 0.0
    for _ in range(40):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        max_abs_q = max(max_abs_q, abs(float(state_0.joint_q.numpy()[q_start + 7])))
    test.assertLess(max_abs_q, 1.0e-3)


def test_elastic_euler_modal_force(test, device):
    length = 0.8
    n = 17
    xs = np.linspace(0.0, length, n, dtype=np.float32)
    sample_points = np.column_stack([xs, np.zeros(n), np.zeros(n)]).astype(np.float32)
    sample_phi = np.zeros((n, 1, 3), dtype=np.float32)
    sample_phi[:, 0, 2] = (xs * xs * (3.0 * length - xs)) / (2.0 * length**3)
    basis = newton.ModalBasis(
        sample_points=sample_points,
        sample_phi=sample_phi,
        sample_mass=np.full(n, 1.0 / n, dtype=np.float32),
        mode_stiffness=[40.0],
        mode_damping=[3.0],
    )

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = builder.add_body(xform=wp.transform_identity(), mass=2.0, inertia=_identity_inertia(), is_kinematic=True)
    beam = builder.add_body_elastic(
        xform=wp.transform_identity(), mass=0.2, inertia=_identity_inertia(), modal_basis=basis
    )
    builder.add_joint_fixed(
        parent=base, child=beam, parent_xform=wp.transform_identity(), child_xform=wp.transform_identity()
    )
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    base_q_start = base_qd_start = -1
    for j in range(len(joint_child)):
        if int(joint_child[j]) == base and int(joint_parent[j]) == -1:
            base_q_start = int(model.joint_q_start.numpy()[j])
            base_qd_start = int(model.joint_qd_start.numpy()[j])
    owner = int(model.elastic_joint.numpy()[0])
    q_index = int(model.joint_q_start.numpy()[owner]) + 7

    stiffness = float(model.elastic_mode_stiffness.numpy()[0])
    coupling_angular = np.array(basis.mode_coupling_angular[0], dtype=np.float64)
    coupling_centrifugal = np.array(basis.mode_coupling_centrifugal[0], dtype=np.float64)

    joint_k = 1.0e7
    solver = newton.solvers.SolverVBD(
        model,
        iterations=24,
        rigid_joint_linear_k_start=joint_k,
        rigid_joint_angular_k_start=joint_k,
        rigid_joint_linear_ke=joint_k,
        rigid_joint_angular_ke=joint_k,
        rigid_joint_linear_kd=0.0,
        rigid_joint_angular_kd=0.0,
        rigid_joint_adaptive_stiffness=False,
    )
    dt = 1.0 / 480.0
    alpha = 2.0

    max_abs_q = 0.0
    t = 0.0
    for _ in range(240):
        omega = alpha * t
        angle = 0.5 * alpha * t * t
        quat = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
        targets = {base: (wp.vec3(0.0, 0.0, 0.0), quat)}
        velocities = {base: (wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, omega, 0.0))}
        apply_kinematic_targets(state_0, {base: base_q_start}, targets, velocities, {base: base_qd_start})
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        max_abs_q = max(max_abs_q, abs(float(state_0.joint_q.numpy()[q_index])))
        t += dt

    q_final = float(state_0.joint_q.numpy()[q_index])
    omega_final = alpha * t
    euler_force = float(np.dot(coupling_angular, np.array([0.0, alpha, 0.0])))
    centrifugal_force = omega_final * omega_final * (coupling_centrifugal[0, 0] + coupling_centrifugal[2, 2])
    q_expected = (euler_force + centrifugal_force) / stiffness

    test.assertGreater(max_abs_q, 5.0e-3)
    test.assertGreater(q_final, 0.0)
    np.testing.assert_allclose(q_final, q_expected, rtol=0.05, atol=1.0e-4)


def _elastic_free_joint_starts(model, body):
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    for j in range(len(joint_child)):
        if int(joint_child[j]) == body and int(joint_parent[j]) == -1:
            return int(model.joint_q_start.numpy()[j]), int(model.joint_qd_start.numpy()[j])
    raise RuntimeError("no free joint")


def test_elastic_centrifugal_modal_force(test, device):
    length = 1.0
    n = 21
    xs = np.linspace(0.0, length, n, dtype=np.float32)
    sample_points = np.column_stack([xs, np.zeros(n), np.zeros(n)]).astype(np.float32)
    sample_phi = np.zeros((n, 1, 3), dtype=np.float32)
    sample_phi[:, 0, 0] = xs / length
    stiffness = 50.0
    basis = newton.ModalBasis(
        sample_points=sample_points,
        sample_phi=sample_phi,
        sample_mass=np.full(n, 1.0 / n, dtype=np.float32),
        mode_stiffness=[stiffness],
        mode_damping=[3.0],
    )
    m_xx = float(basis.mode_coupling_centrifugal[0][0, 0])

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = builder.add_body(xform=wp.transform_identity(), mass=2.0, inertia=_identity_inertia(), is_kinematic=True)
    beam = builder.add_body_elastic(
        xform=wp.transform_identity(), mass=0.2, inertia=_identity_inertia(), modal_basis=basis
    )
    builder.add_joint_fixed(
        parent=base, child=beam, parent_xform=wp.transform_identity(), child_xform=wp.transform_identity()
    )
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    base_q_start, base_qd_start = _elastic_free_joint_starts(model, base)
    owner = int(model.elastic_joint.numpy()[0])
    q_index = int(model.joint_q_start.numpy()[owner]) + 7

    solver = newton.solvers.SolverVBD(model, iterations=24)
    dt = 1.0 / 240.0
    omega = 4.0
    velocities = {base: (wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, omega))}
    t = 0.0
    for _ in range(600):
        quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), omega * t)
        targets = {base: (wp.vec3(0.0, 0.0, 0.0), quat)}
        apply_kinematic_targets(state_0, {base: base_q_start}, targets, velocities, {base: base_qd_start})
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        t += dt

    q_final = float(state_0.joint_q.numpy()[q_index])
    expected = omega * omega * m_xx / stiffness
    test.assertGreater(q_final, 0.0)
    test.assertAlmostEqual(q_final, expected, delta=0.1 * expected)


def test_elastic_coriolis_modal_force(test, device):
    length = 1.0
    n = 21
    xs = np.linspace(0.0, length, n, dtype=np.float32)
    sample_points = np.column_stack([xs, np.zeros(n), np.zeros(n)]).astype(np.float32)
    sample_phi = np.zeros((n, 2, 3), dtype=np.float32)
    sample_phi[:, 0, 2] = xs / length
    sample_phi[:, 1, 1] = xs / length
    basis = newton.ModalBasis(
        sample_points=sample_points,
        sample_phi=sample_phi,
        sample_mass=np.full(n, 1.0 / n, dtype=np.float32),
        mode_stiffness=[30.0, 30.0],
        mode_damping=[0.0, 0.0],
    )

    def max_abs_mode0(omega):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        base = builder.add_body(xform=wp.transform_identity(), mass=2.0, inertia=_identity_inertia(), is_kinematic=True)
        beam = builder.add_body_elastic(
            xform=wp.transform_identity(),
            mass=0.2,
            inertia=_identity_inertia(),
            modal_basis=basis,
            mode_qd=[0.0, 1.0],
        )
        builder.add_joint_fixed(
            parent=base, child=beam, parent_xform=wp.transform_identity(), child_xform=wp.transform_identity()
        )
        builder.color()
        model = builder.finalize(device=device)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        base_q_start, base_qd_start = _elastic_free_joint_starts(model, base)
        owner = int(model.elastic_joint.numpy()[0])
        q0_index = int(model.joint_q_start.numpy()[owner]) + 7
        solver = newton.solvers.SolverVBD(model, iterations=24)
        dt = 1.0 / 240.0
        velocities = {base: (wp.vec3(0.0, 0.0, 0.0), wp.vec3(omega, 0.0, 0.0))}
        peak = 0.0
        t = 0.0
        for _ in range(500):
            quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), omega * t)
            targets = {base: (wp.vec3(0.0, 0.0, 0.0), quat)}
            apply_kinematic_targets(state_0, {base: base_q_start}, targets, velocities, {base: base_qd_start})
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
            peak = max(peak, abs(float(state_0.joint_q.numpy()[q0_index])))
            t += dt
        return peak

    spinning = max_abs_mode0(6.0)
    still = max_abs_mode0(0.0)
    test.assertLess(still, 1.0e-3)
    test.assertGreater(spinning, 1.0e-2)
    test.assertGreater(spinning, 10.0 * still)


def test_elastic_frame_coupling_conserves_com(test, device):
    total_mass = 1.0
    stiffness = 50.0
    deflection = 0.1
    n = 21
    length = 1.0
    xs = np.linspace(-0.5 * length, 0.5 * length, n, dtype=np.float32)
    sample_points = np.column_stack([xs, np.zeros(n), np.zeros(n)]).astype(np.float32)
    sample_phi = np.zeros((n, 1, 3), dtype=np.float32)
    sample_phi[:, 0, 2] = 1.0 - (2.0 * xs / length) ** 2
    basis = newton.ModalBasis(
        sample_points=sample_points,
        sample_phi=sample_phi,
        sample_mass=np.full(n, total_mass / n, dtype=np.float32),
        mode_stiffness=[stiffness],
        mode_damping=[0.0],
    )
    com_factor = float(basis.mode_coupling_linear[0][2]) / total_mass
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    beam = builder.add_body_elastic(
        xform=wp.transform_identity(),
        com=wp.vec3(0.0, 0.0, 0.0),
        mass=total_mass,
        inertia=_identity_inertia(),
        mode_q=[deflection],
        modal_basis=basis,
    )
    builder.color()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    q_start, _ = _elastic_free_joint_starts(model, beam)
    frame_z_index = q_start + 2
    mode_index = q_start + 7

    solver = newton.solvers.SolverVBD(model, iterations=24)
    dt = 1.0 / 240.0

    def frame_and_com():
        jq = state_0.joint_q.numpy()
        frame_z = float(jq[frame_z_index])
        return frame_z, frame_z + com_factor * float(jq[mode_index])

    frame_z0, com0 = frame_and_com()
    max_com_drift = 0.0
    max_frame_move = 0.0
    for _ in range(150):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        frame_z, com = frame_and_com()
        max_com_drift = max(max_com_drift, abs(com - com0))
        max_frame_move = max(max_frame_move, abs(frame_z - frame_z0))

    test.assertGreater(max_frame_move, 0.05)
    test.assertLess(max_com_drift, 0.01 * max_frame_move)


def test_elastic_frame_coupling_conserves_angular_momentum(test, device):
    total_mass = 1.0
    stiffness = 10.0
    deflection = 0.1
    inertia_yy = 0.3
    n = 11
    length = 1.0
    xs = np.linspace(-0.5 * length, 0.5 * length, n, dtype=np.float32)
    sample_points = np.column_stack([xs, np.zeros(n), np.zeros(n)]).astype(np.float32)
    sample_phi = np.zeros((n, 1, 3), dtype=np.float32)
    sample_phi[:, 0, 2] = xs / length
    basis = newton.ModalBasis(
        sample_points=sample_points,
        sample_phi=sample_phi,
        sample_mass=np.full(n, total_mass / n, dtype=np.float32),
        mode_stiffness=[stiffness],
        mode_damping=[0.0],
    )
    angular_y = float(basis.mode_coupling_angular[0][1])
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    beam = builder.add_body_elastic(
        xform=wp.transform_identity(),
        com=wp.vec3(0.0, 0.0, 0.0),
        mass=total_mass,
        inertia=wp.mat33(inertia_yy, 0.0, 0.0, 0.0, inertia_yy, 0.0, 0.0, 0.0, inertia_yy),
        mode_q=[deflection],
        modal_basis=basis,
    )
    builder.color()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    _, qd_start = _elastic_free_joint_starts(model, beam)
    frame_wy_index = qd_start + 4
    mode_vel_index = qd_start + 6

    solver = newton.solvers.SolverVBD(model, iterations=24)
    dt = 1.0 / 240.0
    max_total_h = 0.0
    max_frame_spin = 0.0
    modal_h_scale = 0.0
    for _ in range(150):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        jqd = state_0.joint_qd.numpy()
        frame_wy = float(jqd[frame_wy_index])
        mode_vel = float(jqd[mode_vel_index])
        total_h = inertia_yy * frame_wy - angular_y * mode_vel
        max_total_h = max(max_total_h, abs(total_h))
        max_frame_spin = max(max_frame_spin, abs(frame_wy))
        modal_h_scale = max(modal_h_scale, abs(angular_y * mode_vel))

    test.assertGreater(max_frame_spin, 0.05)
    test.assertLess(max_total_h, 0.02 * max(modal_h_scale, 1.0e-9))


def test_elastic_frame_coupling_conserves_spinning_momentum(test, device):
    total_mass = 1.0
    stiffness = 60.0
    deflection = 0.1
    spin = 5.0
    n = 21
    length = 1.0
    ys = np.linspace(-0.5 * length, 0.5 * length, n, dtype=np.float32)
    sample_points = np.column_stack([np.zeros(n), ys, np.zeros(n)]).astype(np.float32)
    sample_phi = np.zeros((n, 1, 3), dtype=np.float32)
    sample_phi[:, 0, 0] = 1.0 - (2.0 * ys / length) ** 2
    basis = newton.ModalBasis(
        sample_points=sample_points,
        sample_phi=sample_phi,
        sample_mass=np.full(n, total_mass / n, dtype=np.float32),
        mode_stiffness=[stiffness],
        mode_damping=[0.0],
    )
    s_bar = np.array(basis.mode_coupling_linear[0], dtype=np.float64)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    beam = builder.add_body_elastic(
        xform=wp.transform_identity(),
        com=wp.vec3(0.0, 0.0, 0.0),
        mass=total_mass,
        inertia=wp.mat33(0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2),
        mode_q=[deflection],
        modal_basis=basis,
    )
    builder.color()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    q_start, qd_start = _elastic_free_joint_starts(model, beam)
    mode_index = q_start + 7

    qd0 = state_0.joint_qd.numpy()
    qd0[qd_start + 5] = spin
    state_0.joint_qd.assign(qd0)

    solver = newton.solvers.SolverVBD(model, iterations=24)
    dt = 1.0 / 240.0

    def true_com():
        bq = state_0.body_q.numpy()[beam]
        translation = np.array(bq[:3], dtype=np.float64)
        q_mode = float(state_0.joint_q.numpy()[mode_index])
        return translation + quat_rotate(bq[3:7], s_bar * q_mode) / total_mass

    com0 = true_com()
    max_com_drift = 0.0
    max_mode_excursion = 0.0
    max_spin_angle = 0.0
    for _ in range(240):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        max_com_drift = max(max_com_drift, float(np.linalg.norm(true_com() - com0)))
        max_mode_excursion = max(max_mode_excursion, abs(float(state_0.joint_q.numpy()[mode_index])))
        bq = state_0.body_q.numpy()[beam]
        max_spin_angle = max(max_spin_angle, 2.0 * math.atan2(abs(float(bq[5])), abs(float(bq[6]))))

    test.assertGreater(max_spin_angle, 1.0)
    test.assertGreater(max_mode_excursion, 0.5 * deflection)
    test.assertLess(max_com_drift, 0.05)


def test_vbd_revolute_uses_elastic_endpoint(test, device):
    eta = 0.2
    rest_anchor = -0.5

    def shape_fn(x):
        return np.array([[x[0] / abs(rest_anchor), 0.0, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(abs(rest_anchor) + eta, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_q=[eta],
        mode_shape_fn=shape_fn,
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(rest_anchor, 0.0, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    before = _deformed_endpoint_world(model, state_0, joint, "child")
    np.testing.assert_allclose(before, [0.0, 0.0, 0.0], atol=1.0e-6)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=8,
        rigid_compliant_alm=True,
        rigid_joint_linear_ke=1.0e5,
        rigid_joint_angular_ke=1.0e5,
    )
    solver.step(state_0, state_1, control, None, 0.01)

    np.testing.assert_allclose(solver.joint_C0_lin.numpy()[joint], [0.0, 0.0, 0.0], atol=1.0e-6)
    after = _deformed_endpoint_world(model, state_1, joint, "child")
    np.testing.assert_allclose(after, [0.0, 0.0, 0.0], atol=1.0e-4)
    np.testing.assert_allclose(state_1.body_q.numpy()[body, :3], [abs(rest_anchor) + eta, 0.0, 0.0], atol=1.0e-4)


def test_vbd_revolute_constraint_solves_elastic_mode(test, device):
    rest_anchor = -0.5
    target_anchor = 0.1

    def shape_fn(x):
        return np.array([[x[0] / abs(rest_anchor), 0.0, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(abs(rest_anchor), 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_mass=[1.0e-6],
        mode_q=[0.0],
        mode_shape_fn=shape_fn,
        is_kinematic=True,
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(target_anchor, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(rest_anchor, 0.0, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])

    solver = newton.solvers.SolverVBD(
        model,
        iterations=4,
        rigid_joint_linear_k_start=1.0e6,
        rigid_joint_linear_ke=1.0e6,
        rigid_joint_linear_kd=0.0,
    )
    solver.step(state_0, state_1, control, None, 0.01)

    np.testing.assert_allclose(state_1.joint_q.numpy()[q_start + 7], -target_anchor, atol=1.0e-4)
    after = _deformed_endpoint_world(model, state_1, joint, "child")
    np.testing.assert_allclose(after, [target_anchor, 0.0, 0.0], atol=1.0e-4)


def test_vbd_fixed_joint_stiffness_pins_penalties(test, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
    )
    builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
    )
    builder.color()
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=2,
        rigid_compliant_alm=True,
        rigid_joint_linear_k_start=10.0,
        rigid_joint_angular_k_start=5.0,
        rigid_joint_linear_ke=1234.0,
        rigid_joint_angular_ke=567.0,
        rigid_joint_adaptive_stiffness=False,
    )

    np.testing.assert_allclose(solver.joint_penalty_k.numpy(), solver.joint_penalty_k_max.numpy())
    np.testing.assert_allclose(solver.joint_penalty_k_min.numpy(), solver.joint_penalty_k_max.numpy())
    np.testing.assert_allclose(solver.joint_penalty_k.numpy()[:2], [1234.0, 567.0])

    state_0 = model.state()
    state_1 = model.state()
    state_0.body_q.assign([wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity())])
    solver.step(state_0, state_1, model.control(), None, 0.01)

    np.testing.assert_allclose(solver.joint_penalty_k.numpy(), solver.joint_penalty_k_max.numpy())
    test.assertGreater(np.linalg.norm(solver.joint_lambda_lin.numpy()[0]), 0.0)


def test_vbd_elastic_modal_force_matches_joint_projection(test, device):
    for joint_kind in ("ball", "fixed", "revolute", "prismatic"):
        _assert_elastic_modal_projection_matches_joint_force(test, device, joint_kind, elastic_side="child")
    _assert_elastic_modal_projection_matches_joint_force(test, device, "revolute", elastic_side="parent")


def test_vbd_elastic_joint_assembles_one_coupled_block(test, device):
    dt = 0.01
    joint_k = 1000.0
    frame_x = 0.1
    mode_q = -0.04
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[1.0, 0.0, 0.0]]],
        sample_mass=[1.0],
        mode_stiffness=[0.0],
        mode_damping=[0.0],
    )

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(frame_x, 0.0, 0.0), wp.quat_identity()),
        mass=3.0,
        inertia=_identity_inertia(),
        mode_q=[mode_q],
        modal_basis=basis,
    )
    builder.add_joint_fixed(parent=-1, child=body)
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_joint_linear_k_start=joint_k,
        rigid_joint_angular_k_start=joint_k,
        rigid_joint_linear_ke=joint_k,
        rigid_joint_angular_ke=joint_k,
        rigid_joint_linear_kd=0.0,
        rigid_joint_angular_kd=0.0,
        rigid_joint_adaptive_stiffness=False,
    )
    solver.step(state_0, state_1, model.control(), None, dt)

    matrix_upper = solver.elastic_body_block_matrix.numpy()[0]
    matrix = np.triu(matrix_upper) + np.triu(matrix_upper, 1).T
    grad = solver.elastic_body_block_grad.numpy()[0]
    delta = solver.elastic_body_block_delta.numpy()[0]
    expected_frame_diagonal = 3.0 / (dt * dt) + joint_k
    expected_cross = 1.0 / (dt * dt) + joint_k
    expected_modal_diagonal = 1.0 / (dt * dt) + joint_k

    test.assertTrue(bool(solver.elastic_implicit_mass_coupling.numpy()[0]))
    np.testing.assert_allclose(matrix[0, 0], expected_frame_diagonal, rtol=1.0e-6)
    np.testing.assert_allclose(matrix[0, 6], expected_cross, rtol=1.0e-6)
    np.testing.assert_allclose(matrix[6, 6], expected_modal_diagonal, rtol=1.0e-6)
    np.testing.assert_allclose(delta, np.linalg.solve(matrix, -grad), rtol=2.0e-6, atol=1.0e-8)

    body_x = float(state_1.body_q.numpy()[body, 0])
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    solved_mode_q = float(state_1.joint_q.numpy()[q_start + 7])
    np.testing.assert_allclose(body_x, frame_x + delta[0], rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(solved_mode_q, mode_q + delta[6], rtol=1.0e-6, atol=1.0e-7)


def test_vbd_elastic_body_rejects_external_rigid_solver(test, device):
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[1.0, 0.0, 0.0]]],
        sample_mass=[1.0],
        mode_stiffness=[0.0],
        mode_damping=[0.0],
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(mass=3.0, inertia=_identity_inertia(), mode_q=[0.0], modal_basis=basis)
    builder.add_joint_fixed(parent=-1, child=body)
    builder.color()
    model = builder.finalize(device=device)

    with test.assertRaises(NotImplementedError):
        newton.solvers.SolverVBD(model, iterations=1, integrate_with_external_rigid_solver=True)


def test_vbd_elastic_block_solve_falls_back_when_indefinite(test, device):
    """An indefinite block degrades to a Jacobi step instead of writing NaN."""
    width = 7
    kernel = create_solve_elastic_body_tiled(width)

    rng = np.random.default_rng(0)
    matrix = np.zeros((2, width, width), dtype=np.float32)
    grad = np.zeros((2, width), dtype=np.float32)
    for block in range(2):
        a = rng.standard_normal((width, width))
        matrix[block] = np.triu(a @ a.T + width * np.eye(width))
        grad[block] = rng.standard_normal(width)
    matrix[1, 3, 3] = -5.0

    zeros_int = wp.zeros(1, dtype=wp.int32, device=device)
    delta = wp.zeros((2, width), dtype=float, device=device)
    metrics = [wp.zeros(2, dtype=float, device=device) for _ in range(5)]
    wp.launch_tiled(
        kernel=kernel,
        dim=[2],
        inputs=[
            0.01,
            False,
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array([0, 0], dtype=wp.int32, device=device),
            wp.array([1, 1], dtype=wp.int32, device=device),
            wp.zeros(2, dtype=float, device=device),
            wp.zeros(2, dtype=wp.vec3, device=device),
            wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=device),
            zeros_int,
            zeros_int,
            wp.zeros(8, dtype=float, device=device),
            wp.array(grad, dtype=float, device=device),
            delta,
            wp.array(matrix, dtype=float, device=device),
            *metrics,
            1.0,
        ],
        outputs=[
            wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=device),
            wp.zeros(8, dtype=float, device=device),
            wp.zeros(8, dtype=float, device=device),
        ],
        block_dim=32,
        device=device,
    )

    solved = delta.numpy()
    test.assertTrue(bool(np.isfinite(solved).all()))

    definite = np.triu(matrix[0]) + np.triu(matrix[0], 1).T
    np.testing.assert_allclose(solved[0], np.linalg.solve(definite, -grad[0]), rtol=1.0e-4, atol=1.0e-5)

    diagonal = np.diag(matrix[1])
    expected_fallback = np.where(diagonal > 0.0, -grad[1] / np.where(diagonal > 0.0, diagonal, 1.0), 0.0)
    np.testing.assert_allclose(solved[1], expected_fallback, rtol=1.0e-5, atol=1.0e-6)


def test_vbd_elastic_block_solve_supports_representative_widths(test, device):
    """Tiled Cholesky supports the modal widths used by representative assets."""
    rng = np.random.default_rng(11)

    for mode_count in (11, 16, 32):
        width = mode_count + 6
        kernel = create_solve_elastic_body_tiled(width)
        a = rng.standard_normal((width, width))
        matrix = np.triu(a @ a.T + width * np.eye(width)).astype(np.float32)[None]
        grad = rng.standard_normal((1, width)).astype(np.float32)
        delta = wp.zeros((1, width), dtype=float, device=device)
        metrics = [wp.zeros(1, dtype=float, device=device) for _ in range(5)]
        zeros_int = wp.zeros(1, dtype=wp.int32, device=device)

        wp.launch_tiled(
            kernel=kernel,
            dim=[1],
            inputs=[
                0.01,
                False,
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([mode_count], dtype=wp.int32, device=device),
                wp.zeros(1, dtype=float, device=device),
                wp.zeros(1, dtype=wp.vec3, device=device),
                wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
                zeros_int,
                zeros_int,
                wp.zeros(width + 1, dtype=float, device=device),
                wp.array(grad, dtype=float, device=device),
                delta,
                wp.array(matrix, dtype=float, device=device),
                *metrics,
                1.0,
            ],
            outputs=[
                wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
                wp.zeros(width + 1, dtype=float, device=device),
                wp.zeros(width, dtype=float, device=device),
            ],
            block_dim=32,
            device=device,
        )

        dense_matrix = np.triu(matrix[0]) + np.triu(matrix[0], 1).T
        np.testing.assert_allclose(
            delta.numpy()[0],
            np.linalg.solve(dense_matrix, -grad[0]),
            rtol=2.0e-4,
            atol=2.0e-5,
            err_msg=f"mode_count={mode_count}",
        )


def test_vbd_elastic_block_width_rejects_oversized_basis(test, device):
    """A basis too wide for device shared memory fails at construction, not at launch."""
    shared_memory_limit = int(getattr(wp.get_device(device), "max_shared_memory_per_block", 0))
    if shared_memory_limit <= 0:
        test.skipTest("device reports no shared memory budget")

    largest_width = int((np.sqrt(9.0 + 2.0 * shared_memory_limit) - 3.0) / 4.0)
    mode_count = largest_width - 6 + 1

    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=np.tile(np.array([[1.0, 0.0, 0.0]]), (1, mode_count, 1)),
        sample_mass=[1.0],
        mode_stiffness=[1.0] * mode_count,
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), modal_basis=basis, label="oversized")
    builder.color()
    model = builder.finalize(device=device)

    with test.assertRaises(ValueError) as raised:
        newton.solvers.SolverVBD(model, iterations=1)
    test.assertIn("shared memory", str(raised.exception))


def test_vbd_elastic_block_metrics_cover_frame_rows(test, device):
    """Convergence metrics span the whole block, not just the modal rows."""
    width = 7
    kernel = create_solve_elastic_body_tiled(width)

    rng = np.random.default_rng(3)
    a = rng.standard_normal((width, width))
    matrix = np.triu(a @ a.T + width * np.eye(width)).astype(np.float32)[None]
    grad = rng.standard_normal((1, width)).astype(np.float32)

    zeros_int = wp.zeros(1, dtype=wp.int32, device=device)
    delta = wp.zeros((1, width), dtype=float, device=device)
    metrics = [wp.zeros(1, dtype=float, device=device) for _ in range(5)]
    wp.launch_tiled(
        kernel=kernel,
        dim=[1],
        inputs=[
            0.01,
            True,
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.ones(1, dtype=float, device=device),
            wp.zeros(1, dtype=wp.vec3, device=device),
            wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
            zeros_int,
            zeros_int,
            wp.zeros(8, dtype=float, device=device),
            wp.array(grad, dtype=float, device=device),
            delta,
            wp.array(matrix, dtype=float, device=device),
            *metrics,
            1.0,
        ],
        outputs=[
            wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
            wp.zeros(8, dtype=float, device=device),
            wp.zeros(8, dtype=float, device=device),
        ],
        block_dim=32,
        device=device,
    )

    initial_residual = float(metrics[0].numpy()[0])
    modal_only = float(abs(grad[0, 6]))
    test.assertAlmostEqual(initial_residual, float(np.linalg.norm(grad[0])), delta=1.0e-4)
    test.assertGreater(initial_residual, 2.0 * modal_only)

    solve_residual = float(metrics[1].numpy()[0])
    test.assertLess(solve_residual / initial_residual, 1.0e-5)

    update_max = float(metrics[4].numpy()[0])
    test.assertAlmostEqual(update_max, float(np.abs(delta.numpy()[0]).max()), delta=1.0e-5)


def test_vbd_implicit_mass_coupling_requires_spd_block(test, device):
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[1.0, 0.0, 0.0]]],
        sample_mass=[2.0],
        mode_stiffness=[1.0],
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), modal_basis=basis, label="indefinite")
    builder.add_body_elastic(mass=3.0, inertia=_identity_inertia(), modal_basis=basis, label="positive_definite")
    builder.color()
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverVBD(model, iterations=1)
    np.testing.assert_array_equal(solver.elastic_implicit_mass_coupling.numpy(), [False, True])


def test_vbd_craig_bampton_coupled_solve_iteration_invariant(test, device):
    def run(iterations: int) -> np.ndarray:
        interface_positions, mass, stiffness, damping, sample_points, recovery, _ = _build_craig_bampton_test_data()
        generator = newton.ModalGeneratorCraigBampton(
            interface_positions=interface_positions,
            interface_names=["left", "right"],
            mass_matrix=mass,
            stiffness_matrix=stiffness,
            damping_matrix=damping,
            sample_points=sample_points,
            recovery_matrix=recovery,
        )
        basis = generator.build()
        mode_qd = np.zeros(basis.mode_count, dtype=np.float32)
        mode_qd[0] = 1.0

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_body_elastic(
            mass=generator.mass,
            com=wp.vec3(*generator.com),
            inertia=wp.mat33(*generator.inertia.flatten()),
            modal_basis=basis,
            mode_qd=mode_qd,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Adding a FIXED joint.*", category=UserWarning)
            for position in interface_positions:
                xform = wp.transform(wp.vec3(*position), wp.quat_identity())
                builder.add_joint_fixed(parent=-1, child=body, parent_xform=xform, child_xform=xform)
        builder.color()
        model = builder.finalize(device=device)

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        solver = newton.solvers.SolverVBD(
            model,
            iterations=iterations,
            rigid_joint_linear_k_start=1.0e6,
            rigid_joint_angular_k_start=1.0e6,
            rigid_joint_linear_ke=1.0e6,
            rigid_joint_angular_ke=1.0e6,
            rigid_joint_linear_kd=0.0,
            rigid_joint_angular_kd=0.0,
            rigid_joint_adaptive_stiffness=False,
        )

        owner_joint = int(model.elastic_joint.numpy()[0])
        q_start = int(model.joint_q_start.numpy()[owner_joint]) + 7
        trajectory = []
        for _ in range(40):
            solver.step(state_0, state_1, control, None, 1.0e-3)
            state_0, state_1 = state_1, state_0
            trajectory.append(state_0.joint_q.numpy()[q_start : q_start + basis.mode_count].copy())
        return np.asarray(trajectory)

    one_iteration = run(1)
    eight_iterations = run(8)
    relative_difference = np.linalg.norm(one_iteration - eight_iterations) / np.linalg.norm(eight_iterations)
    test.assertLess(relative_difference, 5.0e-5)


def test_vbd_elastic_modal_joint_damping_projection(test, device):
    dt = 0.01
    mode_mass = 1.0
    mode_qd = 0.5
    joint_k = 100.0
    joint_kd = 0.2

    def shape_fn(_x):
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_mass=[mode_mass],
        mode_stiffness=[0.0],
        mode_damping=[0.0],
        mode_q=[0.0],
        mode_qd=[mode_qd],
        mode_shape_fn=shape_fn,
        is_kinematic=True,
    )
    builder.add_joint_ball(
        parent=-1,
        child=body,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_joint_linear_k_start=joint_k,
        rigid_joint_linear_ke=joint_k,
        rigid_joint_linear_kd=joint_kd,
        rigid_joint_adaptive_stiffness=False,
    )
    solver.step(state_0, state_1, model.control(), None, dt)

    q_start = int(model.joint_q_start.numpy()[int(model.elastic_joint.numpy()[0])])
    q_integrated = dt * mode_qd
    modal_inertia = mode_mass / (dt * dt)
    joint_damping_h = joint_kd / dt
    h = modal_inertia + joint_k + joint_damping_h
    grad = joint_k * q_integrated + joint_kd * mode_qd
    q_expected = q_integrated - grad / h

    np.testing.assert_allclose(state_1.joint_q.numpy()[q_start + 7], q_expected, rtol=1.0e-6, atol=1.0e-7)


def test_vbd_elastic_angular_constraint_without_linear_stiffness(test, device):
    dt = 0.01
    mode_mass = 0.05
    mode_q = 0.3
    angular_k = 100.0
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 0.0]]],
        sample_psi=[[[1.0, 0.0, 0.0]]],
        mode_mass=[mode_mass],
        mode_stiffness=[0.0],
        mode_damping=[0.0],
    )

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_q=[mode_q],
        modal_basis=basis,
        is_kinematic=True,
    )
    builder.add_joint_fixed(parent=-1, child=body)
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_joint_linear_k_start=0.0,
        rigid_joint_linear_ke=0.0,
        rigid_joint_angular_k_start=angular_k,
        rigid_joint_angular_ke=angular_k,
        rigid_joint_adaptive_stiffness=False,
    )
    solver.step(state_0, state_1, model.control(), None, dt)

    q_start = int(model.joint_q_start.numpy()[int(model.elastic_joint.numpy()[0])]) + 7
    modal_inertia = mode_mass / (dt * dt)
    expected = mode_q - angular_k * mode_q / (modal_inertia + angular_k)
    np.testing.assert_allclose(state_1.joint_q.numpy()[q_start], expected, rtol=1.0e-6, atol=1.0e-7)


def test_vbd_rigid_angular_damping_includes_modal_velocity(test, device):
    dt = 0.01
    angular_k = 100.0

    def solve_parent_angle(angular_kd: float) -> float:
        basis = newton.ModalBasis(
            sample_points=[[0.0, 0.0, 0.0]],
            sample_phi=[[[0.0, 0.0, 0.0]]],
            sample_psi=[[[1.0, 0.0, 0.0]]],
            mode_mass=[0.05],
            mode_stiffness=[0.0],
            mode_damping=[0.0],
        )
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        parent = builder.add_body(mass=1.0, inertia=_identity_inertia())
        child = builder.add_body_elastic(
            mass=1.0,
            inertia=_identity_inertia(),
            mode_q=[0.0],
            mode_qd=[1.0],
            modal_basis=basis,
            is_kinematic=True,
        )
        builder.add_joint_fixed(parent=parent, child=child)
        builder.color()
        model = builder.finalize(device=device)

        state_0 = model.state()
        state_1 = model.state()
        solver = newton.solvers.SolverVBD(
            model,
            iterations=1,
            rigid_joint_linear_k_start=0.0,
            rigid_joint_linear_ke=0.0,
            rigid_joint_angular_k_start=angular_k,
            rigid_joint_angular_ke=angular_k,
            rigid_joint_angular_kd=angular_kd,
            rigid_joint_adaptive_stiffness=False,
        )
        solver.step(state_0, state_1, model.control(), None, dt)
        q = state_1.body_q.numpy()[parent, 3:7]
        return 2.0 * math.atan2(abs(float(q[0])), abs(float(q[3])))

    undamped_angle = solve_parent_angle(0.0)
    damped_angle = solve_parent_angle(0.2)
    test.assertGreater(
        damped_angle,
        1.1 * undamped_angle,
        f"modal angular damping did not react on the rigid parent: {damped_angle:.3e} vs {undamped_angle:.3e}",
    )


def test_vbd_elastic_angular_projection_uses_exponential_jacobian(test, device):
    dt = 0.01
    mode_mass = np.array([1.0, 1.0], dtype=np.float32)
    mode_q = np.array([0.6, 0.4], dtype=np.float32)
    psi = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    angular_k = 100.0
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[np.zeros((2, 3), dtype=np.float32)],
        sample_psi=[psi],
        mode_mass=mode_mass,
        mode_stiffness=[0.0, 0.0],
        mode_damping=[0.0, 0.0],
    )

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_q=mode_q,
        modal_basis=basis,
        is_kinematic=True,
    )
    builder.add_joint_revolute(parent=-1, child=body, axis=(1.0, 0.0, 0.0))
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_joint_linear_k_start=1.0,
        rigid_joint_linear_ke=1.0,
        rigid_joint_angular_k_start=angular_k,
        rigid_joint_angular_ke=angular_k,
        rigid_joint_adaptive_stiffness=False,
    )
    solver.step(state_0, state_1, model.control(), None, dt)

    theta = psi.T @ mode_q
    projector = np.diag([0.0, 1.0, 1.0])
    expected_grad = angular_k * psi @ projector @ theta
    expected_hessian = np.diag(mode_mass / (dt * dt)) + angular_k * psi @ projector @ psi.T
    actual_grad = solver.elastic_body_block_grad.numpy()[0, 6:8]
    actual_hessian = solver.elastic_body_block_matrix.numpy()[0, 6:8, 6:8]
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=5.0e-4, atol=2.0e-2)
    np.testing.assert_allclose(actual_hessian, expected_hessian, rtol=1.0e-5, atol=6.0e-2)


def test_vbd_revolute_drive_and_limit_project_into_elastic_modes(test, device):
    dt = 0.01
    mode_mass = 1.0
    drive_k = 200.0

    def solve_mode(*, mode_q: float, target_pos: float, target_ke: float, limit_upper: float) -> float:
        basis = newton.ModalBasis(
            sample_points=[[0.0, 0.0, 0.0]],
            sample_phi=[[[0.0, 0.0, 0.0]]],
            sample_psi=[[[1.0, 0.0, 0.0]]],
            mode_mass=[mode_mass],
            mode_stiffness=[0.0],
            mode_damping=[0.0],
        )
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_body_elastic(
            mass=1.0,
            inertia=_identity_inertia(),
            mode_q=[mode_q],
            modal_basis=basis,
            is_kinematic=True,
        )
        builder.add_joint_revolute(
            parent=-1,
            child=body,
            axis=(1.0, 0.0, 0.0),
            target_pos=target_pos,
            target_ke=target_ke,
            target_kd=0.0,
            limit_lower=-1.0,
            limit_upper=limit_upper,
            limit_ke=drive_k,
            limit_kd=0.0,
        )
        builder.color()
        model = builder.finalize(device=device)

        state_0 = model.state()
        state_1 = model.state()
        solver = newton.solvers.SolverVBD(
            model,
            iterations=1,
            rigid_joint_linear_k_start=1.0,
            rigid_joint_linear_ke=1.0,
            rigid_joint_angular_k_start=100.0,
            rigid_joint_angular_ke=100.0,
            rigid_joint_adaptive_stiffness=False,
        )
        solver.step(state_0, state_1, model.control(), None, dt)
        q_start = int(model.joint_q_start.numpy()[int(model.elastic_joint.numpy()[0])]) + 7
        return float(state_1.joint_q.numpy()[q_start])

    modal_inertia = mode_mass / (dt * dt)
    drive_result = solve_mode(mode_q=0.0, target_pos=0.25, target_ke=drive_k, limit_upper=1.0)
    drive_expected = drive_k * 0.25 / (modal_inertia + drive_k)
    np.testing.assert_allclose(drive_result, drive_expected, rtol=1.0e-6, atol=1.0e-7)

    limit_result = solve_mode(mode_q=0.25, target_pos=0.0, target_ke=0.0, limit_upper=0.1)
    limit_expected = 0.25 - drive_k * (0.25 - 0.1) / (modal_inertia + drive_k)
    np.testing.assert_allclose(limit_result, limit_expected, rtol=1.0e-6, atol=1.0e-7)


def test_fourbar_elastic_coupler_geometry(test, device):
    a, b_rest, c, d = 0.2, 0.5, 0.4, 0.5
    eta = 0.06
    b_eff = b_rest + eta
    theta2 = 0.35
    theta3, theta4 = _solve_fourbar(theta2, a, b_eff, c, d)

    A = np.array([0.0, 0.0, 0.0])
    D = np.array([d, 0.0, 0.0])
    e2 = np.array([math.cos(theta2), math.sin(theta2), 0.0])
    e3 = np.array([math.cos(theta3), math.sin(theta3), 0.0])
    e4 = np.array([math.cos(theta4), math.sin(theta4), 0.0])
    B = A + a * e2
    C = B + b_eff * e3
    np.testing.assert_allclose(C, D + c * e4, atol=1.0e-6)

    def axial_shape_fn(x):
        return np.array([[x[0] / b_rest, 0.0, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    crank = builder.add_body(
        xform=wp.transform(wp.vec3(*(A + 0.5 * a * e2)), _quat_from_angle_z(theta2)),
        mass=1.0,
        inertia=_identity_inertia(),
    )
    coupler = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(*(B + 0.5 * b_eff * e3)), _quat_from_angle_z(theta3)),
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_q=[eta],
        mode_shape_fn=axial_shape_fn,
    )
    rocker = builder.add_body(
        xform=wp.transform(wp.vec3(*(D + 0.5 * c * e4)), _quat_from_angle_z(theta4)),
        mass=1.0,
        inertia=_identity_inertia(),
    )

    j_ab = builder.add_joint_revolute(
        parent=crank,
        child=coupler,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(a / 2.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-b_rest / 2.0, 0.0, 0.0), wp.quat_identity()),
    )
    j_bc = builder.add_joint_revolute(
        parent=coupler,
        child=rocker,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(b_rest / 2.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(c / 2.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()

    p_ab_parent = _deformed_endpoint_world(model, state, j_ab, "parent")
    p_ab_child = _deformed_endpoint_world(model, state, j_ab, "child")
    p_bc_parent = _deformed_endpoint_world(model, state, j_bc, "parent")
    p_bc_child = _deformed_endpoint_world(model, state, j_bc, "child")

    np.testing.assert_allclose(p_ab_parent, B, atol=2.0e-6)
    np.testing.assert_allclose(p_ab_child, B, atol=2.0e-6)
    np.testing.assert_allclose(p_bc_parent, C, atol=2.0e-6)
    np.testing.assert_allclose(p_bc_child, C, atol=2.0e-6)
    np.testing.assert_allclose(np.linalg.norm(p_bc_parent - p_ab_child), b_eff, atol=2.0e-6)


def test_vbd_prismatic_rotates_elastic_child(test, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02)
    parent = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=inertia,
    )
    child = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.4, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=inertia,
        mode_count=1,
    )
    parent_joint = builder.add_joint_revolute(
        parent=-1,
        child=parent,
        axis=(0.0, 0.0, 1.0),
    )
    builder.add_joint_prismatic(
        parent=parent,
        child=child,
        axis=(1.0, 0.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.4, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
        target_pos=0.0,
    )
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    parent_qd_start = int(model.joint_qd_start.numpy()[parent_joint])
    joint_f = control.joint_f.numpy()
    joint_f[parent_qd_start] = 4.0
    control.joint_f.assign(joint_f)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=20,
        rigid_joint_linear_k_start=1.0e5,
        rigid_joint_angular_k_start=1.0e5,
        rigid_joint_linear_ke=1.0e6,
        rigid_joint_angular_ke=1.0e6,
    )
    for _ in range(60):
        solver.step(state_0, state_1, control, None, 1.0 / 180.0)
        state_0, state_1 = state_1, state_0

    body_q = state_0.body_q.numpy()
    parent_angle = 2.0 * math.atan2(float(body_q[parent, 5]), float(body_q[parent, 6]))
    child_angle = 2.0 * math.atan2(float(body_q[child, 5]), float(body_q[child, 6]))
    test.assertGreater(abs(parent_angle), 0.1)
    test.assertLess(_quat_angle_error(body_q[parent, 3:7], body_q[child, 3:7]), 1.0e-4)
    np.testing.assert_allclose(child_angle, parent_angle, atol=1.0e-4)


def test_crank_slider_elastic_analytic_geometry(test, device):
    crank_length = 0.24
    rod_rest_length = 0.64
    axial_q = 0.05
    theta = 0.72
    rod_length = rod_rest_length + axial_q
    slider_x = crank_length * math.cos(theta) + math.sqrt(
        rod_length * rod_length - (crank_length * math.sin(theta)) ** 2
    )

    crank_pin = np.array([crank_length * math.cos(theta), crank_length * math.sin(theta), 0.0])
    slider_pin = np.array([slider_x, 0.0, 0.0])
    rod_theta = math.atan2(float(slider_pin[1] - crank_pin[1]), float(slider_pin[0] - crank_pin[0]))

    basis = newton.ModalGeneratorBeam(
        length=rod_rest_length,
        half_width_y=0.03,
        half_width_z=0.02,
        mode_specs=[{"type": newton.ModalGeneratorBeam.Mode.AXIAL}],
    ).build(sample_points=[[-0.5 * rod_rest_length, 0.0, 0.0], [0.5 * rod_rest_length, 0.0, 0.0]])

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    crank = builder.add_body(
        xform=wp.transform(
            wp.vec3(*(0.5 * crank_pin)),
            _quat_from_angle_z(theta),
        ),
        mass=1.0,
        inertia=_identity_inertia(),
    )
    rod = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(*(0.5 * (crank_pin + slider_pin))), _quat_from_angle_z(rod_theta)),
        mass=1.0,
        inertia=_identity_inertia(),
        mode_q=[axial_q],
        modal_basis=basis,
    )
    slider = builder.add_body(
        xform=wp.transform(wp.vec3(*slider_pin), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
    )
    j_crank_rod = builder.add_joint_revolute(
        parent=crank,
        child=rod,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.5 * crank_length, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5 * rod_rest_length, 0.0, 0.0), wp.quat_identity()),
    )
    j_rod_slider = builder.add_joint_revolute(
        parent=rod,
        child=slider,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.5 * rod_rest_length, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()

    p_crank = joint_endpoint_world(model, state, j_crank_rod, "parent")
    p_rod_left = joint_endpoint_world(model, state, j_crank_rod, "child")
    p_rod_right = joint_endpoint_world(model, state, j_rod_slider, "parent")
    p_slider = joint_endpoint_world(model, state, j_rod_slider, "child")

    np.testing.assert_allclose(p_crank, crank_pin, atol=1.0e-6)
    np.testing.assert_allclose(p_rod_left, crank_pin, atol=1.0e-6)
    np.testing.assert_allclose(p_rod_right, slider_pin, atol=1.0e-6)
    np.testing.assert_allclose(p_slider, slider_pin, atol=1.0e-6)
    np.testing.assert_allclose(np.linalg.norm(p_rod_right - p_rod_left), rod_length, atol=1.0e-6)


def test_elastic_multiple_attachment_samples(test, device):
    basis = newton.ModalBasis(
        sample_points=[[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        sample_phi=[
            [[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.25, 0.0]],
            [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        mode_mass=[1.0, 1.0],
    )

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_q=[0.2, -0.4],
        modal_basis=basis,
    )
    j_left = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(-0.6, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
    )
    j_mid = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.1, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )
    j_right = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.6, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()

    endpoint_samples = model.elastic_endpoint_sample.numpy()
    child_endpoints = model.joint_child_elastic_endpoint.numpy()
    sample_ids = [int(endpoint_samples[int(child_endpoints[j])]) for j in (j_left, j_mid, j_right)]
    test.assertEqual(sample_ids, [0, 1, 2])

    np.testing.assert_allclose(joint_endpoint_world(model, state, j_left, "child"), [-0.6, 0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(joint_endpoint_world(model, state, j_mid, "child"), [0.0, 0.1, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(joint_endpoint_world(model, state, j_right, "child"), [0.6, 0.0, 0.0], atol=1.0e-6)


def test_matrix_rom_example(test, device):
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = MatrixROMExample(viewer, None)
        for _ in range(100):
            example.step()
            example.render()
        example.test_final()


def test_dipper_arm_example(test, device):
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = DipperExample(viewer, None)
        for _ in range(120):
            example.step()
            example.render()
        example.test_final()


def test_gravity_coupling_example(test, device):
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = GravityCouplingExample(viewer, None)
        for _ in range(120):
            example.step()
            example.render()
        example.test_final()


def test_base_excitation_example(test, device):
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = BaseExcitationExample(viewer, None)
        for _ in range(120):
            example.step()
            example.render()
        example.test_final()


def test_base_rotation_example(test, device):
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = BaseRotationExample(viewer, None)
        for _ in range(120):
            example.step()
            example.render()
        example.test_final()


def test_clamp_moment_example(test, device):
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = ClampMomentExample(viewer, None)
        for _ in range(150):
            example.step()
            example.render()
        example.test_final()


def test_centrifugal_example(test, device):
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = CentrifugalExample(viewer, None)
        for _ in range(150):
            example.step()
            example.render()
        example.test_final()


def test_coriolis_example(test, device):
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = CoriolisExample(viewer, None)
        for _ in range(180):
            example.step()
            example.render()
        example.test_final()


def test_frame_coupling_example(test, device):
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = FrameCouplingExample(viewer, None)
        for _ in range(180):
            example.step()
            example.render()
        example.test_final()


def test_angular_frame_coupling_example(test, device):
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = AngularFrameCouplingExample(viewer, None)
        for _ in range(180):
            example.step()
            example.render()
        example.test_final()


def _run_reduced_elastic_contact_example(example_cls, frame_count: int, device):
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = example_cls(viewer, None)
        for _ in range(frame_count):
            example.step()
            example.render()
        example.test_final()


def test_elastic_wall_contact_example(test, device):
    _run_reduced_elastic_contact_example(WallContactExample, 90, device)


def test_elastic_gripper_contact_example(test, device):
    _run_reduced_elastic_contact_example(GripperContactExample, 120, device)


def test_elastic_scraper_contact_example(test, device):
    _run_reduced_elastic_contact_example(ScraperContactExample, 120, device)


def test_elastic_chair_stick_slip_example(test, device):
    asset_dir = chair_asset_cache_dir()
    if not all((asset_dir / filename).is_file() for filename in (CHAIR_GLTF_NAME, CHAIR_BIN_NAME)):
        test.skipTest("Poly Haven chair asset is not cached; the offline unit suite does not download it")
    _run_reduced_elastic_contact_example(ChairStickSlipExample, 240, device)


class TestReducedElasticBody(unittest.TestCase):
    pass


for device in devices:
    add_function_test(
        TestReducedElasticBody, "test_modal_basis_add_sample", test_modal_basis_add_sample, devices=[device]
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_basis_add_samples_preserves_mass_invariant",
        test_modal_basis_add_samples_preserves_mass_invariant,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_beam_samples",
        test_modal_generator_beam_samples,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_pod_rank_one",
        test_modal_generator_pod_rank_one,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_pod_explicit_zero_modes",
        test_modal_generator_pod_explicit_zero_modes,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_beam_linear_torsion_properties",
        test_modal_generator_beam_linear_torsion_properties,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_fem_matrix_rom",
        test_modal_generator_fem_matrix_rom,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_craig_bampton_interface_modes",
        test_modal_generator_craig_bampton_interface_modes,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_craig_bampton_single_interface",
        test_modal_generator_craig_bampton_single_interface,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_craig_bampton_nonclassical_damping",
        test_modal_generator_craig_bampton_nonclassical_damping,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_basis_lumped_inertia_coupling",
        test_modal_basis_lumped_inertia_coupling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_basis_coupling_unavailable_without_mass",
        test_modal_basis_coupling_unavailable_without_mass,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_basis_coupling_explicit_override",
        test_modal_basis_coupling_explicit_override,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_fem_inertia_coupling",
        test_modal_generator_fem_inertia_coupling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_pod_inertia_coupling",
        test_modal_generator_pod_inertia_coupling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_pod_coupling_matches_lumped",
        test_modal_generator_pod_coupling_matches_lumped,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_basis_copy_preserves_coupling",
        test_modal_basis_copy_preserves_coupling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_basis_mode_mass_from_sample_mass",
        test_modal_basis_mode_mass_from_sample_mass,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_basis_mode_mass_explicit_wins",
        test_modal_basis_mode_mass_explicit_wins,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_mode_coupling_arrays",
        test_elastic_mode_coupling_arrays,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_coriolis_array_padding",
        test_elastic_coriolis_array_padding,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_mode_coupling_arrays_merge",
        test_elastic_mode_coupling_arrays_merge,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_coriolis_array_merge_different_mode_count",
        test_elastic_coriolis_array_merge_different_mode_count,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_gravity_modal_force",
        test_elastic_gravity_modal_force,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_free_fall_no_modal_force",
        test_elastic_free_fall_no_modal_force,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_euler_modal_force",
        test_elastic_euler_modal_force,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_centrifugal_modal_force",
        test_elastic_centrifugal_modal_force,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_coriolis_modal_force",
        test_elastic_coriolis_modal_force,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_frame_coupling_conserves_com",
        test_elastic_frame_coupling_conserves_com,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_frame_coupling_conserves_angular_momentum",
        test_elastic_frame_coupling_conserves_angular_momentum,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_frame_coupling_conserves_spinning_momentum",
        test_elastic_frame_coupling_conserves_spinning_momentum,
        devices=[device],
    )
    add_function_test(TestReducedElasticBody, "test_elastic_link_layout", test_elastic_link_layout, devices=[device])
    add_function_test(
        TestReducedElasticBody,
        "test_add_joint_elastic_initializes_relative_frame_coordinates",
        test_add_joint_elastic_initializes_relative_frame_coordinates,
        devices=[device],
    )
    add_function_test(TestReducedElasticBody, "test_elastic_fk_sync", test_elastic_fk_sync, devices=[device])
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_fk_ik_com_velocity_round_trip",
        test_elastic_fk_ik_com_velocity_round_trip,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_endpoint_shape_sampling",
        test_elastic_endpoint_shape_sampling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_endpoint_modal_basis_sampling",
        test_elastic_endpoint_modal_basis_sampling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_endpoint_angular_sampling",
        test_elastic_endpoint_angular_sampling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_endpoint_angular_zero_without_beam",
        test_elastic_endpoint_angular_zero_without_beam,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_clamp_moment_reaction",
        test_elastic_clamp_moment_reaction,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_clamp_moment_reaction_parent_side",
        test_elastic_clamp_moment_reaction_parent_side,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_clamp_moment_conserves_angular_momentum",
        test_elastic_clamp_moment_conserves_angular_momentum,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_estimate_sample_psi_rotation_and_translation",
        test_estimate_sample_psi_rotation_and_translation,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_estimate_sample_psi_degenerate_warns",
        test_estimate_sample_psi_degenerate_warns,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_sampled_psi",
        test_modal_generator_sampled_psi,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_pod_derives_psi",
        test_modal_generator_pod_derives_psi,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_fem_derives_psi",
        test_modal_generator_fem_derives_psi,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_basis_shared_by_elastic_bodies",
        test_modal_basis_shared_by_elastic_bodies,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_render_shape_sampling",
        test_elastic_render_shape_sampling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_shape_mesh_sampling",
        test_elastic_shape_mesh_sampling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_shape_box_winding",
        test_elastic_shape_box_winding,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_shape_box_exact_modal_samples",
        test_elastic_shape_box_exact_modal_samples,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_surface_contact_generation",
        test_elastic_surface_contact_generation,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_surface_contact_work_uses_pair_vertex_ranges",
        test_elastic_surface_contact_work_uses_pair_vertex_ranges,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_surface_contact_deterministic_sort_preserves_samples",
        test_elastic_surface_contact_deterministic_sort_preserves_samples,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_surface_contact_reduction_preserves_modal_samples",
        test_elastic_surface_contact_reduction_preserves_modal_samples,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_surface_contact_uses_deformed_vertex",
        test_elastic_surface_contact_uses_deformed_vertex,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_contact_solves_modal_penetration",
        test_vbd_elastic_contact_solves_modal_penetration,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_body_relaxation_is_general_and_validated",
        test_vbd_elastic_body_relaxation_is_general_and_validated,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_compliant_contact_solves_modal_penetration",
        test_vbd_elastic_compliant_contact_solves_modal_penetration,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_compliant_contact_uses_modal_mobility_for_rho_and_forces",
        test_vbd_elastic_compliant_contact_uses_modal_mobility_for_rho_and_forces,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_contact_overflow_assembles_consistent_block",
        test_vbd_elastic_contact_overflow_assembles_consistent_block,
        devices=[device],
        check_output=False,
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_contact_local_mat33_projection_matches_world",
        test_elastic_contact_local_mat33_projection_matches_world,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_strain_visualization_colors",
        test_elastic_strain_visualization_colors,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_rendering_respects_layer_transform_and_visibility",
        test_elastic_rendering_respects_layer_transform_and_visibility,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_torsion_render_shape_sampling",
        test_torsion_render_shape_sampling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_finite_torsion_exemplar_preserves_volume",
        test_finite_torsion_exemplar_preserves_volume,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_finite_torsion_pod_modes_project_exemplar",
        test_finite_torsion_pod_modes_project_exemplar,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_cantilever_tip_load_solution",
        test_cantilever_tip_load_solution,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_cantilever_modal_vibration_solution",
        test_cantilever_modal_vibration_solution,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_modal_implicit_solution_vbd",
        test_elastic_modal_implicit_solution_vbd,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_revolute_uses_elastic_endpoint",
        test_vbd_revolute_uses_elastic_endpoint,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_revolute_constraint_solves_elastic_mode",
        test_vbd_revolute_constraint_solves_elastic_mode,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_fixed_joint_stiffness_pins_penalties",
        test_vbd_fixed_joint_stiffness_pins_penalties,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_modal_force_matches_joint_projection",
        test_vbd_elastic_modal_force_matches_joint_projection,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_joint_assembles_one_coupled_block",
        test_vbd_elastic_joint_assembles_one_coupled_block,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_body_rejects_external_rigid_solver",
        test_vbd_elastic_body_rejects_external_rigid_solver,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_block_solve_falls_back_when_indefinite",
        test_vbd_elastic_block_solve_falls_back_when_indefinite,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_block_solve_supports_representative_widths",
        test_vbd_elastic_block_solve_supports_representative_widths,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_block_width_rejects_oversized_basis",
        test_vbd_elastic_block_width_rejects_oversized_basis,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_block_metrics_cover_frame_rows",
        test_vbd_elastic_block_metrics_cover_frame_rows,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_implicit_mass_coupling_requires_spd_block",
        test_vbd_implicit_mass_coupling_requires_spd_block,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_craig_bampton_coupled_solve_iteration_invariant",
        test_vbd_craig_bampton_coupled_solve_iteration_invariant,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_modal_joint_damping_projection",
        test_vbd_elastic_modal_joint_damping_projection,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_angular_constraint_without_linear_stiffness",
        test_vbd_elastic_angular_constraint_without_linear_stiffness,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_rigid_angular_damping_includes_modal_velocity",
        test_vbd_rigid_angular_damping_includes_modal_velocity,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_angular_projection_uses_exponential_jacobian",
        test_vbd_elastic_angular_projection_uses_exponential_jacobian,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_revolute_drive_and_limit_project_into_elastic_modes",
        test_vbd_revolute_drive_and_limit_project_into_elastic_modes,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_fourbar_elastic_coupler_geometry",
        test_fourbar_elastic_coupler_geometry,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_prismatic_rotates_elastic_child",
        test_vbd_prismatic_rotates_elastic_child,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_crank_slider_elastic_analytic_geometry",
        test_crank_slider_elastic_analytic_geometry,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_multiple_attachment_samples",
        test_elastic_multiple_attachment_samples,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_matrix_rom_example",
        test_matrix_rom_example,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_dipper_arm_example",
        test_dipper_arm_example,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_gravity_coupling_example",
        test_gravity_coupling_example,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_base_excitation_example",
        test_base_excitation_example,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_base_rotation_example",
        test_base_rotation_example,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_clamp_moment_example",
        test_clamp_moment_example,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_centrifugal_example",
        test_centrifugal_example,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_coriolis_example",
        test_coriolis_example,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_frame_coupling_example",
        test_frame_coupling_example,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_angular_frame_coupling_example",
        test_angular_frame_coupling_example,
        devices=[device],
    )
    if wp.get_device(device).is_cuda:
        add_function_test(
            TestReducedElasticBody,
            "test_elastic_wall_contact_example",
            test_elastic_wall_contact_example,
            devices=[device],
        )
        add_function_test(
            TestReducedElasticBody,
            "test_elastic_gripper_contact_example",
            test_elastic_gripper_contact_example,
            devices=[device],
        )
        add_function_test(
            TestReducedElasticBody,
            "test_elastic_scraper_contact_example",
            test_elastic_scraper_contact_example,
            devices=[device],
        )
        add_function_test(
            TestReducedElasticBody,
            "test_elastic_chair_stick_slip_example",
            test_elastic_chair_stick_slip_example,
            devices=[device],
        )


if __name__ == "__main__":
    unittest.main()
