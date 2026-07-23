# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for inverse-dynamics passive terms and joint forces."""

from __future__ import annotations

import unittest
from enum import IntFlag
from typing import ClassVar

import numpy as np
import warp as wp

import newton


class _PassiveOutput(IntFlag):
    """Test-only selector for sharing setup across passive-output tests."""

    MASS_MATRIX = 1 << 0
    GRAVITY_FORCE = 1 << 1
    CORIOLIS_FORCE = 1 << 2
    ALL = MASS_MATRIX | GRAVITY_FORCE | CORIOLIS_FORCE


class _InverseDynamicsArrays:
    """Test-only owner for caller-allocated inverse-dynamics arrays."""

    def __init__(self, model: newton.Model):
        self.mass_matrix = wp.zeros(
            (
                model.articulation_count,
                model.max_dofs_per_articulation,
                model.max_dofs_per_articulation,
            ),
            dtype=wp.float32,
            device=model.device,
        )
        self.gravity_force = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
        self.coriolis_force = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
        self.joint_f = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)


def _eval_inverse_dynamics_passive(
    model: newton.Model,
    state: newton.State,
    eval_type: _PassiveOutput,
    arrays: _InverseDynamicsArrays,
    mask: wp.array[bool] | None = None,
) -> None:
    """Invoke the public passive evaluator for the selected test outputs."""
    newton.eval_inverse_dynamics_passive(
        model,
        state,
        mass_matrix=(arrays.mass_matrix if eval_type & _PassiveOutput.MASS_MATRIX else None),
        gravity_force=(arrays.gravity_force if eval_type & _PassiveOutput.GRAVITY_FORCE else None),
        coriolis_force=(arrays.coriolis_force if eval_type & _PassiveOutput.CORIOLIS_FORCE else None),
        mask=mask,
    )


def _eval_inverse_dynamics_force(
    model: newton.Model,
    state: newton.State,
    arrays: _InverseDynamicsArrays,
    joint_qdd: wp.array[wp.float32],
    mask: wp.array[bool] | None = None,
) -> None:
    """Invoke the public force evaluator with arrays owned by the test."""
    newton.eval_inverse_dynamics_force(
        model,
        state,
        mass_matrix=arrays.mass_matrix,
        joint_qdd=joint_qdd,
        coriolis_force=arrays.coriolis_force,
        gravity_force=arrays.gravity_force,
        joint_f=arrays.joint_f,
        mask=mask,
    )


class TestInverseDynamicsBase:
    """Shared test body. Concrete subclasses set :attr:`device`."""

    device: wp.context.Device | None = None

    # Per-link inertia tensors swept by tests in :class:`TestGravCompForce`
    # to confirm G(q) is genuinely insensitive to the inertia tensor (it
    # depends only on mass and CoM); also used as default inertias in the
    # other test classes to avoid repeating the identity-inertia literal.
    I_UNIT: ClassVar[wp.mat33] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    I_100: ClassVar[wp.mat33] = wp.mat33(100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0)
    INERTIA_PASSES: ClassVar[list[wp.mat33]] = [I_UNIT, I_100]

    @staticmethod
    def _build_two_link_articulation(
        gravity: wp.vec3,
        floating_base: bool,
        joint_type: str,
        joint_axis: wp.vec3,
        link_coms: list[wp.vec3],
        link_masses: list[float],
        joint_frames: list[wp.transform],
        link_inertias: list[wp.mat33],
    ) -> newton.ModelBuilder:
        builder = newton.ModelBuilder(gravity=gravity)

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

        if joint_type == "revolute":
            add_dof_joint = builder.add_joint_revolute
        elif joint_type == "prismatic":
            add_dof_joint = builder.add_joint_prismatic
        else:
            raise ValueError(f"joint_type must be 'revolute' or 'prismatic', got {joint_type!r}.")

        b1 = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=link_masses[0],
            inertia=link_inertias[0],
            com=link_coms[0],
        )
        if floating_base:
            # ``parent_xform.rotation`` is left at identity to avoid a known
            # MuJoCo-bridge convention bug for free joints with a rotated
            # parent frame: https://github.com/newton-physics/newton/issues/2704.
            j1 = builder.add_joint_free(
                parent=-1,
                child=b1,
                parent_xform=identity_xform,
                child_xform=identity_xform,
            )
        else:
            j1 = builder.add_joint_fixed(
                parent=-1,
                child=b1,
                parent_xform=identity_xform,
                child_xform=identity_xform,
            )

        b2 = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=link_masses[1],
            inertia=link_inertias[1],
            com=link_coms[1],
        )
        j2 = add_dof_joint(
            parent=b1,
            child=b2,
            axis=joint_axis,
            parent_xform=joint_frames[0],
            child_xform=joint_frames[1],
        )
        builder.add_articulation([j1, j2], label="pendulum")

        return builder


class TestGravCompForce(TestInverseDynamicsBase):
    """Gravity-force tests for the two-link pendulum harness."""

    @staticmethod
    def _default_joint_q(is_floating_base: list[list[bool]]) -> list[list[list[float]]]:
        """Build the default initial-state ``joint_q`` for a multi-world,
        multi-articulation pendulum: zero position, identity quaternion,
        zero internal q for each floating articulation; a single zero
        internal q for each fixed one.
        """
        default_floating = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        default_fixed = [0.0]
        return [
            [list(default_floating) if floating else list(default_fixed) for floating in row]
            for row in is_floating_base
        ]

    def _test_two_link_grav_comp_force(
        self,
        gravity_vec: wp.vec3,
        joint_type: str,
        is_floating_base: list[list[bool]],
        joint_axis: list[list[wp.vec3]],
        joint_frames: list[list[list[wp.transform]]],
        joint_q: list[list[list[float]]],
        link_coms: list[list[list[wp.vec3]]],
        link_masses: list[list[list[float]]],
        link_inertias: list[list[list[wp.mat33]]],
        expected_grav_comp_forces: list[float],
    ):
        """G(q) is populated correctly for a multi-world, multi-articulation model.

        Args:
            gravity_vec: Axis-aligned gravity (world frame).
            joint_type: ``"revolute"`` or ``"prismatic"`` — shared across every articulation.
            is_floating_base: Per-articulation floating-vs-fixed root flag ``[w][a]``.
            joint_axis: Per-articulation joint axis ``[w][a]`` (same shape as
                ``is_floating_base``).
            joint_frames: Per-joint parent-side anchor transforms ``[w][a][joint]``.
                ``joint[0]`` is the root joint (free or fixed), ``joint[1]`` is
                the internal DOF joint.
            joint_q: Per-articulation initial-state ``joint_q`` shaped
                ``[w][a]``. Each inner list holds the per-articulation
                generalized coordinates: 8 floats for a floating root with
                one internal DOF (3 base position, 4 base quaternion, 1
                internal q) and 1 float for a fixed root with one internal
                DOF (``(q_internal,)``). Written into ``state.joint_q``
                before ``eval_fk`` so the rest pose used during
                gravity-force evaluation reflects this input. Use
                :meth:`_default_joint_q` to build the zero-pose /
                identity-quat default when the test doesn't care about the
                rest pose.
            link_coms: Per-link CoM offsets ``[w][a][link]`` as ``wp.vec3``.
            link_masses: Per-link masses ``[w][a][link]``.
            link_inertias: Per-link body-frame inertia tensors
                ``[w][a][link]`` as ``wp.mat33``.
            expected_grav_comp_forces: Flat expected ``g(q) = ∂U/∂q`` (the
                standard manipulator-equation gravity bias) in the order
                Newton reports them. Equivalently, the joint-space force a
                controller would apply to hold the articulation static under
                gravity.
        """
        # Derive shape constants from the structured inputs.
        num_worlds = len(is_floating_base)
        num_arts_per_world = len(is_floating_base[0])
        num_links_per_articulation = len(link_coms[0][0])
        # _build_two_link_articulation hard-codes a two-link articulation, so the
        # caller's link_coms layout must agree.
        self.assertEqual(num_links_per_articulation, 2)

        # Each articulation contributes 7 DOFs if floating (6 free-joint
        # DOFs + 1 internal) or 1 DOF if fixed. G(q) and
        # expected_grav_comp_forces are sized by total DOF count.
        expected_total_dofs = sum(7 if floating else 1 for row in is_floating_base for floating in row)
        if len(expected_grav_comp_forces) != expected_total_dofs:
            raise ValueError(
                f"expected_grav_comp_forces has length {len(expected_grav_comp_forces)}, "
                f"but is_floating_base implies {expected_total_dofs} total DOFs."
            )

        # Build the model from the structured per-world / per-articulation inputs.
        model_builder = newton.ModelBuilder(gravity=gravity_vec)
        for i in range(0, num_worlds):
            world_builder = newton.ModelBuilder(gravity=gravity_vec)
            for j in range(0, num_arts_per_world):
                articulation_builder = self._build_two_link_articulation(
                    gravity=gravity_vec,
                    joint_type=joint_type,
                    joint_axis=joint_axis[i][j],
                    floating_base=is_floating_base[i][j],
                    link_coms=link_coms[i][j],
                    link_masses=link_masses[i][j],
                    joint_frames=joint_frames[i][j],
                    link_inertias=link_inertias[i][j],
                )
                world_builder.add_builder(articulation_builder)
            model_builder.add_world(world_builder)

        model = model_builder.finalize(device=self.device)
        state = model.state()

        # Patch the per-articulation joint_q ranges in the global state
        # vector. Articulations are appended in (world, articulation) iteration
        # order by the build loop above, and within an articulation the root
        # joint comes first, so the q layout is 7 free-joint values (3 base
        # position + 4 base quaternion) + 1 internal DOF for floating roots,
        # or just 1 internal DOF for fixed roots.
        joint_q_arr = state.joint_q.numpy()
        offset = 0
        for i in range(num_worlds):
            for j in range(num_arts_per_world):
                art_q_size = 8 if is_floating_base[i][j] else 1
                override = joint_q[i][j]
                if len(override) != art_q_size:
                    raise ValueError(
                        f"joint_q[{i}][{j}] has length {len(override)}, "
                        f"expected {art_q_size} for is_floating_base={is_floating_base[i][j]}."
                    )
                joint_q_arr[offset : offset + art_q_size] = override
                offset += art_q_size
        state.joint_q.assign(joint_q_arr)

        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        inverse_dynamics = _InverseDynamicsArrays(model)

        _eval_inverse_dynamics_passive(
            model=model,
            state=state,
            eval_type=_PassiveOutput.GRAVITY_FORCE,
            arrays=inverse_dynamics,
        )

        measured_gravity_comp_force = inverse_dynamics.gravity_force.numpy()
        self.assertTrue(np.all(np.isfinite(measured_gravity_comp_force)))

        # Newton's gravity_force stores the standard
        # manipulator-equation gravity bias g(q) = ∂U/∂q, which equals the
        # joint-space force a controller would apply to hold the articulation
        # static under gravity -- the value listed in expected_grav_comp_forces.
        self.assertEqual(measured_gravity_comp_force.shape, (len(expected_grav_comp_forces),))
        np.testing.assert_allclose(measured_gravity_comp_force, expected_grav_comp_forces, atol=1e-5, rtol=1e-5)

    def test_two_link_grav_comp_force_from_zero_gravity(self):
        """G(q) vanishes everywhere when the model has zero gravity.

        Builds a multi-world, multi-articulation pendulum (2 worlds x 2
        articulations, mixed fixed/floating roots) for each of the supported
        internal joint types (revolute, prismatic) with per-articulation joint
        axes. With gravity set to the zero vector, the generalized gravity
        force must be identically zero on every DOF, independent of joint
        type, joint axis, root type, link mass, CoM offset, or link inertia
        tensor. The outer loop runs once with unit inertias and once with
        inertias scaled by 100 to confirm G(q) is truly insensitive to the
        inertia tensor.
        """
        joint_types = ["revolute", "prismatic"]

        gravity_vec = wp.vec3(0.0, 0.0, 0.0)

        is_floating_base = [
            [False, True],  # World0, articulation0 fixed, articulation1 free
            [False, True],  # World1, articulation0 fixed, articulation1 free
        ]

        prismatic_x = wp.vec3(1.0, 0.0, 0.0)
        prismatic_y = wp.vec3(0.0, 1.0, 0.0)
        prismatic_z = wp.vec3(0.0, 0.0, 1.0)
        joint_axis = [
            [prismatic_x, prismatic_y],  # World0, articulation0/articulation1
            [prismatic_z, prismatic_x],  # World1, articulation0/articulation1
        ]

        # Non-identity anchors on the internal joint — under zero gravity
        # G(q) must still be identically zero, independent of where the
        # joint is anchored in the parent/child bodies or how either frame
        # is oriented. The hand-written quaternion values below encode
        # 45 deg about +z and 60 deg about +y.
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        shift_x = wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity())
        shift_ny = wp.transform(wp.vec3(0.0, -0.3, 0.0), wp.quat_identity())
        shift_z = wp.transform(wp.vec3(0.0, 0.0, 0.7), wp.quat_identity())
        rot_z_45 = wp.transform(wp.vec3(0.1, 0.2, 0.0), wp.quat(0.0, 0.0, 0.3826834, 0.9238795))
        rot_y_60 = wp.transform(wp.vec3(0.0, 0.0, -0.4), wp.quat(0.0, 0.5, 0.0, 0.8660254))
        joint_frames = [
            [
                [shift_x, identity_xform],  # World0, articulation0, internal joint parent/child xforms
                [shift_ny, rot_y_60],  # World0, articulation1, internal joint parent/child xforms
            ],
            [
                [rot_z_45, shift_z],  # World1, articulation0, internal joint parent/child xforms
                [shift_x, rot_z_45],  # World1, articulation1, internal joint parent/child xforms
            ],
        ]

        joint_q = self._default_joint_q(is_floating_base)

        link_coms = [
            [
                [wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 1.0)],  # World0, articulation0, link0/link1
                [wp.vec3(0.0, -2.0, 0.0), wp.vec3(0.0, 1.0, 0.0)],  # World0, articulation1, link0/link1
            ],
            [
                [wp.vec3(2.0, 0.0, 0.0), wp.vec3(0.0, -1.0, 0.0)],  # World1, articulation0, link0/link1
                [wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 3.0, 0.0)],  # World1, articulation1, link0/link1
            ],
        ]
        link_masses = [
            [[1.0, 2.0], [3.0, 4.0]],  # World0,
            [[5.0, 6.0], [7.0, 8.0]],  # World1
        ]

        expected_grav_comp_forces = [
            0.0,  # World 0, fixed root, 1 dof
            0.0,  # World 0, floating root, 6+1 dofs
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # World 1, fixed root, 1 dof
            0.0,  # World 1, floating root, 6+1 dofs
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        for I in self.INERTIA_PASSES:
            link_inertias = [
                [[I, I], [I, I]],  # World0, articulation0/articulation1, link0/link1
                [[I, I], [I, I]],  # World1, articulation0/articulation1, link0/link1
            ]
            for i in range(0, 2):
                self._test_two_link_grav_comp_force(
                    gravity_vec=gravity_vec,
                    joint_type=joint_types[i],
                    is_floating_base=is_floating_base,
                    joint_axis=joint_axis,
                    joint_frames=joint_frames,
                    joint_q=joint_q,
                    link_coms=link_coms,
                    link_masses=link_masses,
                    link_inertias=link_inertias,
                    expected_grav_comp_forces=expected_grav_comp_forces,
                )

    def test_two_link_prismatic_grav_comp_force_from_mass(self):
        """A prismatic DOF aligned with gravity carries ``G(q) = m_distal * g``.

        With gravity along -y and the internal prismatic axis along +y
        (fully aligned), zero CoMs, identity joint frames, and zero
        internal q, each articulation's internal DOF carries
        ``m_distal * |g|`` — the parent link is reacted by either the
        fixed root or the floating base and so contributes nothing on
        the internal slider. Floating-root articulations additionally
        carry ``M_total * |g|`` on the base linear-y entry; angular and
        the other linear base entries are zero (no lever arm with zero
        CoMs). The four articulations sweep distal masses 2, 4, 6, 8 to
        confirm the slider entry scales linearly with ``m_distal``.
        """
        gravity_vec = wp.vec3(0.0, -10.0, 0.0)

        is_floating_base = [
            [False, True],  # World0, articulation0 fixed, articulation1 free
            [False, True],  # World1, articulation0 fixed, articulation1 free
        ]

        prismatic_y = wp.vec3(0.0, 1.0, 0.0)
        joint_axis = [
            [prismatic_y, prismatic_y],  # World0, articulation0/articulation1
            [prismatic_y, prismatic_y],  # World1, articulation0/articulation1
        ]

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        joint_frames = [
            [
                [identity_xform, identity_xform],  # World0, articulation0, root/internal joint
                [identity_xform, identity_xform],  # World0, articulation1, root/internal joint
            ],
            [
                [identity_xform, identity_xform],  # World1, articulation0, root/internal joint
                [identity_xform, identity_xform],  # World1, articulation1, root/internal joint
            ],
        ]

        joint_q = self._default_joint_q(is_floating_base)

        link_coms = [
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation1, link0/link1
            ],
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation1, link0/link1
            ],
        ]
        link_masses = [
            [[1.0, 2.0], [3.0, 4.0]],  # World0,
            [[5.0, 6.0], [7.0, 8.0]],  # World1
        ]

        expected_grav_comp_forces = [
            20.0,  # World 0, fixed root, 1 dof
            0.0,  # World 0, floating root, 6+1 dofs
            70.0,
            0.0,
            0.0,
            0.0,
            0.0,
            40.0,
            60.0,  # World 1, fixed root, 1 dof
            0.0,  # World 1, floating root, 6+1 dofs
            150.0,
            0.0,
            0.0,
            0.0,
            0.0,
            80.0,
        ]

        for I in self.INERTIA_PASSES:
            link_inertias = [
                [[I, I], [I, I]],
                [[I, I], [I, I]],
            ]
            self._test_two_link_grav_comp_force(
                gravity_vec=gravity_vec,
                joint_type="prismatic",
                is_floating_base=is_floating_base,
                joint_axis=joint_axis,
                joint_frames=joint_frames,
                joint_q=joint_q,
                link_coms=link_coms,
                link_masses=link_masses,
                link_inertias=link_inertias,
                expected_grav_comp_forces=expected_grav_comp_forces,
            )

    def test_two_link_prismatic_grav_comp_force_from_rotation(self):
        """Body +X slider with +-90 deg rotation injected via root quaternion or
        internal joint frame.

        Both variants achieve the same world-frame slider direction (body +X
        mapped to world +-Y) and therefore the same prismatic gravity-force entry
        (+-m_2 * |g|), but differ in where the rotation is introduced:

        - ``"rotated_root"`` -- rotation lives in the free-joint quaternion in
          ``joint_q``; all roots are floating.
        - ``"rotated_joint_frame"`` -- rotation lives in the internal joint's
          ``parent_xform``; roots are a mix of fixed and floating.
        """
        gravity_vec = wp.vec3(0.0, -10.0, 0.0)

        prismatic_x = wp.vec3(1.0, 0.0, 0.0)
        joint_axis = [
            [prismatic_x, prismatic_x],  # World0
            [prismatic_x, prismatic_x],  # World1
        ]

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

        link_coms = [
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation0
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation1
            ],
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation0
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation1
            ],
        ]
        link_masses = [
            [[1.0, 2.0], [3.0, 4.0]],  # World0
            [[5.0, 6.0], [7.0, 8.0]],  # World1
        ]

        # Quaternions for +-90 deg rotations about +Z.
        root_quat_pos = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2.0)
        root_quat_neg = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -np.pi / 2.0)
        floating_q_rot_z_90 = [
            0.0,
            0.0,
            0.0,
            root_quat_pos.x,
            root_quat_pos.y,
            root_quat_pos.z,
            root_quat_pos.w,
            0.0,
        ]
        floating_q_rot_z_neg90 = [
            0.0,
            0.0,
            0.0,
            root_quat_neg.x,
            root_quat_neg.y,
            root_quat_neg.z,
            root_quat_neg.w,
            0.0,
        ]

        joint_frame_quat_pos = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2.0)
        joint_frame_quat_neg = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -np.pi / 2.0)
        parent_xform_rot_z_90 = wp.transform(wp.vec3(0.0, 0.0, 0.0), joint_frame_quat_pos)
        parent_xform_rot_z_neg90 = wp.transform(wp.vec3(0.0, 0.0, 0.0), joint_frame_quat_neg)

        cases = [
            (
                "rotated_root",
                # All floating so rotation can live in joint_q.
                [[True, True], [True, True]],
                # Identity joint frames; rotation injected via root quaternion.
                [
                    [[identity_xform, identity_xform], [identity_xform, identity_xform]],
                    [[identity_xform, identity_xform], [identity_xform, identity_xform]],
                ],
                # W0: a0=+90 deg, a1=-90 deg; W1: a0=-90 deg, a1=+90 deg.
                [
                    [floating_q_rot_z_90, floating_q_rot_z_neg90],
                    [floating_q_rot_z_neg90, floating_q_rot_z_90],
                ],
                # 6 base DOFs + 1 internal = 7 DOFs per articulation.
                # lin_y = M_total * 10; prismatic = +-m_2 * 10.
                [
                    0.0,
                    30.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    20.0,  # W0 a0 [1,2]: M=3,  m_2=2  (+90 deg)
                    0.0,
                    70.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -40.0,  # W0 a1 [3,4]: M=7,  m_2=4  (-90 deg)
                    0.0,
                    110.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -60.0,  # W1 a0 [5,6]: M=11, m_2=6  (-90 deg)
                    0.0,
                    150.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    80.0,  # W1 a1 [7,8]: M=15, m_2=8  (+90 deg)
                ],
            ),
            (
                "rotated_joint_frame",
                # Fixed and floating roots: rotation comes from the joint frame.
                [[False, True], [False, True]],
                # Rotation injected through the internal joint parent_xform.
                [
                    [
                        [parent_xform_rot_z_90, identity_xform],  # W0 a0 (+90 deg)
                        [parent_xform_rot_z_neg90, identity_xform],  # W0 a1 (-90 deg)
                    ],
                    [
                        [parent_xform_rot_z_neg90, identity_xform],  # W1 a0 (-90 deg)
                        [parent_xform_rot_z_90, identity_xform],  # W1 a1 (+90 deg)
                    ],
                ],
                None,  # use _default_joint_q
                # Fixed root: 1 DOF (internal only). Floating: 7 DOFs.
                # prismatic = +-m_2 * 10; lin_y = M_total * 10 for floating.
                [
                    20.0,  # W0 a0 [1,2]: fixed,    +90 deg, m_2=2
                    0.0,
                    70.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -40.0,  # W0 a1 [3,4]: floating, -90 deg, M=7,  m_2=4
                    -60.0,  # W1 a0 [5,6]: fixed,    -90 deg, m_2=6
                    0.0,
                    150.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    80.0,  # W1 a1 [7,8]: floating, +90 deg, M=15, m_2=8
                ],
            ),
        ]

        for variant, is_floating_base, joint_frames, joint_q_override, expected_grav_comp_forces in cases:
            joint_q = self._default_joint_q(is_floating_base) if joint_q_override is None else joint_q_override
            with self.subTest(variant=variant):
                for I in self.INERTIA_PASSES:
                    link_inertias = [[[I, I], [I, I]], [[I, I], [I, I]]]
                    self._test_two_link_grav_comp_force(
                        gravity_vec=gravity_vec,
                        joint_type="prismatic",
                        is_floating_base=is_floating_base,
                        joint_axis=joint_axis,
                        joint_frames=joint_frames,
                        joint_q=joint_q,
                        link_coms=link_coms,
                        link_masses=link_masses,
                        link_inertias=link_inertias,
                        expected_grav_comp_forces=expected_grav_comp_forces,
                    )

    def test_two_link_prismatic_grav_comp_force_from_com(self):
        """Lateral CoM offsets produce angular-z entries on floating roots.

        Same setup as
        :meth:`test_two_link_prismatic_grav_comp_force_from_mass` —
        prismatic +y axis aligned with -y gravity, identity joint frames,
        zero internal q — but with non-zero per-link CoM offsets along
        +x. The internal prismatic entry is unchanged (a transverse lever
        arm doesn't project onto an axial slider DOF), and the linear-y
        base entry on floating roots remains ``M_total * |g|``. Floating
        roots additionally develop an angular-z entry equal to
        ``sum_i m_i * x_i * |g|`` from the cross product
        ``r_com x (0, -g, 0)``; angular x and y stay zero because the
        CoM offsets have no y or z component.
        """
        gravity_vec = wp.vec3(0.0, -10.0, 0.0)

        is_floating_base = [
            [False, True],  # World0, articulation0 fixed, articulation1 free
            [False, True],  # World1, articulation0 fixed, articulation1 free
        ]

        prismatic_y = wp.vec3(0.0, 1.0, 0.0)
        joint_axis = [
            [prismatic_y, prismatic_y],  # World0, articulation0/articulation1
            [prismatic_y, prismatic_y],  # World1, articulation0/articulation1
        ]

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        joint_frames = [
            [
                [identity_xform, identity_xform],  # World0, articulation0, root/internal joint
                [identity_xform, identity_xform],  # World0, articulation1, root/internal joint
            ],
            [
                [identity_xform, identity_xform],  # World1, articulation0, root/internal joint
                [identity_xform, identity_xform],  # World1, articulation1, root/internal joint
            ],
        ]

        joint_q = self._default_joint_q(is_floating_base)

        link_coms = [
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation0, link0/link1
                [wp.vec3(0.5, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation1, link0/link1
            ],
            [
                [wp.vec3(0.5, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation0, link0/link1
                [wp.vec3(0.5, 0.0, 0.0), wp.vec3(0.5, 0.0, 0.0)],  # World1, articulation1, link0/link1
            ],
        ]
        link_masses = [
            [[1.0, 2.0], [3.0, 4.0]],  # World0,
            [[5.0, 6.0], [7.0, 8.0]],  # World1
        ]

        expected_grav_comp_forces = [
            20.0,  # World 0, fixed root, 1 dof
            0.0,  # World 0, floating root, 6+1 dofs (linear x, y, z, angular x, y, z, internal q)
            70.0,
            0.0,
            0.0,
            0.0,
            # angular z about root CoM: child (offset from root CoM by
            # (-0.5, 0, 0)) carries gravity force (0, -40, 0); the torque
            # gravity exerts about root CoM is r X F = (-0.5, 0, 0) X
            # (0, -40, 0) = (0, 0, +20). g(q) = ∂U/∂q is the joint-space
            # force needed to hold static, i.e. the counter-torque: -20.
            -20.0,
            40.0,
            60.0,  # World 1, fixed root, 1 dof
            0.0,  # World 1, floating root, 6+1 dofs
            150.0,
            0.0,
            0.0,
            0.0,
            # angular z about root CoM: child CoM coincides with root CoM (both at (0.5, 0, 0))
            # → torque about root CoM = 0
            0.0,
            80.0,
        ]

        for I in self.INERTIA_PASSES:
            link_inertias = [
                [[I, I], [I, I]],
                [[I, I], [I, I]],
            ]
            self._test_two_link_grav_comp_force(
                gravity_vec=gravity_vec,
                joint_type="prismatic",
                is_floating_base=is_floating_base,
                joint_axis=joint_axis,
                joint_frames=joint_frames,
                joint_q=joint_q,
                link_coms=link_coms,
                link_masses=link_masses,
                link_inertias=link_inertias,
                expected_grav_comp_forces=expected_grav_comp_forces,
            )

    def test_two_link_grav_comp_force_zero_internal_dof_degenerate_axis(self):
        """Internal-DOF G(q) is zero when the joint axis cannot do work against gravity.

        Two joint types are checked:

        * prismatic along +x with gravity along -y: the force projection
          g . axis is zero, so the generalized force on the prismatic DOF
          vanishes regardless of distal mass.
        * revolute about +y with gravity along -y: the gravity force on
          the distal link is collinear with the joint axis, so the moment
          (r x F) . axis is identically zero for any lever arm.

        In both cases the per-articulation internal-DOF entry in G(q) must be
        exactly zero. On floating-root articulations the base linear-y entry
        still picks up the total weight, confirming gravity is applied.
        """
        gravity_vec = wp.vec3(0.0, -10.0, 0.0)

        is_floating_base = [
            [False, True],  # World0, articulation0 fixed, articulation1 free
            [False, True],  # World1, articulation0 fixed, articulation1 free
        ]

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        joint_frames = [
            [
                [identity_xform, identity_xform],  # World0, articulation0, internal joint parent/child xforms
                [identity_xform, identity_xform],  # World0, articulation1, internal joint parent/child xforms
            ],
            [
                [identity_xform, identity_xform],  # World1, articulation0, internal joint parent/child xforms
                [identity_xform, identity_xform],  # World1, articulation1, internal joint parent/child xforms
            ],
        ]

        joint_q = self._default_joint_q(is_floating_base)

        link_coms = [
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation1, link0/link1
            ],
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation1, link0/link1
            ],
        ]
        link_masses = [
            [[1.0, 2.0], [1.0, 2.0]],  # World0
            [[1.0, 2.0], [1.0, 2.0]],  # World1
        ]

        # Fixed root: only DOF is the internal joint -> 0 (invariant).
        # Floating root: (v_x, v_y, v_z, omega_x, omega_y, omega_z, q_internal).
        #   Linear y = M_total * |g| = 3 * 10 = 30 (total weight).
        #   Angular and internal = 0 (both CoMs at root origin; degenerate axis).
        expected_grav_comp_forces = [
            0.0,  # World 0, fixed root, 1 dof (internal joint)
            0.0,  # World 0, floating root, 6+1 dofs
            30.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # World 1, fixed root, 1 dof (internal joint)
            0.0,  # World 1, floating root, 6+1 dofs
            30.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        cases = [
            # prismatic perp gravity: g . axis = 0
            ("prismatic", wp.vec3(1.0, 0.0, 0.0)),
            # revolute parallel gravity: (r x F) . axis = 0
            ("revolute", wp.vec3(0.0, 1.0, 0.0)),
        ]
        for joint_type, axis in cases:
            joint_axis = [
                [axis, axis],  # World0, articulation0/articulation1
                [axis, axis],  # World1, articulation0/articulation1
            ]
            with self.subTest(joint_type=joint_type):
                for I in self.INERTIA_PASSES:
                    link_inertias = [
                        [[I, I], [I, I]],
                        [[I, I], [I, I]],
                    ]
                    self._test_two_link_grav_comp_force(
                        gravity_vec=gravity_vec,
                        joint_type=joint_type,
                        is_floating_base=is_floating_base,
                        joint_axis=joint_axis,
                        joint_frames=joint_frames,
                        joint_q=joint_q,
                        link_coms=link_coms,
                        link_masses=link_masses,
                        link_inertias=link_inertias,
                        expected_grav_comp_forces=expected_grav_comp_forces,
                    )

    def test_two_link_fixed_revolute_gravity_force_matches_closed_form(self):
        """Internal revolute DOF matches the closed-form ``m * g * arm_length * cos(q)``.

        Fixed-root arm, internal revolute about +z anchored at the world
        origin. The child body's origin is placed at ``(arm_length, 0, 0)``
        by setting the internal joint's ``child_xform`` to
        ``(-arm_length, 0, 0)``; the distal link's body-frame CoM is the
        origin, so link 1's world CoM at angle ``q`` is
        ``(arm_length * cos q, arm_length * sin q, 0)``. Under gravity
        ``(0, -g, 0)`` the Lagrangian generalized gravity force on the
        internal DOF reduces to
        ``g(q) = ∂U/∂q = m_distal * g * arm_length * cos(q)``, which is
        exactly what Newton's ``gravity_force`` stores, so
        we compare ``tau[0]`` directly against the closed form over a
        pose sweep.
        """
        arm_length = 1.0
        m_distal = 2.0
        g_mag = 10.0

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        child_anchor_back = wp.transform(wp.vec3(-arm_length, 0.0, 0.0), wp.quat_identity())
        for I in self.INERTIA_PASSES:
            builder = self._build_two_link_articulation(
                gravity=wp.vec3(0.0, -g_mag, 0.0),
                floating_base=False,
                joint_type="revolute",
                joint_axis=wp.vec3(0.0, 0.0, 1.0),
                link_coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
                link_masses=[1.0, m_distal],
                joint_frames=[identity_xform, child_anchor_back],
                link_inertias=[I, I],
            )
            model = builder.finalize(device=self.device)
            state = model.state()
            inverse_dynamics = _InverseDynamicsArrays(model)

            sweep = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
            for q in sweep:
                joint_q = state.joint_q.numpy()
                joint_q[0] = q
                state.joint_q.assign(joint_q)
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                _eval_inverse_dynamics_passive(
                    model=model,
                    state=state,
                    eval_type=_PassiveOutput.GRAVITY_FORCE,
                    arrays=inverse_dynamics,
                )
                tau = inverse_dynamics.gravity_force.numpy()

                expected = m_distal * g_mag * arm_length * np.cos(q)
                np.testing.assert_allclose(
                    tau[0],
                    expected,
                    atol=1e-5,
                    rtol=1e-5,
                    err_msg=f"At q = {q}: expected {expected}, got tau = {tau[0]}",
                )

    def test_gravity_three_worlds_different_axes(self):
        """G(q) for a free body in each of three worlds with X, Y, Z gravity vectors.

        Three worlds, each containing a single free body of mass ``m`` at the
        origin with identity orientation and CoM at the body origin. World 0
        has gravity along +X, world 1 along +Y, world 2 along +Z. Under
        Newton's free-joint convention the per-world gravity force must
        equal ``-m * g_world`` in the linear part and zero in the
        angular part, which exercises the per-world ``body_world`` lookup
        inside the RNEA gravity term.
        """
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        m = 1.5
        g_mag = 9.81
        I_body = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        # Build a single-world template containing one free body.
        def build_world() -> newton.ModelBuilder:
            b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
            body = b.add_link(
                xform=identity_xform,
                mass=m,
                inertia=I_body,
                com=wp.vec3(0.0, 0.0, 0.0),
            )
            j = b.add_joint_free(
                parent=-1,
                child=body,
                parent_xform=identity_xform,
                child_xform=identity_xform,
            )
            b.add_articulation([j])
            return b

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        for _ in range(3):
            builder.add_world(build_world())

        model = builder.finalize(device=self.device)
        self.assertEqual(model.world_count, 3)
        self.assertEqual(model.articulation_count, 3)
        self.assertEqual(model.joint_dof_count, 18)  # 3 free joints * 6 DOFs

        # Per-world gravity vectors along the three world axes.
        gravity_per_world = [
            (g_mag, 0.0, 0.0),  # world 0: along +X
            (0.0, g_mag, 0.0),  # world 1: along +Y
            (0.0, 0.0, g_mag),  # world 2: along +Z
        ]
        for w, g in enumerate(gravity_per_world):
            model.set_gravity(g, world=w)

        # Identity pose for every body: translation = 0, rotation = identity quat.
        state = model.state()
        joint_q = state.joint_q.numpy()
        for w in range(3):
            joint_q[w * 7 : (w + 1) * 7] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        state.joint_q.assign(joint_q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        inverse_dynamics = _InverseDynamicsArrays(model)
        _eval_inverse_dynamics_passive(
            model=model,
            state=state,
            eval_type=_PassiveOutput.GRAVITY_FORCE,
            arrays=inverse_dynamics,
        )

        measured = inverse_dynamics.gravity_force.numpy()

        # Free joint: 6 DOFs per articulation -- linear at CoM (parent frame =
        # world for parent=-1), then angular at CoM. Newton's
        # gravity_force stores g(q) = ∂U/∂q, the standard
        # manipulator-equation gravity bias. With U = -m * g_world . x_com,
        # ∂U/∂x_com = -m * g_world, so the linear part equals -m * g_world.
        # The angular part is zero (body CoM at the body origin -> no
        # gravity torque).
        for w, g in enumerate(gravity_per_world):
            with self.subTest(world=w, gravity=g):
                expected_linear = -m * np.asarray(g, dtype=np.float64)
                expected_angular = np.zeros(3)
                np.testing.assert_allclose(measured[w * 6 : w * 6 + 3], expected_linear, atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(measured[w * 6 + 3 : w * 6 + 6], expected_angular, atol=1e-5, rtol=1e-5)

    def test_free_body_rotated_parent_frame_gravity_is_world_frame(self):
        """gravity_force for a free/distance body must be expressed in world frame, not parent frame.

        Regression test for a bug where the RNEA bias (gravity and Coriolis
        contributions) was left in the joint-parent frame rather than rotated
        to the world frame expected by Newton's free-joint convention
        (:attr:`~newton.Model.joint_f` documents world-frame forces at CoM).
        The same code path handles both FREE and DISTANCE joints, so both
        are exercised as subtests.

        Setup: gravity along world +Z, ``parent_xform`` with a 90-degree
        rotation about world X. The parent frame's Y-axis points along
        world +Z, so the buggy parent-frame output would be (0, -m*g, 0)
        while the correct world-frame output is (0, 0, -m*g).

        This test covers only the RNEA bias term (``GRAVITY_FORCE``). The
        companion mass-matrix contribution -- that the ``M(q)*joint_qdd`` wrench
        surfaced by :func:`~newton.eval_inverse_dynamics_force` obeys the same
        world-frame contract for a rotated-parent FREE root under a known
        ``joint_qdd`` (gravity disabled) -- is covered by
        :meth:`test_inverse_dynamics_force_free_root_rotated_parent`.
        """
        m = 2.0
        g_mag = 10.0
        I_body = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        # 90-degree rotation about world X: parent Y → world +Z.
        q_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2.0)
        rotated_parent_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), q_rot)
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

        def add_joint(jtype, builder, body):
            if jtype == "free":
                return builder.add_joint_free(
                    parent=-1,
                    child=body,
                    parent_xform=rotated_parent_xform,
                    child_xform=identity_xform,
                )
            return builder.add_joint_distance(
                parent=-1,
                child=body,
                parent_xform=rotated_parent_xform,
                child_xform=identity_xform,
            )

        for jtype in ("free", "distance"):
            with self.subTest(joint_type=jtype):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
                body = builder.add_link(
                    xform=identity_xform,
                    mass=m,
                    inertia=I_body,
                    com=wp.vec3(0.0, 0.0, 0.0),
                )
                j = add_joint(jtype, builder, body)
                builder.add_articulation([j])

                model = builder.finalize(device=self.device)
                model.set_gravity((0.0, 0.0, g_mag))  # gravity along world +Z

                # Body at rest, identity orientation relative to parent frame.
                # joint_q: [tx, ty, tz, qx, qy, qz, qw] in parent frame (7 DOFs).
                state = model.state()
                joint_q = state.joint_q.numpy()
                joint_q[0:7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                state.joint_q.assign(joint_q)
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                inverse_dynamics = _InverseDynamicsArrays(model)
                _eval_inverse_dynamics_passive(
                    model=model,
                    state=state,
                    eval_type=_PassiveOutput.GRAVITY_FORCE,
                    arrays=inverse_dynamics,
                )

                measured = inverse_dynamics.gravity_force.numpy()

                # Newton convention: gravity_force[0:3] = -m * g_world (world-frame linear force).
                # With gravity along world +Z: expected linear = (0, 0, -m*g_mag).
                # Without the world-frame rotation fix the output stays in parent frame:
                # gravity in parent frame = R_x(-90°) * (0,0,g) = (0,g,0), so the
                # buggy output would be (0, -m*g_mag, 0).
                expected_linear = np.array([0.0, 0.0, -m * g_mag])
                expected_angular = np.zeros(3)
                np.testing.assert_allclose(measured[0:3], expected_linear, atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(measured[3:6], expected_angular, atol=1e-5, rtol=1e-5)


class TestCoriolisCompForce(TestInverseDynamicsBase):
    """Coriolis-force tests for the two-link pendulum harness."""

    def test_coriolis_zero_at_rest(self):
        """C(q, q_dot) must vanish when q_dot = 0."""
        builder = self._build_two_link_articulation(
            gravity=wp.vec3(0.0, -9.81, 0.0),
            floating_base=False,
            joint_type="revolute",
            joint_axis=wp.vec3(0.0, 0.0, 1.0),
            link_coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
            link_masses=[1.0, 2.0],
            joint_frames=[
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            ],
            link_inertias=[self.I_UNIT, self.I_UNIT],
        )
        model = builder.finalize(device=self.device)
        state = model.state()
        joint_q = state.joint_q.numpy()
        joint_q[0] = 0.3
        state.joint_q.assign(joint_q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        state.joint_qd.zero_()

        inverse_dynamics = _InverseDynamicsArrays(model)
        _eval_inverse_dynamics_passive(
            model,
            state,
            _PassiveOutput.CORIOLIS_FORCE,
            inverse_dynamics,
        )

        tau = inverse_dynamics.coriolis_force.numpy()
        np.testing.assert_allclose(tau, np.zeros_like(tau), atol=1e-6)

    def test_coriolis_double_pendulum_matches_analytical(self):
        """C(q, q_dot)*q_dot for a planar 3D double pendulum matches its closed-form values.

        Setup: a fixed-base double pendulum with two revolute joints both
        about world +Y, link length 1.0, and a 25 kg point mass at each
        link midpoint. At ``q = (0, pi/2)`` link 1 points along world +X
        and link 2 along world -Z. Because both joint axes lie along
        world Y, motion is planar in the X-Z plane and the Coriolis term
        collapses to a closed form that is independent of the link
        rotational inertias:

            c_1 = -m * L_1 * l_2c * sin(q2) * (2 * q_dot1 * q_dot2 + q_dot2^2)
            c_2 =  m * L_1 * l_2c * sin(q2) * q_dot1^2

        With ``m = 25``, ``L_1 = 1``, ``l_2c = 0.5`` the prefactor is 12.5;
        the test evaluates these formulas at each ``q_dot`` case below and
        compares against ``coriolis_force`` from
        :func:`newton.eval_inverse_dynamics_passive`, which stores the standard
        manipulator-equation bias term ``+C(q, q_dot)*q_dot``.
        """
        # Per-link mass, link length L_1 (joint-to-joint distance), and link
        # COM offset from the joint l_2c. Defined here so the closed-form
        # Coriolis derivation below uses the same values the links are built
        # with. Joints sit at the link's +/- X end and the COM is at the link
        # center, so l_2c is L_1 / 2.
        m = 25.0
        L_1 = 1.0
        l_2c = L_1 / 2.0

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        pos_half = wp.transform(wp.vec3(l_2c, 0.0, 0.0), wp.quat_identity())
        neg_half = wp.transform(wp.vec3(-l_2c, 0.0, 0.0), wp.quat_identity())
        y_axis = wp.vec3(0.0, 1.0, 0.0)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0), up_axis=newton.Axis.Z)

        b1 = builder.add_link(
            xform=identity_xform,
            mass=m,
            inertia=self.I_UNIT,
            com=wp.vec3(0.0, 0.0, 0.0),
        )
        j1 = builder.add_joint_revolute(
            parent=-1,
            child=b1,
            axis=y_axis,
            parent_xform=identity_xform,
            child_xform=neg_half,
        )
        b2 = builder.add_link(
            xform=identity_xform,
            mass=m,
            inertia=self.I_UNIT,
            com=wp.vec3(0.0, 0.0, 0.0),
        )
        j2 = builder.add_joint_revolute(
            parent=b1,
            child=b2,
            axis=y_axis,
            parent_xform=pos_half,
            child_xform=neg_half,
        )
        builder.add_articulation([j1, j2], label="double_pendulum")

        model = builder.finalize(device=self.device)
        inverse_dynamics = _InverseDynamicsArrays(model)

        # All cases share q = (0, pi/2); only q_dot varies. The expected
        # +C(q, q_dot) * q_dot values come from the closed-form Coriolis terms
        # in the docstring evaluated with this articulation's geometry
        # (m, L_1, l_2c defined above), giving
        # prefactor = m * L_1 * l_2c * sin(q2).
        q1 = 0.0
        q2 = np.pi / 2.0
        prefactor = m * L_1 * l_2c * np.sin(q2)
        qd_cases = [(1.5, 0.0), (1.5, 1.5)]

        for qd1, qd2 in qd_cases:
            expected = (
                -prefactor * (2.0 * qd1 * qd2 + qd2 * qd2),
                prefactor * qd1 * qd1,
            )
            with self.subTest(joint_qd=(qd1, qd2)):
                state = model.state()
                joint_q = state.joint_q.numpy()
                joint_q[:] = (q1, q2)
                state.joint_q.assign(joint_q)
                joint_qd = state.joint_qd.numpy()
                joint_qd[:] = (qd1, qd2)
                state.joint_qd.assign(joint_qd)
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                _eval_inverse_dynamics_passive(
                    model,
                    state,
                    _PassiveOutput.CORIOLIS_FORCE,
                    inverse_dynamics,
                )

                # Newton's coriolis_force stores the standard
                # manipulator-equation Coriolis bias +C(q, q_dot)*q_dot,
                # which is the closed-form ``expected`` above.
                measured = inverse_dynamics.coriolis_force.numpy()
                np.testing.assert_allclose(measured, expected, atol=1e-3, rtol=1e-5)

    def test_coriolis_radial_slider_matches_analytical(self):
        """C(q, q_dot) for a rotating radial slider matches the closed-form values.

        Classic Coriolis textbook setup: a revolute joint about world
        +Z carries an inner link, and a prismatic joint along that
        link's local +X carries a 0.5 kg point-mass slider. With
        ``q = (theta, r)`` and ``q_dot = (omega, v_r)`` the slider
        traces a circle of varying radius and the Coriolis term is:

            c_theta = 2 * m * r * omega * v_r   (Coriolis coupling)
            c_r     = -m * r * omega^2          (centrifugal pull)

        Both formulas are independent of the link rotational inertias,
        because the angular velocity vector is purely along +Z and
        ``I[2, 2]`` (i.e. ``Izz``, the moment of inertia about Z) is
        invariant under rotations about Z. The outer loop sweeps three
        qualitatively different link inertias (negligible, unit, 100x
        unit) to confirm this.
        """
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        z_axis = wp.vec3(0.0, 0.0, 1.0)
        x_axis = wp.vec3(1.0, 0.0, 0.0)

        m_slider = 0.5
        m_base = 1e-6

        omega = 2.0
        v_r = 0.1
        r = 1.0
        theta = 0.7  # arbitrary; system is rotationally symmetric about +Z

        expected = (
            2.0 * m_slider * r * omega * v_r,
            -m_slider * r * omega * omega,
        )

        I_negligible = wp.mat33(1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 0.0, 1e-6)
        for link_inertia in (I_negligible, *self.INERTIA_PASSES):
            with self.subTest(inertia=link_inertia):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)

                base = builder.add_link(
                    xform=identity_xform,
                    mass=m_base,
                    inertia=link_inertia,
                    com=wp.vec3(0.0, 0.0, 0.0),
                )
                j_rot = builder.add_joint_revolute(
                    parent=-1,
                    child=base,
                    axis=z_axis,
                    parent_xform=identity_xform,
                    child_xform=identity_xform,
                    target_ke=0.0,
                    target_kd=0.0,
                    friction=0.0,
                )
                slider = builder.add_link(
                    xform=identity_xform,
                    mass=m_slider,
                    inertia=link_inertia,
                    com=wp.vec3(0.0, 0.0, 0.0),
                )
                j_slide = builder.add_joint_prismatic(
                    parent=base,
                    child=slider,
                    axis=x_axis,
                    parent_xform=identity_xform,
                    child_xform=identity_xform,
                    target_ke=0.0,
                    target_kd=0.0,
                    friction=0.0,
                )
                builder.add_articulation([j_rot, j_slide], label="radial_slider")

                model = builder.finalize(device=self.device)
                inverse_dynamics = _InverseDynamicsArrays(model)

                state = model.state()
                joint_q = state.joint_q.numpy()
                joint_q[:] = (theta, r)
                state.joint_q.assign(joint_q)
                joint_qd = state.joint_qd.numpy()
                joint_qd[:] = (omega, v_r)
                state.joint_qd.assign(joint_qd)
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                _eval_inverse_dynamics_passive(
                    model,
                    state,
                    _PassiveOutput.CORIOLIS_FORCE,
                    inverse_dynamics,
                )

                measured = inverse_dynamics.coriolis_force.numpy()
                np.testing.assert_allclose(measured, expected, atol=1e-4, rtol=1e-5)

    def test_coriolis_anisotropic_gimbal_independent_of_mass(self):
        """C(q, q_dot) for a 2-DOF gimbal depends on rotational inertia but not on link mass.

        Two revolute joints (about world +Z, then the inner link's
        local +Y) share the world origin as their pivot, with both
        link CoMs at the origin. Nothing translates as the joints
        move, so translational kinetic energy vanishes and mass drops
        out of ``M(q)`` entirely. With the outer-link body-frame
        inertia set to ``diag(Ix, Iy, Iz)``, the closed form is:

            c_1 = (Ix - Iz) * sin(2*q2) * q_dot1 * q_dot2
            c_2 = -0.5 * (Ix - Iz) * sin(2*q2) * q_dot1^2

        The outer loop sweeps two qualitatively different link masses
        to confirm empirically that ``C`` does not depend on mass.
        """
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        z_axis = wp.vec3(0.0, 0.0, 1.0)
        y_axis = wp.vec3(0.0, 1.0, 0.0)

        Ix = 2.0
        Iy = 1.5
        Iz = 1.0
        outer_inertia = wp.mat33(Ix, 0.0, 0.0, 0.0, Iy, 0.0, 0.0, 0.0, Iz)

        q1 = 0.0  # arbitrary; system is rotationally symmetric about +Z
        q2 = np.pi / 4.0
        qd1 = 1.0
        qd2 = 1.0

        expected = (
            (Ix - Iz) * np.sin(2.0 * q2) * qd1 * qd2,
            -0.5 * (Ix - Iz) * np.sin(2.0 * q2) * qd1 * qd1,
        )

        for link_mass in (0.5, 50.0):
            with self.subTest(mass=link_mass):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)

                inner = builder.add_link(
                    xform=identity_xform,
                    mass=link_mass,
                    inertia=self.I_UNIT,
                    com=wp.vec3(0.0, 0.0, 0.0),
                )
                j1 = builder.add_joint_revolute(
                    parent=-1,
                    child=inner,
                    axis=z_axis,
                    parent_xform=identity_xform,
                    child_xform=identity_xform,
                    target_ke=0.0,
                    target_kd=0.0,
                    friction=0.0,
                )
                outer = builder.add_link(
                    xform=identity_xform,
                    mass=link_mass,
                    inertia=outer_inertia,
                    com=wp.vec3(0.0, 0.0, 0.0),
                )
                j2 = builder.add_joint_revolute(
                    parent=inner,
                    child=outer,
                    axis=y_axis,
                    parent_xform=identity_xform,
                    child_xform=identity_xform,
                    target_ke=0.0,
                    target_kd=0.0,
                    friction=0.0,
                )
                builder.add_articulation([j1, j2], label="anisotropic_gimbal")

                model = builder.finalize(device=self.device)
                inverse_dynamics = _InverseDynamicsArrays(model)

                state = model.state()
                joint_q = state.joint_q.numpy()
                joint_q[:] = (q1, q2)
                state.joint_q.assign(joint_q)
                joint_qd = state.joint_qd.numpy()
                joint_qd[:] = (qd1, qd2)
                state.joint_qd.assign(joint_qd)
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                _eval_inverse_dynamics_passive(
                    model,
                    state,
                    _PassiveOutput.CORIOLIS_FORCE,
                    inverse_dynamics,
                )

                measured = inverse_dynamics.coriolis_force.numpy()
                np.testing.assert_allclose(measured, expected, atol=1e-4, rtol=1e-5)

    def test_coriolis_floating_root_with_com_offset(self):
        """Pin down Newton's free-joint Coriolis convention with non-zero root CoM.

        Builds a single free-joint body with a non-zero CoM offset and a
        known angular velocity, then compares Newton's
        ``coriolis_force`` against the closed-form spatial
        Coriolis bias under Newton's documented free-joint convention --
        joint_qd's linear part is parent/world-frame CoM velocity, so
        joint_f is a wrench at the body CoM expressed in the world frame.
        With pose = identity (so body frame = world frame and
        ``I_world = I_body``), Newton's second law at the CoM gives:

            linear  = 0                       (m * a_com = F, no Coriolis)
            angular = omega x (I_body * omega)  (gyroscopic)

        Newton's ``coriolis_force`` stores the standard
        manipulator-equation bias ``+C(q, q_dot)*q_dot``, so the test
        compares ``tau`` directly against the closed-form values.

        Both the linear and angular components must match: the angular
        bias is the gyroscopic ``omega x (I * omega)``, and the linear
        bias is zero under Newton's documented v_com convention. The
        latter requires the bias output of Featherstone's spatial RNEA
        to be corrected for the qdd convention mismatch (Featherstone's
        ``a_F = 0`` means classical ``a_com = omega x v_com``, not zero),
        so a residual ``omega x m * v_com`` is subtracted from F_linear
        during the internal-to-public conversion.
        """
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

        m = 1.0
        # Anisotropic body-frame inertia so the gyroscopic angular bias
        # omega x (I * omega) is non-zero for a generic omega.
        I_diag = (2.0, 1.5, 1.0)
        I_body = wp.mat33(I_diag[0], 0.0, 0.0, 0.0, I_diag[1], 0.0, 0.0, 0.0, I_diag[2])
        # Non-zero CoM offset so the conversion's angular correction
        # ``m * r_com x (omega x v_com)`` is exercised in addition to the
        # linear ``omega x m * v_com`` correction.
        r_com = wp.vec3(0.5, 0.2, -0.3)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        body = builder.add_link(
            xform=identity_xform,
            mass=m,
            inertia=I_body,
            com=r_com,
        )
        # ``parent_xform.rotation`` is left at identity to avoid a known
        # MuJoCo-bridge convention bug for free joints with a rotated
        # parent frame: https://github.com/newton-physics/newton/issues/2704.
        j_root = builder.add_joint_free(
            parent=-1,
            child=body,
            parent_xform=identity_xform,
            child_xform=identity_xform,
        )
        builder.add_articulation([j_root], label="floating_body")

        model = builder.finalize(device=self.device)
        state = model.state()
        inverse_dynamics = _InverseDynamicsArrays(model)

        # Sweep two states: stationary CoM with rotation (purely gyroscopic),
        # and translating + rotating CoM. Under Newton's v_com convention both
        # cases have zero linear Coriolis bias at the CoM.
        v_com_cases = [
            (0.0, 0.0, 0.0),
            (0.1, -0.2, 0.05),
        ]
        omega_cases = [
            (0.3, -0.1, 0.2),
            (0.3, -0.1, 0.2),
        ]
        num_cases = len(v_com_cases)

        I_np = np.diag(I_diag)
        for i in range(num_cases):
            v_com_values = v_com_cases[i]
            omega_values = omega_cases[i]
            with self.subTest(v_com=v_com_values, omega=omega_values):
                # Set the initial pos.
                joint_q = state.joint_q.numpy()
                joint_q[:] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # pose = identity
                state.joint_q.assign(joint_q)

                # Set the initial velocity at the CoM.
                joint_qd = state.joint_qd.numpy()
                joint_qd[:] = (*v_com_values, *omega_values)
                state.joint_qd.assign(joint_qd)

                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                _eval_inverse_dynamics_passive(
                    model,
                    state,
                    _PassiveOutput.CORIOLIS_FORCE,
                    inverse_dynamics,
                )
                # Newton stores the standard +C(q, q_dot)*q_dot directly.
                measured_linear = inverse_dynamics.coriolis_force.numpy()[0:3]
                measured_angular = inverse_dynamics.coriolis_force.numpy()[3:6]

                # Closed-form spatial Coriolis bias at the body CoM under
                # Newton's documented free-joint convention. Linear is
                # zero (Newton's second law at the CoM in an inertial
                # frame: m * a_com = F, with no Coriolis term); angular
                # is the standard gyroscopic bias.
                omega = np.asarray(omega_values, dtype=np.float64)
                expected_linear = np.zeros(3)
                expected_angular = np.cross(omega, I_np @ omega)

                np.testing.assert_allclose(measured_linear, expected_linear, atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(measured_angular, expected_angular, atol=1e-5, rtol=1e-5)

    def test_coriolis_floating_root_with_non_identity_child_xform(self):
        """Verify free-joint Coriolis with a non-identity child_xform on the free joint.

        Single free-joint body with a non-identity ``child_xform``
        (translation + rotation about world +Z) on the root joint, plus
        non-zero CoM and anisotropic inertia. The conversion kernels go
        through ``body_q[child]`` to compute ``r_child_com_parent``, so
        non-identity ``child_xform`` changes the body's world pose seen
        by the kernels even when ``joint_q`` is the identity.

        With ``parent_xform = identity``, ``joint_q.rotation = identity``,
        and ``child_xform.rotation = R_c``, the body's world rotation is
        ``R_b = R_c^T``. The body-frame inertia ``I_body`` then maps to
        world frame as ``I_world = R_b * I_body * R_b^T``, and the
        closed-form Coriolis bias at the body CoM under Newton's v_com
        convention is::

            linear  = 0
            angular = omega x (I_world * omega)

        The linear component is zero independently of ``R_b`` (Newton's
        second law at the CoM); the angular component exercises the
        body-frame to world-frame inertia rotation path that
        identity-``child_xform`` tests cannot reach.
        """
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        # 30 deg rotation about world +Z combined with a translation:
        # quat (x, y, z, w) for half-angle 15 deg about +Z.
        half_angle = float(np.pi / 12.0)
        child_xform_quat = wp.quat(0.0, 0.0, float(np.sin(half_angle)), float(np.cos(half_angle)))
        child_xform_offset = wp.transform(wp.vec3(0.3, -0.2, 0.1), child_xform_quat)

        m = 1.0
        I_diag = (2.0, 1.5, 1.0)
        I_body = wp.mat33(I_diag[0], 0.0, 0.0, 0.0, I_diag[1], 0.0, 0.0, 0.0, I_diag[2])
        r_com = wp.vec3(0.5, 0.2, -0.3)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        body = builder.add_link(
            xform=identity_xform,
            mass=m,
            inertia=I_body,
            com=r_com,
        )
        # ``parent_xform.rotation`` is left at identity to avoid a known
        # MuJoCo-bridge convention bug for free joints with a rotated
        # parent frame: https://github.com/newton-physics/newton/issues/2704.
        j_root = builder.add_joint_free(
            parent=-1,
            child=body,
            parent_xform=identity_xform,
            child_xform=child_xform_offset,
        )
        builder.add_articulation([j_root], label="floating_body")

        model = builder.finalize(device=self.device)
        state = model.state()
        inverse_dynamics = _InverseDynamicsArrays(model)

        # Body world rotation R_b = R_c^T = R(z, -2 * half_angle).
        body_z_angle = -2.0 * half_angle
        cos_b, sin_b = float(np.cos(body_z_angle)), float(np.sin(body_z_angle))
        R_b = np.array(
            [
                [cos_b, -sin_b, 0.0],
                [sin_b, cos_b, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        I_world = R_b @ np.diag(I_diag) @ R_b.T

        v_com_cases = [
            (0.0, 0.0, 0.0),
            (0.1, -0.2, 0.05),
        ]
        omega_cases = [
            (0.3, -0.1, 0.2),
            (0.3, -0.1, 0.2),
        ]
        num_cases = len(v_com_cases)

        for i in range(num_cases):
            v_com_values = v_com_cases[i]
            omega_values = omega_cases[i]
            with self.subTest(v_com=v_com_values, omega=omega_values):
                joint_q = state.joint_q.numpy()
                joint_q[:] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # joint_q = identity
                state.joint_q.assign(joint_q)
                joint_qd = state.joint_qd.numpy()
                joint_qd[:] = (*v_com_values, *omega_values)
                state.joint_qd.assign(joint_qd)
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                _eval_inverse_dynamics_passive(
                    model,
                    state,
                    _PassiveOutput.CORIOLIS_FORCE,
                    inverse_dynamics,
                )
                # Newton stores the standard +C(q, q_dot)*q_dot directly.
                measured_linear = inverse_dynamics.coriolis_force.numpy()[0:3]
                measured_angular = inverse_dynamics.coriolis_force.numpy()[3:6]

                omega = np.asarray(omega_values, dtype=np.float64)
                expected_linear = np.zeros(3)
                expected_angular = np.cross(omega, I_world @ omega)

                np.testing.assert_allclose(measured_linear, expected_linear, atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(measured_angular, expected_angular, atol=1e-5, rtol=1e-5)

    def test_coriolis_fixed_root_chain_with_com_offsets(self):
        """Verify Coriolis comp force for a fixed-root revolute chain with non-zero link CoMs.

        Builds a 2-link revolute chain anchored to world (joint axes y, z
        for cross-coupling) with non-zero CoM offsets and anisotropic
        inertias on both links. Under zero gravity and zero applied force
        the manipulator equation reduces to ``M(q) * joint_qdd = -C(q, qd) * qd``,
        so the simulator's ``M * joint_qdd`` after one step must equal the
        negation of Newton's ``coriolis_force``.

        For non-free joints, the joint torque is the scalar projection
        ``S^T * f``, which is reference-point invariant -- so unlike free
        joints, no convention boundary needs crossing and CoM offsets do
        not contaminate the Coriolis output. This locks that property in
        as a regression next to the free-joint case in
        :meth:`test_coriolis_floating_root_with_com_offset`.
        """
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        pos_x = wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity())
        neg_x = wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity())
        y_axis = wp.vec3(0.0, 1.0, 0.0)
        z_axis = wp.vec3(0.0, 0.0, 1.0)

        # Anisotropic inertia and non-zero CoM offsets on both links to
        # exercise the body-CoM term in the spatial-inertia bias.
        I_link = wp.mat33(2.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0)
        com_link = wp.vec3(0.1, 0.2, -0.15)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        b1 = builder.add_link(xform=identity_xform, mass=1.0, inertia=I_link, com=com_link)
        j1 = builder.add_joint_revolute(
            parent=-1,
            child=b1,
            axis=y_axis,
            parent_xform=identity_xform,
            child_xform=neg_x,
        )
        b2 = builder.add_link(xform=identity_xform, mass=1.0, inertia=I_link, com=com_link)
        j2 = builder.add_joint_revolute(
            parent=b1,
            child=b2,
            axis=z_axis,
            parent_xform=pos_x,
            child_xform=neg_x,
        )
        builder.add_articulation([j1, j2], label="fixed_root_chain")

        model = builder.finalize(device=self.device)
        state = model.state()
        state_next = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        inverse_dynamics = _InverseDynamicsArrays(model)
        solver = newton.solvers.SolverMuJoCo(model)
        dt = 1e-4
        zero_bias = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=self.device)

        # Configurations chosen to make C(q, qd) * qd non-trivial: away
        # from singular alignments and with both qd_i non-zero so the
        # cross-coupling between joint 1 and joint 2 is exercised.
        q_cases = [
            (0.0, 0.0),
            (0.3, np.pi / 2.0),
            (-np.pi / 4.0, np.pi / 6.0),
        ]
        qd_cases = [
            (1.5, 1.5),
            (0.7, -1.2),
            (1.0, 0.5),
        ]
        num_cases = len(q_cases)

        for i in range(num_cases):
            q_values = q_cases[i]
            qd_values = qd_cases[i]
            with self.subTest(joint_q=q_values, joint_qd=qd_values):
                joint_q = state.joint_q.numpy()
                joint_q[:] = q_values
                state.joint_q.assign(joint_q)
                joint_qd = state.joint_qd.numpy()
                joint_qd[:] = qd_values
                state.joint_qd.assign(joint_qd)
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                _eval_inverse_dynamics_passive(
                    model,
                    state,
                    _PassiveOutput.MASS_MATRIX | _PassiveOutput.CORIOLIS_FORCE,
                    inverse_dynamics,
                )
                coriolis_comp = inverse_dynamics.coriolis_force.numpy().copy()

                # Step with zero applied force and zero gravity:
                # M * joint_qdd = -C(q, qd) * qd = -coriolis_force.
                solver.step(state, state_next, control, contacts, dt)
                joint_qdd_observed = (state_next.joint_qd.numpy() - np.asarray(joint_qd[:], dtype=np.float64)) / dt

                joint_qdd_arr = wp.array(joint_qdd_observed.astype(np.float32), dtype=wp.float32, device=self.device)
                inverse_dynamics.coriolis_force.assign(zero_bias)
                inverse_dynamics.gravity_force.assign(zero_bias)
                _eval_inverse_dynamics_force(
                    model,
                    state,
                    inverse_dynamics,
                    joint_qdd_arr,
                )
                M_joint_qdd = inverse_dynamics.joint_f.numpy()

                np.testing.assert_allclose(-coriolis_comp, M_joint_qdd, atol=2e-3, rtol=2e-3)


class TestMassMatrix(TestInverseDynamicsBase):
    """Mass-matrix tests for the two-link pendulum harness."""

    def test_mass_matrix_matches_eval_mass_matrix(self):
        """eval_inverse_dynamics_passive(_PassiveOutput.MASS_MATRIX) must match newton.eval_mass_matrix element-wise."""
        builder = self._build_two_link_articulation(
            gravity=wp.vec3(0.0, -9.81, 0.0),
            floating_base=False,
            joint_type="revolute",
            joint_axis=wp.vec3(0.0, 0.0, 1.0),
            link_coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
            link_masses=[1.0, 2.0],
            joint_frames=[
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            ],
            link_inertias=[self.I_UNIT, self.I_UNIT],
        )
        model = builder.finalize(device=self.device)
        state = model.state()
        joint_q = state.joint_q.numpy()
        joint_q[0] = 0.3
        state.joint_q.assign(joint_q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        H_reference = newton.eval_mass_matrix(model, state).numpy()

        inverse_dynamics = _InverseDynamicsArrays(model)
        _eval_inverse_dynamics_passive(model, state, _PassiveOutput.MASS_MATRIX, inverse_dynamics)

        np.testing.assert_allclose(inverse_dynamics.mass_matrix.numpy(), H_reference, rtol=1e-6, atol=1e-6)

    def test_mass_matrix_planar_double_pendulum_matches_analytical(self):
        """M(q) for a planar double pendulum matches the closed-form expression for zero and non-zero CoM.

        Same articulation as
        ``TestCoriolisCompForce.test_coriolis_double_pendulum_matches_analytical``:
        fixed-base, two revolute joints both about world +Y, joint-to-joint
        link length 1, 25 kg per link, identity body-frame inertia. For two
        revolute joints sharing the same axis the mass matrix is:

            M_11 = m1*l1c^2 + m2*(L1^2 + l2c^2 + 2*L1*l2c*cos(q2)) + Iyy_1 + Iyy_2
            M_22 = m2*l2c^2 + Iyy_2
            M_12 = m2*(l2c^2 + L1*l2c*cos(q2)) + Iyy_2

        where ``l_ic`` is the joint-axis-to-link-CoM distance. With body-frame
        ``com = (com_x, 0, 0)`` on each link, the CoM sits at distance
        ``0.5 + com_x`` from the joint axis along the link, so the parallel-
        axis CoM-offset contribution to M scales with ``com_x``. The test
        sweeps ``com_x`` in {0.0, 0.1} (zero-CoM baseline + non-zero-CoM
        regression) and ``q2`` in {0, pi/2, pi} (full range of cos(q2));
        ``M`` is q1-independent (rotational symmetry about world +Y).
        """
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        pos_half = wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity())
        neg_half = wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity())
        y_axis = wp.vec3(0.0, 1.0, 0.0)

        m = 25.0
        Iyy = 1.0  # I_UNIT yy entry
        L1 = 1.0  # joint-to-joint distance

        com_x_cases = [0.0, 0.1]
        q2_cases = [0.0, np.pi / 2.0, np.pi]

        for com_x in com_x_cases:
            with self.subTest(com_x=com_x):
                link_com = wp.vec3(com_x, 0.0, 0.0)
                l1c = 0.5 + com_x
                l2c = 0.5 + com_x

                builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0), up_axis=newton.Axis.Z)
                b1 = builder.add_link(
                    xform=identity_xform,
                    mass=m,
                    inertia=self.I_UNIT,
                    com=link_com,
                )
                j1 = builder.add_joint_revolute(
                    parent=-1,
                    child=b1,
                    axis=y_axis,
                    parent_xform=identity_xform,
                    child_xform=neg_half,
                )
                b2 = builder.add_link(
                    xform=identity_xform,
                    mass=m,
                    inertia=self.I_UNIT,
                    com=link_com,
                )
                j2 = builder.add_joint_revolute(
                    parent=b1,
                    child=b2,
                    axis=y_axis,
                    parent_xform=pos_half,
                    child_xform=neg_half,
                )
                builder.add_articulation([j1, j2], label="double_pendulum")

                model = builder.finalize(device=self.device)
                inverse_dynamics = _InverseDynamicsArrays(model)

                for q2 in q2_cases:
                    with self.subTest(com_x=com_x, q2=q2):
                        state = model.state()
                        joint_q = state.joint_q.numpy()
                        joint_q[:] = (0.7, q2)  # q1 arbitrary; M is q1-independent
                        state.joint_q.assign(joint_q)
                        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                        _eval_inverse_dynamics_passive(model, state, _PassiveOutput.MASS_MATRIX, inverse_dynamics)
                        M = inverse_dynamics.mass_matrix.numpy()[0, :2, :2]

                        cos_q2 = np.cos(q2)
                        M_11 = m * l1c**2 + m * (L1**2 + l2c**2 + 2.0 * L1 * l2c * cos_q2) + 2.0 * Iyy
                        M_22 = m * l2c**2 + Iyy
                        M_12 = m * (l2c**2 + L1 * l2c * cos_q2) + Iyy
                        expected = np.array([[M_11, M_12], [M_12, M_22]])

                        np.testing.assert_allclose(M, expected, atol=1e-3, rtol=1e-5)


class TestManipulatorEquation(TestInverseDynamicsBase):
    """Manipulator-equation tests covering combined inverse-dynamics outputs."""

    def test_eval_inverse_dynamics_passive_finite_on_distance_joint(self):
        """Finite-only guard for the DISTANCE joint type.

        DISTANCE is supported by the inverse-dynamics pipeline (treated as
        FREE), so its outputs are defined. Correctness is not asserted here, but
        a NaN regression on the outputs would go uncaught otherwise.

        CABLE is intentionally excluded: it is not implemented by
        ``jcalc_motion`` / ``jcalc_motion_subspace`` and is not reconstructed by
        ``eval_fk``, so the pipeline reads unreconstructed state for it and the
        outputs are undefined (observed as intermittently non-finite). Asserting
        finiteness there would test undefined behavior rather than a contract.

        Multi-angular-DOF D6 joints are fully supported and are covered by
        analytical assertions in :meth:`test_inverse_dynamics_force_baseline`
        and related tests.
        """
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

        b = newton.ModelBuilder()
        link = b.add_link(
            xform=identity_xform,
            mass=1.0,
            inertia=self.I_UNIT,
            com=wp.vec3(0.0, 0.0, 0.0),
        )
        j = b.add_joint_distance(
            parent=-1,
            child=link,
            parent_xform=identity_xform,
            child_xform=identity_xform,
        )
        b.add_articulation([j])
        model = b.finalize(device=self.device)

        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        # Non-zero velocities so Coriolis paths are exercised.
        if model.joint_dof_count > 0:
            joint_qd = np.ones(model.joint_dof_count, dtype=np.float32) * 0.5
            state.joint_qd.assign(joint_qd)

        inverse_dynamics = _InverseDynamicsArrays(model)
        _eval_inverse_dynamics_passive(
            model,
            state,
            _PassiveOutput.ALL,
            inverse_dynamics,
        )

        self.assertTrue(np.all(np.isfinite(inverse_dynamics.mass_matrix.numpy())))
        self.assertTrue(np.all(np.isfinite(inverse_dynamics.gravity_force.numpy())))
        self.assertTrue(np.all(np.isfinite(inverse_dynamics.coriolis_force.numpy())))

    def test_inverse_dynamics_rejects_cable_joint(self):
        """Both inverse-dynamics entrypoints reject CABLE joints eagerly."""
        identity_xform = wp.transform_identity()
        builder = newton.ModelBuilder()
        link = builder.add_link(
            xform=identity_xform,
            mass=1.0,
            inertia=self.I_UNIT,
            com=wp.vec3(0.0),
        )
        joint = builder.add_joint_cable(
            parent=-1,
            child=link,
            parent_xform=identity_xform,
            child_xform=identity_xform,
        )
        builder.add_articulation([joint])
        model = builder.finalize(device=self.device)
        state = model.state()
        arrays = _InverseDynamicsArrays(model)

        with self.assertRaisesRegex(ValueError, "JointType.CABLE"):
            newton.eval_inverse_dynamics_passive(model, state, mass_matrix=arrays.mass_matrix)

        with self.assertRaisesRegex(ValueError, "JointType.CABLE"):
            newton.eval_inverse_dynamics_force(
                model,
                state,
                mass_matrix=arrays.mass_matrix,
                joint_qdd=wp.zeros_like(state.joint_qd),
                coriolis_force=arrays.coriolis_force,
                gravity_force=arrays.gravity_force,
                joint_f=arrays.joint_f,
            )

    def test_eval_inverse_dynamics_force_hand_crafted_inputs(self):
        """White-box test of ``eval_inverse_dynamics_force`` with hand-chosen inputs.

        Builds a two-articulation model where the articulations have *different*
        DOF counts (1 and 2), then passes synthetic H, joint_qdd, and bias arrays
        whose values are computed independently in NumPy. Comparing the output
        ``tau`` against the NumPy result catches DOF-slicing bugs
        (wrong ``dof_start``, wrong ``art_idx`` into H, off-by-one in
        ``dof_count``) that self-consistent round-trip tests cannot detect.
        """
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        y_axis = wp.vec3(0.0, 1.0, 0.0)

        builder = newton.ModelBuilder()
        # Articulation 0: single revolute → 1 DOF.
        b0 = builder.add_link(xform=identity_xform, mass=1.0, inertia=self.I_UNIT, com=wp.vec3(0.0, 0.0, 0.0))
        j0 = builder.add_joint_revolute(
            parent=-1, child=b0, axis=y_axis, parent_xform=identity_xform, child_xform=identity_xform
        )
        builder.add_articulation([j0])

        # Articulation 1: two revolutes in a chain → 2 DOFs.
        b1 = builder.add_link(xform=identity_xform, mass=1.0, inertia=self.I_UNIT, com=wp.vec3(0.0, 0.0, 0.0))
        j1 = builder.add_joint_revolute(
            parent=-1, child=b1, axis=y_axis, parent_xform=identity_xform, child_xform=identity_xform
        )
        b2 = builder.add_link(xform=identity_xform, mass=1.0, inertia=self.I_UNIT, com=wp.vec3(0.0, 0.0, 0.0))
        j2 = builder.add_joint_revolute(
            parent=b1, child=b2, axis=y_axis, parent_xform=identity_xform, child_xform=identity_xform
        )
        builder.add_articulation([j1, j2])

        model = builder.finalize(device=self.device)
        self.assertEqual(model.articulation_count, 2)
        self.assertEqual(model.joint_dof_count, 3)  # art0: 1, art1: 2
        self.assertEqual(model.max_dofs_per_articulation, 2)

        # Hand-crafted H (2, 2, 2): art0 uses only the [0,0] entry; art1 uses
        # all four.  Values are chosen so any cross-articulation or index mix-up
        # produces a visibly wrong result.
        H_np = np.array(
            [
                [
                    [7.0, 0.0],  # art0 — only [0,0] active; off-diagonal is padding
                    [0.0, 0.0],
                ],
                [
                    [3.0, 1.0],  # art1 — full 2X2 block
                    [0.5, 4.0],
                ],
            ],
            dtype=np.float32,
        )

        joint_qdd_np = np.array([2.0, 5.0, -1.0], dtype=np.float32)  # [art0_dof0, art1_dof0, art1_dof1]
        coriolis_np = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        gravity_np = np.array([1.0, -0.5, 0.25], dtype=np.float32)

        # Reference computed entirely in NumPy, independent of Newton's pipeline.
        # art0 (1 DOF):  tau[0] = H[0,0,0]*joint_qdd[0] + coriolis[0] + gravity[0]
        # art1 (2 DOFs): tau[1:3] = H[1,:2,:2] @ joint_qdd[1:3] + coriolis[1:3] + gravity[1:3]
        tau_expected = np.empty(3, dtype=np.float32)
        tau_expected[0:1] = H_np[0, :1, :1] @ joint_qdd_np[0:1] + coriolis_np[0:1] + gravity_np[0:1]
        tau_expected[1:3] = H_np[1, :2, :2] @ joint_qdd_np[1:3] + coriolis_np[1:3] + gravity_np[1:3]

        H = wp.array(H_np, dtype=wp.float32, device=self.device)
        joint_qdd = wp.array(joint_qdd_np, dtype=wp.float32, device=self.device)
        coriolis = wp.array(coriolis_np, dtype=wp.float32, device=self.device)
        gravity = wp.array(gravity_np, dtype=wp.float32, device=self.device)
        inverse_dynamics = _InverseDynamicsArrays(model)
        inverse_dynamics.mass_matrix.assign(H)
        inverse_dynamics.coriolis_force.assign(coriolis)
        inverse_dynamics.gravity_force.assign(gravity)

        _eval_inverse_dynamics_force(model, model.state(), inverse_dynamics, joint_qdd)

        np.testing.assert_allclose(inverse_dynamics.joint_f.numpy(), tau_expected, atol=1e-6)

    def test_combined_passive_outputs_match_individual_calls(self):
        """A combined request must match three isolated output requests.

        Uses a floating base so the articulation has multi-DOF coupling; a
        fixed root with a single revolute DOF has identically-zero Coriolis
        and would trivially defeat that assertion.

        The cross-check against isolated calls directly catches scratch-buffer
        aliasing: if combining all three outputs causes one pass to corrupt the
        scratch consumed by another, the results will diverge from the
        individual references.
        """
        builder = self._build_two_link_articulation(
            gravity=wp.vec3(0.0, -9.81, 0.0),
            floating_base=True,
            joint_type="revolute",
            joint_axis=wp.vec3(0.0, 0.0, 1.0),
            link_coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
            link_masses=[1.0, 2.0],
            joint_frames=[
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            ],
            link_inertias=[self.I_UNIT, self.I_UNIT],
        )
        model = builder.finalize(device=self.device)
        state = model.state()
        joint_q = state.joint_q.numpy()
        joint_q[0] = 0.3
        joint_q[1] = 0.5
        state.joint_q.assign(joint_q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        joint_qd = state.joint_qd.numpy()
        # Populate linear, angular, and internal DOFs so Coriolis has real
        # coupling: pure linear base motion doesn't couple through C(q, q_dot).
        joint_qd[:] = np.linspace(0.1, 0.7, joint_qd.shape[0])
        state.joint_qd.assign(joint_qd)

        # Reference: three isolated calls, each with fresh internal scratch.
        ref_mm = _InverseDynamicsArrays(model)
        _eval_inverse_dynamics_passive(model, state, _PassiveOutput.MASS_MATRIX, ref_mm)
        ref_gf = _InverseDynamicsArrays(model)
        _eval_inverse_dynamics_passive(model, state, _PassiveOutput.GRAVITY_FORCE, ref_gf)
        ref_cf = _InverseDynamicsArrays(model)
        _eval_inverse_dynamics_passive(model, state, _PassiveOutput.CORIOLIS_FORCE, ref_cf)

        # Combined call under test.
        combined = _InverseDynamicsArrays(model)
        _eval_inverse_dynamics_passive(model, state, _PassiveOutput.ALL, combined)

        np.testing.assert_array_equal(combined.mass_matrix.numpy(), ref_mm.mass_matrix.numpy())
        np.testing.assert_array_equal(combined.gravity_force.numpy(), ref_gf.gravity_force.numpy())
        np.testing.assert_array_equal(combined.coriolis_force.numpy(), ref_cf.coriolis_force.numpy())

    def _test_inverse_dynamics_force(self, non_zero_gravity: bool, non_zero_initial_dof_velocities: bool):
        """Manipulator-equation test parameterized on whether the bias terms are exercised.

        With gravity and ``joint_qd`` both zero, ``tau = M(q)*joint_qdd``. Setting
        ``non_zero_gravity`` switches on a non-zero ``g(q)`` term;
        ``non_zero_initial_dof_velocities`` switches on a non-zero
        ``C(q, q_dot)*q_dot`` term. In every case
        :func:`newton.eval_inverse_dynamics_force` must produce the joint force
        that drives the system to the prescribed ``joint_qdd`` after one
        small-step simulation, recovered from the velocity change.
        """
        gravity_value = wp.vec3(0.0, 0.0, -10.0 if non_zero_gravity else 0.0)

        # Each articulation is a chain of three uniform-density 4x2x2 boxes;
        # the per-articulation density (and so the link mass and inertia) varies
        # to exercise the per-articulation kernel paths with different M(q).
        # I_link = (1/12) * mass * diag(b^2 + c^2, a^2 + c^2, a^2 + b^2)
        #        = (1/12) * mass * diag(8, 20, 20) for full extents (4, 2, 2).
        def box_inertia(mass: float) -> wp.mat33:
            return wp.mat33(
                mass * 8.0 / 12.0,
                0.0,
                0.0,
                0.0,
                mass * 20.0 / 12.0,
                0.0,
                0.0,
                0.0,
                mass * 20.0 / 12.0,
            )

        # A non-identity joint-frame rotation makes the revolute axis no longer
        # aligned with parent body +Z, so the test exercises non-trivial joint
        # geometry. Quaternion (x, y, z, w) for a 30 deg rotation about world +Y
        # is (0, sin(15 deg), 0, cos(15 deg)).
        joint_quat = wp.quat(0.0, float(np.sin(np.pi / 12.0)), 0.0, float(np.cos(np.pi / 12.0)))
        # Root-joint parent_xform with a non-identity translation so the root
        # joint frame is offset from world for every articulation -- exercising
        # the FK path on fixed roots and the world-frame body placement on
        # floating roots. ``parent_xform`` rotation and ``child_xform`` are
        # left at identity on the root joint to avoid a known MuJoCo-bridge
        # convention bug for free joints with a rotated parent frame:
        # https://github.com/newton-physics/newton/issues/2704.
        root_parent_xform = wp.transform(wp.vec3(0.7, -0.4, 0.3), wp.quat_identity())
        # Non-zero CoM offset applied to every link (root and inboard) so the
        # test exercises off-axis mass distribution everywhere -- including
        # the floating-root case where the public free-joint v_com convention
        # interacts non-trivially with the inverse-dynamics RNEA.
        link_com = wp.vec3(0.5, 0.2, -0.3)
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        pos_two = wp.transform(wp.vec3(2.0, 0.0, 0.0), joint_quat)
        neg_two = wp.transform(wp.vec3(-2.0, 0.0, 0.0), joint_quat)
        z_axis = wp.vec3(0.0, 0.0, 1.0)
        x_axis = wp.vec3(1.0, 0.0, 0.0)

        def build_articulation(mass: float, inertia: wp.mat33, root_joint_type: str) -> newton.ModelBuilder:
            b = newton.ModelBuilder(gravity=gravity_value, up_axis=newton.Axis.Z)
            link0 = b.add_link(
                xform=identity_xform,
                mass=mass,
                inertia=inertia,
                com=link_com,
            )
            if root_joint_type == "free":
                j0 = b.add_joint_free(
                    parent=-1,
                    child=link0,
                    parent_xform=root_parent_xform,
                    child_xform=identity_xform,
                )
            elif root_joint_type == "ball":
                j0 = b.add_joint_ball(
                    parent=-1,
                    child=link0,
                    parent_xform=root_parent_xform,
                    child_xform=identity_xform,
                )
            elif root_joint_type == "d6_revolute":
                # D6 with a single angular Z axis -> 1-DOF revolute.
                j0 = b.add_joint_d6(
                    parent=-1,
                    child=link0,
                    parent_xform=root_parent_xform,
                    child_xform=identity_xform,
                    linear_axes=[],
                    angular_axes=[
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z),
                    ],
                )
            elif root_joint_type == "d6_2lin":
                # D6 with 2 linear axes (X, Y) -> planar translation root.
                j0 = b.add_joint_d6(
                    parent=-1,
                    child=link0,
                    parent_xform=root_parent_xform,
                    child_xform=identity_xform,
                    linear_axes=[
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X),
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y),
                    ],
                    angular_axes=[],
                )
            elif root_joint_type == "d6_1lin_1ang":
                # D6 with 1 linear axis (X) + 1 angular axis (Z).
                j0 = b.add_joint_d6(
                    parent=-1,
                    child=link0,
                    parent_xform=root_parent_xform,
                    child_xform=identity_xform,
                    linear_axes=[
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X),
                    ],
                    angular_axes=[
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z),
                    ],
                )
            elif root_joint_type == "d6_2ang":
                # D6 with 2 angular axes (X, Z) -> exercises transform_2d_rotational_axes.
                j0 = b.add_joint_d6(
                    parent=-1,
                    child=link0,
                    parent_xform=root_parent_xform,
                    child_xform=identity_xform,
                    linear_axes=[],
                    angular_axes=[
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X),
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z),
                    ],
                )
            elif root_joint_type == "d6_ball":
                # D6 with 3 angular axes (X, Y, Z) -> ball-equivalent, exercises
                # transform_3d_rotational_axes.
                j0 = b.add_joint_d6(
                    parent=-1,
                    child=link0,
                    parent_xform=root_parent_xform,
                    child_xform=identity_xform,
                    linear_axes=[],
                    angular_axes=[
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X),
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y),
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z),
                    ],
                )
            elif root_joint_type == "fixed":
                j0 = b.add_joint_fixed(
                    parent=-1,
                    child=link0,
                    parent_xform=root_parent_xform,
                    child_xform=identity_xform,
                )
            else:
                raise ValueError(f"Unknown root_joint_type: {root_joint_type!r}")
            link1 = b.add_link(
                xform=identity_xform,
                mass=mass,
                inertia=inertia,
                com=link_com,
            )
            j1 = b.add_joint_revolute(
                parent=link0,
                child=link1,
                axis=z_axis,
                parent_xform=pos_two,
                child_xform=neg_two,
                target_ke=0.0,
                target_kd=0.0,
                friction=0.0,
            )
            link2 = b.add_link(
                xform=identity_xform,
                mass=mass,
                inertia=inertia,
                com=link_com,
            )
            j2 = b.add_joint_prismatic(
                parent=link1,
                child=link2,
                axis=x_axis,
                parent_xform=pos_two,
                child_xform=neg_two,
                target_ke=0.0,
                target_kd=0.0,
                friction=0.0,
            )
            b.add_articulation([j0, j1, j2], label="three_link_chain")
            return b

        num_worlds = 2
        num_arts_per_world = 8
        num_arts = num_worlds * num_arts_per_world

        # Per-articulation root joint type:
        #   ``"free"``: 6 qd / 7 q -- 3 linear + 3 angular for a free body.
        #   ``"ball"``: 3 qd / 4 q -- orientation only (quaternion).
        #   ``"d6_revolute"``: 1 qd / 1 q -- D6 with a single angular Z axis.
        #   ``"d6_2lin"``: 2 qd / 2 q -- D6 with 2 linear axes (X, Y).
        #   ``"d6_1lin_1ang"``: 2 qd / 2 q -- D6 with 1 linear axis (X) + 1
        #       angular axis (Z). Linear DOF precedes angular in the D6 layout.
        #   ``"d6_2ang"``: 2 qd / 2 q -- D6 with 2 angular axes (X, Z).
        #       Exercises the transform_2d_rotational_axes velocity-subspace path.
        #   ``"d6_ball"``: 3 qd / 3 q -- D6 with 3 angular axes (X, Y, Z).
        #       Ball-joint equivalent; exercises transform_3d_rotational_axes.
        #   ``"fixed"``: 0 qd / 0 q.
        # Every world uses the same pattern, exercising all eight root joint
        # types per world.
        root_joint_types_per_world = [
            "fixed",
            "free",
            "ball",
            "d6_revolute",
            "d6_2lin",
            "d6_1lin_1ang",
            "d6_2ang",
            "d6_ball",
        ]
        assert len(root_joint_types_per_world) == num_arts_per_world
        root_joint_types = root_joint_types_per_world * num_worlds
        # Per-type root qd-DOF counts (used to size the expected_dofs check).
        root_qd_len = {
            "fixed": 0,
            "ball": 3,
            "d6_revolute": 1,
            "d6_2lin": 2,
            "d6_1lin_1ang": 2,
            "d6_2ang": 2,
            "d6_ball": 3,
            "free": 6,
        }

        # Per-articulation link mass. The first value (16) corresponds to a
        # density-1 4x2x2 box (mass = density * volume); the others are arbitrary
        # multiples and submultiples to vary M(q) across articulations. Inertia
        # tracks mass for the same shape.
        per_articulation_masses = [
            16.0,
            32.0,
            8.0,
            24.0,
            18.0,
            12.0,
            30.0,
            6.0,  # world 0
            20.0,
            28.0,
            14.0,
            22.0,
            10.0,
            26.0,
            34.0,
            4.0,  # world 1
        ]
        assert len(per_articulation_masses) == num_arts

        builder = newton.ModelBuilder(gravity=gravity_value, up_axis=newton.Axis.Z)
        art_idx = 0
        for _ in range(num_worlds):
            world_builder = newton.ModelBuilder(gravity=gravity_value, up_axis=newton.Axis.Z)
            for _ in range(num_arts_per_world):
                m = per_articulation_masses[art_idx]
                world_builder.add_builder(build_articulation(m, box_inertia(m), root_joint_types[art_idx]))
                art_idx += 1
            builder.add_world(world_builder)

        model = builder.finalize(device=self.device)
        state = model.state()
        state_next = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        inverse_dynamics = _InverseDynamicsArrays(model)
        solvers = {
            "featherstone": newton.solvers.SolverFeatherstone(model),
            "mujoco": newton.solvers.SolverMuJoCo(model),
        }
        dt = 1e-4

        # Each articulation contributes 2 internal DOFs (one revolute j1, one
        # prismatic j2) plus the root joint's qd-DOFs (free=6, ball=3, fixed=0).
        expected_dofs = sum(2 + root_qd_len[t] for t in root_joint_types)
        self.assertEqual(model.body_count, 3 * num_arts)
        self.assertEqual(model.joint_dof_count, expected_dofs)

        initial_joint_positions = [
            (0.0, 0.0),
            (0.3, 0.5),
            (np.pi / 4.0, -np.pi / 3.0),
            (np.pi / 2.0, np.pi / 2.0),
        ]
        # Internal-DOF velocities. Non-zero values let the C(q, q_dot)*q_dot term
        # of the manipulator equation contribute.
        internal_speed = (0.5, -0.3) if non_zero_initial_dof_velocities else (0.0, 0.0)
        initial_joint_speeds = [internal_speed] * 4
        # Per-test-case internal-DOF accelerations. Magnitudes and signs vary so
        # the assertion exercises both small and large q_ddot, including negatives.
        initial_joint_accelerations = [
            (0.02, 0.04),
            (0.5, -0.3),
            (1.0, 1.0),
            (-0.7, 0.2),
        ]

        # Per-type root state used for every test case. The base sits at the
        # joint's parent_xform pose with identity orientation. Root velocity
        # is zero unless ``non_zero_initial_dof_velocities`` flips on the
        # angular DOFs (linear stays zero -- non-zero ``omega x v_com``
        # cross-coupling between root linear and angular velocity surfaces a
        # separate Newton-vs-MuJoCo angular-velocity-frame question on the
        # simulator side, independent of the inverse-dynamics convention this
        # test exercises). Root accelerations are arbitrary non-zero values so
        # the floating root exercises non-trivial M(q)*joint_qdd rows.
        root_omega = (0.3, -0.1, 0.2) if non_zero_initial_dof_velocities else (0.0, 0.0, 0.0)
        root_alpha = (0.2, -0.25, 0.3)
        root_lin_dot = (0.05, -0.1, 0.15)
        # ``d6_revolute`` exercises a single Z-axis angular DOF; its qd uses
        # the z-component of ``root_omega`` and qdd the z-component of
        # ``root_alpha`` so it tracks the same active/zero pattern as the
        # multi-DOF root types. ``d6_2lin`` exercises two linear DOFs (X, Y);
        # linear qd stays zero (matching the rule used for ``free``) and
        # linear qdd takes the X / Y components of ``root_lin_dot``.
        # ``d6_1lin_1ang`` combines a linear X DOF (zero qd, X component of
        # ``root_lin_dot`` for qdd) with an angular Z DOF (z-component of
        # ``root_omega`` / ``root_alpha``); D6 lays linear DOFs out before
        # angular ones. ``d6_2ang`` uses X and Z angular axes (x- and
        # z-components of ``root_omega`` / ``root_alpha``). ``d6_ball`` uses
        # all three angular axes mapped directly from ``root_omega`` / ``root_alpha``.
        root_q_per_type = {
            "fixed": (),
            "ball": (0.0, 0.0, 0.0, 1.0),
            "d6_revolute": (0.0,),
            "d6_2lin": (0.0, 0.0),
            "d6_1lin_1ang": (0.0, 0.0),
            "d6_2ang": (0.0, 0.0),
            "d6_ball": (0.0, 0.0, 0.0),
            "free": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        }
        root_qd_per_type = {
            "fixed": (),
            "ball": root_omega,
            "d6_revolute": (root_omega[2],),
            "d6_2lin": (0.0, 0.0),
            "d6_1lin_1ang": (0.0, root_omega[2]),
            "d6_2ang": (root_omega[0], root_omega[2]),
            "d6_ball": root_omega,
            "free": (0.0, 0.0, 0.0, *root_omega),
        }
        root_qdd_per_type = {
            "fixed": (),
            "ball": root_alpha,
            "d6_revolute": (root_alpha[2],),
            "d6_2lin": (root_lin_dot[0], root_lin_dot[1]),
            "d6_1lin_1ang": (root_lin_dot[0], root_alpha[2]),
            "d6_2ang": (root_alpha[0], root_alpha[2]),
            "d6_ball": root_alpha,
            "free": (*root_lin_dot, *root_alpha),
        }

        # Pick the smallest eval_type that covers the active bias terms:
        # MASS_MATRIX is always required for M*joint_qdd; add GRAVITY_FORCE
        # only when gravity is non-zero, CORIOLIS_FORCE only when
        # joint_qd is non-zero. When both are non-zero this collapses to ALL.
        eval_type = _PassiveOutput.MASS_MATRIX
        if non_zero_gravity:
            eval_type |= _PassiveOutput.GRAVITY_FORCE
        if non_zero_initial_dof_velocities:
            eval_type |= _PassiveOutput.CORIOLIS_FORCE

        for joint_q_values, joint_qd_values, joint_qdd_values in zip(
            initial_joint_positions, initial_joint_speeds, initial_joint_accelerations, strict=True
        ):
            with self.subTest(joint_q=joint_q_values, joint_qd=joint_qd_values, joint_qdd=joint_qdd_values):
                q_pieces: list[float] = []
                qd_pieces: list[float] = []
                qdd_pieces: list[float] = []
                for root_type in root_joint_types:
                    q_pieces.extend(root_q_per_type[root_type])
                    qd_pieces.extend(root_qd_per_type[root_type])
                    qdd_pieces.extend(root_qdd_per_type[root_type])
                    q_pieces.extend(joint_q_values)
                    qd_pieces.extend(joint_qd_values)
                    qdd_pieces.extend(joint_qdd_values)
                joint_q_full = np.asarray(q_pieces, dtype=np.float32)
                joint_qd_full = np.asarray(qd_pieces, dtype=np.float32)
                joint_qdd_target = np.asarray(qdd_pieces, dtype=np.float32)

                joint_q = state.joint_q.numpy()
                joint_q[:] = joint_q_full
                state.joint_q.assign(joint_q)
                joint_qd = state.joint_qd.numpy()
                joint_qd[:] = joint_qd_full
                state.joint_qd.assign(joint_qd)
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                # Manipulator equation: tau = M(q)*joint_qdd + C(q,qdot)*qdot + g(q).
                _eval_inverse_dynamics_passive(model, state, eval_type, inverse_dynamics)
                joint_qdd = wp.array(joint_qdd_target, dtype=wp.float32, device=self.device)
                _eval_inverse_dynamics_force(
                    model,
                    state,
                    inverse_dynamics,
                    joint_qdd,
                )

                control.joint_f.assign(inverse_dynamics.joint_f)

                # The same tau must round-trip through both solvers. Each steps
                # from the same input ``state`` (neither mutates it) into
                # ``state_next``.
                for solver_name, solver in solvers.items():
                    with self.subTest(solver=solver_name):
                        solver.step(state, state_next, control, contacts, dt)

                        # Recover joint_qdd from the velocity change.
                        joint_qdd_observed = (state_next.joint_qd.numpy() - joint_qd_full) / dt
                        np.testing.assert_allclose(joint_qdd_observed, joint_qdd_target, atol=1e-3, rtol=1e-3)

    def test_inverse_dynamics_force_baseline(self):
        """Manipulator equation with zero gravity and zero initial DOF velocities."""
        self._test_inverse_dynamics_force(non_zero_gravity=False, non_zero_initial_dof_velocities=False)

    def test_inverse_dynamics_force_with_gravity(self):
        """Manipulator equation with non-zero gravity (g(q) is non-trivial)."""
        self._test_inverse_dynamics_force(non_zero_gravity=True, non_zero_initial_dof_velocities=False)

    def test_inverse_dynamics_force_with_velocity(self):
        """Manipulator equation with non-zero initial DOF velocities (C(q, q_dot)*q_dot is non-trivial)."""
        self._test_inverse_dynamics_force(non_zero_gravity=False, non_zero_initial_dof_velocities=True)

    def test_inverse_dynamics_force_with_gravity_and_velocity(self):
        """Manipulator equation with non-zero gravity and non-zero initial DOF velocities."""
        self._test_inverse_dynamics_force(non_zero_gravity=True, non_zero_initial_dof_velocities=True)

    def test_inverse_dynamics_force_free_root_rotated_parent(self):
        """FREE root with a rotated parent frame and non-zero ``joint_qdd``.

        The mass matrix is expressed in the joint parent frame while the bias
        terms use the world-frame CoM-wrench convention, so ``H @ joint_qdd`` for the
        FREE root must be rotated to world before summing. This case exercises
        that rotation (a rotated parent + non-zero ``joint_qdd`` -- the base
        ``_test_inverse_dynamics_force`` uses an identity root-parent rotation and
        would not catch it). With zero gravity and zero velocity the bias terms
        vanish, so ``tau`` is purely the world-frame ``M(q)*joint_qdd`` wrench, which
        for a free rigid body with COM at the joint origin is
        ``(m * R * a_com_parent, R * I_local * alpha_parent)``.
        """
        m_mass = 2.0
        I_local = np.diag([0.3, 0.5, 0.4]).astype(np.float64)
        # 90 deg about X: parent +Y -> world +Z, parent +Z -> world -Y.
        qx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2.0)
        R = np.array(wp.quat_to_matrix(qx), dtype=np.float64).reshape(3, 3)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        link = builder.add_link(
            mass=m_mass,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(*I_local.flatten().tolist()),
        )
        joint = builder.add_joint_free(
            parent=-1,
            child=link,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), qx),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([joint])
        q_start = builder.joint_q_start[joint]
        builder.joint_q[q_start : q_start + 7] = list(wp.transform_identity())
        model = builder.finalize(device=self.device)
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        inverse_dynamics = _InverseDynamicsArrays(model)

        # Parent-frame accelerations (COM linear, angular). qd stays zero.
        a_parent = np.array([0.5, -0.3, 0.7], dtype=np.float64)
        alpha_parent = np.array([0.2, 0.4, -0.6], dtype=np.float64)
        joint_qdd_np = np.concatenate([a_parent, alpha_parent]).astype(np.float32)

        _eval_inverse_dynamics_passive(model, state, _PassiveOutput.MASS_MATRIX, inverse_dynamics)
        joint_qdd = wp.array(joint_qdd_np, dtype=wp.float32, device=self.device)
        _eval_inverse_dynamics_force(
            model,
            state,
            inverse_dynamics,
            joint_qdd,
        )

        # World-frame wrench at the COM: force = m*R*a, torque = R*I_local*alpha.
        expected = np.concatenate([m_mass * (R @ a_parent), R @ (I_local @ alpha_parent)]).astype(np.float32)
        np.testing.assert_allclose(inverse_dynamics.joint_f.numpy(), expected, atol=1e-5, rtol=1e-5)

    def test_inverse_dynamics_force_non_root_free_joint_rotated_parent(self):
        """Non-root FREE or DISTANCE joint with a rotated parent frame and non-zero ``joint_qdd``.

        A FIXED root joint makes the second joint non-root. The ``H @ joint_qdd``
        wrench for that joint's six DOFs is in the joint's parent frame and must
        be rotated to world before summing with the world-frame bias terms.

        With zero gravity and zero velocity the bias terms vanish, so ``tau``
        equals the world-frame ``M(q)*joint_qdd`` wrench. For a free rigid body whose
        COM coincides with the joint origin this is
        ``(m * R * a_parent, R * I_local * alpha_parent)`` where ``R`` is the
        rotation from the joint's parent frame to world.

        Tested for both FREE and DISTANCE joint types. The DISTANCE joint uses
        ``max_distance=-1`` (no upper limit) so the distance constraint is inactive.
        """
        m_mass = 2.0
        I_local = wp.mat33(0.3, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.4)
        # 90 deg about X: parent +Y -> world +Z, parent +Z -> world -Y.
        qx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2.0)
        identity = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

        joint_qdd_np = np.array([0.5, -0.3, 0.7, 0.2, 0.4, -0.6], dtype=np.float32)
        a_parent = wp.vec3(*joint_qdd_np[0:3].tolist())
        alpha_parent = wp.vec3(*joint_qdd_np[3:6].tolist())
        f_expected = np.array(wp.quat_rotate(qx, m_mass * a_parent), dtype=np.float32)
        torque_expected = np.array(wp.quat_rotate(qx, I_local * alpha_parent), dtype=np.float32)

        joint_cases = [
            (
                "free",
                lambda b, body1, body2: b.add_joint_free(
                    parent=body1,
                    child=body2,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), qx),
                    child_xform=identity,
                ),
            ),
            (
                "distance",
                lambda b, body1, body2: b.add_joint_distance(
                    parent=body1,
                    child=body2,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), qx),
                    child_xform=identity,
                    max_distance=-1,
                ),
            ),
        ]
        for joint_name, add_joint in joint_cases:
            with self.subTest(joint=joint_name):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
                body1 = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=self.I_UNIT)
                body2 = builder.add_link(mass=m_mass, com=wp.vec3(0.0, 0.0, 0.0), inertia=I_local)
                j_fixed = builder.add_joint_fixed(parent=-1, child=body1, parent_xform=identity, child_xform=identity)
                j_floating = add_joint(builder, body1, body2)
                builder.add_articulation([j_fixed, j_floating])
                q_start = builder.joint_q_start[j_floating]
                builder.joint_q[q_start : q_start + 7] = list(identity)
                model = builder.finalize(device=self.device)
                state = model.state()
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                inverse_dynamics = _InverseDynamicsArrays(model)

                _eval_inverse_dynamics_passive(model, state, _PassiveOutput.MASS_MATRIX, inverse_dynamics)
                joint_qdd = wp.array(joint_qdd_np, dtype=wp.float32, device=self.device)
                _eval_inverse_dynamics_force(
                    model,
                    state,
                    inverse_dynamics,
                    joint_qdd,
                )

                tau = inverse_dynamics.joint_f.numpy()
                np.testing.assert_allclose(tau[0:3], f_expected, atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(tau[3:6], torque_expected, atol=1e-5, rtol=1e-5)

    def test_loop_closing_joint_does_not_contaminate_eval_inverse_dynamics_force(self):
        """A loop-closing joint appended after tree joints must not affect tau.

        Two worlds contain identical fixed-base 2-revolute chains. World 1 also
        has a loop-closing revolute joint (NOT added to the articulation) whose
        single DOF is appended after the tree DOFs. Both
        :func:`newton.eval_inverse_dynamics_passive` and
        :func:`newton.eval_inverse_dynamics_force` must produce identical results
        for both articulations.

        Regression guard for :func:`eval_articulation_inverse_dynamics_force_kernel`
        using ``articulation_end`` (not ``articulation_start[i+1]``) to bound the
        DOF range. With the wrong boundary the kernel includes the loop-closing
        DOF in art 1's ``dof_count`` and reads an off-diagonal entry of the mass
        matrix as a spurious third column, scaled by the large orphan ``joint_qdd``
        value set below.
        """
        gravity_val = wp.vec3(0.0, 0.0, -10.0)
        mass = 2.0
        I = self.I_UNIT
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        pos_one = wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity())
        neg_one = wp.transform(wp.vec3(-1.0, 0.0, 0.0), wp.quat_identity())
        z_axis = wp.vec3(0.0, 0.0, 1.0)

        def build_world(with_loop_closing: bool) -> newton.ModelBuilder:
            b = newton.ModelBuilder(gravity=gravity_val, up_axis=newton.Axis.Z)
            body0 = b.add_link(xform=identity_xform, mass=mass, inertia=I, com=wp.vec3(0.0, 0.0, 0.0))
            body1 = b.add_link(xform=identity_xform, mass=mass, inertia=I, com=wp.vec3(0.0, 0.0, 0.0))
            body2 = b.add_link(xform=identity_xform, mass=mass, inertia=I, com=wp.vec3(0.0, 0.0, 0.0))
            j_fixed = b.add_joint_fixed(parent=-1, child=body0, parent_xform=identity_xform, child_xform=identity_xform)
            j_rev0 = b.add_joint_revolute(
                parent=body0,
                child=body1,
                axis=z_axis,
                parent_xform=pos_one,
                child_xform=neg_one,
                target_ke=0.0,
                target_kd=0.0,
                friction=0.0,
            )
            j_rev1 = b.add_joint_revolute(
                parent=body1,
                child=body2,
                axis=z_axis,
                parent_xform=pos_one,
                child_xform=neg_one,
                target_ke=0.0,
                target_kd=0.0,
                friction=0.0,
            )
            b.add_articulation([j_fixed, j_rev0, j_rev1])
            if with_loop_closing:
                # This joint is NOT passed to add_articulation — it is a
                # loop-closing joint (joint_articulation == -1). Its single DOF
                # must not enter the articulation's DOF range.
                b.add_joint_revolute(
                    parent=body0,
                    child=body2,
                    axis=z_axis,
                    parent_xform=pos_one,
                    child_xform=neg_one,
                    target_ke=0.0,
                    target_kd=0.0,
                    friction=0.0,
                )
            return b

        builder = newton.ModelBuilder(gravity=gravity_val, up_axis=newton.Axis.Z)
        builder.add_world(build_world(with_loop_closing=False))
        builder.add_world(build_world(with_loop_closing=True))
        model = builder.finalize(device=self.device)
        state = model.state()

        # Art0: 2 tree DOFs (slots 0-1); art1: 2 tree DOFs (slots 2-3);
        # loop-closing revolute: 1 DOF (slot 4).
        self.assertEqual(model.joint_dof_count, 5)

        # Identical non-zero angles and velocities for both articulations so
        # all three manipulator-equation terms are non-trivial.
        joint_q = state.joint_q.numpy()
        joint_qd = state.joint_qd.numpy()
        joint_q[0] = np.pi / 4.0  # art0 j_rev0
        joint_q[1] = np.pi / 6.0  # art0 j_rev1
        joint_q[2] = np.pi / 4.0  # art1 j_rev0 (identical)
        joint_q[3] = np.pi / 6.0  # art1 j_rev1 (identical)
        joint_qd[0] = 0.3  # art0 j_rev0 velocity
        joint_qd[1] = -0.2  # art0 j_rev1 velocity
        joint_qd[2] = 0.3  # art1 j_rev0 velocity (identical)
        joint_qd[3] = -0.2  # art1 j_rev1 velocity (identical)
        state.joint_q.assign(joint_q)
        state.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        inverse_dynamics = _InverseDynamicsArrays(model)
        _eval_inverse_dynamics_passive(model, state, _PassiveOutput.ALL, inverse_dynamics)

        H = inverse_dynamics.mass_matrix.numpy()
        g = inverse_dynamics.gravity_force.numpy()
        c = inverse_dynamics.coriolis_force.numpy()

        # Both articulations are physically identical: M(q), g(q), C(q,qd)*qd
        # must be equal.
        np.testing.assert_allclose(H[0], H[1], atol=1e-5, err_msg="mass_matrix differs between articulations")
        np.testing.assert_allclose(g[0:2], g[2:4], atol=1e-5, err_msg="gravity_force differs between articulations")
        np.testing.assert_allclose(c[0:2], c[2:4], atol=1e-5, err_msg="coriolis_force differs between articulations")

        # joint_qdd: same for both tree DOF ranges; large sentinel at the
        # loop-closing slot. With the bug (wrong DOF boundary), the kernel
        # reads H[art1, 0, 2] — which aliases H[art1, 1, 0] (off-diagonal,
        # non-zero for a coupled chain) — multiplied by 99, producing a
        # large detectable contamination in tau[2].
        joint_qdd_np = np.zeros(model.joint_dof_count, dtype=np.float32)
        joint_qdd_np[0] = 0.5  # art0 first DOF acceleration
        joint_qdd_np[1] = 0.3  # art0 second DOF acceleration
        joint_qdd_np[2] = 0.5  # art1 first DOF acceleration (identical)
        joint_qdd_np[3] = 0.3  # art1 second DOF acceleration (identical)
        joint_qdd_np[4] = 99.0  # loop-closing DOF: must NOT enter tau[2:4]
        joint_qdd = wp.array(joint_qdd_np, dtype=wp.float32, device=self.device)
        _eval_inverse_dynamics_force(
            model,
            state,
            inverse_dynamics,
            joint_qdd,
        )

        tau = inverse_dynamics.joint_f.numpy()
        np.testing.assert_allclose(
            tau[0:2],
            tau[2:4],
            atol=1e-5,
            err_msg="tau differs between articulations — loop-closing DOF contaminated result",
        )


class TestInverseDynamicsAPI(TestInverseDynamicsBase):
    """API-surface tests: output dispatch, error paths, and degenerate-model
    edge cases not exercised by the analytical-correctness suites."""

    def test_omitted_mass_matrix_is_not_written(self):
        """Requesting both bias forces leaves the omitted mass matrix untouched.

        A floating-base articulation with non-zero gravity and non-zero
        joint velocities is used so both bias buffers must come back
        non-zero. A sentinel value is stamped into ``mass_matrix`` before
        the call to detect any inadvertent write to it.
        """
        builder = self._build_two_link_articulation(
            gravity=wp.vec3(0.0, -9.81, 0.0),
            floating_base=True,
            joint_type="revolute",
            joint_axis=wp.vec3(0.0, 0.0, 1.0),
            link_coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
            link_masses=[1.0, 2.0],
            joint_frames=[
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            ],
            link_inertias=[self.I_UNIT, self.I_UNIT],
        )
        model = builder.finalize(device=self.device)
        state = model.state()
        joint_q = state.joint_q.numpy()
        joint_q[0] = 0.3
        joint_q[1] = 0.5
        state.joint_q.assign(joint_q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        # Populate every DOF (including the floating-root angular DOFs) so
        # gyroscopic and cross-coupling contributions to C(q, q_dot)*q_dot
        # are non-zero.
        joint_qd = state.joint_qd.numpy()
        joint_qd[:] = np.linspace(0.1, 0.7, joint_qd.shape[0])
        state.joint_qd.assign(joint_qd)

        inverse_dynamics = _InverseDynamicsArrays(model)

        sentinel = np.full(inverse_dynamics.mass_matrix.shape, 7.5, dtype=np.float32)
        inverse_dynamics.mass_matrix.assign(sentinel)

        _eval_inverse_dynamics_passive(
            model,
            state,
            _PassiveOutput.GRAVITY_FORCE | _PassiveOutput.CORIOLIS_FORCE,
            inverse_dynamics,
        )

        np.testing.assert_array_equal(inverse_dynamics.mass_matrix.numpy(), sentinel)
        g = inverse_dynamics.gravity_force.numpy()
        c = inverse_dynamics.coriolis_force.numpy()
        self.assertTrue(np.all(np.isfinite(g)))
        self.assertTrue(np.all(np.isfinite(c)))
        self.assertGreater(float(np.max(np.abs(g))), 1e-6)
        self.assertGreater(float(np.max(np.abs(c))), 1e-6)

    def test_inverse_dynamics_raises_on_shape_mismatch(self):
        """``eval_inverse_dynamics_passive`` and ``eval_inverse_dynamics_force`` raise
        ``ValueError`` when any array's shape disagrees with the model.
        """
        builder = self._build_two_link_articulation(
            gravity=wp.vec3(0.0, 0.0, 0.0),
            floating_base=False,
            joint_type="revolute",
            joint_axis=wp.vec3(0.0, 0.0, 1.0),
            link_coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
            link_masses=[1.0, 1.0],
            joint_frames=[
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            ],
            link_inertias=[self.I_UNIT, self.I_UNIT],
        )
        model = builder.finalize(device=self.device)
        state = model.state()

        cases = [
            (
                "gravity_force",
                _PassiveOutput.GRAVITY_FORCE,
                "gravity_force",
                (model.joint_dof_count + 1,),
            ),
            (
                "coriolis_force",
                _PassiveOutput.CORIOLIS_FORCE,
                "coriolis_force",
                (model.joint_dof_count + 1,),
            ),
            (
                "mass_matrix",
                _PassiveOutput.MASS_MATRIX,
                "mass_matrix",
                (
                    model.articulation_count,
                    model.max_dofs_per_articulation + 1,
                    model.max_dofs_per_articulation + 1,
                ),
            ),
        ]
        for attr, flag, expected_substr, wrong_shape in cases:
            with self.subTest(flag=flag):
                inverse_dynamics = _InverseDynamicsArrays(model)
                setattr(
                    inverse_dynamics,
                    attr,
                    wp.zeros(wrong_shape, dtype=wp.float32, device=self.device),
                )
                with self.assertRaises(ValueError) as ctx:
                    _eval_inverse_dynamics_passive(model, state, flag, inverse_dynamics)
                msg = str(ctx.exception)
                self.assertIn(expected_substr, msg)
                self.assertIn(str(wrong_shape), msg)

        inverse_dynamics = _InverseDynamicsArrays(model)
        joint_qdd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=self.device)
        force_args = {
            "mass_matrix": inverse_dynamics.mass_matrix,
            "joint_qdd": joint_qdd,
            "coriolis_force": inverse_dynamics.coriolis_force,
            "gravity_force": inverse_dynamics.gravity_force,
            "joint_f": inverse_dynamics.joint_f,
        }
        force_cases = [
            (
                "mass_matrix",
                (
                    model.articulation_count,
                    model.max_dofs_per_articulation + 1,
                    model.max_dofs_per_articulation + 1,
                ),
            ),
            ("joint_qdd", (model.joint_dof_count + 1,)),
            ("coriolis_force", (model.joint_dof_count + 1,)),
            ("gravity_force", (model.joint_dof_count + 1,)),
            ("joint_f", (model.joint_dof_count + 1,)),
        ]
        for name, wrong_shape in force_cases:
            with self.subTest(force_array=name):
                args = force_args.copy()
                args[name] = wp.zeros(wrong_shape, dtype=wp.float32, device=self.device)
                with self.assertRaises(ValueError) as ctx:
                    newton.eval_inverse_dynamics_force(model, state, **args)
                msg = str(ctx.exception)
                self.assertIn(name, msg)
                self.assertIn(str(wrong_shape), msg)

        wrong_mask_shape = (model.articulation_count + 1,)
        wrong_mask = wp.zeros(wrong_mask_shape, dtype=bool, device=self.device)
        with self.subTest(passive_array="mask"):
            with self.assertRaises(ValueError) as ctx:
                newton.eval_inverse_dynamics_passive(
                    model,
                    state,
                    gravity_force=inverse_dynamics.gravity_force,
                    mask=wrong_mask,
                )
            msg = str(ctx.exception)
            self.assertIn("mask", msg)
            self.assertIn(str(wrong_mask_shape), msg)

        with self.subTest(force_array="mask"):
            with self.assertRaises(ValueError) as ctx:
                newton.eval_inverse_dynamics_force(model, state, **force_args, mask=wrong_mask)
            msg = str(ctx.exception)
            self.assertIn("mask", msg)
            self.assertIn(str(wrong_mask_shape), msg)

    def test_eval_inverse_dynamics_passive_raises_without_outputs(self):
        """The passive evaluator rejects a call that requests no outputs."""
        builder = self._build_two_link_articulation(
            gravity=wp.vec3(0.0, 0.0, 0.0),
            floating_base=False,
            joint_type="revolute",
            joint_axis=wp.vec3(0.0, 0.0, 1.0),
            link_coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
            link_masses=[1.0, 1.0],
            joint_frames=[
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            ],
            link_inertias=[self.I_UNIT, self.I_UNIT],
        )
        model = builder.finalize(device=self.device)
        state = model.state()
        with self.assertRaisesRegex(ValueError, "At least one inverse-dynamics output"):
            newton.eval_inverse_dynamics_passive(model, state)

    def test_inverse_dynamics_arrays_are_keyword_only(self):
        """All inverse-dynamics arrays must be passed by keyword."""
        builder = self._build_two_link_articulation(
            gravity=wp.vec3(0.0, 0.0, 0.0),
            floating_base=False,
            joint_type="revolute",
            joint_axis=wp.vec3(0.0, 0.0, 1.0),
            link_coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
            link_masses=[1.0, 1.0],
            joint_frames=[wp.transform_identity(), wp.transform_identity()],
            link_inertias=[self.I_UNIT, self.I_UNIT],
        )
        model = builder.finalize(device=self.device)
        state = model.state()
        arrays = _InverseDynamicsArrays(model)

        with self.assertRaises(TypeError):
            newton.eval_inverse_dynamics_passive(model, state, arrays.mass_matrix)
        with self.assertRaises(TypeError):
            newton.eval_inverse_dynamics_force(
                model,
                state,
                arrays.mass_matrix,
                wp.zeros_like(state.joint_qd),
                arrays.coriolis_force,
                arrays.gravity_force,
                arrays.joint_f,
            )

    def test_eval_inverse_dynamics_passive_cuda_graph_capture(self):
        """Internally allocated scratch remains valid in a captured graph."""
        if self.device is None or not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        if not wp.is_mempool_enabled(self.device):
            self.skipTest("CUDA graph capture requires the memory pool")

        builder = self._build_two_link_articulation(
            gravity=wp.vec3(0.0, -9.81, 0.0),
            floating_base=False,
            joint_type="revolute",
            joint_axis=wp.vec3(0.0, 0.0, 1.0),
            link_coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
            link_masses=[1.0, 1.0],
            joint_frames=[wp.transform_identity(), wp.transform_identity()],
            link_inertias=[self.I_UNIT, self.I_UNIT],
        )
        model = builder.finalize(device=self.device)
        state = model.state()
        arrays = _InverseDynamicsArrays(model)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        with wp.ScopedCapture(device=self.device) as capture:
            newton.eval_inverse_dynamics_passive(
                model,
                state,
                mass_matrix=arrays.mass_matrix,
                gravity_force=arrays.gravity_force,
                coriolis_force=arrays.coriolis_force,
            )

        wp.capture_launch(capture.graph)
        wp.capture_launch(capture.graph)
        self.assertTrue(np.all(np.isfinite(arrays.mass_matrix.numpy())))
        self.assertTrue(np.all(np.isfinite(arrays.gravity_force.numpy())))
        self.assertTrue(np.all(np.isfinite(arrays.coriolis_force.numpy())))

    def test_eval_inverse_dynamics_force_zero_articulations_preserves_joint_f(self):
        """``eval_inverse_dynamics_force`` short-circuits on a model with
        zero articulations without touching ``joint_f``.

        The ``articulation_count == 0`` guard returns before the in-place
        ``joint_f.zero_()``, so a sentinel previously written into ``joint_f`` must
        be preserved -- a regression check that the early return stays in
        place ahead of the zero pass.
        """
        builder = newton.ModelBuilder()
        model = builder.finalize(device=self.device)
        self.assertEqual(model.articulation_count, 0)

        n = 4
        sentinel = np.array([7.0, -2.5, 0.1, 99.0], dtype=np.float32)
        joint_f = wp.array(sentinel, dtype=wp.float32, device=self.device)
        # Buffer sizes here are otherwise irrelevant: the kernel never runs.
        H = wp.zeros((1, 1, 1), dtype=wp.float32, device=self.device)
        joint_qdd = wp.zeros(n, dtype=wp.float32, device=self.device)
        zero_bias = wp.zeros(n, dtype=wp.float32, device=self.device)
        inverse_dynamics = _InverseDynamicsArrays(model)
        inverse_dynamics.mass_matrix = H
        inverse_dynamics.coriolis_force = zero_bias
        inverse_dynamics.gravity_force = zero_bias
        inverse_dynamics.joint_f = joint_f

        _eval_inverse_dynamics_force(model, model.state(), inverse_dynamics, joint_qdd)

        np.testing.assert_array_equal(joint_f.numpy(), sentinel)

    def test_eval_inverse_dynamics_passive_zero_articulations_no_error(self):
        """Requesting every output on an empty model leaves consistent arrays."""
        builder = newton.ModelBuilder()
        model = builder.finalize(device=self.device)
        self.assertEqual(model.articulation_count, 0)
        self.assertEqual(model.joint_dof_count, 0)

        state = model.state()
        inverse_dynamics = _InverseDynamicsArrays(model)

        _eval_inverse_dynamics_passive(
            model,
            state,
            _PassiveOutput.ALL,
            inverse_dynamics,
        )

        self.assertEqual(inverse_dynamics.gravity_force.shape, (0,))
        self.assertEqual(inverse_dynamics.coriolis_force.shape, (0,))
        self.assertEqual(inverse_dynamics.mass_matrix.shape[0], 0)

    def test_articulation_view_masks_inverse_dynamics(self):
        """``ArticulationView.eval_inverse_dynamics_passive`` restricts the computation to selected articulations.

        Builds a 2-world model where each world has two articulations
        labelled ``"A"`` and ``"B"`` with distinct masses / inertias.
        Creates an :class:`ArticulationView` selecting only ``"A"``
        articulations and runs ``view.eval_inverse_dynamics_passive``. Asserts:

        - Slots in ``mass_matrix`` / ``gravity_force`` /
          ``coriolis_force`` corresponding to selected
          ``A`` articulations match an unmasked reference run.
        - Slots corresponding to unselected ``B`` articulations are
          exactly zero (matching the convention of
          :func:`~newton.eval_mass_matrix`).

        Repeats with a 1-D per-world submask to verify view-local
        filtering composes with the label-pattern selection.
        """
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        pos_half = wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity())
        neg_half = wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity())
        y_axis = wp.vec3(0.0, 1.0, 0.0)

        def _add_two_link_pendulum(builder, m_first, m_second, label):
            # Planar double pendulum: two revolutes about +Y, end-to-end
            # link layout (link half-length 0.5). Two DOFs per articulation
            # so M(q) has q-dependence and C(q, q_dot)*q_dot has real
            # cross-coupling at non-zero qd, and Z-up gravity gives
            # non-trivial g(q) (mass sweeps vertically in the XZ plane).
            b1 = builder.add_link(xform=identity_xform, mass=m_first, inertia=self.I_UNIT, com=wp.vec3(0.0, 0.0, 0.0))
            j1 = builder.add_joint_revolute(
                parent=-1,
                child=b1,
                axis=y_axis,
                parent_xform=identity_xform,
                child_xform=neg_half,
            )
            b2 = builder.add_link(xform=identity_xform, mass=m_second, inertia=self.I_UNIT, com=wp.vec3(0.0, 0.0, 0.0))
            j2 = builder.add_joint_revolute(
                parent=b1,
                child=b2,
                axis=y_axis,
                parent_xform=pos_half,
                child_xform=neg_half,
            )
            builder.add_articulation([j1, j2], label=label)

        # Per-world: two articulations "A" and "B" with distinct masses so
        # the M / g / C signatures differ between the two.
        world = newton.ModelBuilder()
        _add_two_link_pendulum(world, m_first=1.0, m_second=2.0, label="A")
        _add_two_link_pendulum(world, m_first=3.0, m_second=5.0, label="B")

        # Replicate to 2 worlds → 4 articulations: [A0, B0, A1, B1].
        scene = newton.ModelBuilder()
        scene.replicate(world, world_count=2)
        model = scene.finalize(device=self.device)
        self.assertEqual(model.articulation_count, 4)

        # Per-DOF layout: [w0.A.q1, q2, w0.B.q1, q2, w1.A.q1, q2, w1.B.q1, q2].
        # Distinct non-zero qd keeps C(q, q_dot)*q_dot non-trivial on the
        # selected slots so the parity asserts below aren't vacuous.
        joint_q = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        joint_qd = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        state = model.state()
        state.joint_q.assign(joint_q)
        state.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        # Layout: 2 worlds, 2 articulations [A, B] per world, 2 DOFs per articulation. →
        #   articulation index  0    1    2    3
        #   label               A    B    A    B
        #   DOF range           0,1  2,3  4,5  6,7

        # Reference: unmasked run → full-model M, g, C.
        reference_id = _InverseDynamicsArrays(model)
        _eval_inverse_dynamics_passive(model, state, _PassiveOutput.ALL, reference_id)
        H_ref = reference_id.mass_matrix.numpy()
        g_ref = reference_id.gravity_force.numpy()
        c_ref = reference_id.coriolis_force.numpy()
        joint_qdd = wp.array(np.linspace(0.1, 0.8, model.joint_dof_count), dtype=wp.float32, device=self.device)
        _eval_inverse_dynamics_force(model, state, reference_id, joint_qdd)
        tau_ref = reference_id.joint_f.numpy()

        # Ensure no entry in H_ref / g_ref / c_ref is (numerically) zero
        # — we use zero later as the signal that a slot was masked out, so
        # the parity asserts below would pass trivially if the reference
        # itself were zero anywhere.
        for art_id in range(model.articulation_count):
            for i in range(model.max_dofs_per_articulation):
                for j in range(model.max_dofs_per_articulation):
                    self.assertGreater(abs(H_ref[art_id, i, j]), 1e-6)
        for dof_idx in range(model.joint_dof_count):
            self.assertGreater(abs(g_ref[dof_idx]), 1e-6)
            self.assertGreater(abs(c_ref[dof_idx]), 1e-6)

        # View pattern is shared across cases; the per-world submask and
        # the resulting selected articulations / DOFs differ per case.
        view = newton.selection.ArticulationView(model, "A", verbose=False)
        np.testing.assert_array_equal(view.articulation_mask.numpy(), [True, False, True, False])

        # Parallel per-case data: row `i` describes case `i`.
        per_world_masks = [
            None,
            wp.array(np.asarray([True, False], dtype=bool), dtype=bool, device=self.device),
        ]
        # (n_cases, articulation_count): True at articulations expected to match
        # the unmasked reference; False ones must come back as zero.
        articulation_selected = np.array(
            [
                [True, False, True, False],  # no submask → A in both worlds
                [True, False, False, False],  # per-world [T, F] → A in world 0 only
            ]
        )
        # (n_cases, joint_dof_count): per-DOF version of the above.
        dof_selected = np.array(
            [
                [True, True, False, False, True, True, False, False],
                [True, True, False, False, False, False, False, False],
            ]
        )

        # All-deselected case: per-world mask is all-False, so no articulation
        # is selected and every output slot must be exactly zero.
        with self.subTest(case_idx="all_deselected"):
            inverse_dynamics = _InverseDynamicsArrays(model)
            all_false_mask = wp.array(np.asarray([False, False], dtype=bool), dtype=bool, device=self.device)
            view.eval_inverse_dynamics_passive(
                state,
                mass_matrix=inverse_dynamics.mass_matrix,
                gravity_force=inverse_dynamics.gravity_force,
                coriolis_force=inverse_dynamics.coriolis_force,
                mask=all_false_mask,
            )
            np.testing.assert_array_equal(
                inverse_dynamics.mass_matrix.numpy(),
                np.zeros_like(inverse_dynamics.mass_matrix.numpy()),
            )
            np.testing.assert_array_equal(
                inverse_dynamics.gravity_force.numpy(),
                np.zeros_like(inverse_dynamics.gravity_force.numpy()),
            )
            np.testing.assert_array_equal(
                inverse_dynamics.coriolis_force.numpy(),
                np.zeros_like(inverse_dynamics.coriolis_force.numpy()),
            )
            joint_qdd_nan = wp.array(
                np.full(model.joint_dof_count, np.nan, dtype=np.float32),
                dtype=wp.float32,
                device=self.device,
            )
            view.eval_inverse_dynamics_force(
                state,
                mass_matrix=inverse_dynamics.mass_matrix,
                joint_qdd=joint_qdd_nan,
                coriolis_force=inverse_dynamics.coriolis_force,
                gravity_force=inverse_dynamics.gravity_force,
                joint_f=inverse_dynamics.joint_f,
                mask=all_false_mask,
            )
            np.testing.assert_array_equal(
                inverse_dynamics.joint_f.numpy(),
                np.zeros_like(inverse_dynamics.joint_f.numpy()),
            )

        for case_idx in range(len(per_world_masks)):
            with self.subTest(case_idx=case_idx):
                inverse_dynamics = _InverseDynamicsArrays(model)
                view.eval_inverse_dynamics_passive(
                    state,
                    mass_matrix=inverse_dynamics.mass_matrix,
                    gravity_force=inverse_dynamics.gravity_force,
                    coriolis_force=inverse_dynamics.coriolis_force,
                    mask=per_world_masks[case_idx],
                )

                H = inverse_dynamics.mass_matrix.numpy()
                g = inverse_dynamics.gravity_force.numpy()
                c = inverse_dynamics.coriolis_force.numpy()
                view.eval_inverse_dynamics_force(
                    state,
                    mass_matrix=inverse_dynamics.mass_matrix,
                    joint_qdd=joint_qdd,
                    coriolis_force=inverse_dynamics.coriolis_force,
                    gravity_force=inverse_dynamics.gravity_force,
                    joint_f=inverse_dynamics.joint_f,
                    mask=per_world_masks[case_idx],
                )
                tau = inverse_dynamics.joint_f.numpy()

                # Mass matrix: selected articulations match reference; the rest are zero.
                for art_id in range(model.articulation_count):
                    for i in range(model.max_dofs_per_articulation):
                        for j in range(model.max_dofs_per_articulation):
                            if articulation_selected[case_idx, art_id]:
                                self.assertAlmostEqual(float(H[art_id, i, j]), float(H_ref[art_id, i, j]), delta=1e-5)
                            else:
                                self.assertAlmostEqual(float(H[art_id, i, j]), 0.0, delta=1e-6)

                # Per-DOF bias buffers: selected DOFs match reference; the rest are zero.
                for dof_idx in range(model.joint_dof_count):
                    if dof_selected[case_idx, dof_idx]:
                        self.assertAlmostEqual(float(g[dof_idx]), float(g_ref[dof_idx]), delta=1e-5)
                        self.assertAlmostEqual(float(c[dof_idx]), float(c_ref[dof_idx]), delta=1e-5)
                    else:
                        self.assertAlmostEqual(float(g[dof_idx]), 0.0, delta=1e-6)
                        self.assertAlmostEqual(float(c[dof_idx]), 0.0, delta=1e-6)

                    if dof_selected[case_idx, dof_idx]:
                        self.assertAlmostEqual(float(tau[dof_idx]), float(tau_ref[dof_idx]), delta=1e-5)
                    else:
                        self.assertAlmostEqual(float(tau[dof_idx]), 0.0, delta=1e-6)

    def test_articulation_view_masks_inverse_dynamics_2d(self):
        """``ArticulationView.eval_inverse_dynamics_passive`` accepts a 2-D ``wp.array2d[bool]`` mask.

        A 2-D mask of shape ``[world_count, count_per_world]`` selects
        individual articulations per world independently, which cannot be
        expressed as a 1-D per-world mask.  This test uses a diagonal
        pattern — world 0 selects its first articulation (A), world 1
        selects its second (B) — to verify that the 2-D path produces the
        correct per-articulation zeroing behaviour.

        Model layout (2 worlds X 2 articulations):
            art index   0    1    2    3
            label       A0   B0   A1   B1
            DOF range  0,1  2,3  4,5  6,7
        Selected by 2-D mask: A0 (index 0) and B1 (index 3).
        """
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        pos_half = wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity())
        neg_half = wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity())
        y_axis = wp.vec3(0.0, 1.0, 0.0)

        def _add_two_link_pendulum(builder, m_first, m_second, label):
            b1 = builder.add_link(xform=identity_xform, mass=m_first, inertia=self.I_UNIT, com=wp.vec3(0.0, 0.0, 0.0))
            j1 = builder.add_joint_revolute(
                parent=-1,
                child=b1,
                axis=y_axis,
                parent_xform=identity_xform,
                child_xform=neg_half,
            )
            b2 = builder.add_link(xform=identity_xform, mass=m_second, inertia=self.I_UNIT, com=wp.vec3(0.0, 0.0, 0.0))
            j2 = builder.add_joint_revolute(
                parent=b1,
                child=b2,
                axis=y_axis,
                parent_xform=pos_half,
                child_xform=neg_half,
            )
            builder.add_articulation([j1, j2], label=label)

        world = newton.ModelBuilder()
        _add_two_link_pendulum(world, m_first=1.0, m_second=2.0, label="A")
        _add_two_link_pendulum(world, m_first=3.0, m_second=5.0, label="B")
        scene = newton.ModelBuilder()
        scene.replicate(world, world_count=2)
        model = scene.finalize(device=self.device)

        joint_q = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        joint_qd = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        state = model.state()
        state.joint_q.assign(joint_q)
        state.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        # Unmasked reference.
        reference_id = _InverseDynamicsArrays(model)
        _eval_inverse_dynamics_passive(model, state, _PassiveOutput.ALL, reference_id)
        H_ref = reference_id.mass_matrix.numpy()
        g_ref = reference_id.gravity_force.numpy()
        c_ref = reference_id.coriolis_force.numpy()
        joint_qdd = wp.array(np.linspace(0.1, 0.8, model.joint_dof_count), dtype=wp.float32, device=self.device)
        _eval_inverse_dynamics_force(model, state, reference_id, joint_qdd)
        tau_ref = reference_id.joint_f.numpy()

        # View covers all four articulations (A and B in both worlds),
        # giving count_per_world=2.  The fnmatch pattern "[AB]" matches
        # both single-character labels.
        view = newton.selection.ArticulationView(model, "[AB]", verbose=False)
        self.assertEqual(view.count_per_world, 2)

        # 1-D mask [True, False]: selects all articulations in world 0 (A0 and B0),
        # deselects all in world 1 (A1 and B1).  With count_per_world=2 this
        # verifies that a single True/False entry governs the whole world, not
        # just one articulation slot — which a count_per_world=1 view cannot test.
        mask1d = wp.array(np.array([True, False], dtype=bool), dtype=bool, device=self.device)

        inverse_dynamics = _InverseDynamicsArrays(model)
        view.eval_inverse_dynamics_passive(
            state,
            mass_matrix=inverse_dynamics.mass_matrix,
            gravity_force=inverse_dynamics.gravity_force,
            coriolis_force=inverse_dynamics.coriolis_force,
            mask=mask1d,
        )

        H = inverse_dynamics.mass_matrix.numpy()
        g = inverse_dynamics.gravity_force.numpy()
        c = inverse_dynamics.coriolis_force.numpy()

        # art 0 = A0 (selected), art 1 = B0 (selected), art 2 = A1 (not), art 3 = B1 (not)
        articulation_selected_1d = [True, True, False, False]
        dof_selected_1d = [True, True, True, True, False, False, False, False]

        for art_id in range(model.articulation_count):
            for i in range(model.max_dofs_per_articulation):
                for j in range(model.max_dofs_per_articulation):
                    if articulation_selected_1d[art_id]:
                        self.assertAlmostEqual(float(H[art_id, i, j]), float(H_ref[art_id, i, j]), delta=1e-5)
                    else:
                        self.assertAlmostEqual(float(H[art_id, i, j]), 0.0, delta=1e-6)

        for dof_idx in range(model.joint_dof_count):
            if dof_selected_1d[dof_idx]:
                self.assertAlmostEqual(float(g[dof_idx]), float(g_ref[dof_idx]), delta=1e-5)
                self.assertAlmostEqual(float(c[dof_idx]), float(c_ref[dof_idx]), delta=1e-5)
            else:
                self.assertAlmostEqual(float(g[dof_idx]), 0.0, delta=1e-6)
                self.assertAlmostEqual(float(c[dof_idx]), 0.0, delta=1e-6)

        # Diagonal 2-D mask: select A0 (world 0, slot 0) and B1 (world 1, slot 1).
        # This pattern cannot be expressed as a 1-D per-world mask.
        mask2d = wp.array(
            np.array([[True, False], [False, True]], dtype=bool),
            dtype=bool,
            device=self.device,
        )

        inverse_dynamics = _InverseDynamicsArrays(model)
        view.eval_inverse_dynamics_passive(
            state,
            mass_matrix=inverse_dynamics.mass_matrix,
            gravity_force=inverse_dynamics.gravity_force,
            coriolis_force=inverse_dynamics.coriolis_force,
            mask=mask2d,
        )
        view.eval_inverse_dynamics_force(
            state,
            mass_matrix=inverse_dynamics.mass_matrix,
            joint_qdd=joint_qdd,
            coriolis_force=inverse_dynamics.coriolis_force,
            gravity_force=inverse_dynamics.gravity_force,
            joint_f=inverse_dynamics.joint_f,
            mask=mask2d,
        )

        H = inverse_dynamics.mass_matrix.numpy()
        g = inverse_dynamics.gravity_force.numpy()
        c = inverse_dynamics.coriolis_force.numpy()
        tau = inverse_dynamics.joint_f.numpy()

        # art 0 = A0 (selected), art 1 = B0 (not), art 2 = A1 (not), art 3 = B1 (selected)
        articulation_selected_2d = [True, False, False, True]
        # DOF ranges: A0→[0,1], B0→[2,3], A1→[4,5], B1→[6,7]
        dof_selected_2d = [True, True, False, False, False, False, True, True]

        for art_id in range(model.articulation_count):
            for i in range(model.max_dofs_per_articulation):
                for j in range(model.max_dofs_per_articulation):
                    if articulation_selected_2d[art_id]:
                        self.assertAlmostEqual(float(H[art_id, i, j]), float(H_ref[art_id, i, j]), delta=1e-5)
                    else:
                        self.assertAlmostEqual(float(H[art_id, i, j]), 0.0, delta=1e-6)

        for dof_idx in range(model.joint_dof_count):
            if dof_selected_2d[dof_idx]:
                self.assertAlmostEqual(float(g[dof_idx]), float(g_ref[dof_idx]), delta=1e-5)
                self.assertAlmostEqual(float(c[dof_idx]), float(c_ref[dof_idx]), delta=1e-5)
            else:
                self.assertAlmostEqual(float(g[dof_idx]), 0.0, delta=1e-6)
                self.assertAlmostEqual(float(c[dof_idx]), 0.0, delta=1e-6)

            if dof_selected_2d[dof_idx]:
                self.assertAlmostEqual(float(tau[dof_idx]), float(tau_ref[dof_idx]), delta=1e-5)
            else:
                self.assertAlmostEqual(float(tau[dof_idx]), 0.0, delta=1e-6)


class TestGravCompForceCPU(TestGravCompForce, unittest.TestCase):
    device = wp.get_device("cpu")


@unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
class TestGravCompForceCUDA(TestGravCompForce, unittest.TestCase):
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else None


class TestCoriolisCompForceCPU(TestCoriolisCompForce, unittest.TestCase):
    device = wp.get_device("cpu")


@unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
class TestCoriolisCompForceCUDA(TestCoriolisCompForce, unittest.TestCase):
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else None


class TestMassMatrixCPU(TestMassMatrix, unittest.TestCase):
    device = wp.get_device("cpu")


@unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
class TestMassMatrixCUDA(TestMassMatrix, unittest.TestCase):
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else None


class TestManipulatorEquationCPU(TestManipulatorEquation, unittest.TestCase):
    device = wp.get_device("cpu")


@unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
class TestManipulatorEquationCUDA(TestManipulatorEquation, unittest.TestCase):
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else None


class TestInverseDynamicsAPICPU(TestInverseDynamicsAPI, unittest.TestCase):
    device = wp.get_device("cpu")


@unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
class TestInverseDynamicsAPICUDA(TestInverseDynamicsAPI, unittest.TestCase):
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else None


if __name__ == "__main__":
    unittest.main()
