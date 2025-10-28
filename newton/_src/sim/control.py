# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from .model import Model
from .joints import JointType
from .state import State


@wp.kernel
def zero_array_kernel(arr: wp.array(dtype=wp.float32)):
    i = wp.tid()
    arr[i] = 0.0


class Control:
    """Time-varying control data for a :class:`Model`.

    Time-varying control data includes joint torques, control inputs, muscle activations,
    and activation forces for triangle and tetrahedral elements.

    The exact attributes depend on the contents of the model. Control objects
    should generally be created using the :func:`newton.Model.control()` function.
    """

    def __init__(self):
        self.joint_f: wp.array | None = None
        """
        Array of generalized joint forces with shape ``(joint_dof_count,)`` and type ``float``.

        The degrees of freedom for free joints are included in this array and have the same
        convention as the :attr:`newton.State.body_f` array where the 6D wrench is defined as
        ``(f_x, f_y, f_z, t_x, t_y, t_z)``, where ``f_x``, ``f_y``, and ``f_z`` are the components
        of the force vector (linear) and ``t_x``, ``t_y``, and ``t_z`` are the
        components of the torque vector (angular). Both linear forces and angular torques applied to free joints are
        applied in world frame (same as :attr:`newton.State.body_f`).
        """

        self.joint_f_total: wp.array | None = None
        """Total joint forces, shape ``(joint_dof_count,)``, type ``float``."""

        self.joint_target_pos: wp.array | None = None
        """Per-DOF position targets, shape ``(joint_dof_count,)``, type ``float`` (optional)."""

        self.joint_target_vel: wp.array | None = None
        """Per-DOF velocity targets, shape ``(joint_dof_count,)``, type ``float`` (optional)."""

        self.tri_activations: wp.array | None = None
        """Array of triangle element activations with shape ``(tri_count,)`` and type ``float``."""

        self.tet_activations: wp.array | None = None
        """Array of tetrahedral element activations with shape with shape ``(tet_count,) and type ``float``."""

        self.muscle_activations: wp.array | None = None
        """
        Array of muscle activations with shape ``(muscle_count,)`` and type ``float``.

        .. note::
            Support for muscle dynamics is not yet implemented.
        """

    def clear(self) -> None:
        """Reset the control inputs to zero."""

        if self.joint_f is not None:
            self.joint_f.zero_()
        if self.tri_activations is not None:
            self.tri_activations.zero_()
        if self.tet_activations is not None:
            self.tet_activations.zero_()
        if self.muscle_activations is not None:
            self.muscle_activations.zero_()

        if self.joint_target_pos is not None:
            self.joint_target_pos.zero_()
        if self.joint_target_vel is not None:
            self.joint_target_vel.zero_()

    def compute_actuator_forces(self, model: Model, state: State) -> None:
        """Compute and accumulate forces from all actuators into joint_f_total."""
        if self.joint_f_total is None:
            self.joint_f_total = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
        else:
            wp.launch(
                zero_array_kernel,
                dim=model.joint_dof_count,
                inputs=[self.joint_f_total],
                device=model.device,
            )

        for actuator in model.actuators:
            actuator.compute_force(model, state, self)


@wp.kernel
def pd_actuator_kernel(
    kp_dof: wp.array(dtype=wp.float32),
    kd_dof: wp.array(dtype=wp.float32),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    q_target: wp.array(dtype=wp.float32),
    qd_target: wp.array(dtype=wp.float32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_actuator_start: wp.array(dtype=wp.int32),
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    joint_type: wp.array(dtype=wp.int32),
    joint_f: wp.array(dtype=wp.float32),
    gear_ratio: wp.array(dtype=wp.float32),
    joint_effort_limit:wp.array(dtype=wp.float32),
    # outputs
    joint_f_total: wp.array(dtype=wp.float32),
):
    joint_id = wp.tid()
    qi = joint_q_start[joint_id]
    qdi = joint_qd_start[joint_id]
    actuator_start = joint_actuator_start[joint_id]
    dim = joint_dof_dim[joint_id, 0] + joint_dof_dim[joint_id, 1]

    if joint_type[joint_id] == JointType.FREE:
        for j in range(dim):
            qdj = qdi + j
            force = joint_f[qdj]
            joint_f_total[qdj] = force
    else:
        for j in range(dim):
            qj = qi + j
            qdj = qdi + j
            q = joint_q[qj]
            qd = joint_qd[qdj]

            tq = q_target[actuator_start + j]
            tqd = qd_target[actuator_start + j]
            Kp = kp_dof[actuator_start + j]
            Kd = kd_dof[actuator_start + j]
            tq = Kp * (tq - q) + Kd * (tqd - qd)
            force = gear_ratio[qdj] * (tq + joint_f[qdj])
            limit = joint_effort_limit[qdj]
            joint_f_total[qdj] += wp.clamp(force, -limit, limit)


class Actuator(ABC):
    """Abstract base class for actuators that apply forces/torques to joint DOFs."""

    def __init__(self):
        self.gear_ratio: wp.array | None = None

    @abstractmethod
    def compute_force(self, model: Model, state: State, control: Control) -> None:
        """Compute and apply actuator forces to the control.joint_f_total array.

        Args:
            model: The physics model containing joint parameters
            state: Current state of joints (positions, velocities)
            control: Control targets and force output
        """
        pass


class PDActuator(Actuator):
    """PD actuator acting on a set of joint DOFs.

    This actuator computes torques/forces for the specified DOF indices using a PD law:

        tau = kp * (q_target - q) + kd * (qd_target - qd)

    Attributes:
        gear_ratio: Array of gear ratios per DOF. Contains the actual gear ratio when this
                   actuator type is applied to a DOF, otherwise 0.0. This allows multiple
                   actuators to control different subsets of DOFs.
    """

    def __init__(self):
        super().__init__()

    def compute_force(self, model: Model, state: State, control: Control) -> None:
        """Compute PD control forces for all DOFs controlled by this actuator."""
        wp.launch(
            pd_actuator_kernel,
            dim=model.joint_count,
            inputs=[
                model.joint_target_ke,
                model.joint_target_kd,
                state.joint_q,
                state.joint_qd,
                control.joint_target_pos,
                control.joint_target_vel,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_actuator_start,
                model.joint_dof_dim,
                model.joint_type,
                control.joint_f,
                self.gear_ratio,
                model.joint_effort_limit,
            ],
            outputs=[
                control.joint_f_total,
            ],
            device=model.device,
        )