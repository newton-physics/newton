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

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from .model import Model
from .joints import JointType
from .state import State


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

        self.joint_pos_target: wp.array | None = None
        # should thos be joint_dof_count +1 dim
        """Per-DOF position targets, shape ``(joint_dof_count,)``, type ``float`` (optional)."""

        self.joint_vel_target: wp.array | None = None
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

        # Actuator list. Each actuator can contribute forces into ``joint_f``.
        self.actuators: list[Actuator] = []

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

        if self.joint_pos_target is not None:
            self.joint_pos_target.zero_()
        if self.joint_vel_target is not None:
            self.joint_vel_target.zero_()

    def compute_actuator_forces(self, model: Model, state: State, nworld: int, axes_per_env: int) -> None:
        """Compute and accumulate forces from all actuators into ``joint_f``.

        Notes:
            - This method zeros ``joint_f`` before accumulation.
            - For convenience, if ``joint_f`` is ``None`` and the model has joints,
              a zero array will be allocated on the model's device.
        """
        self.joint_f_total = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)

        for actuator in self.actuators:
            actuator.compute_force(model, state, self, nworld, axes_per_env)


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
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    joint_type: wp.array(dtype=wp.int32),
    joint_f: wp.array(dtype=wp.float32),
    # outputs
    joint_f_total: wp.array(dtype=wp.float32),
):
    joint_id = wp.tid()
    qi = joint_q_start[joint_id]
    qdi = joint_qd_start[joint_id]
    dim = joint_dof_dim[joint_id, 0] + joint_dof_dim[joint_id, 1]

    if joint_type[joint_id] == int(JointType.FREE):
        for j in range(dim):
            qdj = qdi + j
            joint_f_total[qdj] = joint_f[qdj]
    else:

        for j in range(dim):
            qj = qi + j
            qdj = qdi + j
            q = joint_q[qj]
            qd = joint_qd[qdj]

            tq = q_target[qdj]
            tqd = qd_target[qdj]
            Kp = kp_dof[qdj]
            Kd = kd_dof[qdj]
            tq = Kp * (tq - q) + Kd * (tqd - qd)
            joint_f_total[qdj] = tq + joint_f[qdj]



class Actuator:
    """Simple PD actuator acting on a set of joint DOFs.

    This actuator computes torques/forces for the specified DOF indices using a PD law:

        tau = kp * (q_target - q) + kd * (qd_target - qd)

    Position tracking is only applied for joints where the number of coordinates equals
    the number of DOFs (e.g., revolute, prismatic, D6). For joints like FREE or BALL,
    where coordinate and DOF dimensions differ, only the velocity term is applied.
    """

    def compute_force(self, model: Model, state: State, control: Control, nworld: int, axes_per_env: int) -> None:

        wp.launch(
            pd_actuator_kernel,
            dim=model.joint_count,
            inputs=[
                model.joint_target_ke,
                model.joint_target_kd,
                state.joint_q,
                state.joint_qd,
                control.joint_pos_target,
                control.joint_vel_target,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.joint_type,
                control.joint_f,
            ],
            outputs=[
                control.joint_f_total,
            ],
            device=model.device,
        )
