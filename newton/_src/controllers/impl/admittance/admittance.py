# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerAdmittance — task-space admittance control with variable gains.

Every port is per-robot: exactly one entry per robot. The controller needs no
kinematics or dynamics of its own — it turns a measured contact wrench into a
compliant tool pose and twist, which a motion controller such as
:class:`~newton.controllers.ControllerOperationalSpace` or
:class:`~newton.controllers.ControllerDifferentialIK` then tracks — so there is
no model-based/model-free pair, unlike the other controller families.

Admittance law, per axis of the operational frame (terms enabled at
construction):

    M · ë + D · ė + K · e = [desired_wrench if use_desired_wrench else 0] - measured_wrench

``e`` is the compliant tool pose's displacement from the reference tool pose:
a position offset added to the reference position, and a rotation vector
whose rotation is applied to the reference orientation about the
operational frame's axes. ``ė`` is its twist. ``M``/``D``/``K`` are diagonal,
per-axis in the operational frame, and may change every step.

Each step advances ``(e, ė)`` by one backward-Euler step, solving the spring
and damper implicitly, axis by axis:

    ė⁺ = (M · ė + dt · (w - K · e)) / (M + dt · D + dt² · K)
    e⁺ = e ⊞ dt · ė⁺

where ``w`` is the right-hand side above and ``⊞`` adds the position half
and composes the rotation half on the rotation group. For constant
non-negative gains this step never increases the virtual energy
``½ ėᵀ M ė + ½ eᵀ K e`` when ``w = 0``, for any ``dt``, so gains may be
changed every step — e.g. by a learned policy — without the integration
going unstable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from ...controller import ControllerBase
from ...utils import _validate_array
from .._common import _port_destination, _port_source, _write_port
from ..operational_space._common import _rotate_spatial_vector
from ..operational_space.model_free import _validate_gain_argument, _validate_transform_argument

_IDENTITY_TRANSFORM = wp.transform()


@wp.func
def _quat_to_rotation_vector(q: wp.quat) -> wp.vec3:
    """Rotation vector (axis times angle, angle in [0, pi]) of a unit quaternion.

    Same small-angle treatment as :func:`_pose_error_kernel`: Warp's
    ``quat_to_axis_angle`` divides by the vector-part norm unguarded, which
    is NaN at the rest state, the most common state of all.
    """
    sign = 1.0
    if q[3] < 0.0:
        sign = -1.0
    vector_part = sign * wp.vec3(q[0], q[1], q[2])
    vector_part_norm = wp.length(vector_part)
    if vector_part_norm > 1.0e-8:
        return vector_part * (2.0 * wp.atan2(vector_part_norm, sign * q[3]) / vector_part_norm)
    return 2.0 * vector_part


@wp.func
def _rotation_vector_to_quat(rotation_vector: wp.vec3) -> wp.quat:
    """Unit quaternion of a rotation vector (axis times angle)."""
    angle = wp.length(rotation_vector)
    if angle > 1.0e-8:
        return wp.quat_from_axis_angle(rotation_vector / angle, angle)
    half = 0.5 * rotation_vector
    return wp.normalize(wp.quat(half[0], half[1], half[2], 1.0))


@wp.kernel
def _admittance_step_kernel(
    operational_frame_pose_world: wp.array[wp.transform],  # (robot_count,)
    reference_tool_pose_operational: wp.array[wp.transform],  # (robot_count,)
    reference_twist_operational: wp.array[wp.spatial_vector],  # (robot_count,)
    measured_wrench_world: wp.array[wp.spatial_vector],  # (robot_count,) tool on environment
    desired_wrench_world: wp.array[wp.spatial_vector],  # (robot_count,) zeros when disabled
    virtual_stiffness: wp.array[wp.spatial_vector],  # (robot_count,) operational-frame-local
    virtual_damping: wp.array[wp.spatial_vector],  # (robot_count,) operational-frame-local
    virtual_mass: wp.array[wp.spatial_vector],  # (robot_count,) operational-frame-local
    displacement_operational: wp.array[wp.spatial_vector],  # (robot_count,) (position offset, rotation vector)
    displacement_twist_operational: wp.array[wp.spatial_vector],  # (robot_count,)
    dt: wp.array[wp.float32],  # (1,)
    # outputs
    compliant_tool_pose_operational: wp.array[wp.transform],  # (robot_count,)
    compliant_tool_pose_world: wp.array[wp.transform],  # (robot_count,)
    compliant_twist_operational: wp.array[wp.spatial_vector],  # (robot_count,)
    displacement_operational_out: wp.array[wp.spatial_vector],  # (robot_count,)
    displacement_twist_operational_out: wp.array[wp.spatial_vector],  # (robot_count,)
):
    # One thread owns one robot and reads every input slot before writing any
    # output slot, so binding an output to the same array as its input (to
    # advance the displacement in place) is safe.
    robot = wp.tid()
    h = dt[0]

    operational_frame = operational_frame_pose_world[robot]
    reference_pose = reference_tool_pose_operational[robot]
    reference_twist = reference_twist_operational[robot]
    displacement = displacement_operational[robot]
    displacement_twist = displacement_twist_operational[robot]
    stiffness = virtual_stiffness[robot]
    damping = virtual_damping[robot]
    mass = virtual_mass[robot]

    quat_operational_from_world = wp.quat_inverse(wp.transform_get_rotation(operational_frame))
    wrench = _rotate_spatial_vector(
        quat_operational_from_world, desired_wrench_world[robot] - measured_wrench_world[robot]
    )

    # Backward Euler, one decoupled axis at a time: the spring is evaluated at
    # the end-of-step displacement e + h * ė⁺, which is what puts the dt² K
    # term in the denominator. An axis with no positive gain has no defined
    # response, so it is held still rather than divided by zero.
    displacement_twist_next = wp.spatial_vector()
    for axis in range(6):
        k = wp.max(stiffness[axis], 0.0)
        d = wp.max(damping[axis], 0.0)
        m = wp.max(mass[axis], 0.0)
        denominator = m + h * d + h * h * k
        if denominator > 0.0:
            displacement_twist_next[axis] = (
                m * displacement_twist[axis] + h * (wrench[axis] - k * displacement[axis])
            ) / denominator

    position_offset_next = wp.spatial_top(displacement) + h * wp.spatial_top(displacement_twist_next)
    # The angular twist is about the operational frame's fixed axes, so its
    # increment composes on the left of the current rotation offset.
    rotation_offset_next = wp.normalize(
        _rotation_vector_to_quat(h * wp.spatial_bottom(displacement_twist_next))
        * _rotation_vector_to_quat(wp.spatial_bottom(displacement))
    )

    compliant_pose = wp.transform(
        wp.transform_get_translation(reference_pose) + position_offset_next,
        wp.normalize(rotation_offset_next * wp.transform_get_rotation(reference_pose)),
    )
    # d/dt (q_offset * q_reference) = ½ (ω_offset + R_offset ω_reference) * (q_offset * q_reference).
    compliant_twist = wp.spatial_vector(
        wp.spatial_top(reference_twist) + wp.spatial_top(displacement_twist_next),
        wp.spatial_bottom(displacement_twist_next)
        + wp.quat_rotate(rotation_offset_next, wp.spatial_bottom(reference_twist)),
    )

    compliant_tool_pose_operational[robot] = compliant_pose
    compliant_tool_pose_world[robot] = operational_frame * compliant_pose
    compliant_twist_operational[robot] = compliant_twist
    displacement_operational_out[robot] = wp.spatial_vector(
        position_offset_next, _quat_to_rotation_vector(rotation_offset_next)
    )
    displacement_twist_operational_out[robot] = displacement_twist_next


def _validate_non_negative_gain(value: Any, name: str) -> np.ndarray | None:
    """Reject a baked gain with a negative entry; return its per-robot values, or ``None`` when live."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        values = np.full((1, 6), float(value))
    elif isinstance(value, wp.spatial_vector):
        values = np.array(value, dtype=np.float64).reshape(1, 6)
    else:
        values = value.numpy().reshape(-1, 6)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must be finite and non-negative on every axis.")
    return values


class ControllerAdmittance(ControllerBase):
    """Task-space admittance controller with variable virtual mass, damping, and stiffness.

    Implements the admittance control law. Where an impedance controller
    (e.g. :class:`ControllerOperationalSpace`) maps a pose error to a wrench,
    this controller maps a measured contact wrench to a compliant motion:
    each robot's tool behaves as a virtual mass-spring-damper attached to its
    reference pose, displaced by the contact wrench. The resulting compliant
    tool pose and twist are this controller's output, to be tracked by any
    motion controller — e.g. bound to
    :class:`ControllerOperationalSpace`'s
    ``inputs.desired_tool_pose_operational``/``inputs.desired_twist_operational``,
    or composed into :class:`ControllerDifferentialIK`'s
    ``inputs.desired_tool_pose_world`` for a position-controlled robot.

    Per axis of the operational frame, the controller integrates

    .. math::

        M \\ddot{e} + D \\dot{e} + K e = w_{des} - w_{meas},

    where :math:`e` is the compliant pose's displacement from the reference
    pose (a position offset and a rotation vector) and :math:`w_{meas}` the
    measured wrench the tool exerts on its environment. Each :meth:`step`
    advances the displacement by one backward-Euler step, which stays stable
    for any non-negative gains and any ``dt``. Gains passed as ``None`` are
    read from the input struct every step, so they can be varied
    online — variable admittance — e.g. stiff in free space and compliant in
    contact, or output by a learned policy.

    With ``virtual_stiffness`` zero on an axis and ``use_desired_wrench``
    enabled, the steady state on that axis is ``measured = desired``: the
    controller then performs explicit force tracking, moving the reference
    along that axis until the contact wrench matches the setpoint (e.g.
    pressing a tool onto a surface of unknown height).

    The displacement and its twist are the controller's only state, and they
    are ports rather than internal buffers: ``inputs.displacement_operational``/
    ``inputs.displacement_twist_operational`` are read, and their advanced
    values written to the identically named outputs. Bind each output to its
    input's array to advance in place, and write zeros into it to reset a
    robot (e.g. on an environment reset). Zero is the rest state, so the
    zero-initialised arrays from :meth:`input` start every robot at rest.

    Every port is **per-robot**: a 1-D array with one entry per robot. Every
    port, of any dtype, may be bound either to a plain array or to an indexed
    view of a larger array.

    The controller reads input ports and overwrites output ports. Plain arrays
    are read and written directly; indexed views use gather/scatter buffers.
    Unlike the other controllers, an output may be bound to the same array as
    an input, as the displacement outputs are to advance in place, because
    each robot's inputs are read before its outputs are written. Any other
    overlap between ports, including through views, is not supported and is
    not validated.

    Array shapes and devices are validated on each direct call to
    :meth:`step`, but not when a captured graph is replayed, since the checks
    run in Python at capture time only.

    The rotational half of the displacement is a rotation vector, so it
    represents rotations of at most pi; the rotational spring always pulls
    back along the shorter way around.

    Args:
        controlled_robot_count: Number of robots, the length of every port.
        virtual_stiffness: Virtual spring stiffness K per axis, in the
            operational frame, [N/m] on the position axes and [N·m/rad] on
            the orientation axes. Must be non-negative. Pass a scalar to
            apply the same gain to every axis of every robot, a
            ``wp.spatial_vector`` to apply the same 6 per-axis gains to every
            robot, an array of shape [controlled_robot_count] to set them
            individually, or ``None`` to read ``inputs.virtual_stiffness``
            each step.
        virtual_damping: Virtual damping D per axis, in the operational
            frame, [N·s/m] on the position axes and [N·m·s/rad] on the
            orientation axes. Must be non-negative. Same format as
            ``virtual_stiffness``.
        virtual_mass: Virtual mass M per axis, in the operational frame,
            [kg] on the position axes and [kg·m²] on the orientation axes.
            Must be non-negative; zero gives a first-order (damper-spring)
            admittance on that axis. Same format as ``virtual_stiffness``.
            Every axis needs at least one of ``virtual_stiffness``,
            ``virtual_damping``, and ``virtual_mass`` positive. A live gain
            is clamped to zero if negative, and a live axis left with all
            three zero has its displacement twist held at zero.
        operational_frame_pose_world: World pose of the operational frame —
            the frame the reference and compliant poses/twists are expressed
            relative to, and the gains are interpreted in. Pass a
            ``wp.transform`` to apply the same fixed pose to every robot, an
            array of shape [controlled_robot_count] to set them individually,
            or ``None`` to read ``inputs.operational_frame_pose_world`` each
            step for a time-varying frame. Defaults to identity (coincides
            with world frame).
        use_desired_wrench: Add a desired contact wrench setpoint,
            ``inputs.desired_wrench_world``, to the right-hand side, so the
            controller regulates the measured wrench toward it rather than
            toward zero.
        device: Warp device.
        requires_grad: Not supported at this time; must be ``False``.
    """

    class Inputs:
        """Input struct returned by :meth:`~ControllerAdmittance.input`.

        Every field is per-robot, shape [controlled_robot_count]. Optional
        fields are ``None`` when the corresponding feature is disabled or
        its value is baked at construction.
        """

        reference_tool_pose_operational: wp.array[wp.transform] | wp.indexedarray[wp.transform]
        """Reference (nominal, contact-free) tool pose, relative to the operational frame [m, unitless quaternion], shape [controlled_robot_count]."""
        reference_twist_operational: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector]
        """Reference tool twist (linear, angular), components expressed in the operational frame [m/s, rad/s], shape [controlled_robot_count]."""
        measured_wrench_world: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector]
        """Measured contact wrench (force, moment about the tool point) the tool exerts on its environment, in world coordinates [N, N·m], shape [controlled_robot_count]."""
        desired_wrench_world: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector] | None
        """Desired contact wrench (force, moment about the tool point) the tool exerts on its environment, in world coordinates [N, N·m], shape [controlled_robot_count]. ``None`` unless ``use_desired_wrench=True``."""
        displacement_operational: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector]
        """Compliant pose's displacement from the reference pose at the start of the step (position offset, rotation vector), operational frame [m, rad], shape [controlled_robot_count]. Zero is the rest state."""
        displacement_twist_operational: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector]
        """Rate of ``displacement_operational`` (linear, angular), operational frame [m/s, rad/s], shape [controlled_robot_count]."""
        operational_frame_pose_world: wp.array[wp.transform] | wp.indexedarray[wp.transform] | None
        """World pose of the operational frame, shape [controlled_robot_count]. ``None`` when fixed at construction."""
        virtual_stiffness: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector] | None
        """Virtual stiffness K per axis, operational-frame-local [N/m, N·m/rad], shape [controlled_robot_count]. ``None`` when baked at construction."""
        virtual_damping: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector] | None
        """Virtual damping D per axis, operational-frame-local [N·s/m, N·m·s/rad], shape [controlled_robot_count]. ``None`` when baked at construction."""
        virtual_mass: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector] | None
        """Virtual mass M per axis, operational-frame-local [kg, kg·m²], shape [controlled_robot_count]. ``None`` when baked at construction."""

    class Outputs:
        """Output struct returned by :meth:`~ControllerAdmittance.output`."""

        compliant_tool_pose_operational: wp.array[wp.transform] | wp.indexedarray[wp.transform]
        """Compliant tool pose, relative to the operational frame [m, unitless quaternion], shape [controlled_robot_count]."""
        compliant_tool_pose_world: wp.array[wp.transform] | wp.indexedarray[wp.transform]
        """Compliant tool pose in world frame [m, unitless quaternion], shape [controlled_robot_count]."""
        compliant_twist_operational: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector]
        """Compliant tool twist (linear, angular), components expressed in the operational frame [m/s, rad/s], shape [controlled_robot_count]."""
        displacement_operational: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector]
        """Displacement at the end of the step [m, rad], shape [controlled_robot_count]. See :attr:`Inputs.displacement_operational`."""
        displacement_twist_operational: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector]
        """Displacement twist at the end of the step [m/s, rad/s], shape [controlled_robot_count]."""

    def __init__(
        self,
        *,
        controlled_robot_count: int,
        virtual_stiffness: wp.array[wp.spatial_vector] | wp.spatial_vector | float | None,
        virtual_damping: wp.array[wp.spatial_vector] | wp.spatial_vector | float | None,
        virtual_mass: wp.array[wp.spatial_vector] | wp.spatial_vector | float | None,
        operational_frame_pose_world: wp.array[wp.transform] | wp.transform | None = _IDENTITY_TRANSFORM,
        use_desired_wrench: bool = False,
        device: Any = None,
        requires_grad: bool = False,
    ):
        self._device = wp.get_device(device)

        if isinstance(controlled_robot_count, bool) or not isinstance(controlled_robot_count, (int, np.integer)):
            raise TypeError(f"controlled_robot_count must be an int, got {type(controlled_robot_count).__name__}.")
        if controlled_robot_count < 1:
            raise ValueError(f"controlled_robot_count must be positive, got {controlled_robot_count}.")
        if requires_grad:
            raise ValueError("requires_grad=True is not supported at this time.")
        controlled_robot_count = int(controlled_robot_count)

        gains = (
            ("virtual_stiffness", virtual_stiffness),
            ("virtual_damping", virtual_damping),
            ("virtual_mass", virtual_mass),
        )
        baked_values = []
        for name, value in gains:
            _validate_gain_argument(value, name, controlled_robot_count, self._device)
            baked_values.append(_validate_non_negative_gain(value, name))
        # A fully baked axis with no positive gain has no defined response, so
        # reject it here instead of silently holding it still every step.
        if all(values is not None for values in baked_values):
            gain_sum = sum(baked_values)
            if np.any(gain_sum <= 0.0):
                raise ValueError(
                    "every axis needs a positive virtual_stiffness, virtual_damping, or virtual_mass; "
                    f"got zero on all three for axes {sorted(set(np.nonzero(gain_sum <= 0.0)[1].tolist()))}."
                )
        _validate_transform_argument(
            operational_frame_pose_world, "operational_frame_pose_world", controlled_robot_count, self._device
        )

        self._controlled_robot_count = controlled_robot_count
        self._use_desired_wrench = bool(use_desired_wrench)
        self._requires_grad = requires_grad

        self._stiffness_baked = self._bake(virtual_stiffness, wp.spatial_vector)
        self._damping_baked = self._bake(virtual_damping, wp.spatial_vector)
        self._mass_baked = self._bake(virtual_mass, wp.spatial_vector)
        self._operational_frame_baked = self._bake(operational_frame_pose_world, wp.transform)

        def _buf(dtype):
            return wp.zeros(controlled_robot_count, dtype=dtype, device=self._device)

        # Gather targets for ports bound to indexed views; a port bound to a
        # plain array is passed to the kernel as it is. Allocated up front
        # because allocation is not allowed during graph capture.
        self._input_bufs = {
            "reference_tool_pose_operational": _buf(wp.transform),
            "reference_twist_operational": _buf(wp.spatial_vector),
            "measured_wrench_world": _buf(wp.spatial_vector),
            "desired_wrench_world": _buf(wp.spatial_vector),
            "displacement_operational": _buf(wp.spatial_vector),
            "displacement_twist_operational": _buf(wp.spatial_vector),
            "operational_frame_pose_world": _buf(wp.transform),
            "virtual_stiffness": _buf(wp.spatial_vector),
            "virtual_damping": _buf(wp.spatial_vector),
            "virtual_mass": _buf(wp.spatial_vector),
        }
        self._output_bufs = {
            "compliant_tool_pose_operational": _buf(wp.transform),
            "compliant_tool_pose_world": _buf(wp.transform),
            "compliant_twist_operational": _buf(wp.spatial_vector),
            "displacement_operational": _buf(wp.spatial_vector),
            "displacement_twist_operational": _buf(wp.spatial_vector),
        }
        # Stands in for desired_wrench_world when use_desired_wrench=False.
        self._zero_wrench = _buf(wp.spatial_vector)
        self._dt_buf = wp.zeros(1, dtype=wp.float32, device=self._device)

    def _bake(self, value: Any, dtype: Any) -> wp.array | None:
        """Broadcast a scalar or Warp value, or copy a per-robot array, into a fresh buffer.

        Returns ``None`` for a live value, which is read from the input struct
        each step instead. A wp.array is already validated.
        """
        if value is None:
            return None
        if isinstance(value, (int, float)):
            v = float(value)
            value = wp.spatial_vector(v, v, v, v, v, v)
        if isinstance(value, wp.array):
            baked = wp.zeros(self._controlled_robot_count, dtype=dtype, device=self._device)
            wp.copy(baked, value)
            return baked
        return wp.full(self._controlled_robot_count, value, dtype=dtype, device=self._device)

    @property
    def controlled_robot_count(self) -> int:
        """Number of robots, the length of every port."""
        return self._controlled_robot_count

    @property
    def device(self):
        return self._device

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    def is_graphable(self) -> bool:
        return True

    def input(self) -> Inputs:
        """Return a pre-allocated :class:`Inputs` with zero-initialised arrays.

        The displacement ports start at zero, the rest state. The reference
        pose starts as an all-zero transform, which is not a valid pose — assign
        it before the first :meth:`step`.
        """
        n, d = self._controlled_robot_count, self._device

        def _port(dtype, enabled: bool = True):
            return wp.zeros(n, dtype=dtype, device=d) if enabled else None

        inputs = ControllerAdmittance.Inputs()
        inputs.reference_tool_pose_operational = _port(wp.transform)
        inputs.reference_twist_operational = _port(wp.spatial_vector)
        inputs.measured_wrench_world = _port(wp.spatial_vector)
        inputs.desired_wrench_world = _port(wp.spatial_vector, self._use_desired_wrench)
        inputs.displacement_operational = _port(wp.spatial_vector)
        inputs.displacement_twist_operational = _port(wp.spatial_vector)
        inputs.operational_frame_pose_world = _port(wp.transform, self._operational_frame_baked is None)
        inputs.virtual_stiffness = _port(wp.spatial_vector, self._stiffness_baked is None)
        inputs.virtual_damping = _port(wp.spatial_vector, self._damping_baked is None)
        inputs.virtual_mass = _port(wp.spatial_vector, self._mass_baked is None)
        return inputs

    def output(self) -> Outputs:
        """Return a pre-allocated :class:`Outputs`."""
        n, d = self._controlled_robot_count, self._device
        outputs = ControllerAdmittance.Outputs()
        outputs.compliant_tool_pose_operational = wp.zeros(n, dtype=wp.transform, device=d)
        outputs.compliant_tool_pose_world = wp.zeros(n, dtype=wp.transform, device=d)
        outputs.compliant_twist_operational = wp.zeros(n, dtype=wp.spatial_vector, device=d)
        outputs.displacement_operational = wp.zeros(n, dtype=wp.spatial_vector, device=d)
        outputs.displacement_twist_operational = wp.zeros(n, dtype=wp.spatial_vector, device=d)
        return outputs

    def step(
        self,
        *,
        inputs: Inputs,
        outputs: Outputs,
        dt: float | wp.array[wp.float32],
    ) -> None:
        """Advance every robot's admittance by one step and write its compliant pose and twist.

        Args:
            inputs: Populated :class:`Inputs` struct.
            outputs: :class:`Outputs` struct to write into. Its displacement
                fields may be bound to the same arrays as the inputs'.
            dt: Step duration [s], a positive float or a ``wp.array`` of
                shape (1,) so a captured graph can be replayed with a
                different step.
        """
        n = self._controlled_robot_count

        # Every check runs before the first launch, so a rejected call leaves
        # the outputs -- possibly aliased to the inputs -- untouched.
        live = {
            "desired_wrench_world": (self._use_desired_wrench, "use_desired_wrench"),
            "operational_frame_pose_world": (
                self._operational_frame_baked is None,
                "a live operational_frame_pose_world",
            ),
            "virtual_stiffness": (self._stiffness_baked is None, "a live virtual_stiffness"),
            "virtual_damping": (self._damping_baked is None, "a live virtual_damping"),
            "virtual_mass": (self._mass_baked is None, "a live virtual_mass"),
        }
        # A port belonging to a disabled/baked feature is never read, so
        # writing one would go unnoticed. getattr because a caller may leave
        # the field unset rather than None.
        for name, (enabled, switch) in live.items():
            if not enabled and getattr(inputs, name, None) is not None:
                raise ValueError(
                    f"inputs.{name} is set, but the controller was built without {switch}, so the value "
                    f"would be ignored."
                )

        input_ports = [
            ("reference_tool_pose_operational", wp.transform),
            ("reference_twist_operational", wp.spatial_vector),
            ("measured_wrench_world", wp.spatial_vector),
            ("displacement_operational", wp.spatial_vector),
            ("displacement_twist_operational", wp.spatial_vector),
        ]
        input_ports += [
            (name, wp.transform if name == "operational_frame_pose_world" else wp.spatial_vector)
            for name, (enabled, _) in live.items()
            if enabled
        ]
        for name, dtype in input_ports:
            _validate_array(
                array=getattr(inputs, name, None),
                name=f"inputs.{name}",
                dtype=dtype,
                shape=(n,),
                device=self._device,
                allow_indexed=True,
            )
        for name, buf in self._output_bufs.items():
            _validate_array(
                array=getattr(outputs, name, None),
                name=f"outputs.{name}",
                dtype=buf.dtype,
                shape=(n,),
                device=self._device,
                allow_indexed=True,
            )
        if isinstance(dt, wp.array):
            _validate_array(array=dt, name="dt", dtype=wp.float32, shape=(1,), device=self._device)
        elif isinstance(dt, bool) or not isinstance(dt, (int, float)) or not dt > 0.0:
            raise ValueError(f"dt must be a positive float or a wp.array of shape (1,), got {dt!r}.")

        sources = {
            name: _port_source(getattr(inputs, name), self._input_bufs[name], n, self._device)
            for name, _ in input_ports
        }

        if isinstance(dt, wp.array):
            dt_buf = dt
        else:
            self._dt_buf.fill_(float(dt))
            dt_buf = self._dt_buf

        destinations = {name: _port_destination(getattr(outputs, name), buf) for name, buf in self._output_bufs.items()}

        wp.launch(
            _admittance_step_kernel,
            dim=n,
            inputs=[
                sources.get("operational_frame_pose_world", self._operational_frame_baked),
                sources["reference_tool_pose_operational"],
                sources["reference_twist_operational"],
                sources["measured_wrench_world"],
                sources.get("desired_wrench_world", self._zero_wrench),
                sources.get("virtual_stiffness", self._stiffness_baked),
                sources.get("virtual_damping", self._damping_baked),
                sources.get("virtual_mass", self._mass_baked),
                sources["displacement_operational"],
                sources["displacement_twist_operational"],
                dt_buf,
            ],
            outputs=[
                destinations["compliant_tool_pose_operational"],
                destinations["compliant_tool_pose_world"],
                destinations["compliant_twist_operational"],
                destinations["displacement_operational"],
                destinations["displacement_twist_operational"],
            ],
            device=self._device,
        )

        for name, buf in self._output_bufs.items():
            _write_port(getattr(outputs, name), buf, n, self._device)
