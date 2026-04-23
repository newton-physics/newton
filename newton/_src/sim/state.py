# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp


def _copy_arrays(dst: object, src: object, prefix: str = "") -> None:
    """Copy ``wp.array`` attributes from ``src`` into ``dst``.

    Walks both objects' ``__dict__``, matches attributes by name, and copies
    ``wp.array`` values via ``dst_array.assign(src_array)``. Raises
    :class:`ValueError` on presence mismatch (one side has an array where
    the other does not).

    Args:
        dst: Destination object (its ``wp.array`` attributes will be overwritten).
        src: Source object.
        prefix: Prefix prepended to attribute names in error messages
            (e.g. ``"mujoco."`` for namespaced attributes).
    """
    attributes = set(dst.__dict__).union(src.__dict__)
    for attr in attributes:
        val_dst = getattr(dst, attr, None)
        val_src = getattr(src, attr, None)

        if val_dst is None and val_src is None:
            continue

        array_dst = isinstance(val_dst, wp.array)
        array_src = isinstance(val_src, wp.array)

        if not array_dst and not array_src:
            continue

        qualified = f"{prefix}{attr}"
        if val_dst is None or not array_dst:
            raise ValueError(f"State is missing array for '{qualified}' which is present in the other state.")

        if val_src is None or not array_src:
            raise ValueError(f"Other state is missing array for '{qualified}' which is present in this state.")

        val_dst.assign(val_src)


class State:
    """
    Represents the time-varying state of a :class:`Model` in a simulation.

    The State object holds all dynamic quantities that change over time during simulation,
    such as particle and rigid body positions, velocities, and forces, as well as joint coordinates.

    State objects are typically created via :meth:`newton.Model.state()` and are used to
    store and update the simulation's current configuration and derived data.
    """

    EXTENDED_ATTRIBUTES: frozenset[str] = frozenset(
        (
            "body_qdd",
            "body_parent_f",
            "mujoco:qfrc_actuator",
        )
    )
    """
    Names of optional extended state attributes that are not allocated by default.

    These can be requested via :meth:`newton.ModelBuilder.request_state_attributes` or
    :meth:`newton.Model.request_state_attributes` before calling :meth:`newton.Model.state`.

    See :ref:`extended_state_attributes` for details and usage.
    """

    @classmethod
    def validate_extended_attributes(cls, attributes: tuple[str, ...]) -> None:
        """Validate names passed to request_state_attributes().

        Only extended state attributes listed in :attr:`EXTENDED_ATTRIBUTES` are accepted.

        Args:
            attributes: Tuple of attribute names to validate.

        Raises:
            ValueError: If any attribute name is not in :attr:`EXTENDED_ATTRIBUTES`.
        """
        if not attributes:
            return

        invalid = sorted(set(attributes).difference(cls.EXTENDED_ATTRIBUTES))
        if invalid:
            allowed = ", ".join(sorted(cls.EXTENDED_ATTRIBUTES))
            bad = ", ".join(invalid)
            raise ValueError(f"Unknown extended state attribute(s): {bad}. Allowed: {allowed}.")

    def __init__(self) -> None:
        """
        Initialize an empty State object.
        To ensure that the attributes are properly allocated create the State object via :meth:`newton.Model.state` instead.
        """

        self.particle_q: wp.array | None = None
        """3D positions of particles [m], shape (particle_count,), dtype :class:`vec3`."""

        self.particle_qd: wp.array | None = None
        """3D velocities of particles [m/s], shape (particle_count,), dtype :class:`vec3`."""

        self.particle_f: wp.array | None = None
        """3D forces on particles [N], shape (particle_count,), dtype :class:`vec3`."""

        self.body_q: wp.array | None = None
        """Rigid body transforms (7-DOF) [m, unitless quaternion], shape (body_count,), dtype :class:`transform`."""

        self.body_qd: wp.array | None = None
        """Rigid body velocities (spatial) [m/s, rad/s], shape (body_count,), dtype :class:`spatial_vector`.
        First three entries: linear velocity [m/s] relative to the body's center of mass in world frame;
        last three: angular velocity [rad/s] in world frame.
        See :ref:`Twist conventions in Newton <Twist conventions>` for more information."""

        self.body_q_prev: wp.array | None = None
        """Previous rigid body transforms [m, unitless quaternion] for finite-difference velocity computation."""

        self.body_qdd: wp.array | None = None
        """Rigid body accelerations (spatial) [m/s², rad/s²], shape (body_count,), dtype :class:`spatial_vector`.
        First three entries: linear acceleration [m/s²] relative to the body's center of mass in world frame;
        last three: angular acceleration [rad/s²] in world frame.

        This is an extended state attribute; see :ref:`extended_state_attributes` for more information.
        """

        self.body_f: wp.array | None = None
        """Rigid body forces (spatial) [N, N·m], shape (body_count,), dtype :class:`spatial_vector`.
        First three entries: linear force [N] in world frame applied at the body's center of mass (COM).
        Last three: torque (moment) [N·m] in world frame.

        .. note::
            :attr:`body_f` represents an external wrench in world frame with the body's center of mass (COM) as reference point.
        """

        self.body_parent_f: wp.array | None = None
        """Parent interaction forces [N, N·m], shape (body_count,), dtype :class:`spatial_vector`.
        First three entries: linear force [N]; last three: torque [N·m].

        This is an extended state attribute; see :ref:`extended_state_attributes` for more information.

        .. note::
            :attr:`body_parent_f` represents incoming joint wrenches in world frame, referenced to the body's center of mass (COM).
        """

        self.joint_q: wp.array | None = None
        """Generalized joint position coordinates [m or rad, depending on joint type], shape (joint_coord_count,), dtype float."""

        self.joint_qd: wp.array | None = None
        """Generalized joint velocity coordinates [m/s or rad/s, depending on joint type], shape (joint_dof_count,), dtype float.
        For FREE and DISTANCE joints, the linear entries are child-COM velocity in the joint parent frame and the angular entries are angular velocity in that same frame."""

    def clear_forces(self) -> None:
        """
        Clear all force arrays (for particles and bodies) in the state object.

        Sets all entries of :attr:`particle_f` and :attr:`body_f` to zero, if present.
        """
        with wp.ScopedTimer("clear_forces", False):
            if self.particle_count:
                self.particle_f.zero_()

            if self.body_count:
                self.body_f.zero_()

    def assign(self, other: State) -> None:
        """
        Copies the array attributes of another State object into this one.

        This can be useful for swapping states in a simulation when using CUDA graphs.
        If the number of substeps is odd, the last state needs to be explicitly copied for the graph to be captured correctly:

        .. code-block:: python

            # Assume we are capturing the following simulation loop in a CUDA graph
            for i in range(sim_substeps):
                state_0.clear_forces()

                solver.step(state_0, state_1, control, contacts, sim_dt)

                # Swap states - handle CUDA graph case specially
                if sim_substeps % 2 == 1 and i == sim_substeps - 1:
                    # Swap states by copying the state arrays for graph capture
                    state_0.assign(state_1)
                else:
                    # We can just swap the state references
                    state_0, state_1 = state_1, state_0

        Args:
            other: The source State object to copy from.

        Raises:
            ValueError: If the states have mismatched attributes (one has an array allocated where the other is None).
        """
        from .model import Model  # noqa: PLC0415

        # Top-level array attributes.
        _copy_arrays(self, other)

        # Discover all AttributeNamespace containers on either state and
        # descend into each. This uniformly covers both EXTENDED_ATTRIBUTES
        # (e.g. ``mujoco.qfrc_actuator``) and custom namespaced attributes
        # registered via ``ModelBuilder.add_custom_attribute``.
        ns_self = {k: v for k, v in self.__dict__.items() if isinstance(v, Model.AttributeNamespace)}
        ns_other = {k: v for k, v in other.__dict__.items() if isinstance(v, Model.AttributeNamespace)}

        for ns_name in ns_self.keys() | ns_other.keys():
            dst_ns = ns_self.get(ns_name)
            src_ns = ns_other.get(ns_name)

            # If the namespace container is missing on one side, only raise
            # when the other side actually holds arrays inside it.
            if dst_ns is None:
                if any(isinstance(v, wp.array) for v in src_ns.__dict__.values()):
                    raise ValueError(
                        f"State is missing namespace '{ns_name}' which contains arrays in the other state."
                    )
                continue
            if src_ns is None:
                if any(isinstance(v, wp.array) for v in dst_ns.__dict__.values()):
                    raise ValueError(
                        f"Other state is missing namespace '{ns_name}' which contains arrays in this state."
                    )
                continue

            _copy_arrays(dst_ns, src_ns, prefix=f"{ns_name}.")

    @property
    def requires_grad(self) -> bool:
        """Indicates whether the state arrays have gradient computation enabled."""
        if self.particle_q:
            return self.particle_q.requires_grad
        if self.body_q:
            return self.body_q.requires_grad
        return False

    @property
    def body_count(self) -> int:
        """The number of bodies represented in the state."""
        if self.body_q is None:
            return 0
        return len(self.body_q)

    @property
    def particle_count(self) -> int:
        """The number of particles represented in the state."""
        if self.particle_q is None:
            return 0
        return len(self.particle_q)

    @property
    def joint_coord_count(self) -> int:
        """The number of generalized joint position coordinates represented in the state."""
        if self.joint_q is None:
            return 0
        return len(self.joint_q)

    @property
    def joint_dof_count(self) -> int:
        """The number of generalized joint velocity coordinates represented in the state."""
        if self.joint_qd is None:
            return 0
        return len(self.joint_qd)
