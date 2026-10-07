# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared validation for coupled neural drives."""


def _validate_network_dof_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("network_dof_count must be a positive integer")
    return value


def _network_count(num_actuators: int, network_dof_count: int) -> int:
    if num_actuators < 1 or num_actuators % network_dof_count:
        raise ValueError("Actuator DOF count must be positive and divisible by network_dof_count")
    return num_actuators // network_dof_count
