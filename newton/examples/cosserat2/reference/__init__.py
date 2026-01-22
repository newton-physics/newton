# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""NumPy reference implementation of Position Based Cosserat Rods.

This module provides a pure NumPy implementation of the paper:
"Position And Orientation Based Cosserat Rods"
(Kugelstadt & Schömer, 2016)
https://animation.rwth-aachen.de/publication/0550/

The implementation closely follows the pbd_rods C++ reference code,
providing a non-GPU, sequential solver for validation and testing
against the Warp-based GPU implementation in cosserat2.

Example:
    >>> from newton.examples.cosserat2.reference import (
    ...     CosseratRodNumpy,
    ...     SolverCosseratNumpy,
    ...     SolverConfig,
    ... )
    >>> import numpy as np
    >>>
    >>> # Create a straight rod
    >>> rod = CosseratRodNumpy.create_straight_rod(
    ...     num_particles=10,
    ...     start_pos=np.array([0.0, 0.0, 1.0]),
    ...     direction=np.array([1.0, 0.0, 0.0]),
    ...     segment_length=0.1,
    ...     fixed_particles=[0],
    ... )
    >>>
    >>> # Create solver
    >>> config = SolverConfig(dt=1 / 60, substeps=4)
    >>> solver = SolverCosseratNumpy(rod, config)
    >>>
    >>> # Simulate
    >>> for _ in range(100):
    ...     solver.step()
"""

from newton.examples.cosserat2.reference.cosserat_rod_numpy import CosseratRodNumpy
from newton.examples.cosserat2.reference.quaternion_ops import (
    compute_darboux_vector,
    quat_conjugate,
    quat_e3_bar,
    quat_multiply,
    quat_normalize,
    quat_rotate_e3,
    quat_rotate_vector,
    quat_rotate_vector_inv,
    quat_to_rotation_matrix,
)
from newton.examples.cosserat2.reference.solver_numpy import (
    PositionBasedCosseratRods,
    SolverConfig,
    SolverCosseratNumpy,
)

__all__ = [
    "CosseratRodNumpy",
    "PositionBasedCosseratRods",
    "SolverConfig",
    "SolverCosseratNumpy",
    "compute_darboux_vector",
    "quat_conjugate",
    "quat_e3_bar",
    "quat_multiply",
    "quat_normalize",
    "quat_rotate_e3",
    "quat_rotate_vector",
    "quat_rotate_vector_inv",
    "quat_to_rotation_matrix",
]
