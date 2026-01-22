# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cosserat rod kernels for XPBD constraint solving."""

from newton.examples.cosserat2.kernels.utilities import (
    quat_rotate_e3,
    quat_e3_bar,
    compute_darboux_vector,
    zero_vec3_kernel,
    zero_quat_kernel,
    apply_particle_corrections_kernel,
    apply_quaternion_corrections_kernel,
)

from newton.examples.cosserat2.kernels.integration import (
    integrate_particles_kernel,
    update_velocities_kernel,
    apply_velocity_damping_kernel,
)

from newton.examples.cosserat2.kernels.stretch_shear import (
    solve_stretch_shear_constraint_kernel,
    compute_stretch_constraint_data_kernel,
    assemble_stretch_tridiagonal_system_kernel,
    assemble_stretch_global_system_kernel,
    apply_stretch_corrections_kernel,
)

from newton.examples.cosserat2.kernels.bend_twist import (
    solve_bend_twist_constraint_kernel,
    solve_bend_twist_with_strain_rate_damping_kernel,
    solve_bend_twist_with_dahl_friction_kernel,
    compute_bend_constraint_data_kernel,
    assemble_bend_global_system_kernel,
    apply_bend_corrections_kernel,
    compute_current_kappa_kernel,
)

from newton.examples.cosserat2.kernels.direct_solvers import (
    thomas_solve_kernel,
    cholesky_solve_kernel,
    TILE,
    BLOCK_DIM,
)

from newton.examples.cosserat2.kernels.collision import (
    solve_ground_collision_kernel,
    compute_static_tri_aabbs_kernel,
    compute_triangle_normal,
    collide_particles_vs_triangles_bvh_kernel,
)

from newton.examples.cosserat2.kernels.visualization import (
    compute_director_lines_kernel,
    update_rest_darboux_kernel,
    update_tip_rest_darboux_kernel,
)

__all__ = [
    # Utilities
    "quat_rotate_e3",
    "quat_e3_bar",
    "compute_darboux_vector",
    "zero_vec3_kernel",
    "zero_quat_kernel",
    "apply_particle_corrections_kernel",
    "apply_quaternion_corrections_kernel",
    # Integration
    "integrate_particles_kernel",
    "update_velocities_kernel",
    "apply_velocity_damping_kernel",
    # Stretch/Shear
    "solve_stretch_shear_constraint_kernel",
    "compute_stretch_constraint_data_kernel",
    "assemble_stretch_tridiagonal_system_kernel",
    "assemble_stretch_global_system_kernel",
    "apply_stretch_corrections_kernel",
    # Bend/Twist
    "solve_bend_twist_constraint_kernel",
    "solve_bend_twist_with_strain_rate_damping_kernel",
    "solve_bend_twist_with_dahl_friction_kernel",
    "compute_bend_constraint_data_kernel",
    "assemble_bend_global_system_kernel",
    "apply_bend_corrections_kernel",
    "compute_current_kappa_kernel",
    # Direct solvers
    "thomas_solve_kernel",
    "cholesky_solve_kernel",
    "TILE",
    "BLOCK_DIM",
    # Collision
    "solve_ground_collision_kernel",
    "compute_static_tri_aabbs_kernel",
    "compute_triangle_normal",
    "collide_particles_vs_triangles_bvh_kernel",
    # Visualization
    "compute_director_lines_kernel",
    "update_rest_darboux_kernel",
    "update_tip_rest_darboux_kernel",
]
