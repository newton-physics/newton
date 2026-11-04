# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Construct the benchmark's ordinary, unmodified public VBD solver."""

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from .contact_stiffness import apply_contact_policy


def create_solver(
    model,
    *,
    soft_contact_gap=0.01,
    full_surface_contact=False,
    contact_matching="disabled",
    **solver_options,
):
    """Return a public solver and its owned collision pipeline.

    The contact cap compares numerical coefficients for tetrahedral materials;
    their units differ from contact stiffness. It is not a fidelity criterion.
    Kinematic bodies use authored endpoint poses and ordinary internal rigid
    integration. The solver owns all collision passes and history updates.
    """
    apply_contact_policy(model, {"max_ratio": 10.0, "tet_policy": "numeric"})
    pipeline = newton.CollisionPipeline(
        model,
        soft_contact_gap=soft_contact_gap,
        enable_rigid_soft_full_surface_contact=full_surface_contact,
        contact_matching=contact_matching,
    )
    options = dict(solver_options)
    options.update(
        integrate_with_external_rigid_solver=False,
        rigid_soft_enable_dat=bool(model.particle_count),
        enable_fast_math=False,
        collision_pipeline=pipeline,
    )
    solver = SolverVBD(model, **options)
    model.collision_fidelity_solver_options = {
        key: value for key, value in options.items() if key != "collision_pipeline"
    }
    model.collision_fidelity_solver_options["rigid_compliant_alm"] = solver.rigid_compliant_alm
    if model.particle_count:
        active = (model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0
        free = active & (model.particle_mass.numpy() > 0)
        model.collision_fidelity_initial_fixed_particles = int(np.count_nonzero(~free))
    else:
        model.collision_fidelity_initial_fixed_particles = 0
    return solver, pipeline


class SceneInputs:
    """Expose authored inputs without deriving task success from a smoke run."""

    def metrics(self):
        """Return actual numerical settings and model sizes for the runner."""
        model = self.model
        physics = {
            "frame_dt_s": self.frame_dt,
            "substeps": self.substeps,
            "substep_dt_s": self.dt,
            "iterations": self.iterations,
            "soft_contact_ke": model.soft_contact_ke,
            "soft_contact_kd": model.soft_contact_kd,
            "soft_contact_mu": model.soft_contact_mu,
            "solver_options": model.collision_fidelity_solver_options,
            "contact_policy": model.collision_benchmark_contact_policy,
        }
        for name in ("closed_opening", "grip_x", "grip_z_offset", "particle_radius", "table_top"):
            if hasattr(self, name):
                physics[name] = getattr(self, name)
        return {
            "physics_parameters": physics,
            "particle_count": model.particle_count,
            "triangle_count": model.tri_count,
            "tetrahedron_count": model.tet_count,
            "body_count": model.body_count,
            "shape_count": model.shape_count,
            "all_particles_free_initially": model.collision_fidelity_initial_fixed_particles == 0
            if model.particle_count
            else None,
            "initial_fixed_particle_count": model.collision_fidelity_initial_fixed_particles,
            "fixture_notes": getattr(self, "fixture_notes", []),
        }


@wp.kernel
def _set_command_time(clock: wp.array[float], value: float):
    clock[0] = value


def set_command_time(clock, time_s):
    """Set the scene's device command clock without touching physical state."""
    wp.launch(_set_command_time, dim=1, inputs=[clock, float(time_s)], device=clock.device)
