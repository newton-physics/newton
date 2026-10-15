Add experimental ``SolverVBD`` options ``rigid_axis_support_armature``, ``rigid_limit_rho_floor`` and
``rigid_external_force_mode`` that keep joint limits from overshooting on light, armature-heavy links at large time
steps, and apply external body forces and joint forces inside the rigid solve.
