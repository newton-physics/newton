Correct angular inertial targets in local `SolverVBD` solves to avoid
attenuated or reversed torque responses from large predicted rotations.
Safeguard iterative rotation updates against crossing the angle-branch boundary.
