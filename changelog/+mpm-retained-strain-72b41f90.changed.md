Retain leftover elastic strain by default in the implicit MPM solver so an
incomplete solve does not discard part of the kinematic strain increment.
Existing configurations require no changes; material deformation and contact
forces can change when the solve has not fully converged.
