# Area-scaled elastic column friction

`elastic_coulomb` sets each column's tangential stiffness to
`G_eq A / L`, where `G_eq` is the shoe material's equilibrium shear modulus,
`A` is the column area, and `L` is its rest length. The elastic bristle uses a
Coulomb cap and has no dashpot or relaxation state. `column_maxwell` keeps the
same equilibrium stiffness and adds a Maxwell branch using
`(G_inst - G_eq) A / L` and the material relaxation time. This makes the
unsaturated force response scale with the modeled foam geometry and preserves
the summed stiffness when a footprint is refined into smaller columns.

The shoe, Cartesian leg engine, and generic `FoundationConfig` use
`elastic_coulomb` by default. Explicitly selecting `maxwell` retains configured
per-column stiffness and viscosity. The legacy anchored spring remains
available.

This is a shear estimate derived from the fitted normal material, not an
independent outsole shear calibration. The linear shear-layer approximation
can predict large displacements during sliding. Earlier `column_maxwell` trial
replays reduced the peak AP release rate while retaining nearly the same
braking impulse and zero-crossing time. Those runs do not validate the new
elastic default. Large release/ringing transients remain, and neither model
establishes an independent calibration of the braking-force transient. Contact
velocity handling and reported forces are unchanged.
