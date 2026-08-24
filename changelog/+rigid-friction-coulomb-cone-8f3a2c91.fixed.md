Fix rigid contact friction exceeding the Coulomb cone in `SolverSemiImplicit` and
`SolverFeatherstone`. The shared penalty kernel normalized the tangential velocity with
`wp.norm_huber`, which returns `0.5 * |v_t|^2` below its delta and so scaled the friction
bound by `2 / |v_t|`, reaching 100x the cone at 20 mm/s of slip. It now uses
`wp.norm_pseudo_huber`, which is a real norm.
