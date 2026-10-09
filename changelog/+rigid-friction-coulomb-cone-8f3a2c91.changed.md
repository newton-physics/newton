Change the `friction_smoothing` default on `SolverSemiImplicit` and `SolverFeatherstone`
from 1.0 to 1.0e-3. It is a slip speed in m/s below which friction fades out, and friction
is scaled by `|v_t| / sqrt(friction_smoothing^2 + |v_t|^2)`, so the old default left 2% of
the Coulomb limit at 20 mm/s of slip. Scenes that set it explicitly should scale their
value down by the same factor and keep it well under the slip speeds in the scene. Values
<= 0 now raise `ValueError` instead of producing NaN forces.
