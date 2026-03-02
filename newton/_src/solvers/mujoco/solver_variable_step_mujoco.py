# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ...utils.benchmark import event_scope
from .solver_mujoco import SolverMuJoCo


class SolverVariableStepMuJoCo(SolverMuJoCo):
    """
    A variant of :class:`SolverMuJoCo` that supports per-world variable time steps.

    This solver is GPU-only (``mujoco_warp`` backend) and always uses separate MuJoCo
    worlds, making it well-suited for scenarios where different simulated environments
    need to advance at different rates—such as for error-controlled integration.

    The key difference from :class:`SolverMuJoCo` is the :meth:`step` method, which
    accepts ``dt`` as either a scalar ``float`` (applied uniformly to all worlds) or a
    ``wp.array`` of shape ``[world_count]`` (one timestep per world).

    .. note::

        - ``use_mujoco_cpu`` is not supported; this solver always uses the
          ``mujoco_warp`` GPU backend.
        - ``separate_worlds`` is always ``True``; each Newton world is mapped to an
          independent MuJoCo world.

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverVariableStepMuJoCo(model)

        # Per-world timesteps
        dts = wp.array([0.01, 0.005], dtype=wp.float32)

        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dts)
            state_in, state_out = state_out, state_in
    """

    def __init__(
        self,
        model: Model,
        *,
        njmax: int | None = None,
        nconmax: int | None = None,
        iterations: int | None = None,
        ls_iterations: int | None = None,
        ccd_iterations: int | None = None,
        sdf_iterations: int | None = None,
        sdf_initpoints: int | None = None,
        solver: int | str | None = None,
        integrator: int | str | None = None,
        cone: int | str | None = None,
        jacobian: int | str | None = None,
        impratio: float | None = None,
        tolerance: float | None = None,
        ls_tolerance: float | None = None,
        ccd_tolerance: float | None = None,
        density: float | None = None,
        viscosity: float | None = None,
        wind: tuple | None = None,
        magnetic: tuple | None = None,
        disable_contacts: bool = False,
        update_data_interval: int = 1,
        save_to_mjcf: str | None = None,
        ls_parallel: bool = False,
        use_mujoco_contacts: bool = True,
        include_sites: bool = True,
        skip_visual_only_geoms: bool = True,
    ):
        """
        Args:
            model: The model to be simulated.
            njmax: Maximum number of constraints per world. If None, a default value is estimated from the initial state.
            nconmax: Number of contact points per world. If None, a default value is estimated from the initial state.
            iterations: Number of solver iterations. If None, uses model custom attribute or MuJoCo's default (100).
            ls_iterations: Number of line search iterations for the solver. If None, uses model custom attribute or MuJoCo's default (50).
            ccd_iterations: Maximum CCD iterations. If None, uses model custom attribute or MuJoCo's default (35).
            sdf_iterations: Maximum SDF iterations. If None, uses model custom attribute or MuJoCo's default (10).
            sdf_initpoints: Number of SDF initialization points. If None, uses model custom attribute or MuJoCo's default (40).
            solver: Solver type. Can be ``"cg"`` or ``"newton"``, or their corresponding MuJoCo integer constants.
            integrator: Integrator type. Can be ``"euler"``, ``"rk4"``, or ``"implicitfast"``, or their corresponding MuJoCo integer constants.
            cone: The type of contact friction cone. Can be ``"pyramidal"`` or ``"elliptic"``.
            jacobian: Jacobian computation method. Can be ``"dense"``, ``"sparse"``, or ``"auto"``.
            impratio: Frictional-to-normal constraint impedance ratio. If None, uses model custom attribute or MuJoCo's default (1.0).
            tolerance: Solver tolerance for early termination.
            ls_tolerance: Line search tolerance for early termination.
            ccd_tolerance: Continuous collision detection tolerance.
            density: Medium density for lift and drag forces [kg/m³].
            viscosity: Medium viscosity for lift and drag forces [Pa·s].
            wind: Wind velocity vector ``(x, y, z)`` [m/s].
            magnetic: Global magnetic flux vector ``(x, y, z)`` [T].
            disable_contacts: If True, disable contact computation in MuJoCo.
            update_data_interval: Frequency (in simulation steps) at which to update the MuJoCo Data object from the Newton state.
            save_to_mjcf: Optional path to save the generated MJCF model file.
            ls_parallel: If True, enable parallel line search in MuJoCo.
            use_mujoco_contacts: If True, use the MuJoCo contact solver. If False, Newton contacts must be passed via the step function.
            include_sites: If True, Newton shapes marked with ``ShapeFlags.SITE`` are exported as MuJoCo sites.
            skip_visual_only_geoms: If True, visualization-only geometries are excluded from the exported MuJoCo spec.
        """
        super().__init__(
            model,
            separate_worlds=True,
            njmax=njmax,
            nconmax=nconmax,
            iterations=iterations,
            ls_iterations=ls_iterations,
            ccd_iterations=ccd_iterations,
            sdf_iterations=sdf_iterations,
            sdf_initpoints=sdf_initpoints,
            solver=solver,
            integrator=integrator,
            cone=cone,
            jacobian=jacobian,
            impratio=impratio,
            tolerance=tolerance,
            ls_tolerance=ls_tolerance,
            ccd_tolerance=ccd_tolerance,
            density=density,
            viscosity=viscosity,
            wind=wind,
            magnetic=magnetic,
            use_mujoco_cpu=False,
            disable_contacts=disable_contacts,
            update_data_interval=update_data_interval,
            save_to_mjcf=save_to_mjcf,
            ls_parallel=ls_parallel,
            use_mujoco_contacts=use_mujoco_contacts,
            include_sites=include_sites,
            skip_visual_only_geoms=skip_visual_only_geoms,
        )

    @event_scope
    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float | wp.array,
    ):
        """Advance the simulation by one step.

        Args:
            state_in: Input state.
            state_out: Output state (modified in place).
            control: Control inputs.
            contacts: Contact data (used when ``use_mujoco_contacts`` is False).
            dt: Time step [s]. Can be a scalar ``float`` applied to all worlds, or a
                ``wp.array`` of shape ``[world_count]`` with per-world timesteps.
        """
        self._enable_rne_postconstraint(state_out)
        self._apply_mjc_control(self.model, state_in, control, self.mjw_data)
        if self.update_data_interval > 0 and self._step % self.update_data_interval == 0:
            self._update_mjc_data(self.mjw_data, self.model, state_in)

        if isinstance(dt, float):
            self.mjw_model.opt.timestep.fill_(dt)
        else:
            self.mjw_model.opt.timestep = dt

        with wp.ScopedDevice(self.model.device):
            if self.mjw_model.opt.run_collision_detection:
                self._mujoco_warp_step()
            else:
                self._convert_contacts_to_mjwarp(self.model, state_in, contacts)
                self._mujoco_warp_step()

        self._update_newton_state(self.model, state_out, self.mjw_data)
        self._step += 1
        return state_out