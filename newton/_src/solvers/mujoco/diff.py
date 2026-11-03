# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bridge Newton tapes to the optional MuJoCo Warp analytic step adjoint."""

from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from dataclasses import fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

import warp as wp

if TYPE_CHECKING:
    from mujoco_warp import Data as MjWarpData
    from mujoco_warp import Model as MjWarpModel

    from ...sim import State
    from .solver_mujoco import SolverMuJoCo


def _clone_data(value: Any) -> Any:
    if isinstance(value, wp.array):
        return wp.clone(value)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: _clone_data(getattr(value, field.name)) for field in fields(value)})
    return value


@contextmanager
def _pause_tape() -> Iterator[wp.Tape | None]:
    # Warp currently exposes the active tape only through its runtime.
    runtime = wp._src.context.runtime
    tape = runtime.tape
    runtime.tape = None
    try:
        yield tape
    finally:
        runtime.tape = tape


@wp.kernel
def _copy_mass(mapping: wp.array2d[int], mass: wp.array[float], mass_out: wp.array2d[float]):
    world, body = wp.tid()
    index = mapping[world, body]
    if index >= 0:
        mass_out[world, body] = mass[index]


@wp.kernel
def _copy_dof_parameters(
    mapping: wp.array2d[int],
    armature: wp.array[float],
    damping: wp.array[float],
    actuator_damping: wp.array[float],
    armature_out: wp.array2d[float],
    damping_out: wp.array2d[float],
):
    world, dof = wp.tid()
    index = mapping[world, dof]
    if index >= 0:
        armature_out[world, dof] = armature[index]
        value = damping[index]
        if actuator_damping:
            value += actuator_damping[dof]
        damping_out[world, dof] = value


class DiffContextMuJoCo:
    """Own persistent per-output-state snapshots and a shared backward workspace."""

    def __init__(self, solver: "SolverMuJoCo"):
        self.solver = solver
        self.mjw = solver._mujoco_warp
        model = solver.mjw_model
        if model.opt.solver != self.mjw.SolverType.NEWTON:
            raise ValueError("Differentiable SolverMuJoCo requires solver='newton'.")
        if model.opt.integrator not in (self.mjw.IntegratorType.EULER, self.mjw.IntegratorType.IMPLICITFAST):
            raise ValueError("Differentiable SolverMuJoCo requires Euler or implicitfast integration.")
        if solver.enable_sleeping or model.neq or model.ntendon or model.nflex or model.na or model.nmocap:
            raise ValueError(
                "Differentiable SolverMuJoCo does not support sleeping, equality constraints, tendons, flex, "
                "actuator activation state, or mocap bodies. Use kinematic_switchable_roots=False."
            )
        self.context = self.mjw.create_backward_context(model, solver.mjw_data)
        self.template = solver.mjw_data
        self.buffers: WeakKeyDictionary[State, tuple[MjWarpData, MjWarpData, MjWarpModel]] = WeakKeyDictionary()

    def _buffers(self, state: "State") -> tuple["MjWarpData", "MjWarpData", "MjWarpModel"]:
        if state not in self.buffers:
            if self.solver.device.is_capturing:
                raise RuntimeError("Warm up the differentiable rollout once before capturing a CUDA graph.")
            with _pause_tape():
                data_in, data_out = _clone_data(self.template), _clone_data(self.template)
                for name in ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied"):
                    getattr(data_in, name).requires_grad = True
                parameters = {
                    name: wp.clone(getattr(self.solver.mjw_model, name), requires_grad=True)
                    for name in ("body_mass", "dof_armature", "dof_damping")
                }
                model = copy(self.solver.mjw_model)
                model.opt = copy(model.opt)
                model.opt.timestep = wp.clone(model.opt.timestep)
                for name, array in parameters.items():
                    setattr(model, name, array)
            self.buffers[state] = data_in, data_out, model
        return self.buffers[state]

    def prepare(self, state_in: "State", state_out: "State", dt: float) -> tuple["MjWarpModel", "MjWarpData"]:
        """Select independent storage before the solver pushes Newton inputs."""
        if state_in is state_out:
            raise ValueError("Differentiable SolverMuJoCo requires distinct input and output states.")
        if not state_in.requires_grad or not state_out.requires_grad:
            raise ValueError("Differentiable SolverMuJoCo requires states allocated with requires_grad=True.")
        solver = self.solver
        data_in, data_out, model = self._buffers(state_out)

        # Snapshot parameters and dt as well as state: another step may use different
        # values before this step's backward executes.
        model.opt.timestep.fill_(dt)
        wp.launch(
            _copy_mass,
            solver.mjc_body_to_newton.shape,
            inputs=[solver.mjc_body_to_newton, solver.model.body_mass],
            outputs=[model.body_mass],
        )
        wp.launch(
            _copy_dof_parameters,
            solver.mjc_dof_to_newton_dof.shape,
            inputs=[
                solver.mjc_dof_to_newton_dof,
                solver.model.joint_armature,
                solver.model.joint_damping,
                solver._actuator_dof_damping,
            ],
            outputs=[model.dof_armature, model.dof_damping],
        )

        # No hidden warm-start/force history may leak between independent rollouts.
        data_in.qacc_warmstart.zero_()
        data_in.ctrl.zero_()
        data_in.qfrc_applied.zero_()
        data_in.xfrc_applied.zero_()
        solver.mjw_data = data_in
        return model, data_out

    def step(self, model: "MjWarpModel", data_out: "MjWarpData") -> None:
        """Record the native step with the solver's reusable backward workspace."""
        solver = self.solver
        data_in = solver.mjw_data
        data_out.overflow.zero_()

        with _pause_tape() as tape:
            if tape is None:
                self.mjw.step(model, data_in, data_out)
            else:
                inner = wp.Tape()
                with inner:
                    self.mjw.step(model, data_in, data_out)

        if tape is not None:

            def backward():
                with wp.ScopedDevice(solver.device), self.mjw.backward_context(self.context):
                    inner.backward()

            tape.record_func(
                backward,
                [*inner.gradients, model.body_mass, model.dof_armature, model.dof_damping],
            )

        solver.mjw_data = data_out
