# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver construction and native two-way coupling for the scene editor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..sim import BodyFlags
from .authoring_deformables import MPM_VOXEL_SIZE

if TYPE_CHECKING:
    from ..sim import Model
    from ..solvers.coupled.model_view import ModelView
    from ..solvers.implicit_mpm import SolverImplicitMPM
    from ..solvers.mujoco.solver_mujoco import SolverMuJoCo
    from ..solvers.solver import SolverBase
    from ..solvers.vbd import SolverVBD
    from .scene_authoring import _Runtime


def create_mujoco_solver(model: Model | ModelView, contact_max: int, *, cpu: bool = False) -> SolverMuJoCo:
    from ..solvers.mujoco.solver_mujoco import SolverMuJoCo  # noqa: PLC0415

    return SolverMuJoCo(
        model,
        use_mujoco_cpu=cpu,
        use_mujoco_contacts=cpu,
        integrator="implicitfast",
        nconmax=contact_max,
        strict_capacity=not cpu,
        kinematic_switchable_roots=False,
    )


def create_deformable_solver(
    model: Model | ModelView, *, mpm: bool, vbd_self_contact: bool
) -> SolverImplicitMPM | SolverVBD:
    if mpm:
        from ..solvers.implicit_mpm import SolverImplicitMPM  # noqa: PLC0415

        config = SolverImplicitMPM.Config(voxel_size=MPM_VOXEL_SIZE)
        if model.device.is_cpu:
            config.solver = "jacobi"
            config.max_iterations = 10
        return SolverImplicitMPM(model, config=config)

    from ..solvers.vbd import SolverVBD  # noqa: PLC0415

    return SolverVBD(
        model,
        iterations=8,
        particle_enable_self_contact=vbd_self_contact and bool(model.tri_count),
        particle_self_contact_margin=0.01,
        particle_self_contact_gap=0.015,
    )


def create_coupled_solver(
    runtime: _Runtime, *, mpm: bool, vbd_self_contact: bool
) -> tuple[SolverBase | None, SolverBase.Observables | None]:
    """Partition whole authored objects and use Newton's native proxy coupling.

    The editor supplies complete outer Newton contacts before each step. Lagged
    proxies use those same begin poses, so the native coupler can filter and
    remap the checked contacts without a second, unchecked collision pipeline.
    The returned coupled observables retain MuJoCo's capacity status in
    ``entry_observables["rigid"].overflow``.
    """
    from newton.solvers.experimental.coupled import SolverCoupledProxy  # noqa: PLC0415

    model = runtime.model
    rigid_bodies, rigid_joints = [], []
    deformable_bodies, deformable_joints = [], []
    for binding in runtime.bindings.values():
        if binding.vbd:
            deformable_bodies.extend(range(*binding.bodies))
            deformable_joints.extend(range(*binding.joints))
        else:
            rigid_bodies.extend(range(*binding.bodies))
            rigid_joints.extend(range(*binding.joints))

    def rigid_solver(view: Model | ModelView) -> SolverMuJoCo:
        return create_mujoco_solver(view, runtime.contacts.rigid_contact_max)

    def deformable_solver(view: Model | ModelView) -> SolverImplicitMPM | SolverVBD:
        return create_deformable_solver(view, mpm=mpm, vbd_self_contact=vbd_self_contact)

    has_deformables = bool(model.particle_count or deformable_bodies)
    if not has_deformables:
        if not model.joint_count:
            return None, None
        solver = rigid_solver(model)
        return solver, solver.observables({solver.ObservableKind.OVERFLOW})

    flags = model.body_flags.numpy() if rigid_bodies else ()
    if not any(
        int(flags[body]) & int(BodyFlags.DYNAMIC) and not int(flags[body]) & int(BodyFlags.KINEMATIC)
        for body in rigid_bodies
    ):
        # Stationary rigid colliders need no rigid integrator or proxy feedback.
        return deformable_solver(model), None

    destination = "mpm" if mpm else "vbd"
    solver = SolverCoupledProxy(
        model,
        entries=[
            SolverCoupledProxy.Entry(name="rigid", solver=rigid_solver, bodies=rigid_bodies, joints=rigid_joints),
            SolverCoupledProxy.Entry(
                name=destination,
                solver=deformable_solver,
                bodies=deformable_bodies,
                joints=deformable_joints,
                particles=range(model.particle_count),
                in_place=mpm,
            ),
        ],
        coupling=SolverCoupledProxy.Config(
            proxies=[
                SolverCoupledProxy.Proxy(source="rigid", destination=destination, bodies=rigid_bodies, mode="lagged")
            ],
            iterations=1,
        ),
    )
    observables = solver.observables(())
    rigid = solver.solver("rigid")
    observables.entry_observables["rigid"] = rigid.observables({rigid.ObservableKind.OVERFLOW})
    return solver, observables
