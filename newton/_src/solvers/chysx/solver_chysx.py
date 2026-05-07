# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton SolverBase wrapper around the standalone ``chysx`` CUDA simulator.

This module is intentionally tiny: ``chysx`` is a real (if toy) external
physics engine living in its own folder/wheel; everything Newton-specific
stays here.

Bridging strategy
-----------------

ChysX exposes a :class:`chysx.ClothSimulator` that owns:

* a :class:`chysx.ClothMaterial` (mu, lambda, density, gravity, ...)
* a small set of externally-owned device pointer slots (pos / vel)

So the bridge looks like this:

1. **At construction**, build a ``ClothMaterial`` from Newton's
   ``Model`` (gravity, eventually density / Lamé from ``model.tri_*``)
   and copy it into the simulator with ``set_material()``.

2. **Each step**, take Newton's particle pointers (``state_out.particle_q.ptr``,
   ``state_out.particle_qd.ptr``) and assign them to the simulator with
   ``set_external_buffers()``.  No data is copied — the kernel writes
   directly back into Newton's tensors.

3. Call ``ClothSimulator.step(dt)`` and let chysx run its kernel.

This is the standard "values get copied, pointers get referenced"
contract that `SolverUIPC` and similar plug-in solvers use.
"""

from __future__ import annotations

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase


class SolverChysX(SolverBase):
    """Plug the toy ``chysx`` CUDA backend into Newton.

    Only particle dynamics are integrated (a single semi-implicit Euler
    kernel with gravity).  Rigid bodies, joints, and contacts are
    ignored.

    Args:
        model: The Newton model to simulate.
        gravity: Optional gravity override ``(gx, gy, gz)`` [m/s²].
            When ``None`` (default), reads :attr:`Model.gravity` for
            world 0.
        damping: Optional velocity damping ``[1/s]`` applied as
            ``v *= exp(-damping * dt)``.  Default 0 (no damping).
    """

    def __init__(
        self,
        model: Model,
        gravity: tuple[float, float, float] | None = None,
        damping: float = 0.0,
    ):
        super().__init__(model=model)

        # Lazy import so Newton can be imported without chysx installed.
        try:
            import chysx  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "SolverChysX requires the standalone `chysx` package. "
                "Build and install it from `ChysX/` in this repo: "
                "`uv pip install ./ChysX --no-build-isolation`."
            ) from e

        if gravity is None:
            g_np = model.gravity.numpy().reshape(-1, 3)[0]
            gravity = (float(g_np[0]), float(g_np[1]), float(g_np[2]))

        # Build the chysx-side material from Newton's model and *copy*
        # it into the simulator.  Mutating `material` after this call
        # has no effect on the simulator state.
        material = chysx.ClothMaterial()
        material.gx, material.gy, material.gz = gravity
        material.damping = float(damping)
        # Lamé / bending / density placeholders — wired in once we move
        # past the free-fall demo.

        self._sim = chysx.ClothSimulator()
        self._sim.set_material(material)

        self._device = wp.get_device(str(model.device))

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        del control, contacts  # unused

        n = self.model.particle_count
        if n == 0:
            return

        if not self._device.is_cuda:
            raise RuntimeError(
                f"SolverChysX requires a CUDA device, got {self._device}. "
                "The chysx kernel only supports GPU execution."
            )

        # Newton callers double-buffer state.  Seed state_out with
        # state_in and then mutate it in place — same convention
        # SolverUIPC uses.
        if state_out is not state_in:
            wp.copy(state_out.particle_q, state_in.particle_q)
            wp.copy(state_out.particle_qd, state_in.particle_qd)

        # Hand raw CUDA device pointers to the external engine.  The
        # chysx kernel writes directly into Newton's particle buffers;
        # no data round-trip.
        self._sim.set_external_buffers(
            pos_ptr=state_out.particle_q.ptr,
            vel_ptr=state_out.particle_qd.ptr,
            particle_count=n,
        )
        self._sim.step(dt=float(dt))
