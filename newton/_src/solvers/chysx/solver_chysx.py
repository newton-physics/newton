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

from collections.abc import Sequence

import numpy as np
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
        spring_stiffness: **Deprecated.**  Per-edge Hookean spring
            stiffness ``k`` [N/m].  Mass-spring edges are no longer
            assembled into the solve because BW98 stretch + shear FEM
            elements already cover that response (and at higher
            fidelity); leaving them on would just double-count
            in-plane stiffness.  Pass ``0.0`` (default); any non-zero
            value triggers a ``DeprecationWarning`` and is ignored.
        fem_stretch_stiffness: Per-area Baraff-Witkin triangle stretch
            stiffness ``k`` [N/m^2].  Default ``0.0`` disables the FEM
            membrane element; ``1e2`` is a reasonable starting value
            for soft cotton-like cloth.
        fem_shear_stiffness: Per-area Baraff-Witkin triangle *shear*
            stiffness ``k`` [N/m^2].  Internally this reuses the same
            kernels as the stretch element with the material (u, v)
            axes rotated 45 degrees, matching cuda-cloth's
            ``KernelComputeStretchShearForceAndHessianFast``.
            Default ``0.0`` disables shear; set to a value comparable
            to ``fem_stretch_stiffness`` for a fully BW98-compliant
            membrane.
        bending_stiffness: Dihedral bending stiffness ``k_bending``
            shared by every interior mesh edge (Bridson / BW98
            discrete bending energy, matching cuda-cloth's
            ``KernelComputeDihedralForcesAndHessianFast``).  Default
            ``0.0`` disables bending entirely.  Realistic cloth has
            ``k_bending`` *six to seven* orders of magnitude smaller
            than the in-plane stretch / shear stiffness — e.g.
            ``4e-5`` against ``1e2`` for cotton-like behaviour — which
            is what gives a real sheet its drape: stiff against
            stretching, very compliant against bending.
        pin_indices: Optional iterable of particle indices to pin in
            place.  Each pinned particle has its position frozen at
            its initial value (pulled from
            :attr:`Model.particle_q`) and its velocity zeroed every
            step.  Use this to attach a cloth corner to a frame, etc.
        pin_stiffness: Penalty stiffness for the pin constraint when
            running the PCG implicit-Euler step.  Larger values yield
            harder pins.
        pcg_iterations: Maximum number of PCG iterations per step.
            ``50`` is the chysx default and works well for the cloth
            scales targeted here; reduce for cheaper-but-less-accurate
            steps, or increase if the solve fails to converge.
        surface_density: Optional uniform surface density [kg/m^2].
            When set, chysx overwrites Newton's per-particle
            ``inv_mass`` with an area-weighted lumped-mass distribution
            (each triangle contributes ``surface_density * area`` split
            equally across its three vertices), matching the cuda-
            cloth finite-element convention.  Requires the model to
            have a triangle mesh.  When ``None`` (default), per-
            particle masses are taken from Newton's ``inv_mass`` as-is
            (typically uniform when the model came from
            ``add_cloth_grid(mass=...)``).
        self_collision_enabled: When ``True``, run chysx's brute-force
            vertex-face self-collision detector before each Newton
            iteration and emit penalty contact contributions into the
            linear system.  v1 only handles VF (vertex penetrating
            triangle) -- edge-edge contacts will be added once we
            replace the broadphase with a spatial hash.  Off by
            default.
        self_collision_thickness: Contact distance threshold ``h``
            (same units as positions).  A vertex within ``h`` of any
            non-incident triangle becomes a contact with penetration
            depth ``h - dist``.  cuda-cloth's twist case uses
            ``h ~ 0.2 * average_edge_length``.
        self_collision_stiffness: Per-contact penalty stiffness ``k``.
            Default ``1e3`` matches cuda-cloth's ``m_4_k = 1000``;
            scale up for stiffer contact response at the cost of PCG
            conditioning.  Hessian contributions live in a COO
            sidecar (a ``ContactSpMVOp``) the PCG solver consumes
            directly, so even very large ``k`` does not pollute the
            CSR topology between frames.
        self_collision_max_contacts_factor: Multiplier on
            ``particle_count`` used to size the device-side
            *narrow-phase* contact result buffer (the
            ``(Vec4i, ContactWeights)`` stream emitted by
            ``cull_ef_to_vfee_kernel`` / ``cull_ee_adjacent_kernel``
            after geometric filtering).  Default ``8`` is plenty for
            typical cloth (most particles touch at most a couple of
            contacts at once).  Increase if you see "contact buffer
            overflow" warnings or visible self-penetration once the
            cloth gets densely wrung up.
        self_collision_max_ef_candidates_factor: Multiplier on
            ``particle_count`` used to size the *broad-phase*
            ``(edge_id, face_id)`` candidate buffer the LBVH self-EF
            query writes into.  Broad-phase output is typically an
            order of magnitude larger than the narrow-phase contact
            count -- one edge AABB can overlap dozens of face AABBs
            in the wrung-up state, but most pairs get rejected by the
            depth / barycentric filters in
            ``cull_ef_to_vfee_kernel``.  Default ``32`` (i.e. 4x the
            narrow-phase factor) keeps the broad-phase from clipping
            without bloating the per-contact output stream.  If the
            BVH query saturates this cap (the candidate list silently
            truncates), narrow-phase will miss real contacts even
            though its own buffer has room to spare.
    """

    def __init__(
        self,
        model: Model,
        gravity: tuple[float, float, float] | None = None,
        damping: float = 0.0,
        spring_stiffness: float = 0.0,
        fem_stretch_stiffness: float = 0.0,
        fem_shear_stiffness: float = 0.0,
        bending_stiffness: float = 0.0,
        pin_indices: Sequence[int] | None = None,
        pin_stiffness: float = 1.0e6,
        pcg_iterations: int = 50,
        surface_density: float | None = None,
        self_collision_enabled: bool = False,
        self_collision_thickness: float = 0.0,
        self_collision_stiffness: float = 1.0e3,
        self_collision_max_contacts_factor: int = 8,
        self_collision_max_ef_candidates_factor: int = 32,
    ):
        super().__init__(model=model)

        if spring_stiffness != 0.0:
            import warnings  # noqa: PLC0415

            warnings.warn(
                "SolverChysX(spring_stiffness=...) is deprecated and "
                "ignored: BW98 fem_stretch + fem_shear already cover "
                "edge-stretch resistance, so adding springs would "
                "double-count in-plane stiffness.  Switch to "
                "`fem_stretch_stiffness` / `fem_shear_stiffness`.",
                DeprecationWarning,
                stacklevel=2,
            )
            spring_stiffness = 0.0

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
        self._sim.set_pcg_iterations(int(pcg_iterations))

        self._device = wp.get_device(str(model.device))

        # Bind initial particle pointers so the simulator can read
        # the rest configuration when installing mesh-derived springs.
        # These pointers will be re-bound every step() call anyway.
        if model.particle_count > 0:
            inv_mass_ptr = (
                model.particle_inv_mass.ptr
                if getattr(model, "particle_inv_mass", None) is not None
                else 0
            )
            self._sim.set_external_buffers(
                pos_ptr=model.particle_q.ptr,
                vel_ptr=model.particle_qd.ptr,
                particle_count=model.particle_count,
                inv_mass_ptr=inv_mass_ptr,
            )

        # Mesh + FEM topology — installed once at construction.
        # Newton's `model.tri_indices` is a wp.array of int32 with
        # shape (M, 3); pass straight through to chysx.
        wants_mesh = (
            fem_stretch_stiffness > 0.0
            or fem_shear_stiffness > 0.0
            or bending_stiffness > 0.0
            or surface_density is not None
        )
        if (
            wants_mesh
            and getattr(model, "tri_indices", None) is not None
            and model.tri_count > 0
        ):
            tris_np = np.ascontiguousarray(
                model.tri_indices.numpy().reshape(-1, 3), dtype=np.int32
            )
            self._sim.set_mesh(tris_np)

            # Area-weighted lumped vertex mass.  Done before the
            # constraint installs so callers reading particle_inv_mass
            # downstream of construction see the correct distribution.
            # Newton's particle_inv_mass storage is overwritten in
            # place — this is intentional, since boundary particles
            # genuinely should not have the same lumped mass as
            # interior ones.
            if (
                surface_density is not None
                and surface_density > 0.0
                and getattr(model, "particle_inv_mass", None) is not None
            ):
                self._sim.redistribute_mass_area_weighted(
                    surface_density=float(surface_density),
                    inv_mass_ptr=model.particle_inv_mass.ptr,
                    particle_count=model.particle_count,
                )

            if fem_stretch_stiffness > 0.0:
                self._sim.build_fem_stretch_from_current_positions(
                    stiffness=float(fem_stretch_stiffness)
                )
            if fem_shear_stiffness > 0.0:
                self._sim.build_fem_shear_from_current_positions(
                    stiffness=float(fem_shear_stiffness)
                )
            if bending_stiffness > 0.0:
                self._sim.build_bending_from_current_positions(
                    stiffness=float(bending_stiffness)
                )

        # Pin configuration: targets are read once from the model's
        # initial particle_q so the user can express pinning purely
        # by index.  In the PCG implicit-Euler step pin energy
        # 1/2 k |x - target|^2 contributes a k*I diagonal block to
        # the global Hessian (and a k*(target - x_tilde) entry to
        # the RHS), so a sufficiently large `pin_stiffness` produces
        # a hard pin.
        self._pin_indices_np: np.ndarray | None = None
        if pin_indices is not None and len(pin_indices) > 0 and model.particle_count > 0:
            indices_np = np.asarray(list(pin_indices), dtype=np.int32)
            if indices_np.ndim != 1:
                raise ValueError("pin_indices must be a 1-D iterable of ints")
            if (indices_np < 0).any() or (indices_np >= model.particle_count).any():
                raise ValueError(f"pin_indices out of range [0, {model.particle_count})")
            q_np = model.particle_q.numpy().reshape(-1, 3)
            targets_np = np.ascontiguousarray(q_np[indices_np], dtype=np.float32)
            self._sim.set_pins(indices_np, targets_np, float(pin_stiffness))
            self._pin_indices_np = indices_np

        # Self-collision (DCD).  Disabled by default: brute-force VF
        # broadphase is O(n_verts * n_tris) and only practical for
        # ~25x25 cloth in the v1 cut; turn it on per-example with
        # `self_collision_enabled=True` once you've sized things
        # appropriately.  The contact buffer is allocated up front so
        # the simulation loop is alloc-free.
        if self_collision_enabled and model.particle_count > 0:
            if self_collision_thickness <= 0.0:
                raise ValueError(
                    "SolverChysX(self_collision_enabled=True) requires "
                    "self_collision_thickness > 0; cuda-cloth's twist case "
                    "uses ~0.2 * average_edge_length."
                )
            self._sim.set_self_collision_enabled(True)
            self._sim.set_self_collision_thickness(float(self_collision_thickness))
            self._sim.set_self_collision_stiffness(float(self_collision_stiffness))
            # Narrow-phase cap: actual VF/EE contacts kept after geometric
            # filtering.  Lower bound 1024 so tiny meshes still get a
            # workable buffer.
            narrow_cap = max(
                int(self_collision_max_contacts_factor) * int(model.particle_count),
                1024,
            )
            # Broad-phase cap: (edge_id, face_id) pairs from the LBVH
            # self-EF query before any geometric culling.  Sized
            # independently because in dense / wrung-up configurations
            # one edge AABB overlaps dozens of face AABBs even though
            # very few survive the narrow-phase distance & barycentric
            # tests.  Always >= narrow_cap so we never starve narrow-
            # phase by truncating its input.
            ef_cap = max(
                int(self_collision_max_ef_candidates_factor) * int(model.particle_count),
                narrow_cap,
            )
            self._sim.set_self_collision_max_contacts(narrow_cap, ef_cap)

    # ---- pin animation ------------------------------------------------

    def update_pin_targets(self, targets: np.ndarray) -> None:
        """Update the world-space targets of the currently installed pins.

        ``targets`` must be a ``(n_pins, 3)`` float32 numpy array whose
        row ordering matches the ``pin_indices`` passed to the
        constructor.  This is the cheap per-frame update path -- it
        only does an H2D memcpy of the target buffer, leaving the pin
        index set (and therefore the Hessian topology and the cached
        PCG CUDA Graph) untouched.

        Use this for animations like the twist scene where pin
        positions move every frame but the pin set itself stays fixed.
        """

        if self._pin_indices_np is None:
            raise RuntimeError(
                "update_pin_targets called but no pin set was installed at "
                "construction time"
            )
        targets = np.ascontiguousarray(targets, dtype=np.float32)
        if targets.shape != (self._pin_indices_np.shape[0], 3):
            raise ValueError(
                f"targets must have shape ({self._pin_indices_np.shape[0]}, 3), "
                f"got {targets.shape}"
            )
        self._sim.update_pin_targets(targets)

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
                f"SolverChysX requires a CUDA device, got {self._device}. The chysx kernel only supports GPU execution."
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
        inv_mass_ptr = (
            self.model.particle_inv_mass.ptr
            if getattr(self.model, "particle_inv_mass", None) is not None
            else 0
        )
        self._sim.set_external_buffers(
            pos_ptr=state_out.particle_q.ptr,
            vel_ptr=state_out.particle_qd.ptr,
            particle_count=n,
            inv_mass_ptr=inv_mass_ptr,
        )

        # Issue every chysx kernel onto the current Warp stream so the
        # whole step joins any wp.ScopedCapture() the user wrapped us
        # in.  Passing a non-zero stream also disables chysx's
        # per-kernel `cudaStreamSynchronize` fallbacks (those only
        # trigger when `cuda_stream == 0`), which is what makes
        # graph capture viable.
        stream = wp.get_stream(self._device).cuda_stream
        self._sim.step(dt=float(dt), cuda_stream=stream)
