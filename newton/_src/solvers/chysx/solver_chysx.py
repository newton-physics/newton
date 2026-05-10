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


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a Newton-style ``(x, y, z, w)`` quaternion into a 3x3
    rotation matrix.

    Used by the static-shape contact registration path to pull the
    world-space orientation of a Newton ``shape_transform`` into the
    three orthonormal column vectors chysx's ``BoxShape`` expects.
    """

    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz),       2.0 * (xy - wz),       2.0 * (xz + wy)],
            [      2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz),       2.0 * (yz - wx)],
            [      2.0 * (xz - wy),       2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


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
        static_contact_enabled: When ``True``, scan ``model`` for
            world-static plane (``GeoType.PLANE``) and box
            (``GeoType.BOX``) shapes (``shape_body == -1``) at
            construction time and register each one with chysx's
            static-shape contact set.  Each step the simulator runs
            a per-particle DCD against every registered primitive,
            picks the deepest penetration, and adds the resulting
            penalty contribution directly to the diagonal of the
            implicit-Euler Hessian and to the right-hand side of the
            linear system.  No off-diagonal sidecar is needed, so
            adding ground / table primitives does not invalidate the
            captured PCG graph.
        static_contact_thickness: Contact distance threshold ``h``
            for static-shape contacts (same units as positions).  A
            particle within ``h`` of any registered plane / box
            becomes a contact with penetration depth ``h - dist``.
            Use a small value comparable to the cloth's typical edge
            length (e.g. ``5e-3`` m for a 1 m square cloth at
            21x21 resolution).  Required when
            ``static_contact_enabled=True``.
        static_contact_stiffness: Per-contact penalty stiffness ``k``
            [N/m] for static-shape contacts.  Default ``1e4`` is a
            reasonable starting value for a stiff ground that
            shouldn't visibly compress; raise for an even harder
            response at the cost of PCG conditioning.
        static_contact_friction: Viscous tangential friction
            coefficient ``μ_v`` [N·s/m] for static-shape contacts.
            For every active contact the implicit-Euler Hessian
            picks up an extra block ``(μ_v / dt) * (I - n n^T)``
            that drives the tangential velocity towards zero --
            visually equivalent to dry friction without the cost of
            a Coulomb-cone projection.  Zero (default) disables
            friction; values around ``surface_density * area``
            give visibly "grippy" cloth.
        untangle_enabled: When ``True``, run the 5-vertex
            edge-face tangle pass after proximity self-collision.
            For every (edge, face) pair where the edge has actually
            crossed through the face -- detected via a per-pair
            ray-triangle intersection -- a 5-vertex penalty contact
            (two edge endpoints + three face vertices) is emitted
            and pushes the edge back to the un-crossed side along
            the cross-product of the two face normals.  Reuses the
            BVH the proximity pass already built for its broadphase,
            so requires ``self_collision_enabled=True``.  Off by
            default; cuda-cloth uses it for the ``Untangle`` case
            where the rest pose is intentionally tangled.  Diagonal-
            only contribution to the implicit-Euler linear system,
            so toggling this on / off never invalidates the captured
            PCG graph.
        untangle_thickness: Per-tangle restoring depth (world units)
            applied as a constant penalty for every detected EF
            crossing.  Larger values produce stronger restoring
            forces; cuda-cloth's Untangle case uses ``1e-2`` for a
            1 m cloth.  Required when ``untangle_enabled=True``.
        untangle_stiffness: Per-tangle penalty stiffness ``k`` [N/m].
            Defaults to ``2 * self_collision_stiffness`` default
            (``2e3``) to mirror style3d's ``stiff_ef / stiff_vf = 2.0``
            ratio (see ``newton/_src/solvers/style3d/collision/
            collision.py``).  The 2x split matters: an EF / proximity
            ratio < 1 lets the proximity term hold the cloth in the
            tangled equilibrium and untangle never recovers.
            cuda-cloth's UntangleCase uses ``k = 100`` for both -- a
            1:1 ratio is OK only when the proximity branch sees no
            real contacts (which is the case for that scene) and is
            unsafe for general drape / drop scenarios.
        untangle_max_contacts_factor: Multiplier on ``particle_count``
            used to size the device-side untangle 5-vertex contact
            buffer.  Default ``8`` matches the proximity narrow-phase
            cap, which is a loose-but-safe upper bound (the typical
            steady-state tangle count is much lower).
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
        static_contact_enabled: bool = False,
        static_contact_thickness: float = 0.0,
        static_contact_stiffness: float = 1.0e4,
        static_contact_friction: float = 0.0,
        untangle_enabled: bool = False,
        untangle_thickness: float = 0.0,
        untangle_stiffness: float = 2.0e3,
        untangle_max_contacts_factor: int = 8,
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

        # Untangle (5-vertex EF tangle) -- consumes the BVH the
        # proximity pass just built, so requires
        # ``self_collision_enabled=True``.  Same allocation pattern as
        # the proximity narrow-phase: cap defaults to 8 * particle
        # count which is loose but safe.
        if untangle_enabled and model.particle_count > 0:
            if not self_collision_enabled:
                raise ValueError(
                    "SolverChysX(untangle_enabled=True) requires "
                    "self_collision_enabled=True; the untangle pass "
                    "reuses the BVH built by the proximity self-"
                    "collision detector."
                )
            if untangle_thickness <= 0.0:
                raise ValueError(
                    "SolverChysX(untangle_enabled=True) requires "
                    "untangle_thickness > 0; cuda-cloth's Untangle "
                    "case uses ~1e-2 for a 1 m cloth."
                )
            self._sim.set_untangle_enabled(True)
            self._sim.set_untangle_thickness(float(untangle_thickness))
            self._sim.set_untangle_stiffness(float(untangle_stiffness))
            untangle_cap = max(
                int(untangle_max_contacts_factor) * int(model.particle_count),
                1024,
            )
            self._sim.set_untangle_max_contacts(untangle_cap)

        # Static-shape contact: scan the model for world-static (body == -1)
        # plane / box shapes and register them with chysx.  This wires
        # cloth ⇄ ground / table contact straight into the implicit-Euler
        # linear system (penalty contributions land on A's diagonal block
        # and on the RHS — no off-diagonal sidecar, so the captured PCG
        # graph stays valid frame-to-frame).
        if static_contact_enabled and model.particle_count > 0:
            if static_contact_thickness <= 0.0:
                raise ValueError(
                    "SolverChysX(static_contact_enabled=True) requires "
                    "static_contact_thickness > 0; pick a value comparable "
                    "to the cloth's typical edge length (e.g. 5e-3 for a "
                    "1 m square cloth at 21x21 resolution)."
                )
            self._sim.set_static_contact_thickness(float(static_contact_thickness))
            self._sim.set_static_contact_stiffness(float(static_contact_stiffness))
            self._sim.set_static_contact_friction(float(static_contact_friction))
            self._register_static_shapes_from_model()

    def _register_static_shapes_from_model(self) -> None:
        """Walk ``self.model.shape_*`` and push every world-static plane /
        box shape into chysx's :class:`StaticContactSet`.

        Newton encodes a plane as ``GeoType.PLANE`` with the surface
        normal stored along the local +Z axis of ``shape_transform``;
        the world-space normal is therefore ``rotate(quat, +Z)`` and
        the offset ``d`` such that ``dot(n, x) + d == 0`` is
        ``-dot(n, pos)``.

        Boxes (``GeoType.BOX``) carry their world-space transform in
        ``shape_transform`` (position + quaternion) and their
        half-extents in ``shape_scale``.  We extract the rotation
        matrix's three columns as the chysx ``ex / ey / ez`` axes.
        """

        from ...geometry.types import GeoType  # noqa: PLC0415

        model = self.model
        if model.shape_count == 0:
            return

        shape_type = model.shape_type.numpy()
        shape_body = (
            model.shape_body.numpy()
            if getattr(model, "shape_body", None) is not None
            else np.full(model.shape_count, -1, dtype=np.int32)
        )
        shape_transform = model.shape_transform.numpy().reshape(model.shape_count, 7)
        shape_scale = model.shape_scale.numpy().reshape(model.shape_count, 3)

        n_planes = 0
        n_boxes = 0
        for s in range(model.shape_count):
            if int(shape_body[s]) != -1:
                # Skip dynamic shapes: chysx's StaticContactSet only
                # handles primitives whose pose is constant across the
                # simulation.  Moving rigid bodies would require us to
                # re-upload the shape table every step, which we don't
                # support yet.
                continue
            t = int(shape_type[s])
            xform = shape_transform[s]
            pos = xform[:3].astype(np.float32)
            quat = xform[3:7].astype(np.float32)  # (x, y, z, w)
            R = _quat_to_matrix(quat)
            if t == int(GeoType.PLANE):
                # Newton: plane normal is the local +Z axis of
                # shape_transform.  World-space n = R · (0, 0, 1) =
                # R[:, 2].  Plane equation dot(n, x) + d == 0 with
                # x = pos on the plane gives d = -dot(n, pos).
                n = R[:, 2]
                d = float(-np.dot(n, pos))
                self._sim.add_static_plane(
                    n=np.ascontiguousarray(n, dtype=np.float32),
                    d=d,
                )
                n_planes += 1
            elif t == int(GeoType.BOX):
                self._sim.add_static_box(
                    center=np.ascontiguousarray(pos, dtype=np.float32),
                    half_ext=np.ascontiguousarray(shape_scale[s], dtype=np.float32),
                    ex=np.ascontiguousarray(R[:, 0], dtype=np.float32),
                    ey=np.ascontiguousarray(R[:, 1], dtype=np.float32),
                    ez=np.ascontiguousarray(R[:, 2], dtype=np.float32),
                )
                n_boxes += 1
            # Other shape types (sphere / capsule / mesh / ...) are
            # silently skipped for now — chysx's StaticContactSet only
            # supports plane + box primitives.

        if n_planes == 0 and n_boxes == 0:
            import warnings  # noqa: PLC0415

            warnings.warn(
                "SolverChysX(static_contact_enabled=True) found no "
                "world-static plane / box shapes in the model; cloth "
                "will fall through any other geometry.  Use "
                "ModelBuilder.add_ground_plane() / add_shape_box(body=-1, "
                "...) to register obstacles.",
                stacklevel=2,
            )

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
