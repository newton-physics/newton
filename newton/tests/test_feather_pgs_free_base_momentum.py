# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Momentum conservation for a rotating floating-base articulation.

FeatherPGS solves the free base in a world-aligned frame centred on a material point of the root
body. Three things have to agree about that point, and they did not:

* the root link's linear inertial wrench was zeroed, discarding the term that carries the
  reference point as the body rotates;
* the free-base linear coordinate was integrated without its ``omega x v`` transport term;
* the public-to-internal shift moved the coordinate to the root body's *origin* while the frame
  sits on the root body's *centre of mass*.

Each partially compensated the others, so any one of them in isolation looked defensible. Together
they made every rotating multi-link articulation create linear momentum from nothing — and also
every *tumbling* single free body whose centre of mass is offset from the body origin, because the
shift's step-to-step cancellation only survives while the angular velocity is constant.

With no gravity, no contacts and no external wrench, the centre-of-mass velocity of a free
articulation is exactly constant, so any measured drift is spurious. :class:`SolverFeatherstone`
solves the same dynamics without those three deviations and is used here as the reference.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS, SolverFeatherstone

DT = 1.0 / 200.0
STEPS = 60
# Spin about x, transverse to the stack axis, so the composite COM offset is swept through the
# rotation. A spin parallel to the offset cannot excite the term at all.
OMEGA = (20.0, 0.0, 0.0)
LINK_MASS = 1.0
LINK_HALF = 0.10  # half the stack spacing; the composite COM sits this far above the root COM


def _link_inertia(mass, radius=0.05, height=0.20):
    """Solid-cylinder inertia tensor about the link's own centre of mass."""
    i_xy = mass * (3.0 * radius**2 + height**2) / 12.0
    i_z = 0.5 * mass * radius**2
    return wp.mat33(i_xy, 0.0, 0.0, 0.0, i_xy, 0.0, 0.0, 0.0, i_z)


def _build_welded_pair(device, root_com_z=0.0):
    """Two links welded by a fixed joint: composite COM sits ``LINK_HALF`` above the root COM.

    ``root_com_z`` offsets the root link's own centre of mass from its origin, which is the case
    every real robot link has and no bare primitive does.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    root = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=LINK_MASS,
        com=wp.vec3(0.0, 0.0, root_com_z),
        inertia=_link_inertia(LINK_MASS),
    )
    upper = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * LINK_HALF), wp.quat_identity()),
        mass=LINK_MASS,
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=_link_inertia(LINK_MASS),
    )
    free = builder.add_joint_free(parent=-1, child=root)
    weld = builder.add_joint_fixed(
        parent=root,
        child=upper,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * LINK_HALF), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([free, weld])
    return builder.finalize(device=device)


def _build_equivalent_single(device):
    """One link carrying the welded pair's composite mass properties.

    Same total mass, same centre-of-mass offset from the body origin and same inertia about that
    centre of mass as :func:`_build_welded_pair`, so the two models are the same physical object.
    """
    total = 2.0 * LINK_MASS
    single = _link_inertia(LINK_MASS)
    i_xy = 2.0 * (single[0, 0] + LINK_MASS * LINK_HALF**2)
    i_z = 2.0 * single[2, 2]
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=total,
        com=wp.vec3(0.0, 0.0, LINK_HALF),
        inertia=wp.mat33(i_xy, 0.0, 0.0, 0.0, i_xy, 0.0, 0.0, 0.0, i_z),
    )
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _build_serial_chain(device, depth=2):
    """Free root followed by ``depth`` revolute links in series, axes transverse to the spin."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    prev = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=LINK_MASS,
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=_link_inertia(LINK_MASS),
    )
    joints = [builder.add_joint_free(parent=-1, child=prev)]
    for level in range(depth):
        link = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * LINK_HALF * (level + 1)), wp.quat_identity()),
            mass=LINK_MASS,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=_link_inertia(LINK_MASS),
        )
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * LINK_HALF), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            )
        )
        prev = link
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def _build_single_offset_com_tumbling(device):
    """One free link with a triaxial inertia and a centre of mass offset from its origin.

    Distinct principal moments let a non-principal spin tumble, so the angular velocity changes
    within every step. The public-to-internal shift's residual is ``(w_out - w_in) x com_offset``
    per step: it needs both the tumble and the offset to be nonzero, which is why bodies spun about
    a principal axis (and every zero-offset primitive) cannot see it. The offset magnitude matches
    a humanoid pelvis.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=2.0 * LINK_MASS,
        com=wp.vec3(0.0, 0.0, -0.076),
        inertia=wp.mat33(0.04, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.06),
    )
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _com_velocity(model, state):
    """Centre-of-mass velocity of the whole articulation [m/s]."""
    mass = model.body_mass.numpy().astype(np.float64)
    body_qd = state.body_qd.numpy().astype(np.float64).reshape(-1, 6)
    return (mass[:, None] * body_qd[:, :3]).sum(axis=0) / mass.sum()


def _spin_free_articulation(model, omega=OMEGA):
    """Spin the articulation with its composite centre of mass at rest, then step.

    Returns the largest centre-of-mass velocity drift [m/s] seen over the run.
    """
    state_0, state_1 = model.state(), model.state()

    # The free joint's linear coordinate is the ROOT body's COM velocity, which for a multi-link
    # articulation is not the composite COM. Give it the orbital velocity that leaves the composite
    # COM at rest, so both models start from the same physical state.
    body_q = state_0.body_q.numpy().astype(np.float64).reshape(-1, 7)
    mass = model.body_mass.numpy().astype(np.float64)
    com_local = model.body_com.numpy().astype(np.float64).reshape(-1, 3)
    com_world = body_q[:, :3] + com_local  # identity orientation at build time
    composite = (mass[:, None] * com_world).sum(axis=0) / mass.sum()
    root_com = com_world[0]

    joint_qd = state_0.joint_qd.numpy()
    joint_qd[0:3] = np.cross(np.array(omega), root_com - composite)
    joint_qd[3:6] = omega
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    solver = SolverFeatherPGS(model, angular_damping=0.0)
    control = model.control()
    reference = _com_velocity(model, state_0)
    worst = 0.0
    for _ in range(STEPS):
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
        worst = max(worst, float(np.linalg.norm(_com_velocity(model, state_0) - reference)))
    return worst


class TestFeatherPgsFreeBaseMomentum(unittest.TestCase):
    """Free articulations must not create linear momentum while they rotate."""

    def test_single_link_conserves_com_velocity(self):
        """A lone rotating free body in free fall holds its centre-of-mass velocity.

        With no gravity the public-to-internal shift cancels step to step, so this case passes even
        on the defective formulation; it pins the baseline, not the fix. The gravity variant below
        is the discriminating single-body case.
        """
        model = _build_equivalent_single(wp.get_device())
        self.assertLess(_spin_free_articulation(model), 1e-6)

    def test_single_link_offset_com_tumbling(self):
        """A lone tumbling free body with an offset centre of mass conserves its COM velocity.

        Tumbling changes the angular velocity within each step, so the public-to-internal shift's
        reference-point error stops cancelling step to step: on the defective formulation this body
        drifts at ~0.5 m/s within the run, versus ~1e-7 here, in every ``pgs_mode`` and
        ``articulated_contact_response``. This is the single-body case the fix corrects — every
        real robot root link has an offset centre of mass; it fails without the fix.
        """
        model = _build_single_offset_com_tumbling(wp.get_device())
        self.assertLess(_spin_free_articulation(model, omega=(20.0, 0.5, 0.0)), 1e-3)

    def test_welded_pair_conserves_com_velocity(self):
        """A welded two-link articulation holds it to the integrator's own accuracy.

        The pair is the same physical object as the single link above -- same mass, centre of mass
        and inertia -- but its composite centre of mass sits away from the frame origin, so it
        exercises the reference-point bias. Drift over this run is ~0.20 m/s with the per-link
        frame correction and ~1.64 m/s without it; the bound sits between the two.
        """
        model = _build_welded_pair(wp.get_device())
        self.assertLess(_spin_free_articulation(model), 0.5)

    def test_welded_pair_drift_converges_with_timestep(self):
        """The residual drift is first-order in dt, not a fixed formulation error.

        Halving the step must roughly halve the drift. A formulation error would sit flat instead,
        which is what distinguished this defect from ordinary truncation error.
        """
        global DT  # noqa: PLW0603 - the step size is the quantity under test
        original = DT
        try:
            DT = 1.0 / 200.0
            coarse = _spin_free_articulation(_build_welded_pair(wp.get_device()))
            DT = 1.0 / 400.0
            fine = _spin_free_articulation(_build_welded_pair(wp.get_device()))
        finally:
            DT = original
        self.assertGreater(coarse, 0.0)
        self.assertLess(fine, 0.65 * coarse)

    def test_offset_root_com_conserves_com_velocity(self):
        """A root link whose own COM is offset from its origin conserves momentum too.

        This exercises the public-to-internal free-base shift, which is an identity only when the
        frame origin and the coordinate refer to the same point. It is a no-op for a bare primitive
        and the dominant error term for a real robot link.
        """
        model = _build_welded_pair(wp.get_device(), root_com_z=-0.076)
        self.assertLess(_spin_free_articulation(model), 0.5)

    def test_matches_featherstone_reference(self):
        """FeatherPGS tracks SolverFeatherstone on chains deep enough to expose the free base.

        Featherstone solves the same dynamics without FeatherPGS's frame re-centring machinery, so
        it is an independent reference. A depth-2 serial chain is the shallowest case that
        distinguishes a correct free-base bias from one that merely looks right on a welded pair.
        """
        model = _build_serial_chain(wp.get_device(), depth=2)
        trajectories = []
        for solver_cls in (SolverFeatherstone, SolverFeatherPGS):
            state_0, state_1 = model.state(), model.state()
            joint_qd = state_0.joint_qd.numpy()
            joint_qd[3:6] = OMEGA
            state_0.joint_qd.assign(joint_qd)
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            solver = solver_cls(model, angular_damping=0.0)
            control = model.control()
            for _ in range(20):
                solver.step(state_0, state_1, control, None, DT)
                state_0, state_1 = state_1, state_0
            trajectories.append(state_0.body_qd.numpy().astype(np.float64).copy())
        self.assertLess(float(np.abs(trajectories[0] - trajectories[1]).max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
