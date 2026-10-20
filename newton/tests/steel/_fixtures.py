# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scenes, solver settings and helpers shared by the SolverSteel test modules."""

import warnings
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.friction_patches import (
    _FrictionPatchState,
)
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_FRICTION,
)
from newton.solvers import SolverBase, SolverSteel

CONTACT_MAX = 8192


def _heap(device, nx=6, ny=6, nz=3, pitch=0.042, half=0.02, kinematic_tray=False):
    """A dense heap of touching boxes on the ground, optionally on one kinematic tray."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.rigid_gap = 0.002
    builder.add_ground_plane()
    if kinematic_tray:
        # Every box of the bottom layer rests on the tray, a hub of their contact units.
        tray = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.01), wp.quat_identity()), is_kinematic=True)
        builder.add_shape_box(tray, hx=nx * pitch, hy=ny * pitch, hz=0.01)
    z0 = 0.04 if kinematic_tray else 0.021
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                body = builder.add_body(
                    xform=wp.transform(
                        wp.vec3((ix - nx / 2) * pitch, (iy - ny / 2) * pitch, z0 + iz * (2 * half + 0.001)),
                        wp.quat_identity(),
                    )
                )
                builder.add_shape_box(body, hx=half, hy=half, hz=half)
    model = builder.finalize(device=device)
    # Sizes the solver's contact scratch, so it is set before the solver is built.
    model.rigid_contact_max = CONTACT_MAX
    return model


def _build_four_bar():
    """Build a planar four-bar: two grounded revolute chains closed by a BALL loop joint.

    Ground pivots at x=0 and x=0.4; crank and rocker hang down 0.2 m; the coupler
    connects the crank tip to the rocker tip through the loop closure. The crank is
    position-driven; the rocker is undriven, so any coherent motion it does comes from
    the closed loop.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    z0 = 0.6

    crank = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, z0 - 0.1), wp.quat_identity()))
    b.add_shape_box(crank, hx=0.02, hy=0.02, hz=0.1)
    j_crank = b.add_joint_revolute(
        parent=-1,
        child=crank,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, z0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
    )
    coupler = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, z0 - 0.2), wp.quat_identity()))
    b.add_shape_box(coupler, hx=0.2, hy=0.02, hz=0.02)
    j_coupler = b.add_joint_revolute(
        parent=crank,
        child=coupler,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    rocker = b.add_link(xform=wp.transform(wp.vec3(0.4, 0.0, z0 - 0.1), wp.quat_identity()))
    b.add_shape_box(rocker, hx=0.02, hy=0.02, hz=0.1)
    j_rocker = b.add_joint_revolute(
        parent=-1,
        child=rocker,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.4, 0.0, z0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
    )
    b.add_articulation([j_crank, j_coupler, j_rocker], label="four_bar")

    # Loop closure: coupler tip pinned to rocker tip (a trailing BALL loop joint,
    # matching how the MJCF importer closes `connect` equalities).
    b.add_joint_ball(
        parent=coupler,
        child=rocker,
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
    )

    # Drive the crank only.
    b.joint_target_ke[0] = 50.0
    b.joint_target_kd[0] = 5.0
    b.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
    return b


def _loop_anchor_gap(model, state) -> float:
    """World-space distance between the loop joint's parent and child anchors [m]."""
    bq = state.body_q.numpy().reshape(-1, 7).astype(np.float64)
    jt = model.joint_type.numpy()
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()
    Xp = model.joint_X_p.numpy()
    Xc = model.joint_X_c.numpy()
    j = int(np.nonzero(jt == int(newton.JointType.BALL))[0][-1])

    def anchor(body, X):
        t = wp.transform(wp.vec3(*bq[body, :3]), wp.quat(*bq[body, 3:]))
        a = wp.transform(wp.vec3(*X[:3]), wp.quat(*X[3:]))
        w = wp.transform_multiply(t, a)
        return np.array([w.p[0], w.p[1], w.p[2]])

    return float(np.linalg.norm(anchor(int(jp[j]), Xp[j]) - anchor(int(jc[j]), Xc[j])))


_CHILD_ANCHORS = ((0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0))


_REL_P = np.array([0.0, 0.0, -0.2])


def _build_carried_load(*, enabled: bool = True, world_parent: bool = False):
    """A kinematic carrier holding a dynamic box through three BALL loop joints.

    Three non-collinear point closures pin all six relative degrees of freedom, so the
    load must follow the carrier as if welded at ``_REL_P`` below it. With
    ``world_parent`` the carrier is the world and the anchors are world points.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    carrier_p = np.array([0.0, 0.0, 1.0])
    carrier = -1
    if not world_parent:
        carrier = b.add_body(xform=wp.transform(wp.vec3(*carrier_p), wp.quat_identity()), is_kinematic=True)
        b.add_shape_box(carrier, hx=0.03, hy=0.03, hz=0.03)
    load = b.add_body(xform=wp.transform(wp.vec3(*(carrier_p + _REL_P)), wp.quat_identity()))
    b.add_shape_box(load, hx=0.04, hy=0.04, hz=0.04, cfg=newton.ModelBuilder.ShapeConfig(density=500.0))
    joints = []
    # The closures intentionally parallel the load's free joint and each other.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*another joint already connects these bodies", category=UserWarning
        )
        for c in _CHILD_ANCHORS:
            p = (_REL_P if not world_parent else carrier_p + _REL_P) + np.array(c)
            joints.append(
                b.add_joint_ball(
                    parent=carrier,
                    child=load,
                    parent_xform=wp.transform(wp.vec3(*p), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(*c), wp.quat_identity()),
                    enabled=enabled,
                )
            )
    return b, carrier, load, joints


def run_fixture(
    *,
    device,
    articulated,
    enabled,
    steps=200,
    dt=0.005,
    iterations=8,
    stock=False,
    lateral_force=0.0,
    friction_scale=1.0,
    stiffness=3000.0,
    height=0.05,
    kinematic=False,
    restitution=0.0,
    solver_options=None,
):
    """Run sphere/plane contacts with prescribed material arrays, one physical step per tick."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.rigid_gap = 0.005
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
        xform = wp.transform(wp.vec3(0, 0, height), wp.quat_identity())
        if articulated:
            body = builder.add_link(xform=xform)
            if lateral_force:
                joint = builder.add_joint_d6(
                    parent=-1,
                    child=body,
                    parent_xform=xform,
                    linear_axes=[
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X),
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z),
                    ],
                )
            else:
                joint = builder.add_joint_prismatic(parent=-1, child=body, axis=newton.Axis.Z, parent_xform=xform)
            builder.add_articulation([joint])
        else:
            body = builder.add_body(xform=xform)
        builder.add_shape_sphere(
            body,
            radius=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.3 / (4 / 3 * np.pi * 0.05**3), mu=0.5, restitution=restitution
            ),
        )
        model = builder.finalize()
        if kinematic:
            flags = model.body_flags.numpy()
            flags[body] |= int(newton.BodyFlags.KINEMATIC)
            model.body_flags.assign(flags)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
        model.rigid_contact_max = 32
        contacts = pipeline.contacts()
        for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
            setattr(contacts, name, wp.zeros(32, dtype=float))
        extra = {} if stock else {"contact_compliance": enabled}
        options = dict(
            # Isolate the normal material law from positional patch friction.
            friction_anchor_beta=0.0,
            iterations=iterations,
            velocity_iterations=0,
            articulation_constraint_max=32,
            free_body_constraint_max=32,
            baumgarte=0.05,
            **extra,
        )
        options.update(solver_options or {})
        solver = SolverSteel(model, solve_mode="matrix_free", **options)
        state, next_state = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        trace = []
        seen_paths = set()
        control = model.control()
        for step in range(steps):
            state.clear_forces()
            pipeline.collide(state, contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            contacts.rigid_contact_stiffness.fill_(stiffness)
            contacts.rigid_contact_damping.fill_(20.0)
            contacts.rigid_contact_friction.fill_(friction_scale)
            if lateral_force and step >= steps // 2:
                control.joint_f.assign(np.array([lateral_force, 0.0], dtype=np.float32))
            solver.step(state, next_state, control, contacts, dt)
            state, next_state = next_state, state
            seen_paths.update(solver._contact_path.numpy()[:count].tolist())
            trace.append(state.body_q.numpy()[body].copy())
        return np.asarray(trace), seen_paths, solver, float(model.body_mass.numpy()[body])


# Persistent friction patches (the default friction law).
PATCH_OPTIONS = {"friction_anchor_beta": 0.2}


def set_torsion_radius(model, radius, shapes=None):
    """Set ``steel:min_torsional_patch_radius`` [m] of ``shapes`` (every shape for ``None``), zero elsewhere.

    ``radius`` may also be one value per selected shape.
    """
    radii = np.zeros(model.shape_count, dtype=np.float32)
    radii[slice(None) if shapes is None else list(shapes)] = radius
    namespace = getattr(model, "steel", None)
    if namespace is not None and getattr(namespace, "min_torsional_patch_radius", None) is not None:
        namespace.min_torsional_patch_radius.assign(radii)
    else:
        model.add_attribute(
            "min_torsional_patch_radius",
            wp.array(radii, dtype=wp.float32, device=model.device),
            frequency=newton.Model.AttributeFrequency.SHAPE,
            namespace="steel",
        )


def fixture(
    radius=None,
    *,
    device="cuda:0",
    mu=0.5,
    closing=0.1,
    spin=1.0,
    sliding=0.0,
    sat=False,
    separation=0.0,
    center_only=False,
    center_count=1,
    restitution=0.0,
    dt=0.0025,
    row_limit=None,
    model_arrays=None,
    step=True,
    torsion_shapes=None,
    **solver_overrides,
):
    """Solve two equal articulated rectangular pads touching face to face.

    ``model_arrays`` maps model attribute names to values assigned before the solver is built.
    ``radius`` is the torsion radius of ``torsion_shapes`` (both pads for ``None``); ``()`` builds the
    torsion solver and then clears every radius, so no contact is admitted. With ``step=False`` the solver is returned before its first step, without a result.
    """
    b = newton.ModelBuilder(gravity=(0, 0, 0))
    for side in (-1, 1):
        pose = wp.transform(wp.vec3(0, 0, side * (0.025 + separation / 2)), wp.quat_identity())
        body = b.add_link(
            xform=pose, mass=0.3, inertia=wp.mat33(np.diag([0.00012, 0.00012, 0.00008]).astype(np.float32))
        )
        axes = [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(*a)) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        joint = b.add_joint_d6(-1, body, parent_xform=pose, linear_axes=axes, angular_axes=axes)
        b.add_articulation([joint])
        b.add_shape_box(
            body,
            hx=0.02,
            hy=0.015,
            hz=0.025,
            cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=mu, restitution=restitution),
        )
    model = b.finalize(device=device)
    for name, values in (model_arrays or {}).items():
        getattr(model, name).assign(values)
    a, z = model.state(), model.state()
    a.joint_qd.assign(np.array([sliding, 0, closing, 0, 0, spin, -sliding, 0, -closing, 0, 0, -spin], np.float32))
    newton.eval_fk(model, a.joint_q, a.joint_qd, a)
    pipeline_options = {"box_box_sat": True} if sat else {}
    pipeline = newton.CollisionPipeline(
        model,
        contact_matching="latest",
        reduce_contacts=False,
        broad_phase="nxn",
        rigid_contact_max=64,
        **pipeline_options,
    )
    contacts = pipeline.contacts()
    pipeline.collide(a, contacts)
    if center_only and int(contacts.rigid_contact_count.numpy()[0]) > 0:
        # Deliberately represent the known planar disk by one central witness:
        # this isolates the new material spin law from native point lever arms.
        shapes = [contacts.rigid_contact_shape0.numpy()[0], contacts.rigid_contact_shape1.numpy()[0]]
        poses = a.body_q.numpy()
        for side, name in enumerate(("rigid_contact_point0", "rigid_contact_point1")):
            values = getattr(contacts, name).numpy()
            body = model.shape_body.numpy()[shapes[side]]
            values[:center_count] = -poses[body, :3]
            values[:center_count, 2] += np.sign(poses[body, 2]) * separation / 2
            getattr(contacts, name).assign(values)
        contacts.rigid_contact_count.assign(np.array([center_count], np.int32))
    kwargs = {
        # Keep this material-law control on the same point-friction baseline
        # whether the optional spin radius is zero or positive.
        "friction_anchor_beta": 0.0,
        "iterations": 128,
        "velocity_iterations": 0,
        "baumgarte": 0.05,
        "cfm": 0,
        "contact_regularization": 0,
        "enable_joint_velocity_limits": True,
        "contact_warmstart": False,
        "articulation_constraint_max": 64,
        "free_body_constraint_max": 16,
    }
    admit_none = torsion_shapes == ()
    if radius is not None:
        set_torsion_radius(model, radius, None if admit_none else torsion_shapes)
    kwargs.update(solver_overrides)
    if kwargs["friction_anchor_beta"] is None:
        # Use the constructor default (persistent friction patches).
        del kwargs["friction_anchor_beta"]
    solver = SolverSteel(model, solve_mode="matrix_free", **kwargs)
    solver._rigid_body_angular_damping.zero_()
    if row_limit is not None:
        solver._articulation_constraint_max = row_limit
    if admit_none:
        # Torsion stays enabled, but no shape pair has a radius left to admit.
        set_torsion_radius(model, 0.0)
    if not step:
        return None, solver, model, a, contacts
    observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
    solver.step(a, z, model.control(), contacts, dt, observables=observables)
    result = {name: getattr(z, name).numpy() for name in ("body_q", "body_qd", "joint_q", "joint_qd")}
    result["contact_f"] = observables.contact_f.numpy()
    result.update(
        {
            name: getattr(solver, f"_{name}").numpy()
            for name in ("impulses", "row_type", "row_parent", "row_mu", "J_world", "Y_world", "v_hat", "v_out")
        }
    )
    result["count"] = solver._constraint_count.numpy()
    return result, solver, model, a, contacts


_MU_JAW, _MU_BOX = 5.0, 0.5


_BOX_HALF, _JAW_HALF_T, _GAP0 = 0.02, 0.005, 0.001


def _build_v_jaws(tilt_deg: float, geometry: str = "box", device=None):
    """Fixed base and two independently driven prismatic jaws pinching a free 0.1 kg object.

    Both jaws are driven toward the object with equal gains and mirrored targets. Their
    faces tilt toward +z by ``tilt_deg``, so the two normal rows' depenetration biases
    share a +z tangential component: the geometry that leaks tangential drift through
    velocity-only friction rows.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    b.rigid_gap = 0.005
    base = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), label="base")
    b.add_shape_box(
        base,
        hx=0.05,
        hy=0.05,
        hz=0.005,
        cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, has_shape_collision=False),
    )
    root = b.add_joint_fixed(-1, base, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
    jaw_cfg = newton.ModelBuilder.ShapeConfig(density=2000.0, mu=_MU_JAW)
    x0 = _BOX_HALF + _JAW_HALF_T + _GAP0
    jaws, joints = [], []
    for name, sign in (("jaw_L", -1.0), ("jaw_R", 1.0)):
        body = b.add_link(xform=wp.transform(wp.vec3(sign * x0, 0.0, 0.5), wp.quat_identity()), label=name)
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(-sign * np.radians(tilt_deg)))
        b.add_shape_box(body, hx=_JAW_HALF_T, hy=0.02, hz=0.02, xform=wp.transform(wp.vec3(0.0), rot), cfg=jaw_cfg)
        joints.append(
            b.add_joint_prismatic(
                base,
                body,
                parent_xform=wp.transform(wp.vec3(sign * x0, 0.0, 0.0), wp.quat_identity()),
                axis=newton.Axis.X,
                limit_lower=-0.05,
                limit_upper=0.05,
                target_pos=-0.01 * sign,
                target_ke=100.0,
                target_kd=10.0,
            )
        )
        jaws.append(body)
    b.add_articulation([root, *joints], label="gripper")
    for dof in range(len(b.joint_effort_limit)):
        b.joint_effort_limit[dof] = 10.0
    box = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), label="box")
    if geometry == "sphere":
        b.add_shape_sphere(
            box,
            radius=_BOX_HALF,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.1 / (4.0 / 3.0 * np.pi * _BOX_HALF**3), mu=_MU_BOX),
        )
    elif geometry == "capsule":
        b.add_shape_capsule(
            box,
            radius=_BOX_HALF,
            half_height=_BOX_HALF,
            xform=wp.transform(wp.vec3(0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2)),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.1 / (np.pi * _BOX_HALF**2 * 2 * _BOX_HALF + 4.0 / 3.0 * np.pi * _BOX_HALF**3), mu=_MU_BOX
            ),
        )
    else:
        b.add_shape_box(
            box,
            hx=_BOX_HALF,
            hy=_BOX_HALF,
            hz=_BOX_HALF,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.1 / (2 * _BOX_HALF) ** 3, mu=_MU_BOX),
        )
    return b.finalize(device=device), jaws, box


# Few iterations and a soft normal law leave the velocity-only point friction rows
# under-converged, so the tilted pinch leaks drift unless anchors correct it.
_SQUEEZE_SOLVER = {
    "iterations": 8,
    "velocity_iterations": 0,
    "baumgarte": 0.05,
    "contact_regularization": 0.1,
    "contact_friction_gap_max": 0.001,
    "articulation_constraint_max": 64,
}


# Steel and its patch kernels are exercised on CUDA, the solver's only device.
_DEVICE = "cuda:0"


def _patch_fixture(
    points, *, shape0=None, normals=None, materials=(0.5, 0.5, 0.5), shape_bodies=(0, 1, 0), device=_DEVICE
):
    """Construct patch geometry without a collision matcher or a solver step."""
    n = len(points)
    body_count = max(shape_bodies) + 1
    model = SimpleNamespace(
        device=wp.get_device(device),
        body_count=body_count,
        body_world=wp.zeros(body_count, dtype=int, device=device),
        shape_body=wp.array(shape_bodies, dtype=int, device=device),
        shape_collision_radius=wp.array([0.2] * len(shape_bodies), dtype=float, device=device),
        shape_transform=wp.array([wp.transform_identity()] * len(shape_bodies), dtype=wp.transform, device=device),
        shape_scale=wp.ones(len(shape_bodies), dtype=wp.vec3, device=device),
        shape_type=wp.zeros(len(shape_bodies), dtype=int, device=device),
        shape_source_ptr=wp.zeros(len(shape_bodies), dtype=wp.uint64, device=device),
        shape_margin=wp.zeros(len(shape_bodies), dtype=float, device=device),
        shape_gap=wp.zeros(len(shape_bodies), dtype=float, device=device),
        shape_is_solid=wp.ones(len(shape_bodies), dtype=bool, device=device),
        shape_material_mu=wp.array(materials, dtype=float, device=device),
    )
    state = SimpleNamespace(body_q=wp.array([wp.transform_identity()] * body_count, dtype=wp.transform, device=device))
    contacts = SimpleNamespace(
        rigid_contact_count=wp.array([n], dtype=int, device=device),
        rigid_contact_shape0=wp.array(shape0 if shape0 is not None else [0] * n, dtype=int, device=device),
        rigid_contact_shape1=wp.ones(n, dtype=int, device=device),
        rigid_contact_point0=wp.array(points, dtype=wp.vec3, device=device),
        rigid_contact_point1=wp.array(points, dtype=wp.vec3, device=device),
        rigid_contact_normal=wp.array(
            normals if normals is not None else [[0.0, 0.0, -1.0]] * n, dtype=wp.vec3, device=device
        ),
        rigid_contact_margin0=wp.zeros(n, dtype=float, device=device),
        rigid_contact_margin1=wp.zeros(n, dtype=float, device=device),
    )
    patches = _FrictionPatchState(model, max(n, 8), True, wp.zeros(max(n, 8), dtype=wp.vec2, device=device))
    patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
    return model, state, contacts, patches


def _tree(device, links, *, branched=False, worlds=2, loop=False):
    """Driven revolute chains (or a two-branch tree) hanging above the ground."""
    template = newton.ModelBuilder()
    joints = []
    bodies = []
    for i in range(links):
        parent = -1 if i == 0 else bodies[(i - 1) // 2 if branched else i - 1]
        link = template.add_link(mass=1.0 + 0.1 * i, inertia=wp.mat33(np.eye(3) * (0.01 + 0.002 * i)))
        template.add_shape_capsule(link, radius=0.03, half_height=0.08)
        side = 1.0 if i % 2 else -1.0
        offset = wp.vec3(0.0, 0.0, 1.5) if parent < 0 else wp.vec3(0.05 * side if branched else 0.0, 0.0, -0.2)
        joints.append(
            template.add_joint_revolute(
                parent,
                link,
                axis=wp.vec3(0.0, 1.0, 0.0) if i % 3 else wp.vec3(1.0, 0.0, 0.0),
                parent_xform=wp.transform(offset, wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
                target_ke=20.0 + i,
                target_kd=1.0,
                target_pos=0.2,
                armature=0.01,
            )
        )
        bodies.append(link)
    template.add_articulation(joints)
    if loop:
        template.add_joint_ball(
            bodies[0], bodies[-1], child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity())
        )
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds)
    return builder.finalize(device=device)


def _trajectory(model, steps=40, **solver_kwargs):
    solver = SolverSteel(model, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    qd = np.linspace(-0.5, 0.5, model.joint_dof_count).astype(np.float32)
    state_0.joint_qd.assign(qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    return solver, state_0.joint_q.numpy(), state_0.joint_qd.numpy()


DT = 1.0 / 240.0


def _build_contact_scene(device, *, mu=0.6, restitution=0.0, worlds=2):
    """Build a sliding articulated box with a driven arm and two sliding free boxes per world."""
    scene = newton.ModelBuilder()
    scene.default_shape_cfg.mu = mu
    scene.default_shape_cfg.restitution = restitution
    scene.add_ground_plane()
    for i in range(2):
        body = scene.add_body(xform=wp.transform(wp.vec3(0.4 * i, 0.6, 0.099), wp.quat_rpy(0.0, 0.0, 0.2 * i)))
        scene.add_shape_box(body, hx=0.1, hy=0.08, hz=0.1)
    xform = wp.transform(wp.vec3(0.0, 0.0, 0.049), wp.quat_identity())
    base = scene.add_link(xform=xform)
    scene.add_shape_box(base, hx=0.15, hy=0.1, hz=0.05)
    arm = scene.add_link()
    scene.add_shape_capsule(arm, radius=0.03, half_height=0.08)
    axis = newton.ModelBuilder.JointDofConfig
    scene.add_articulation(
        [
            scene.add_joint_d6(
                parent=-1,
                child=base,
                parent_xform=xform,
                linear_axes=[axis(axis=newton.Axis.X), axis(axis=newton.Axis.Z)],
            ),
            scene.add_joint_revolute(
                base,
                arm,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.12), wp.quat_identity()),
                target_ke=20.0,
                target_kd=1.0,
                target_pos=0.4,
            ),
        ]
    )
    builder = newton.ModelBuilder()
    builder.replicate(scene, worlds)
    return builder.finalize(device=device)


def _set_box_velocities(model, state, speed=1.5):
    """Give every free box and articulated box a horizontal velocity so friction does work."""
    joint_qd = state.joint_qd.numpy()
    starts = model.joint_qd_start.numpy()
    for joint, joint_type in enumerate(model.joint_type.numpy()):
        if joint_type in (int(newton.JointType.FREE), int(newton.JointType.D6)):
            joint_qd[starts[joint]] = speed
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def _friction_impulses(solver):
    """Return every friction row impulse of the dense, free-body and propagation row families."""
    values = []
    families = [
        (solver._constraint_count, solver._row_type, solver._impulses),
        (solver._mf_constraint_count, solver._mf_row_type, solver._mf_impulses),
    ]
    if solver._propagation_active:
        families.append(
            (solver._propagation_constraint_count, solver._propagation_row_type, solver._propagation_impulses)
        )
    for count_array, type_array, impulse_array in families:
        row_type = type_array.numpy()
        impulses = impulse_array.numpy()
        for world, rows in enumerate(count_array.numpy()):
            friction = row_type[world, :rows] == PGS_CONSTRAINT_TYPE_FRICTION
            values.append(impulses[world, :rows][friction])
    return np.concatenate(values) if values else np.zeros(0)


def _run(model, *, steps=12, init=None, overrides=None, capture=False, **options):
    """Step ``model`` and return the solver, the final joint state and each step's friction impulses."""
    with kernel_overrides(overrides or {}):
        solver = SolverSteel(model, **options)
    state_in, state_out = model.state(), model.state()
    if init is not None:
        init(model, state_in)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
    contacts = pipeline.contacts()
    control = model.control()
    friction = []

    def step():
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, DT)
        for src, dst in (
            (state_out.joint_q, state_in.joint_q),
            (state_out.joint_qd, state_in.joint_qd),
            (state_out.body_q, state_in.body_q),
            (state_out.body_qd, state_in.body_qd),
        ):
            wp.copy(dst, src)

    if capture:
        step()
        friction.append(_friction_impulses(solver))
        with wp.ScopedCapture(model.device) as graph:
            step()
        for _ in range(steps - 1):
            wp.capture_launch(graph.graph)
            friction.append(_friction_impulses(solver))
    else:
        for _ in range(steps):
            step()
            friction.append(_friction_impulses(solver))
    return solver, state_in.joint_q.numpy(), state_in.joint_qd.numpy(), friction


POISON = -123.25


def _build_model(device="cpu", *, locked_d6=False, chain=False, leaves=7):
    """Build a tree of prismatic leaves with nontrivial anchors and COMs, plus unrelated moving bodies."""
    builder = newton.ModelBuilder(gravity=(0.7, -1.2, -9.1))

    def body(index):
        return builder.add_link(
            mass=0.5 + 0.1 * index,
            com=wp.vec3(0.07, -0.03, 0.02),
            inertia=wp.mat33(0.3, 0.02, 0.01, 0.02, 0.4, 0.03, 0.01, 0.03, 0.5),
        )

    base = body(0)
    root_kwargs = {
        "parent": -1,
        "child": base,
        "parent_xform": wp.transform(wp.vec3(2.0, -1.0, 0.6), wp.quat_rpy(0.3, -0.2, 0.4)),
        "child_xform": wp.transform(wp.vec3(0.2, -0.1, 0.05), wp.quat_rpy(-0.1, 0.2, 0.0)),
    }
    root = builder.add_joint_d6(**root_kwargs) if locked_d6 else builder.add_joint_fixed(**root_kwargs)
    joints = [root]
    leaf_bodies = []
    for index in range(leaves):
        child = body(index + 1)
        leaf_bodies.append(child)
        joints.append(
            builder.add_joint_prismatic(
                parent=(base if index == 0 else leaf_bodies[index - 1]) if chain else base,
                child=child,
                axis=wp.vec3(0.2 + 0.1 * index, 0.5, 0.9),
                parent_xform=wp.transform(wp.vec3(0.1 * index, -0.05, 0.13), wp.quat_rpy(0.2, -0.1 * index, 0.3)),
                child_xform=wp.transform(wp.vec3(-0.09, 0.04, 0.08), wp.quat_rpy(0.4, 0.2, -0.1)),
                target_ke=3.0,
                target_kd=0.2,
                limit_lower=-0.5,
                limit_upper=0.5,
            )
        )
    builder.add_articulation(joints)
    arm = body(leaves + 1)
    hinge = builder.add_joint_revolute(
        parent=-1,
        child=arm,
        axis=newton.Axis.Y,
        parent_xform=wp.transform(wp.vec3(-0.8, 0.1, 1.0), wp.quat_identity()),
    )
    builder.add_articulation([hinge])
    free = body(leaves + 2)
    builder.add_articulation([builder.add_joint_free(free)])
    model = builder.finalize(device=device)
    model.rigid_contact_max = 1
    q = model.joint_q.numpy()
    q[: leaves + 1] = np.linspace(-0.31, 0.27, leaves + 1)
    model.joint_q.assign(q)
    model.joint_qd.assign(np.linspace(-0.7, 0.8, model.joint_dof_count, dtype=np.float32))
    # Joint axes need not be unit length.
    axes = model.joint_axis.numpy()
    axes[0] *= 1.4
    model.joint_axis.assign(axes)
    return model, joints, leaf_bodies


def _build_branched_model(device="cpu", *, floating_root=False):
    """Interleave terminal joints and internal parents with nonzero parent motion."""
    builder = newton.ModelBuilder(gravity=(0.7, -1.2, -9.1))
    bodies = [
        builder.add_link(
            mass=0.7 + 0.1 * index,
            com=wp.vec3(0.03, -0.04, 0.02),
            inertia=wp.mat33(0.3, 0.02, 0.01, 0.02, 0.4, 0.03, 0.01, 0.03, 0.5),
        )
        for index in range(7)
    ]
    root_args = {"parent": -1, "child": bodies[0]}
    root = builder.add_joint_free(**root_args) if floating_root else builder.add_joint_revolute(**root_args)
    joints = [root]
    # Joint 1 is terminal although it precedes the internal joint 2; later leaves use both moving parents.
    for index, kind in enumerate(("revolute", "revolute", "prismatic", "ball", "d6", "fixed"), start=1):
        args = {
            "parent": bodies[2] if index in (4, 5) else bodies[0],
            "child": bodies[index],
            "parent_xform": wp.transform(wp.vec3(0.1 * index, -0.07, 0.13), wp.quat_rpy(0.2, -0.1, 0.3)),
            "child_xform": wp.transform(wp.vec3(-0.03, 0.04, 0.02), wp.quat_rpy(-0.1, 0.2, 0.1)),
        }
        if kind in ("revolute", "prismatic"):
            args["axis"] = wp.vec3(0.2, 0.5, 0.9)
        if kind == "d6":
            args["linear_axes"] = [newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X)]
            args["angular_axes"] = [newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y)]
        joints.append(getattr(builder, f"add_joint_{kind}")(**args))
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    q = model.joint_q.numpy()
    starts = model.joint_q_start.numpy()
    for joint in (1, 2, 3, 5):
        q[starts[joint] : starts[joint + 1]] = 0.07 * joint
    if floating_root:
        q[:7] = (0.4, -0.2, 0.6, *wp.quat_rpy(0.3, -0.2, 0.1))
    else:
        q[0] = 0.31
    model.joint_q.assign(q)
    model.joint_qd.assign(np.linspace(-0.9, 0.8, model.joint_dof_count, dtype=np.float32))
    return model, joints, bodies


def _build_contact_model(device):
    """Load three terminal joints against the ground under a moving root."""
    builder = newton.ModelBuilder()
    root = builder.add_link(mass=1.0, inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1))
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=root,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.185), wp.quat_identity()),
        )
    ]
    for index, kind in enumerate(("revolute", "prismatic", "ball")):
        child = builder.add_link(mass=0.5, inertia=wp.mat33(0.03, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.03))
        args = {
            "parent": root,
            "child": child,
            "parent_xform": wp.transform(wp.vec3(0.5 * (index - 1), 0.3, 0.0), wp.quat_identity()),
        }
        if kind != "ball":
            args["axis"] = newton.Axis.Y if kind == "revolute" else newton.Axis.Z
        joints.append(getattr(builder, f"add_joint_{kind}")(**args))
        builder.add_shape_sphere(child, radius=0.2)
    builder.add_articulation(joints)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.joint_qd.assign(np.linspace(-0.1, 0.15, model.joint_dof_count, dtype=np.float32))
    return model


def _solver(model, *, enabled, velocity_limits=False, **options):
    """Build a solver with serial (``enabled=False``) or parallel tree traversal."""
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        iterations=8,
        update_mass_matrix_interval=2,
        enable_joint_limits=True,
        enable_joint_velocity_limits=velocity_limits,
        parallel_tree=enabled,
        **options,
    )
    if enabled:
        assert solver._tree_plan is not None
    else:
        assert solver._tree_plan is None
        assert solver._tree_net_wrenches == ()
    solver._prepare_augmented_state(model.state())
    return solver


def _dynamics_fields(solver, state):
    """Arrays written by the forward pass of stage 1."""
    return {
        "body_q": state.body_q,
        "body_q_com": solver._body_q_com,
        "origin": solver._articulation_origin,
        "S": solver._joint_S_s,
        "v": solver._body_v_s,
        "a": solver._body_a_s,
        "I": solver._body_I_s,
        "f": solver._body_f_s,
    }


def _poison(fields, keep=()):
    for name, array in fields.items():
        if name not in keep:
            array.fill_(POISON)


def _run_stage1(solver, state, following):
    """Run stage 1 forward and backward passes; return the predictor input velocity."""
    qd = solver._stage1_fk_id(state, solver, following)
    solver._stage1_joint_tau(state, solver, following, solver.model.control(), 1.0 / 240.0)
    return qd


def _build_mixed_response_model(
    device,
    world_count=1,
    *,
    dof_count=13,
    friction=0.0,
    restitution=0.0,
    static_plane=False,
    static_support=False,
    free_body_first=False,
    free_body_velocity_limit=None,
):
    """Build one serial articulation contacting one free rigid body.

    ``static_support`` rests the free body on a static ledge that clears the arm, so the free body also
    produces matrix-free contact rows. ``free_body_first`` builds the free body before the articulation,
    which packs the world DOFs free-body first. ``free_body_velocity_limit`` caps the free body's linear
    and angular velocity [m/s, rad/s], producing matrix-free velocity-limit rows once exceeded.
    ``static_plane`` adds a ground plane under the free body.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if free_body_velocity_limit is not None:
        SolverSteel.register_custom_attributes(builder)
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e3
    builder.default_shape_cfg.mu = friction
    builder.default_shape_cfg.restitution = restitution
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    def add_free_body():
        custom_attributes = None
        if free_body_velocity_limit is not None:
            custom_attributes = {
                "steel:rigid_body_max_linear_velocity": free_body_velocity_limit,
                "steel:rigid_body_max_angular_velocity": free_body_velocity_limit,
            }
        box = builder.add_link(
            xform=wp.transform(wp.vec3(0.7, 0.0, 0.5695), wp.quat_identity()), custom_attributes=custom_attributes
        )
        builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05)
        builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])
        if static_support:
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(0.7, 0.11, 0.47), wp.quat_identity()),
                hx=0.05,
                hy=0.05,
                hz=0.05,
            )

    if free_body_first:
        add_free_body()
    arm = builder.add_link()
    builder.add_shape_box(arm, hx=0.4, hy=0.05, hz=0.02)
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=arm,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.4, 0.0, 0.0), wp.quat_identity()),
        )
    ]
    parent = arm
    for index in range(dof_count - 1):
        child = builder.add_link(
            mass=0.05,
            inertia=wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-3),
            lock_inertia=True,
        )
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=child,
                axis=(newton.Axis.X, newton.Axis.Y, newton.Axis.Z)[index % 3],
            )
        )
        parent = child
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[1], joints[0])

    if not free_body_first:
        add_free_body()
    if static_plane:
        builder.add_shape_plane(plane=(0.0, 0.0, 1.0, -0.53))
    if world_count == 1:
        return builder.finalize(device=device)
    replicated = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    replicated.replicate(builder, world_count, spacing=(3.0, 0.0, 0.0))
    return replicated.finalize(device=device)


def _build_two_revolute_chain(coef0: float, coef1: float, legacy: bool = False):
    """Build a fixed-base chain of two revolute Z-joints with a mimic between them.

    The leader joint is position-driven; the follower joint has no drive and no spring,
    so any tracking it does comes from the mimic row alone.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    # add_link (not add_body): add_body wraps each body in its own free-joint articulation.
    link_a = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_box(link_a, hx=0.1, hy=0.02, hz=0.02)
    j_leader = b.add_joint_revolute(
        parent=-1,
        child=link_a,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    link_b = b.add_link(xform=wp.transform(wp.vec3(0.6, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_box(link_b, hx=0.1, hy=0.02, hz=0.02)
    j_follower = b.add_joint_revolute(
        parent=link_a,
        child=link_b,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    b.add_articulation([j_leader, j_follower], label="mimic_chain")
    b.joint_target_ke[0] = 50.0
    b.joint_target_kd[0] = 5.0
    b.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
    # follower: q_follower = coef0 + coef1 * q_leader
    if legacy:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            b.add_constraint_mimic(joint0=j_follower, joint1=j_leader, coef0=coef0, coef1=coef1)
    else:
        b.set_joint_mimic(j_follower, j_leader, coeffs=(coef0, coef1))
    return b, j_leader, j_follower


def _build_scissor_model(device):
    """One articulation: base link + two sibling links whose boxes overlap.

    The siblings are not directly jointed, so collision produces a
    link-link contact within a single articulation (common ancestor = base).
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e3
    builder.default_shape_cfg.mu = 0.75
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    base = builder.add_link()
    builder.add_shape_box(base, hx=0.05, hy=0.05, hz=0.05)
    j_base = builder.add_joint_revolute(
        parent=-1,
        child=base,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )

    joints = [j_base]
    for side in (1.0, -1.0):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.12, hy=0.03, hz=0.04)
        joints.append(
            builder.add_joint_revolute(
                parent=base,
                child=link,
                axis=wp.vec3(0.0, 0.0, 1.0),
                parent_xform=wp.transform(wp.vec3(0.06, side * 0.028, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
            )
        )
    builder.add_articulation(joints)
    return builder.finalize(device=device)


RADIUS = 0.05


CONTACT_GAP = 0.1


def _diagonal_inertia(value):
    """Return an isotropic inertia tensor."""
    return wp.mat33(value, 0.0, 0.0, 0.0, value, 0.0, 0.0, 0.0, value)


def _sphere_inertia(mass):
    """Return the inertia of a solid sphere about its centre."""
    return _diagonal_inertia(0.4 * mass * RADIUS * RADIUS)


def _shape_cfg(restitution):
    """Return an inertialess, frictionless test material."""
    return newton.ModelBuilder.ShapeConfig(
        density=0.0,
        mu=0.0,
        restitution=restitution,
        margin=0.0,
        gap=CONTACT_GAP,
    )


def _build_plane_model(
    device,
    *,
    separation,
    restitution,
    plane_restitution=None,
    mass=1.0,
    gravity=(0.0, 0.0, 0.0),
    scene="free",
):
    """Build one sphere above a plane, optionally as a two-link articulation."""
    builder = newton.ModelBuilder(gravity=gravity)
    xform = wp.transform(wp.vec3(0.0, 0.0, RADIUS + separation), wp.quat_identity())
    if scene == "free":
        body = builder.add_body(
            xform=xform,
            mass=mass,
            inertia=_sphere_inertia(mass),
            lock_inertia=True,
        )
    elif scene == "articulated":
        # A concentric, shape-free fixed child makes this a real floating
        # articulation without adding a second contact or a changing contact
        # point velocity to the analytical one-row problem.
        body = builder.add_link(
            xform=xform,
            mass=mass,
            inertia=_sphere_inertia(mass),
            lock_inertia=True,
        )
        child_mass = 0.25 * mass
        child = builder.add_link(
            xform=xform,
            mass=child_mass,
            inertia=_sphere_inertia(child_mass),
            lock_inertia=True,
        )
        root_joint = builder.add_joint_free(child=body)
        fixed_joint = builder.add_joint_fixed(parent=body, child=child)
        builder.add_articulation([root_joint, fixed_joint])
    else:
        raise ValueError(f"unknown scene {scene!r}")

    builder.add_shape_sphere(body, radius=RADIUS, cfg=_shape_cfg(restitution))
    builder.add_ground_plane(cfg=_shape_cfg(restitution if plane_restitution is None else plane_restitution))
    return builder.finalize(device=device), body


def _reset_state(model, state, vertical_velocity):
    """Restore the authored pose and set a free root's world-z velocity."""
    wp.copy(state.joint_q, model.joint_q)
    joint_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    joint_qd[2] = vertical_velocity
    state.joint_qd.assign(joint_qd)
    state.clear_forces()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


# A manipulation profile without contact torsion; the torsion variant is in test_sleeping.
PROFILE = {
    "iterations": 32,
    "contact_regularization": 0.01,
    "friction_anchor_beta": 0.2,
    "contact_friction_gap_max": 0.001,
    "articulation_constraint_max": 512,
    "free_body_constraint_max": 512,
    "enable_sleeping": True,
    "sleep_quiet_time": 0.1,
}


def _articulations(device, tiles=False, profile=None, torsion_radius=None):
    """Two undriven two-link articulations resting on the ground, which route contacts to dense rows.

    With ``tiles``, each base is a grid of small boxes so its ground pair exceeds the warp-flood threshold.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for x in (0.0, 1.0):
        base = builder.add_link(xform=wp.transform((x, 0.0, 0.1), wp.quat_identity()))
        if tiles:
            for i in range(4):
                for j in range(3):
                    offset = wp.transform((-0.15 + 0.1 * i, -0.067 + 0.067 * j, 0.0), wp.quat_identity())
                    builder.add_shape_box(base, xform=offset, hx=0.05, hy=0.033, hz=0.1)
        else:
            builder.add_shape_box(base, hx=0.2, hy=0.1, hz=0.1)
        tip = builder.add_link(xform=wp.transform((x + 0.3, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(tip, hx=0.1, hy=0.1, hz=0.1)
        hinge = wp.transform((0.3, 0.0, 0.0), wp.quat_identity())
        builder.add_articulation(
            [
                builder.add_joint_free(child=base),
                builder.add_joint_revolute(parent=base, child=tip, parent_xform=hinge, axis=(0.0, 1.0, 0.0)),
            ]
        )
    model = builder.finalize(device=device)
    if torsion_radius is not None:
        set_torsion_radius(model, torsion_radius)
    # Deterministic contact order, as in production, so A/B trajectories compare bitwise.
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=512, deterministic=True)
    solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free", **(profile or PROFILE))
    return model, pipeline, solver, [model.state(), model.state()], model.control()


def _advance(pipeline, solver, states, control, steps, *, clear=True, contacts=None):
    contacts = contacts or pipeline.contacts()
    for _ in range(steps):
        if clear:
            states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, 0.005)
        states.reverse()


def _build_sparse_contact_friction_model(num_branches=16, num_worlds=2, *, device=None):
    """Build the sparse pair model with tilted sliding branches resting on a static ground plane.

    Each branch slides along ``(1, 0, 1) / sqrt(2)`` and carries a small box that touches the ground, so every
    contact normal and tangent row acts on that branch's single coordinate and friction is active.
    """
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    scene.default_shape_cfg.ke = 1.0e5
    scene.default_shape_cfg.kd = 1.0e3
    scene.default_shape_cfg.mu = 0.8
    scene.default_shape_cfg.margin = 0.0
    scene.default_shape_cfg.gap = 0.0
    scene.add_ground_plane()
    base = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    star_joints = [
        scene.add_joint_fixed(
            parent=-1, child=base, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity())
        )
    ]
    for branch in range(num_branches):
        child = scene.add_link(mass=1.0 + 0.01 * branch, inertia=wp.mat33(np.eye(3)))
        # Four boxes keep the four-point patches within the dense row capacity of the test solvers.
        if branch % 4 == 0:
            scene.add_shape_box(child, hx=0.01, hy=0.01, hz=0.01)
        star_joints.append(
            scene.add_joint_prismatic(
                parent=base,
                child=child,
                axis=wp.normalize(wp.vec3(1.0, 0.0, 1.0)),
                parent_xform=wp.transform(wp.vec3(0.05 * branch, 0.0, -0.4905), wp.quat_identity()),
                limit_lower=-0.1,
                limit_upper=0.1,
            )
        )
    scene.add_articulation(star_joints)

    chain_joints = []
    parent = -1
    for _link_index in range(3):
        child = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        chain_joints.append(
            scene.add_joint_revolute(
                parent=parent,
                child=child,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.0, 1.0, 1.0), wp.quat_identity()),
                limit_lower=-0.2,
                limit_upper=0.2,
            )
        )
        parent = child
    scene.add_articulation(chain_joints)

    replicated = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    replicated.replicate(scene, num_worlds, spacing=(3.0, 3.0, 0.0))
    return replicated.finalize(device=device)


def _assert_budget(test, result, solver):
    """Check the shared final-load budget and nonnegative normal impulses."""
    for group in solver._torsion_stats["groups"]:
        impulse = result["impulses"][group["world"]]
        normal = sum(max(float(impulse[n]), 0.0) for n in group["normal_rows"])
        sliding = sum(float(np.linalg.norm(impulse[n + 1 : n + 3])) for n in group["anchor_rows"])
        spin = abs(float(impulse[group["row"]])) / group["effective_radius_m"]
        test.assertLessEqual(sliding + spin, group["mu"] * normal + 3e-6)
