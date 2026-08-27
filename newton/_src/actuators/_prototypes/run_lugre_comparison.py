# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""LuGre friction: MuJoCo's own actuator against Newton's.

THE CLAIM THIS SUPPORTS
    LuGre and Dahl friction can be modelled in Newton today, by writing a
    custom actuator controller. No change to Newton is required.

    One approach reproduces MuJoCo's modelling choices. Another evaluates the
    force using the deflection advanced within the step, which may improve
    behaviour for some models and timesteps. This comparison illustrates the
    alternatives; it does not establish that either is generally more accurate.

    Any improvement would be a property of the modelling choice, not of the
    solver, and depends on the timestep resolving the bristle stiffness. The
    relevant condition, dt*sqrt(sigma0/M), is printed on every run.

THE TEST
    One joint follows the same applied-torque sequence in both simulations:
    unloaded, below Coulomb friction, above static friction, reversed, then
    unloaded again. This exercises stiction, breakaway, sliding, and reversal.

MUJOCO
    MuJoCo's real `<dcmotor lugre=...>` actuator, driven through the `mujoco`
    package. The motor constant is set near zero so the only actuator force is
    friction.

NEWTON
    A custom Controller running in Newton's implicit effort mode, stepped with
    SolverMuJoCo. Four variants cover two modelling choices: how the contact
    deflection is stepped, and which deflection the force is built from. The
    ``ZOH, old z`` variant uses MuJoCo's pair of choices and closely agrees
    with MuJoCo's own actuator in this test.

SCOPE AND LIMITATIONS
    Both sides treat the deflection state explicitly and the velocity
    implicitly, so the friction model itself is not the difference. The
    difference is how far the implicitness reaches.

    MuJoCo forms one velocity derivative for the whole system and folds it into
    the system solve, so every force present in the step is inside the implicit
    update. Newton's actuator instead predicts the end-of-step velocity as
    ``qd + A p``, from its own impulse alone. Gravity, contacts, other actuators
    on the same articulation and joint drive applied outside the actuator are
    all absent from that prediction, and the response ``A`` itself omits joint
    damping, limits, friction, contacts and loop closures. These are the
    documented limits of the implicit effort mode; see the "Effort Modes"
    section of the Newton actuator documentation.

    This benchmark does not expose them. There is no gravity and no contact,
    and the applied torque is passed as feedforward so that it does enter the
    prediction. For the implications in scenes with contacts, gravity, several
    actuators, or other external forces, refer to the documented implicit
    effort-mode limitations cited above. Note also that
    :meth:`Controller.prepare_implicit` is not given the feedforward array, so
    a state update cannot depend on it.

Run::

    uv run python -m newton._src.actuators._prototypes.run_lugre_comparison
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton.actuators import Actuator, ResponseOracle

from .controller_lugre import ControllerLuGre

DT = 0.01  # timestep [s], 100 Hz
STEPS = 160
RADIUS = 0.1  # sphere radius
ARMATURE = 16.0  # reflected gearbox inertia [kg m^2]; dominates the link inertia
INTEGRATOR = "implicitfast"  # same on both sides

SIGMA0 = 1.0e5  # contact stiffness [N m / rad]
SIGMA1 = 1.0  # contact damping [N m s / rad]
SIGMA2 = 0.6  # viscous [N m s / rad]; passivity needs sigma2 > sigma1*(Fs-Fc)/Fc
FC = 1.0  # Coulomb level [N m]
FS = 1.5  # static level [N m]
VS = 1.0e-3  # Stribeck velocity [rad/s]

NEWTON_VARIANTS = {
    "Newton (ZOH, old z)": (("zoh", "old"), "#e8913a", "--"),
    "Newton (BE, old z)": (("be", "old"), "#f0c419", "--"),
    "Newton (ZOH, new z)": (("zoh", "new"), "#4c9be8", "--"),
    "Newton (BE, new z)": (("be", "new"), "#1f77b4", "-"),
}


def applied_torque(step):
    """Applied torque sequence: stiction, breakaway, reversal, then release."""
    t = step * DT
    if t < 0.2:
        return 0.0
    if t < 0.6:
        return 0.5
    if t < 1.0:
        return 2.0
    if t < 1.4:
        return -2.0
    return 0.0


MUJOCO_XML = f"""<mujoco>
  <option timestep="{DT}" gravity="0 0 0" integrator="{INTEGRATOR}"/>
  <worldbody>
    <body>
      <joint name="j" type="hinge" axis="0 0 1" armature="{ARMATURE}"/>
      <geom type="sphere" size="{RADIUS}"/>
    </body>
  </worldbody>
  <actuator>
    <dcmotor name="a" joint="j" motorconst="1e-9 1e-9" resistance="1.0"
             lugre="{SIGMA0} {SIGMA1} {FC} {FS} {VS}" damping="{SIGMA2}"/>
  </actuator>
</mujoco>"""


def run_mujoco():
    """MuJoCo's own LuGre actuator."""
    import mujoco

    m = mujoco.MjModel.from_xml_string(MUJOCO_XML)
    d = mujoco.MjData(m)
    v = np.empty(STEPS + 1)
    v[0] = 0.0
    for k in range(STEPS):
        d.qfrc_applied[0] = applied_torque(k)
        mujoco.mj_step(m, d)
        v[k + 1] = d.qvel[0]
    return v, float(m.dof_M0[0])


def run_newton(z_method="zoh", force_z="old"):
    """Newton's actuator in the implicit effort mode, stepped by SolverMuJoCo."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))
    link = builder.add_link()
    joint = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z, armature=ARMATURE)
    builder.add_shape_sphere(body=link, radius=RADIUS)
    builder.add_articulation([joint])
    model = builder.finalize()
    dof = int(builder.joint_qd_start[joint])

    one = lambda x: wp.array([x], dtype=wp.float32)  # noqa: E731
    ctrl = ControllerLuGre(
        sigma0=one(SIGMA0),
        sigma1=one(SIGMA1),
        sigma2=one(SIGMA2),
        coulomb_friction=one(FC),
        static_friction=one(FS),
        stribeck_velocity=one(VS),
        z_method=z_method,
        force_z=force_z,
    )
    actuator = Actuator(wp.array([dof], dtype=wp.uint32), controller=ctrl)
    oracle = ResponseOracle(model)
    # Default tolerances stop at iteration 0 for this law, returning the explicit
    # warm start. Tighten them so the numbers come from the implicit solve.
    actuator.set_effort_mode_implicit(
        response=oracle,
        options=Actuator.ImplicitOptions(residual_tol=1e-12, update_tol=1e-12),
    )

    solver = newton.solvers.SolverMuJoCo(model, integrator=INTEGRATOR)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    act_0, act_1 = actuator.state(), actuator.state()

    # The torque goes in as feedforward, not added to joint_f afterwards, so the
    # actuator's implicit prediction includes it. Adding it after actuator.step
    # would leave it out of the prediction (see effort_mode_implicit.py:19-22)
    # and the comparison with MuJoCo would no longer be like for like.
    v = np.empty(STEPS + 1)
    v[0] = 0.0
    for k in range(STEPS):
        control.joint_act.assign(np.full(control.joint_act.shape, applied_torque(k), dtype=np.float32))
        oracle.refresh(state_0)
        control.joint_f.zero_()
        actuator.step(state_0, control, act_0, act_1, dt=DT)
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
        act_0, act_1 = act_1, act_0
        v[k + 1] = float(state_0.joint_qd.numpy()[dof])
    return v, float(1.0 / oracle.inverse_blocks.numpy()[0, 0, 0])


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wp.init()
    t = np.arange(STEPS + 1) * DT
    torque = np.array([applied_torque(k) for k in range(STEPS + 1)])

    v_mj, inertia_mj = run_mujoco()
    results = {"MuJoCo (its own actuator)": (v_mj, "#c1440e", "-")}
    inertia_nt = None
    for name, (cfg, color, ls) in NEWTON_VARIANTS.items():
        v, inertia_nt = run_newton(*cfg)
        results[name] = (v, color, ls)

    print(f"integrator = {INTEGRATOR}")
    print(f"dt = {DT}, Coulomb level = {FC} N·m, static level = {FS} N·m")
    print(f"dt*sqrt(sigma0/M): {DT * (SIGMA0 / inertia_mj) ** 0.5:.2f}  (needs to be near 1 or below)")
    print(f"inertia: MuJoCo {inertia_mj:.7f}, Newton {inertia_nt:.7f} kg·m²")
    print(f"{'':>26} | {'peak |v| [rad/s]':>17} | {'final v [rad/s]':>17} | {'max diff vs MJ':>14}")
    for name, (v, _color, _ls) in results.items():
        print(f"{name:>26} | {np.max(np.abs(v)):17.6e} | {v[-1]:17.6e} | {np.max(np.abs(v - v_mj)):14.3e}")

    fig, (ax_torque, ax) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    ax_torque.step(t, torque, where="post", color="black")
    ax_torque.axhline(FC, color="gray", ls=":", label="Coulomb level")
    ax_torque.axhline(-FC, color="gray", ls=":")
    ax_torque.set_ylabel("applied torque [N·m]")
    ax_torque.legend(fontsize=9)
    ax_torque.grid(alpha=0.3)

    for name, (v, color, ls) in results.items():
        lw = 2.4 if ls == "-" else 1.6
        ax.plot(t, v, color=color, ls=ls, marker="o", ms=3, lw=lw, label=name)
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("joint velocity [rad/s]")
    ax.set_title(
        f"LuGre friction step response, dt = {DT} s\n"
        "stiction, breakaway, sliding, reversal, and release"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig("lugre_comparison.png", dpi=110)
    print("\nwrote lugre_comparison.png")


if __name__ == "__main__":
    main()
