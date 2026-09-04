# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Show a case where implicit actuation makes a clear difference.

This example uses a low joint inertia (0.1 kg·m²) and strong
velocity-dependent friction (``sigma1 + sigma2 = 10``).

Three runs are compared:

* MuJoCo's LuGre actuator
* the Newton controller in implicit mode
* the same Newton controller in explicit mode

All runs use the same LuGre state update. Newton's implicit result should stay
close to MuJoCo, while the explicit result has larger transient errors.

The scene has no gravity or contacts, and the torque drives the joint beyond
breakaway. Gravity, contacts, and other forces can require a more complete
velocity prediction than the one tested here.

Run::

    uv run python -m newton._src.actuators._prototypes.run_lugre_implicit_regime
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton.actuators import Actuator, ResponseOracle

from .controller_lugre import ControllerLuGre

DT = 0.01  # timestep [s], 100 Hz
STEPS = 160
RADIUS = 0.02  # small link; the armature dominates
INTEGRATOR = "implicitfast"  # same on both sides

ARMATURE = 0.1
"""Rotor inertia [kg m^2]. Direct drive, so there is no gearbox reflection."""

FC = 5.0
"""Coulomb (sliding) friction level [N m]."""

BREAKAWAY = 1.0e-3
"""Deflection before the contact starts to slide [rad]. A compliant contact."""

SIGMA0 = FC / BREAKAWAY
"""Contact stiffness [N m / rad] = 5000. Fixed by the friction level."""

SIGMA1 = 1.0
"""Contact damping [N m s / rad]."""

SIGMA2 = 9.0
"""Viscous friction [N m s / rad]. Gives dt*k*A = 1; passivity needs sigma2 > 0.5."""

FS = 1.5 * FC
"""Static (stiction) friction level [N m]."""

VS = 1.0e-3
"""Stribeck velocity [rad/s]."""

RUNS = {
    "MuJoCo (its own actuator)": ("mujoco", "#c1440e", "-"),
    "Newton implicit (ZOH, old z)": ("implicit", "#1f77b4", "--"),
    "Newton explicit (ZOH, old z)": ("explicit", "#7f4fc0", "--"),
}


def applied_torque(step):
    """Drive the joint past breakaway, reverse it, and then release it."""
    t = step * DT
    if t < 0.2:
        return 0.0
    if t < 0.6:
        return 5.0 * FC  # fast sliding, viscous dominated
    if t < 1.0:
        return -5.0 * FC  # reversal
    if t < 1.4:
        return 2.0 * FC  # slower sliding
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


def run_newton(mode):
    """Newton's controller with MuJoCo's modelling choices, in either effort mode."""
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
        z_method="zoh",
        force_z="old",
    )
    actuator = Actuator(wp.array([dof], dtype=wp.uint32), controller=ctrl)
    oracle = ResponseOracle(model)
    if mode == "implicit":
        # Default tolerances stop the solve early; tighten them so the numbers
        # come from the converged implicit solve.
        actuator.set_effort_mode_implicit(
            response=oracle,
            options=Actuator.ImplicitOptions(residual_tol=1e-12, update_tol=1e-12),
        )

    solver = newton.solvers.SolverMuJoCo(model, integrator=INTEGRATOR)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    act_0, act_1 = actuator.state(), actuator.state()

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

    results = {}
    inertia_mj = inertia_nt = None
    for name, (mode, color, ls) in RUNS.items():
        if mode == "mujoco":
            v, inertia_mj = run_mujoco()
        else:
            v, inertia_nt = run_newton(mode)
        results[name] = (v, color, ls)

    v_mj = results["MuJoCo (its own actuator)"][0]
    v_imp = results["Newton implicit (ZOH, old z)"][0]
    v_exp = results["Newton explicit (ZOH, old z)"][0]

    a = 1.0 / inertia_nt
    k = SIGMA1 + SIGMA2
    dtka = DT * k * a

    print(f"integrator = {INTEGRATOR}")
    print(f"dt = {DT}, Coulomb level = {FC} N·m, static level = {FS} N·m")
    print(f"sigma0 = {SIGMA0:.6g}, sigma1 = {SIGMA1}, sigma2 = {SIGMA2} N·m·s/rad")
    print(f"inertia: MuJoCo {inertia_mj:.7f}, Newton {inertia_nt:.7f} kg·m²")
    print(f"dt*sqrt(sigma0/M) = {DT * (SIGMA0 / inertia_mj) ** 0.5:.2f}   (deflection resolution)")
    print()
    print("How much the implicit effort mode can matter here:")
    print(f"  k = sigma1 + sigma2 = {k:.1f} N·m·s/rad")
    print(f"  A = 1/inertia       = {a:.4f}")
    print(f"  dt*k*A              = {dtka:.4f}")
    print(f"  F_implicit/F_explicit = 1/(1+dt*k*A) = {1.0 / (1.0 + dtka):.4f}")
    print(f"  so the implicit correction is {100.0 * dtka / (1.0 + dtka):.1f} % of the force")
    print("  (the stiction test has dt*k*A = 0.001, i.e. 0.1 %)")
    print()
    print(f"{'':>30} | {'peak |v| [rad/s]':>17} | {'final v [rad/s]':>17} | {'max diff vs MJ':>14}")
    for name, (v, _color, _ls) in results.items():
        print(f"{name:>30} | {np.max(np.abs(v)):17.6e} | {v[-1]:17.6e} | {np.max(np.abs(v - v_mj)):14.3e}")
    print()
    print(f"implicit contribution, max |explicit - implicit| = {np.max(np.abs(v_exp - v_imp)):.6e} rad/s")
    print(
        f"                       relative to peak velocity = {100.0 * np.max(np.abs(v_exp - v_imp)) / np.max(np.abs(v_imp)):.2f} %"
    )
    print(f"fidelity difference,   max |Newton - MuJoCo|     = {np.max(np.abs(v_imp - v_mj)):.6e} rad/s")

    fig, (ax_t, ax_v, ax_d) = plt.subplots(
        3, 1, figsize=(10, 9.5), sharex=True, layout="constrained", height_ratios=[1, 2, 1.4]
    )

    ax_t.step(t, torque, where="post", color="black")
    ax_t.axhline(FC, color="gray", ls=":", label="Coulomb level")
    ax_t.axhline(-FC, color="gray", ls=":")
    ax_t.set_ylabel("applied torque [N·m]")
    ax_t.legend(fontsize=9)
    ax_t.grid(alpha=0.3)

    for name, (v, color, ls) in results.items():
        lw = 3.2 if ls == "-" else 1.8
        ax_v.plot(t, v, color=color, ls=ls, marker="o", ms=3, lw=lw, label=name)
    ax_v.axhline(0.0, color="gray", lw=0.8)
    ax_v.set_ylabel("joint velocity [rad/s]")
    ax_v.legend(fontsize=9)
    ax_v.grid(alpha=0.3)

    ax_d.plot(t, v_exp - v_imp, color="#7f4fc0", lw=2.0, label="explicit - implicit  (the implicit contribution)")
    ax_d.plot(t, v_imp - v_mj, color="#1f77b4", lw=2.0, label="Newton implicit - MuJoCo  (fidelity difference)")
    ax_d.axhline(0.0, color="gray", lw=0.8)
    ax_d.set_xlabel("time [s]")
    ax_d.set_ylabel("velocity difference [rad/s]")
    ax_d.legend(fontsize=9)
    ax_d.grid(alpha=0.3)

    fig.suptitle(
        f"LuGre friction, regime where the implicit effort mode matters: dt*k*A = {dtka:.2f}\n"
        f"same modelling choices (ZOH, old z) in all three runs; only the effort mode differs",
        fontsize=12,
    )
    fig.savefig("lugre_implicit_regime.png", dpi=110)
    print("\nwrote lugre_implicit_regime.png")


if __name__ == "__main__":
    main()
