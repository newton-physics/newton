"""Reproduce UR5e + Robotiq 2f85 simulation explosion.

Tests both native MuJoCo (CPU) and MuJoCo Warp (GPU) paths.
The CPU path has mj_checkAcc safety checks; the GPU path does not,
so NaN values propagate and the simulation explodes.

Reference: https://github.com/google-deepmind/mujoco_warp/discussions/1112
"""

import numpy as np

XML_PATH = "/home/adenzler/git/newton-2/repro_ur5e_explosion.xml"
NUM_STEPS = 500


def test_native_mujoco():
    """Run with native MuJoCo (CPU) - has mj_checkAcc safety checks."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    print(f"Native MuJoCo: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"  integrator={model.opt.integrator} (3=implicitfast)")
    print(f"  timestep={model.opt.timestep}")

    for i in range(NUM_STEPS):
        mujoco.mj_step(model, data)

        qpos_max = np.max(np.abs(data.qpos))
        qvel_max = np.max(np.abs(data.qvel))
        qacc_max = np.max(np.abs(data.qacc))

        has_nan = np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel))
        has_inf = np.any(np.isinf(data.qpos)) or np.any(np.isinf(data.qvel))
        warnings_total = int(data.warning.number.sum())

        if i % 100 == 0 or has_nan or has_inf or qpos_max > 1e6:
            print(
                f"  step {i:4d}: |qpos|_max={qpos_max:.6e}, "
                f"|qvel|_max={qvel_max:.6e}, |qacc|_max={qacc_max:.6e}"
                f"{' NAN!' if has_nan else ''}{' INF!' if has_inf else ''}"
                f" warnings={warnings_total}"
            )

        if has_nan or has_inf:
            print("  => Native MuJoCo produced NaN/Inf!")
            return

    print(f"  => Native MuJoCo completed {NUM_STEPS} steps OK")
    print(f"  Final: |qpos|_max={np.max(np.abs(data.qpos)):.6e}, warnings={int(data.warning.number.sum())}")


def test_mujoco_warp():
    """Run with MuJoCo Warp (GPU) - lacks mj_checkAcc safety checks."""
    import mujoco
    import mujoco_warp as mjw

    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    mj_data = mujoco.MjData(mj_model)

    mjw_model = mjw.put_model(mj_model)
    mjw_data = mjw.put_data(mj_model, mj_data)

    print(f"\nMuJoCo Warp: nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}")
    print(f"  integrator={mj_model.opt.integrator} (3=implicitfast)")

    for i in range(NUM_STEPS):
        mjw.step(mjw_model, mjw_data)

        # Pull data back periodically to check
        if i % 25 == 0 or i < 10:
            mjw.get_data_into(mj_data, mj_model, mjw_data)

            qpos_max = np.max(np.abs(mj_data.qpos))
            qvel_max = np.max(np.abs(mj_data.qvel))

            has_nan = np.any(np.isnan(mj_data.qpos)) or np.any(np.isnan(mj_data.qvel))
            has_inf = np.any(np.isinf(mj_data.qpos)) or np.any(np.isinf(mj_data.qvel))

            print(
                f"  step {i:4d}: |qpos|_max={qpos_max:.6e}, "
                f"|qvel|_max={qvel_max:.6e}"
                f"{' NAN!' if has_nan else ''}{' INF!' if has_inf else ''}"
            )

            if has_nan or has_inf or qpos_max > 1e6:
                print("  => MuJoCo Warp simulation EXPLODED!")
                return

    mjw.get_data_into(mj_data, mj_model, mjw_data)
    print(f"  => MuJoCo Warp completed {NUM_STEPS} steps OK")


def test_newton_gpu():
    """Run through Newton's SolverMuJoCo GPU path."""
    import warp as wp

    import newton

    wp.init()

    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_mjcf(
        XML_PATH,
        parse_meshes=False,
        parse_visuals=False,
        parse_sites=False,
    )
    model = builder.finalize()

    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    contacts = model.collide(state_0)

    print(f"\nNewton (GPU): bodies={model.body_count}, joints={model.joint_count}")

    # Check initial state
    body_q0 = state_0.body_q.numpy()
    print(f"  Initial body_q: max={np.max(np.abs(body_q0)):.6e}, nan={np.any(np.isnan(body_q0))}")

    # Also check the MJWarp data state before stepping
    mjw_qpos = solver.mjw_data.qpos.numpy()
    mjw_qvel = solver.mjw_data.qvel.numpy()
    print(f"  MJWarp initial qpos: {mjw_qpos.flatten()}")
    print(f"  MJWarp initial qvel: max={np.max(np.abs(mjw_qvel)):.6e}")

    dt_arr = solver.mjw_model.opt.timestep.numpy()
    dt = float(dt_arr.flatten()[0])
    print(f"  timestep={dt}")

    for i in range(NUM_STEPS):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

        if i % 25 == 0 or i < 10:
            body_q = state_0.body_q.numpy()
            body_q_max = np.max(np.abs(body_q))
            has_nan = np.any(np.isnan(body_q))
            has_inf = np.any(np.isinf(body_q))

            # Also check mjw_data internals
            mjw_qpos = solver.mjw_data.qpos.numpy()
            mjw_qvel = solver.mjw_data.qvel.numpy()
            mjw_qacc = solver.mjw_data.qacc.numpy()
            mjw_qpos_max = np.max(np.abs(mjw_qpos))
            mjw_qvel_max = np.max(np.abs(mjw_qvel))
            mjw_qacc_max = np.max(np.abs(mjw_qacc))
            mjw_nan = np.any(np.isnan(mjw_qpos)) or np.any(np.isnan(mjw_qvel)) or np.any(np.isnan(mjw_qacc))

            print(
                f"  step {i:4d}: |body_q|_max={body_q_max:.6e}, "
                f"|mjw_qpos|={mjw_qpos_max:.6e}, |mjw_qvel|={mjw_qvel_max:.6e}, "
                f"|mjw_qacc|={mjw_qacc_max:.6e}"
                f"{' NAN!' if (has_nan or mjw_nan) else ''}"
                f"{' INF!' if has_inf else ''}"
            )

            if has_nan or has_inf or body_q_max > 1e6:
                # Dump more details
                if has_nan or mjw_nan:
                    print(f"    mjw_qpos={mjw_qpos.flatten()}")
                    print(f"    mjw_qvel={mjw_qvel.flatten()}")
                    print(f"    mjw_qacc={mjw_qacc.flatten()}")
                print("  => Newton GPU simulation EXPLODED!")
                return

    print(f"  => Newton GPU completed {NUM_STEPS} steps OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Reproducing UR5e + Robotiq 2f85 simulation explosion")
    print("=" * 60)

    print("\n--- Test 1: Native MuJoCo (CPU, has mj_checkAcc) ---")
    try:
        test_native_mujoco()
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n--- Test 2: MuJoCo Warp (GPU, no mj_checkAcc) ---")
    try:
        test_mujoco_warp()
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n--- Test 3: Newton SolverMuJoCo (GPU path) ---")
    try:
        test_newton_gpu()
    except Exception as e:
        import traceback

        print(f"  ERROR: {e}")
        traceback.print_exc()
