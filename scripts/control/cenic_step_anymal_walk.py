import sys

import numpy as np
import torch
import warp as wp

wp.config.enable_backward = False

import newton
import newton.examples
import newton.utils
from newton import GeoType

num_worlds = 3

# Policy fires every POLICY_DT seconds of simulation time (zero-order hold).
POLICY_DT = 0.002  # 2 ms

# Joint index remapping between lab convention and MuJoCo convention
lab_to_mujoco = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
mujoco_to_lab = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[..., 3]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def compute_obs_for_world(
    world,
    actions_w,
    state,
    joint_pos_initial,
    torch_device,
    lab_indices,
    gravity_vec,
    command,
    q_offset,
    qd_offset,
    coords_per_world,
    dofs_per_world,
):
    """Build the policy observation for a single world."""
    q = q_offset
    qd = qd_offset

    root_quat = torch.tensor(state.joint_q[q + 3 : q + 7], device=torch_device, dtype=torch.float32).unsqueeze(0)
    root_lin_vel = torch.tensor(state.joint_qd[qd : qd + 3], device=torch_device, dtype=torch.float32).unsqueeze(0)
    root_ang_vel = torch.tensor(state.joint_qd[qd + 3 : qd + 6], device=torch_device, dtype=torch.float32).unsqueeze(0)
    joint_pos = torch.tensor(
        state.joint_q[q + 7 : q + coords_per_world], device=torch_device, dtype=torch.float32
    ).unsqueeze(0)
    joint_vel = torch.tensor(
        state.joint_qd[qd + 6 : qd + dofs_per_world], device=torch_device, dtype=torch.float32
    ).unsqueeze(0)

    vel_b = quat_rotate_inverse(root_quat, root_lin_vel)
    ang_vel_b = quat_rotate_inverse(root_quat, root_ang_vel)
    grav = quat_rotate_inverse(root_quat, gravity_vec)

    joint_pos_rel = torch.index_select(joint_pos - joint_pos_initial, 1, lab_indices)
    joint_vel_rel = torch.index_select(joint_vel, 1, lab_indices)

    return torch.cat([vel_b, ang_vel_b, grav, command, joint_pos_rel, joint_vel_rel, actions_w], dim=1)


device = wp.get_device()
torch_device = wp.device_to_torch(device)

robot = newton.ModelBuilder()
newton.solvers.SolverMuJoCoCENIC.register_custom_attributes(robot)
robot.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
    armature=0.06,
    limit_ke=1.0e3,
    limit_kd=1.0e1,
)
robot.default_shape_cfg.ke = 5.0e4
robot.default_shape_cfg.kd = 5.0e2
robot.default_shape_cfg.kf = 1.0e3
robot.default_shape_cfg.mu = 0.75

asset_path = newton.utils.download_asset("anybotics_anymal_c")
robot.add_urdf(
    str(asset_path / "urdf" / "anymal.urdf"),
    xform=wp.transform(wp.vec3(0.0, 0.0, 0.62), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)),
    floating=True,
    enable_self_collisions=False,
    collapse_fixed_joints=True,
    ignore_inertial_definitions=False,
)

for i in range(len(robot.shape_type)):
    if robot.shape_type[i] == GeoType.SPHERE:
        r = robot.shape_scale[i][0]
        robot.shape_scale[i] = (r * 2.0, 0.0, 0.0)

initial_q = {
    "RH_HAA": 0.0,
    "RH_HFE": -0.4,
    "RH_KFE": 0.8,
    "LH_HAA": 0.0,
    "LH_HFE": -0.4,
    "LH_KFE": 0.8,
    "RF_HAA": 0.0,
    "RF_HFE": 0.4,
    "RF_KFE": -0.8,
    "LF_HAA": 0.0,
    "LF_HFE": 0.4,
    "LF_KFE": -0.8,
}
for name, value in initial_q.items():
    idx = next((i for i, lbl in enumerate(robot.joint_label) if lbl.endswith(f"/{name}")), None)
    if idx is None:
        raise ValueError(f"Joint '{name}' not found")
    robot.joint_q[idx + 6] = value

for i in range(len(robot.joint_target_ke)):
    robot.joint_target_ke[i] = 150
    robot.joint_target_kd[i] = 5

scene = newton.ModelBuilder()
scene.replicate(robot, num_worlds, spacing=(1.5, 0.0, 0.0))
scene.add_ground_plane()
model = scene.finalize()

coords_per_world = model.joint_coord_count // num_worlds
dofs_per_world = model.joint_dof_count // num_worlds

solver = newton.solvers.SolverMuJoCoCENIC(
    model,
    tol=1e-3,
    dt_init=0.005,
    dt_min=2e-7,
    dt_max=0.02,
    njmax=50,
    nconmax=100,
    solver="newton",
    ls_parallel=False,
    ls_iterations=50,
)

state_0 = model.state()
state_1 = model.state()
control = model.control()

newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

policy = torch.jit.load(
    str(asset_path / "rl_policies" / "anymal_walking_policy_physx.pt"),
    map_location=torch_device,
)

joint_pos_initial = torch.tensor(
    state_0.joint_q[7:coords_per_world],
    device=torch_device,
    dtype=torch.float32,
).unsqueeze(0)

actions = torch.zeros(num_worlds, 12, device=torch_device, dtype=torch.float32)

lab_indices = torch.tensor(lab_to_mujoco, device=torch_device)
mujoco_indices = torch.tensor(mujoco_to_lab, device=torch_device)
gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=torch_device, dtype=torch.float32)
command = torch.zeros((1, 3), device=torch_device, dtype=torch.float32)
command[0, 0] = 1.0

all_targets = torch.zeros(num_worlds * dofs_per_world, device=torch_device, dtype=torch.float32)

LOG_EVERY_N_STEPS = 5
_grid_lines_written = 0


def print_status_grid(solver, step):
    global _grid_lines_written

    sim_times = solver.sim_time.numpy()
    dts = solver.dt.numpy()
    errors = solver.last_error.numpy()
    accepted = solver.accepted.numpy()

    col = 16
    bar = "+" + ("-" * col + "+") * 5
    header = f"{'world':>{col}}{'sim_time (s)':>{col}}{'dt (s)':>{col}}{'RMS error':>{col}}{'status':>{col}}"

    lines = [f"  step {step}  (tol={solver._tol:.1e})", bar, header, bar]
    for i in range(len(sim_times)):
        status = "ok" if accepted[i] else "REJECT"
        lines.append(
            f"{f'world {i}':>{col}}{sim_times[i]:>{col}.4f}{dts[i]:>{col}.6f}{errors[i]:>{col}.3e}{status:>{col}}"
        )
    lines.append(bar)
    if not any(accepted):
        lines.append("  WARNING: all worlds rejected — tol may be too tight or dt_min too small")

    if _grid_lines_written > 0:
        sys.stdout.write(f"\033[{_grid_lines_written}A\033[0J")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()
    _grid_lines_written = len(lines)


print(
    f"CENIC ready — {num_worlds} worlds  tol={solver._tol:.1e}  "
    f"dt_init={solver._dt.numpy()[0]:.4f}  coords/world={coords_per_world}  dofs/world={dofs_per_world}",
    flush=True,
)

viewer = newton.viewer.ViewerGL(headless=False)
viewer.set_model(model)

t = 0.0
outer_step = 0

next_policy_time = np.zeros(num_worlds, dtype=np.float32)

while viewer.is_running():
    with wp.ScopedTimer("policy", active=False):
        with torch.no_grad():
            for w in range(num_worlds):
                q_offset = w * coords_per_world
                qd_offset = w * dofs_per_world

                obs_w = compute_obs_for_world(
                    w,
                    actions[w : w + 1],
                    state_0,
                    joint_pos_initial,
                    torch_device,
                    lab_indices,
                    gravity_vec,
                    command,
                    q_offset,
                    qd_offset,
                    coords_per_world,
                    dofs_per_world,
                )
                act_w = policy(obs_w)
                actions[w] = act_w[0]

                rearranged = torch.gather(act_w, 1, mujoco_indices.unsqueeze(0))
                targets = joint_pos_initial + 0.5 * rearranged
                all_targets[w * dofs_per_world : (w + 1) * dofs_per_world] = torch.cat(
                    [
                        torch.zeros(6, device=torch_device, dtype=torch.float32),
                        targets.squeeze(0),
                    ]
                )

            wp.copy(control.joint_target_pos, wp.from_torch(all_targets, dtype=wp.float32))

    with wp.ScopedTimer("simulate", active=False):
        while True:
            state_0.clear_forces()
            viewer.apply_forces(state_0)
            solver.step(state_0, state_1, control, contacts=None)
            state_0, state_1 = state_1, state_0
            if np.all(solver.sim_time.numpy() >= next_policy_time):
                break

        next_policy_time += POLICY_DT
        t += POLICY_DT
        outer_step += 1

    if outer_step % LOG_EVERY_N_STEPS == 0:
        print_status_grid(solver, outer_step)

    with wp.ScopedTimer("render", active=False):
        viewer.begin_frame(t)
        viewer.log_state(state_0)
        viewer.end_frame()
