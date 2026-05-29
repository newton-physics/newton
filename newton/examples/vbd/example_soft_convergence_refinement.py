# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Soft Convergence Under Refinement
#
# Runs the same gravity-extension scenario (a beam pinned at the top and
# hanging under its own weight) at three mesh resolutions that all discretize
# the same 1 m beam. As the mesh refines, the discrete tip extension must
# converge to the continuum linear-elastic self-weight solution
# delta = rho * g * L^2 / (2 E): the per-vertex mass lumping in add_soft_grid
# over-counts the load by (dim+1)^3 / dim^3, a factor that vanishes with
# refinement, so the extension converges DOWN to the continuum value. The test
# asserts the extension decreases monotonically toward that value and that
# successive refinement steps shrink (Cauchy convergence).
#
# Each resolution is solved with iterations proportional to its resolution:
# VBD's Gauss-Seidel sweep propagates information ~one element per iteration, so
# stress needs O(dim_z) iterations to reach the free end of the beam. The
# refinement study runs in test_final so interactive viewing stays fast.
#
# Command: python -m newton.examples vbd.example_soft_convergence_refinement
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags

# Shared material parameters. The stable Neo-Hookean material is calibrated to
# these linear Lame parameters (see
# particle_vbd_kernels.evaluate_volumetric_neo_hookean_force_and_hessian), so the
# Young's modulus used for the continuum prediction is E = mu(3*lambda+2*mu)/(lambda+mu).
_DENSITY = 1000.0
_K_MU = 5.0e4
_K_LAMBDA = 5.0e4

# Mesh resolutions: each discretizes the same 1 m beam (dim_z * cell == 1.0).
_REFINEMENT_CONFIGS = (
    {"name": "coarse", "dim_xy": 2, "dim_z": 10, "cell": 0.10},
    {"name": "medium", "dim_xy": 4, "dim_z": 20, "cell": 0.05},
    {"name": "fine", "dim_xy": 8, "dim_z": 40, "cell": 0.025},
)


def _run_extension(dim_xy: int, dim_z: int, cell: float, n_frames: int, iterations: int) -> float:
    """Run gravity extension to equilibrium and return the bottom-layer displacement."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 2.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim_xy,
        dim_y=dim_xy,
        dim_z=dim_z,
        cell_x=cell,
        cell_y=cell,
        cell_z=cell,
        density=_DENSITY,
        k_mu=_K_MU,
        k_lambda=_K_LAMBDA,
        k_damp=0.1,
    )
    builder.color()
    model = builder.finalize()
    model.soft_contact_ke = 1.0e2
    model.soft_contact_kd = 1.0e0
    model.soft_contact_mu = 1.0

    q_np = model.particle_q.numpy()
    beam_height = dim_z * cell
    top_z = 2.0 + beam_height
    top_mask = np.abs(q_np[:, 2] - top_z) < 1e-6
    flags = model.particle_flags.numpy()
    for i in np.where(top_mask)[0]:
        flags[i] = flags[i] & ~int(ParticleFlags.ACTIVE)
    model.particle_flags = wp.array(flags)

    solver = newton.solvers.SolverVBD(model=model, iterations=iterations, particle_enable_self_contact=False)
    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()
    contacts = model.contacts()
    # Even substep count keeps the CUDA-graph capture parity-safe: an odd count
    # leaves the state_0/state_1 ping-pong replaying from the wrong start buffer,
    # under-integrating each replayed frame and skewing the convergence metric.
    dt = 1.0 / 60 / 6

    if wp.get_device().is_cuda:
        with wp.ScopedCapture() as capture:
            model.collide(s0, contacts)
            for _ in range(6):
                s0.clear_forces()
                solver.step(s0, s1, ctrl, contacts, dt)
                s0, s1 = s1, s0
        graph = capture.graph
        for _ in range(n_frames):
            wp.capture_launch(graph)
    else:
        for _ in range(n_frames):
            model.collide(s0, contacts)
            for _ in range(6):
                s0.clear_forces()
                solver.step(s0, s1, ctrl, contacts, dt)
                s0, s1 = s1, s0

    q2 = s0.particle_q.numpy()
    bot_mask = np.abs(q_np[:, 2] - 2.0) < 1e-6
    return float(2.0 - np.mean(q2[bot_mask, 2]))


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        # Even substep count keeps the CUDA-graph capture parity-safe (state ping-pong).
        self.sim_substeps = 6
        self.sim_dt = self.frame_dt / self.sim_substeps
        # The visual mesh is the medium resolution; match the iteration count the
        # refinement study uses for it (5 * dim_z) so the displayed beam is converged.
        self.iterations = 100

        # Visual: medium mesh. The refinement study itself runs in test_final.
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 2.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=4,
            dim_y=4,
            dim_z=20,
            cell_x=0.05,
            cell_y=0.05,
            cell_z=0.05,
            density=_DENSITY,
            k_mu=_K_MU,
            k_lambda=_K_LAMBDA,
            k_damp=0.1,
        )
        builder.color()
        self.model = builder.finalize()

        q_np = self.model.particle_q.numpy()
        top_mask = np.abs(q_np[:, 2] - 3.0) < 1e-6
        flags = self.model.particle_flags.numpy()
        for i in np.where(top_mask)[0]:
            flags[i] = flags[i] & ~int(ParticleFlags.ACTIVE)
        self.model.particle_flags = wp.array(flags)

        self.solver = newton.solvers.SolverVBD(
            model=self.model, iterations=self.iterations, particle_enable_self_contact=False
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.model.soft_contact_ke = 1e2
        self.model.soft_contact_kd = 1e0
        self.model.soft_contact_mu = 1.0

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.model.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        # Refinement study: solve each resolution to equilibrium with iterations
        # scaled to the resolution (VBD propagates ~one element per iteration, so
        # the beam needs O(dim_z) iterations to converge).
        names: list[str] = []
        deltas: list[float] = []
        for cfg in _REFINEMENT_CONFIGS:
            delta = _run_extension(cfg["dim_xy"], cfg["dim_z"], cfg["cell"], n_frames=150, iterations=5 * cfg["dim_z"])
            names.append(cfg["name"])
            deltas.append(delta)

        for name, delta in zip(names, deltas, strict=True):
            if not np.isfinite(delta) or delta <= 0.0:
                raise ValueError(f"{name}: invalid displacement {delta}")

        # Continuum (linear-elastic) self-weight extension of the 1 m beam. The
        # discrete results over-shoot it because add_soft_grid lumps a full cell's
        # mass onto every vertex (over-load (dim+1)^3 / dim^3 -> 1 as the mesh
        # refines), so the extension must converge DOWN to this value.
        young = _K_MU * (3.0 * _K_LAMBDA + 2.0 * _K_MU) / (_K_LAMBDA + _K_MU)
        gravity_z = abs(float(self.model.gravity.numpy().reshape(-1)[2]))
        length = 1.0
        continuum = _DENSITY * gravity_z * length**2 / (2.0 * young)

        # Convergence to the continuum: the extension decreases monotonically toward
        # it from above as the mesh refines.
        if not (deltas[0] > deltas[1] > deltas[2] > continuum):
            raise ValueError(
                f"extension does not converge monotonically toward the continuum ({continuum:.4f} m): "
                + ", ".join(f"{n}={d:.4f}" for n, d in zip(names, deltas, strict=True))
            )

        # Successive refinement steps must shrink (Cauchy convergence under refinement).
        step_coarse_medium = deltas[0] - deltas[1]
        step_medium_fine = deltas[1] - deltas[2]
        if step_medium_fine >= step_coarse_medium:
            raise ValueError(
                f"refinement steps do not shrink: coarse->medium {step_coarse_medium:.4f} m, "
                f"medium->fine {step_medium_fine:.4f} m"
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer=viewer, args=args)
    newton.examples.run(example, args)
