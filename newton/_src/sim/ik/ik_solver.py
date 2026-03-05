# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frontend wrapper for inverse-kinematics optimizers with sampling/selection."""

from collections.abc import Sequence
from enum import Enum

import numpy as np
import warp as wp

from .ik_common import IKJacobianType
from .ik_lbfgs_optimizer import IKOptimizerLBFGS
from .ik_lm_optimizer import IKOptimizerLM
from .ik_objectives import IKObjective


class IKOptimizer(str, Enum):
    """Optimization backend selection for :class:`IKSolver`."""

    LM = "lm"
    """Use a Levenberg-Marquardt optimizer."""

    LBFGS = "lbfgs"
    """Use an L-BFGS quasi-Newton optimizer."""


class IKSampler(str, Enum):
    """Initial-seed sampling strategy."""

    NONE = "none"
    """Disable sampling and use the input seed as-is."""

    GAUSS = "gauss"
    """Perturb the input seed with Gaussian noise, clamped to joint limits."""

    ROBERTS = "roberts"
    """Use a deterministic low-discrepancy (Roberts) sequence over joint limits."""

    UNIFORM = "uniform"
    """Sample each bounded joint uniformly within its limits."""


@wp.kernel
def _sample_none_kernel(
    joint_q_in: wp.array2d(dtype=wp.float32),
    n_seeds: int,
    n_coords: int,
    joint_q_out: wp.array2d(dtype=wp.float32),
):
    expanded_idx = wp.tid()
    problem_idx = expanded_idx // n_seeds

    for coord in range(n_coords):
        joint_q_out[expanded_idx, coord] = joint_q_in[problem_idx, coord]


@wp.kernel
def _sample_gauss_kernel(
    joint_q_in: wp.array2d(dtype=wp.float32),
    n_seeds: int,
    n_coords: int,
    noise_std: float,
    joint_lower: wp.array1d(dtype=wp.float32),
    joint_upper: wp.array1d(dtype=wp.float32),
    joint_bounded: wp.array1d(dtype=wp.int32),
    base_seed: wp.array1d(dtype=wp.uint32),
    joint_q_out: wp.array2d(dtype=wp.float32),
):
    expanded_idx = wp.tid()
    problem_idx = expanded_idx // n_seeds
    seed_idx = expanded_idx % n_seeds

    for coord in range(n_coords):
        base = joint_q_in[problem_idx, coord]
        if seed_idx == 0:
            val = base
        else:
            seed = wp.int32(base_seed[0])
            offset = wp.int32(expanded_idx) * wp.int32(n_coords) + wp.int32(coord)
            state = wp.rand_init(seed, offset)
            val = base + wp.randn(state) * noise_std
            if joint_bounded[coord]:
                lo = joint_lower[coord]
                hi = joint_upper[coord]
                val = wp.min(wp.max(val, lo), hi)
        joint_q_out[expanded_idx, coord] = val


@wp.kernel
def _sample_uniform_kernel(
    n_coords: int,
    joint_lower: wp.array1d(dtype=wp.float32),
    joint_upper: wp.array1d(dtype=wp.float32),
    joint_bounded: wp.array1d(dtype=wp.int32),
    base_seed: wp.array1d(dtype=wp.uint32),
    joint_q_out: wp.array2d(dtype=wp.float32),
):
    expanded_idx = wp.tid()

    for coord in range(n_coords):
        if joint_bounded[coord]:
            lo = joint_lower[coord]
            hi = joint_upper[coord]
            span = hi - lo
            seed = wp.int32(base_seed[0])
            offset = wp.int32(expanded_idx) * wp.int32(n_coords) + wp.int32(coord)
            state = wp.rand_init(seed, offset)
            val = lo + wp.randf(state) * span
        else:
            val = 0.0
        joint_q_out[expanded_idx, coord] = val


@wp.kernel
def _sample_roberts_kernel(
    n_seeds: int,
    n_coords: int,
    roberts_basis: wp.array1d(dtype=wp.float32),
    joint_lower: wp.array1d(dtype=wp.float32),
    joint_upper: wp.array1d(dtype=wp.float32),
    joint_bounded: wp.array1d(dtype=wp.int32),
    joint_q_out: wp.array2d(dtype=wp.float32),
):
    expanded_idx = wp.tid()
    seed_idx = expanded_idx % n_seeds

    for coord in range(n_coords):
        if joint_bounded[coord]:
            lo = joint_lower[coord]
            hi = joint_upper[coord]
            span = hi - lo
            basis = roberts_basis[coord]
            val = lo + wp.mod(float(seed_idx) * basis, 1.0) * span
        else:
            val = 0.0
        joint_q_out[expanded_idx, coord] = val


@wp.kernel
def _select_best_seed_indices(
    costs: wp.array1d(dtype=wp.float32),
    n_seeds: int,
    best: wp.array1d(dtype=wp.int32),
):
    problem_idx = wp.tid()
    base = problem_idx * n_seeds
    best_seed = wp.int32(0)
    best_cost = wp.float32(costs[base])

    for seed_idx in range(1, n_seeds):
        idx = base + seed_idx
        cost = wp.float32(costs[idx])
        if cost < best_cost:
            best_cost = cost
            best_seed = wp.int32(seed_idx)

    best[problem_idx] = best_seed


@wp.kernel
def _gather_best_seed(
    joint_q_expanded: wp.array2d(dtype=wp.float32),
    best: wp.array1d(dtype=wp.int32),
    n_seeds: int,
    n_coords: int,
    joint_q_out: wp.array2d(dtype=wp.float32),
):
    problem_idx, coord_idx = wp.tid()
    best_seed = best[problem_idx]
    expanded_idx = problem_idx * n_seeds + best_seed
    joint_q_out[problem_idx, coord_idx] = joint_q_expanded[expanded_idx, coord_idx]


@wp.kernel
def _pull_seed(
    seed_state: wp.array1d(dtype=wp.uint32),
    out_seed: wp.array1d(dtype=wp.uint32),
):
    out_seed[0] = seed_state[0]
    seed_state[0] = seed_state[0] + wp.uint32(1)


@wp.kernel
def _set_seed(
    seed_state: wp.array1d(dtype=wp.uint32),
    value: wp.uint32,
):
    seed_state[0] = value


class IKSolver:
    """Flat IK front-end that handles sampling, optimization, and selection."""

    def __init__(
        self,
        model,
        n_problems: int,
        objectives: Sequence[IKObjective],
        *,
        optimizer: IKOptimizer | str = IKOptimizer.LM,
        jacobian_mode: IKJacobianType | str = IKJacobianType.AUTODIFF,
        sampler: IKSampler | str = IKSampler.NONE,
        n_seeds: int = 1,
        noise_std: float = 0.1,
        rng_seed: int = 12345,
        # LM parameters
        lambda_initial: float = 0.1,
        lambda_factor: float = 2.0,
        lambda_min: float = 1e-5,
        lambda_max: float = 1e10,
        rho_min: float = 1e-3,
        # L-BFGS parameters
        history_len: int = 10,
        h0_scale: float = 1.0,
        line_search_alphas=None,
        wolfe_c1: float = 1e-4,
        wolfe_c2: float = 0.9,
    ):
        if isinstance(optimizer, str):
            optimizer = IKOptimizer(optimizer)
        if isinstance(jacobian_mode, str):
            jacobian_mode = IKJacobianType(jacobian_mode)
        if isinstance(sampler, str):
            sampler = IKSampler(sampler)

        if n_seeds < 1:
            raise ValueError("n_seeds must be >= 1")
        if sampler is IKSampler.NONE and n_seeds != 1:
            raise ValueError("sampler 'none' requires n_seeds == 1")

        self.model = model
        self.device = model.device
        self.objectives = objectives
        self.optimizer_type = optimizer
        self.sampler = sampler
        self.n_problems = n_problems
        self.n_seeds = n_seeds
        self.n_expanded = n_problems * n_seeds
        self.n_coords = model.joint_coord_count
        self.noise_std = noise_std
        self._rng_seed = np.uint32(rng_seed)

        self.joint_q_expanded = wp.zeros((self.n_expanded, self.n_coords), dtype=wp.float32, device=self.device)
        self.best_indices = wp.zeros(self.n_problems, dtype=wp.int32, device=self.device)
        self._seed_state = wp.array(np.array([self._rng_seed], dtype=np.uint32), dtype=wp.uint32, device=self.device)
        self._seed_tmp = wp.zeros(1, dtype=wp.uint32, device=self.device)

        base_idx_np = np.repeat(np.arange(self.n_problems, dtype=np.int32), self.n_seeds)
        self.problem_idx_expanded = wp.array(base_idx_np, dtype=wp.int32, device=self.device)

        lower_np = model.joint_limit_lower.numpy()[: self.n_coords].astype(np.float32)
        upper_np = model.joint_limit_upper.numpy()[: self.n_coords].astype(np.float32)
        span_np = upper_np - lower_np
        bounded_mask_np = (np.isfinite(lower_np) & np.isfinite(upper_np) & (np.abs(span_np) < 1.0e5)).astype(np.int32)
        lower_np = np.where(bounded_mask_np, lower_np, 0.0).astype(np.float32)
        upper_np = np.where(bounded_mask_np, upper_np, 0.0).astype(np.float32)

        self.joint_lower = wp.array(lower_np, dtype=wp.float32, device=self.device)
        self.joint_upper = wp.array(upper_np, dtype=wp.float32, device=self.device)
        self.joint_bounded = wp.array(bounded_mask_np, dtype=wp.int32, device=self.device)

        if sampler is IKSampler.ROBERTS:
            roberts_basis = self._compute_roberts_basis(self.n_coords)
            self.roberts_basis = wp.array(roberts_basis, dtype=wp.float32, device=self.device)
        else:
            self.roberts_basis = None

        if optimizer is IKOptimizer.LM:
            self._impl = IKOptimizerLM(
                model,
                self.n_expanded,
                objectives,
                problem_idx=self.problem_idx_expanded,
                lambda_initial=lambda_initial,
                jacobian_mode=jacobian_mode,
                lambda_factor=lambda_factor,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                rho_min=rho_min,
            )
        elif optimizer is IKOptimizer.LBFGS:
            self._impl = IKOptimizerLBFGS(
                model,
                self.n_expanded,
                objectives,
                problem_idx=self.problem_idx_expanded,
                jacobian_mode=jacobian_mode,
                history_len=history_len,
                h0_scale=h0_scale,
                line_search_alphas=line_search_alphas,
                wolfe_c1=wolfe_c1,
                wolfe_c2=wolfe_c2,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        self.costs_expanded = self._impl.costs

    def step(self, joint_q_in: wp.array2d, joint_q_out: wp.array2d, iterations=50, step_size=1.0):
        if joint_q_in.shape != (self.n_problems, self.n_coords):
            raise ValueError("joint_q_in has incompatible shape")
        if joint_q_out.shape != (self.n_problems, self.n_coords):
            raise ValueError("joint_q_out has incompatible shape")

        self._sample(joint_q_in)

        self._impl.reset()

        if self.optimizer_type is IKOptimizer.LM:
            self._impl.step(self.joint_q_expanded, self.joint_q_expanded, iterations=iterations, step_size=step_size)
        elif self.optimizer_type is IKOptimizer.LBFGS:
            self._impl.step(self.joint_q_expanded, self.joint_q_expanded, iterations=iterations)
        else:
            raise RuntimeError(f"Unsupported optimizer: {self.optimizer_type}")

        self._impl.compute_costs(self.joint_q_expanded)

        if self.n_seeds == 1:
            if joint_q_out.ptr != self.joint_q_expanded.ptr:
                wp.copy(joint_q_out, self.joint_q_expanded)
            return

        wp.launch(
            _select_best_seed_indices,
            dim=self.n_problems,
            inputs=[self.costs_expanded, self.n_seeds],
            outputs=[self.best_indices],
            device=self.device,
        )
        wp.launch(
            _gather_best_seed,
            dim=[self.n_problems, self.n_coords],
            inputs=[self.joint_q_expanded, self.best_indices, self.n_seeds, self.n_coords],
            outputs=[joint_q_out],
            device=self.device,
        )

    def reset(self):
        self._impl.reset()
        self.best_indices.zero_()
        wp.launch(
            _set_seed,
            dim=1,
            inputs=[self._seed_state, int(self._rng_seed)],
            device=self.device,
        )

    @property
    def joint_q(self):
        return self.joint_q_expanded

    @property
    def costs(self):
        return self.costs_expanded

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def _sample(self, joint_q_in: wp.array2d):
        wp.launch(
            _pull_seed,
            dim=1,
            inputs=[self._seed_state],
            outputs=[self._seed_tmp],
            device=self.device,
        )

        if self.sampler is IKSampler.NONE:
            wp.launch(
                _sample_none_kernel,
                dim=self.n_expanded,
                inputs=[joint_q_in, self.n_seeds, self.n_coords],
                outputs=[self.joint_q_expanded],
                device=self.device,
            )
            return

        if self.sampler is IKSampler.GAUSS:
            wp.launch(
                _sample_gauss_kernel,
                dim=self.n_expanded,
                inputs=[
                    joint_q_in,
                    self.n_seeds,
                    self.n_coords,
                    self.noise_std,
                    self.joint_lower,
                    self.joint_upper,
                    self.joint_bounded,
                    self._seed_tmp,
                ],
                outputs=[self.joint_q_expanded],
                device=self.device,
            )
            return

        if self.sampler is IKSampler.UNIFORM:
            wp.launch(
                _sample_uniform_kernel,
                dim=self.n_expanded,
                inputs=[
                    self.n_coords,
                    self.joint_lower,
                    self.joint_upper,
                    self.joint_bounded,
                    self._seed_tmp,
                ],
                outputs=[self.joint_q_expanded],
                device=self.device,
            )
            return

        if self.sampler is IKSampler.ROBERTS:
            wp.launch(
                _sample_roberts_kernel,
                dim=self.n_expanded,
                inputs=[
                    self.n_seeds,
                    self.n_coords,
                    self.roberts_basis,
                    self.joint_lower,
                    self.joint_upper,
                    self.joint_bounded,
                ],
                outputs=[self.joint_q_expanded],
                device=self.device,
            )
            return

        raise RuntimeError(f"Unsupported sampler: {self.sampler}")

    @staticmethod
    def _compute_roberts_basis(n_coords: int) -> np.ndarray:
        x = 1.5
        for _ in range(20):
            f = x ** (n_coords + 1) - x - 1.0
            df = (n_coords + 1) * x**n_coords - 1.0
            x_next = x - f / df
            if abs(x_next - x) < 1.0e-12:
                break
            x = x_next
        basis = 1.0 - 1.0 / x ** (1 + np.arange(n_coords))
        return basis.astype(np.float32)
