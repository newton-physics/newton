# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Score a population of candidates against the recorded goals.

:class:`CableEvaluator` connects a search to the simulation. The search gives it
a list of candidates and gets one objective value per candidate back. The search
does not need to know about rods, solvers, cameras or losses, so it can come
from outside Newton, together with its own dependencies.

The evaluator scores all candidates of a population together. One
:class:`~.model.CableWorld` holds every candidate as a separate world, so each
goal group needs one build, one settle and one rollout per population, not one
per candidate. The population size is therefore the main cost of a search step.

The evaluator prepares the goals once, in the constructor. Data that depends only
on the recording, such as a distance field, is the same for every candidate, so
it is not computed again for each one.

:meth:`CableEvaluator.record` runs the same rollout for one candidate and keeps
the frames and the per-frame losses. It returns arrays and writes no files, so
image formats and plotting libraries stay outside this package.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .goal import CableGoal, CableGoalGroup, group_goals
from .model import CableWorld


@dataclass
class CableCandidate:
    """One candidate parameter set."""

    angles: list[tuple[float, float]]
    """Rest configuration as ``[(alpha, beta), ...]`` [rad], one pair per capsule."""

    bend_stiffness: float
    """Per-joint bend stiffness [N·m/rad]."""

    twist_stiffness: float
    """Per-joint twist stiffness [N·m/rad]."""

    bend_damping: float
    """Per-joint bend damping [N·m·s/rad]."""

    twist_damping: float
    """Per-joint twist damping [N·m·s/rad]."""


@dataclass
class CableTraceView:
    """One view's record of a single candidate's rollout."""

    label: str
    """Label of the goal this view comes from."""

    crop: list[int]
    """``[x0, y0, x1, y1]`` region scored in this view [px]."""

    weight: float
    """This view's share of the objective; see :func:`~.goal.group_goals`."""

    loss: float
    """The view's objective value, before weighting."""

    frames: list[np.ndarray]
    """Rendered sensor frames, one per recorded simulation frame."""

    masks: list[np.ndarray]
    """Full-frame uint8 cable masks (0 or 255), aligned with :attr:`frames`."""

    goal_indices: list[int]
    """Index of the reference frame that pairs with each recorded frame."""

    frame_losses: list[tuple[float, float]]
    """``(time [s], loss)`` of each scored frame, taken from the rollout.

    Their sum divided by the number of reference frames is :attr:`loss`. A loss
    recomputed from the saved masks need not match.
    """


def _goal_index_for_sim_frame(f: int, frame_times: Sequence[float], fps: float) -> int:
    """Return the reference frame captured nearest to simulation frame ``f``.

    This is the reverse of the pairing :meth:`~.model.CableWorld.run_sequence`
    uses.

    Args:
        f: Simulation frame index.
        frame_times: Capture time of each reference frame [s].
        fps: Simulation frame rate [Hz].

    Returns:
        Index of the reference frame.
    """
    t = f / fps
    return min(range(len(frame_times)), key=lambda j: abs(frame_times[j] - t))


class CableEvaluator:
    """Scores candidate populations against a fixed set of goals.

    Args:
        goals: The :class:`~.goal.CableGoal` entries to score against.
        loss: The objective, a ``TuningLoss``.
        settle_frames: Maximum number of frames spent settling before frame 0.
            0 does not settle, so each rollout starts from the straight initial
            cable.
        settle_mode: How to reach the initial equilibrium. Only ``"dynamic"``
            is supported.
        sim_iterations: Solver iterations per substep.
        stretch_stiffness: Per-joint stretch stiffness of the rod [N/m]. It is
            held fixed, not searched.
        num_elements: Capsule count. It is also the number of rest-angle pairs
            per candidate.
        segment_length: Length of each capsule [m].
        cable_radius: Rod radius [m].
        cable_mass: Total cable mass [kg].
        angle_parametrization: How a rest-angle pair becomes a joint rotation.
            Only ``"exp_map"`` is supported.
        fps: Simulation frame rate [Hz]; see :class:`~.model.CableWorld`.
        sim_substeps: Solver substeps per frame; see :class:`~.model.CableWorld`.
        settle_check_every: Number of frames between two settle convergence
            checks; see :class:`~.model.CableWorld`.
        settle_move_tol: Settle convergence tolerance [m]; see
            :class:`~.model.CableWorld`.

    Raises:
        ValueError: If ``goals`` is empty, a goal fails
            :meth:`~.goal.CableGoal.validate` or has no ``sensor_quat``, or
            :func:`~.goal.group_goals` rejects the goals.
    """

    def __init__(
        self,
        goals: Sequence[CableGoal],
        loss: Any,
        *,
        settle_frames: int,
        settle_mode: str,
        sim_iterations: int,
        stretch_stiffness: float,
        num_elements: int,
        segment_length: float,
        cable_radius: float,
        cable_mass: float,
        angle_parametrization: str,
        fps: int = 60,
        sim_substeps: int = 20,
        settle_check_every: int = 25,
        settle_move_tol: float = 1.0e-3,
    ) -> None:
        if not goals:
            raise ValueError("CableEvaluator needs at least one goal.")
        for goal in goals:
            goal.validate()
            if goal.sensor_quat is None:
                raise ValueError(f"goal '{goal.label}': the simulated camera needs sensor_quat.")
        self.loss = loss
        self.settle_frames = settle_frames
        self.settle_mode = settle_mode
        self.sim_iterations = sim_iterations
        self.stretch_stiffness = stretch_stiffness
        # The geometry sizes the decision vector, so every candidate of a
        # population must share it.
        self.num_elements = num_elements
        self.segment_length = segment_length
        self.cable_radius = cable_radius
        self.cable_mass = cable_mass
        self.angle_parametrization = angle_parametrization

        self.fps = fps
        self.sim_substeps = sim_substeps
        self.settle_check_every = settle_check_every
        self.settle_move_tol = settle_move_tol
        self.groups: list[CableGoalGroup] = group_goals(goals, fps, sim_substeps)
        # Prepared once and shared by every candidate in every generation.
        self.goal_reprs: list[list[list[Any]]] = [
            [[loss.prepare(m) for m in v.masks] for v in g.views] for g in self.groups
        ]

    def evaluate(self, candidates: Sequence[CableCandidate]) -> list[float]:
        """Return the objective value of each candidate.

        Args:
            candidates: The :class:`CableCandidate` population to score.

        Returns:
            One objective value per candidate, in the order given. An empty
            population gives an empty list.

        Raises:
            ValueError: If :class:`~.model.CableWorld` rejects the
                configuration, for example an unsupported
                ``angle_parametrization`` or a ``clamp_position`` outside the cable.
        """
        if not candidates:
            return []
        n = len(candidates)
        totals = [0.0] * n

        for gi, group in enumerate(self.groups):
            world = self._build_world(group, candidates)
            world.settle(self.settle_frames)
            reprs, crops, times = group.run_args(self.goal_reprs[gi])
            losses, _, _ = world.run_sequence(group.num_frames, reprs, crops, self.loss, goal_times=times)
            for weight, per_world in zip(group.weights, group.per_view(losses), strict=True):
                for i, value in enumerate(per_world):
                    totals[i] += weight * value
        return totals

    def record(self, candidate: CableCandidate, *, record_fps: float = math.inf) -> list[CableTraceView]:
        """Roll out one candidate and keep its frames and per-frame losses.

        Runs a full rollout per goal group and renders every recorded frame,
        also for a loss that does not use the render. Use it to inspect one
        candidate, not to score a population.

        Args:
            candidate: The :class:`CableCandidate` to run.
            record_fps: Recorded frames per second [Hz]. A rate of ``fps``
                or more, such as the default ``math.inf``, records every
                simulated frame.

        Returns:
            One :class:`CableTraceView` per view, in group then view order.

        Raises:
            ValueError: If :class:`~.model.CableWorld` rejects the
                configuration, for example an unsupported
                ``angle_parametrization`` or a ``clamp_position`` outside the
                cable, or if ``record_fps`` is not positive.
        """
        views = []
        for gi, group in enumerate(self.groups):
            world = self._build_world(group, [candidate])
            world.settle(self.settle_frames)
            reprs, crops, times = group.run_args(self.goal_reprs[gi])
            losses, recorded, recorded_masks = world.run_sequence(
                group.num_frames,
                reprs,
                crops,
                self.loss,
                record_fps=record_fps,
                goal_times=times,
            )
            rec_idx = world.record_indices(group.num_frames, record_fps)
            for ci, (view, weight, per_world_frames, per_world_masks, per_world_loss) in enumerate(
                zip(
                    group.views,
                    group.weights,
                    group.per_view(recorded),
                    group.per_view(recorded_masks),
                    group.per_view(losses),
                    strict=True,
                )
            ):
                views.append(
                    CableTraceView(
                        label=view.label,
                        crop=list(view.crop),
                        weight=weight,
                        loss=per_world_loss[0],
                        frames=per_world_frames[0],
                        masks=per_world_masks[0],
                        goal_indices=[_goal_index_for_sim_frame(f, view.frame_times, self.fps) for f in rec_idx],
                        frame_losses=[(float(t), float(v)) for t, v in world.last_frame_losses[ci]],
                    )
                )
        return views

    def _build_world(self, group: CableGoalGroup, candidates: Sequence[CableCandidate]) -> CableWorld:
        """Build one :class:`~.model.CableWorld` that holds the whole population."""
        return CableWorld(
            group.cable_start,
            [c.angles for c in candidates],
            [c.bend_stiffness for c in candidates],
            group.transform_buffer,
            attachment_transform=group.attachment_transform,
            clamp_position=group.clamp_position,
            cable_axis=group.cable_axis,
            cameras=group.cameras(),
            bend_damping_list=[c.bend_damping for c in candidates],
            twist_stiffness_list=[c.twist_stiffness for c in candidates],
            twist_damping_list=[c.twist_damping for c in candidates],
            settle_mode=self.settle_mode,
            sim_iterations=self.sim_iterations,
            stretch_stiffness=self.stretch_stiffness,
            num_elements=self.num_elements,
            segment_length=self.segment_length,
            cable_radius=self.cable_radius,
            cable_mass=self.cable_mass,
            angle_parametrization=self.angle_parametrization,
            fps=self.fps,
            sim_substeps=self.sim_substeps,
            settle_check_every=self.settle_check_every,
            settle_move_tol=self.settle_move_tol,
        )
