# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example Steel Sleeping

Drops four box towers from different heights and visualizes Steel's
passive-island sleeping. The towers do not touch, so each one is its own
island and falls asleep once it has settled. A block is then thrown into
one tower and wakes only that tower. Awake bodies use their tower's
color and sleeping bodies are slate blue.

Requires a CUDA device. Use the GL viewer to observe the live sleeping colors.

Success criteria:
- Four separate box towers fall and settle on the ground.
- Settled towers turn slate blue and remain still while asleep.
- After about five seconds, the block strikes one tower; only that tower wakes and regains its color.
- The other towers stay asleep and stationary, without persistent jitter or explosive motion.

Runs:
- `python -m newton.examples steel_sleeping`
"""

import math
import warnings

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.exceptions import NewtonWarning
from newton.solvers import SolverSteel

TOWER_HEIGHTS = (6, 7, 5, 6)
TOWER_DROP_HEIGHTS = (0.0, 0.5, 1.0, 1.5)
TOWER_SPACING = 1.2
BOX_HALF_EXTENT = 0.1
BLOCK_HALF_EXTENT = 0.12
TARGET_TOWER = 1
KICK_FRAME = 300
KICK_VELOCITY = (0.0, 5.0, 3.0)

# Newton's standard shape palette (Paul Tol's Bright scheme); the thrown block uses its red entry.
TOWER_COLORS = (
    (238.0 / 255.0, 153.0 / 255.0, 51.0 / 255.0),
    (34.0 / 255.0, 136.0 / 255.0, 51.0 / 255.0),
    (204.0 / 255.0, 187.0 / 255.0, 68.0 / 255.0),
    (170.0 / 255.0, 51.0 / 255.0, 119.0 / 255.0),
)
BLOCK_COLOR = (238.0 / 255.0, 102.0 / 255.0, 119.0 / 255.0)
SLEEP_COLOR = (0.36, 0.43, 0.58)


@wp.kernel(enable_backward=False)
def apply_kick(
    frame_index: wp.array[wp.int32],
    kick_frame: int,
    body: int,
    force: wp.spatial_vector,
    body_f: wp.array[wp.spatial_vector],
):
    """Add ``force`` to ``body`` when the frame counter reaches ``kick_frame``."""
    if frame_index[0] == kick_frame:
        body_f[body] = body_f[body] + force


@wp.kernel(enable_backward=False)
def advance_frame_index(frame_index: wp.array[wp.int32]):
    """Count the simulated frames on the device so the captured graph can schedule the kick."""
    frame_index[0] = frame_index[0] + 1


@wp.kernel(enable_backward=False)
def update_shape_colors(
    shape_body: wp.array[wp.int32],
    body_awake: wp.array[wp.bool],
    awake_color: wp.array[wp.vec3],
    sleep_color: wp.vec3,
    shape_color: wp.array[wp.vec3],
):
    """Color the shapes of sleeping bodies with ``sleep_color`` and the rest with their awake color."""
    shape_id = wp.tid()
    body_id = shape_body[shape_id]
    if body_id >= 0 and not body_awake[body_id]:
        shape_color[shape_id] = sleep_color
    else:
        shape_color[shape_id] = awake_color[shape_id]


def build_scene() -> tuple[newton.ModelBuilder, np.ndarray, int]:
    """Build the towers and the block; return the builder, each body's tower (``-1`` for the block) and the block."""
    scene = newton.ModelBuilder()
    scene.default_shape_cfg.mu = 0.7
    body_tower = []
    x0 = -0.5 * (len(TOWER_HEIGHTS) - 1) * TOWER_SPACING
    for tower, (height, drop) in enumerate(zip(TOWER_HEIGHTS, TOWER_DROP_HEIGHTS, strict=True)):
        x = x0 + tower * TOWER_SPACING
        for level in range(height):
            z = BOX_HALF_EXTENT + drop + level * (2.0 * BOX_HALF_EXTENT + 0.002)
            yaw = math.radians(12.0) * (level % 2)
            body = scene.add_body(
                xform=wp.transform((x, 0.0, z), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)),
                label=f"tower_{tower}_box_{level}",
            )
            scene.add_shape_box(
                body, hx=BOX_HALF_EXTENT, hy=BOX_HALF_EXTENT, hz=BOX_HALF_EXTENT, color=TOWER_COLORS[tower]
            )
            body_tower.append(tower)

    block = scene.add_body(
        xform=wp.transform((x0 + TARGET_TOWER * TOWER_SPACING, -1.4, BLOCK_HALF_EXTENT), wp.quat_identity()),
        label="block",
    )
    scene.add_shape_box(block, hx=BLOCK_HALF_EXTENT, hy=BLOCK_HALF_EXTENT, hz=BLOCK_HALF_EXTENT, color=BLOCK_COLOR)
    body_tower.append(-1)
    return scene, np.asarray(body_tower), block


class Example:
    def __init__(self, viewer, args):
        """Build the towers, the sleeping Steel solver and the kick of the thrown block."""
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        # An even substep count leaves state_0 holding the frame's result after graph replay.
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0

        if not wp.get_device().is_cuda:
            raise RuntimeError("The Steel sleeping example requires a CUDA device.")

        scene, self.body_tower, self.block = build_scene()
        builder = newton.ModelBuilder()
        # Replicate once so the bodies belong to world 0 rather than the global world.
        builder.replicate(scene, 1)
        builder.add_ground_plane()
        self.model = builder.finalize()

        self.solver = SolverSteel(
            self.model,
            solve_mode="matrix_free",
            articulated_contact_response="immediate",
            enable_sleeping=True,
        )
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.observables = self.solver.observables({SolverSteel.ObservableKind.BODY_AWAKE})

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # A one-substep force that changes the block's velocity by KICK_VELOCITY.
        block_mass = float(self.model.body_mass.numpy()[self.block])
        kick = block_mass * np.asarray(KICK_VELOCITY) / self.sim_dt
        self.kick_force = wp.spatial_vector(*kick, 0.0, 0.0, 0.0)
        self.frame_index = wp.zeros(1, dtype=wp.int32)

        self.awake_shape_color = wp.clone(self.model.shape_color)
        self.sleep_color = wp.vec3(*SLEEP_COLOR)

        if isinstance(self.viewer, newton.viewer.ViewerRTX):
            warnings.warn(
                "ViewerRTX keeps the first frame's shape colors, so sleeping bodies keep their awake color.",
                NewtonWarning,
                stacklevel=2,
            )
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -5.2, 2.0), pitch=-14.0, yaw=90.0)

        # Per-frame history for the test checks.
        self.previous_body_q = None
        self.previous_body_awake = None
        self.towers_asleep_at_kick = False
        self.towers_woken_by_kick = set()
        self.sleeping_body_moved_frame = None

        self.capture()

    def capture(self):
        """Capture one frame, including the kick of the thrown block, in a CUDA graph."""
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        """Launch one rendered frame of substeps; ``capture()`` records it into the CUDA graph."""
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if substep == 0:
                wp.launch(
                    apply_kick,
                    dim=1,
                    inputs=[self.frame_index, KICK_FRAME, self.block, self.kick_force],
                    outputs=[self.state_0.body_f],
                )
            self.collision_pipeline.collide(self.state_0, self.contacts)
            # Only the last substep's flags are rendered.
            observables = self.observables if substep == self.sim_substeps - 1 else None
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt, observables=observables
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(advance_frame_index, dim=1, outputs=[self.frame_index])

    def step(self):
        """Advance the towers by one rendered frame by replaying the CUDA graph."""
        wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt
        self.frame += 1

    def render(self):
        """Render awake bodies in their tower's color and sleeping bodies in slate blue."""
        wp.launch(
            update_shape_colors,
            dim=self.model.shape_count,
            inputs=[self.model.shape_body, self.observables.body_awake, self.awake_shape_color, self.sleep_color],
            outputs=[self.model.shape_color],
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Record which towers wake after the kick and whether a sleeping body changes its pose."""
        body_q = self.state_0.body_q.numpy()
        body_awake = self.observables.body_awake.numpy()
        if self.previous_body_q is not None:
            asleep = ~body_awake & ~self.previous_body_awake
            frozen = body_q[asleep].tobytes() == self.previous_body_q[asleep].tobytes()
            if not frozen and self.sleeping_body_moved_frame is None:
                self.sleeping_body_moved_frame = self.frame
        if self.frame == KICK_FRAME:
            self.towers_asleep_at_kick = not body_awake[self.body_tower >= 0].any()
        if self.frame > KICK_FRAME:
            self.towers_woken_by_kick.update(int(tower) for tower in np.unique(self.body_tower[body_awake]))
        self.previous_body_q = body_q
        self.previous_body_awake = body_awake

    def test_final(self):
        """Verify every tower slept before the kick, only the hit tower woke and sleeping bodies did not move."""
        if self.frame <= KICK_FRAME:
            raise AssertionError(f"Run more than {KICK_FRAME} frames to reach the kick.")
        if np.any(self.solver.constraint_overflow.numpy()):
            raise AssertionError("The solver dropped constraint rows.")
        if not self.towers_asleep_at_kick:
            raise AssertionError("A tower was still awake when the block was thrown.")
        self.towers_woken_by_kick.discard(-1)
        if self.towers_woken_by_kick != {TARGET_TOWER}:
            raise AssertionError(
                f"The kick woke towers {sorted(self.towers_woken_by_kick)}, expected [{TARGET_TOWER}]."
            )
        if self.sleeping_body_moved_frame is not None:
            raise AssertionError(f"A sleeping body changed its pose at frame {self.sleeping_body_moved_frame}.")


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
