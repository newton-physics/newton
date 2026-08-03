# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MuJoCo Sleeping
#
# Drops four nearby stacks of ants and visualizes MuJoCo Warp's sleeping
# islands. The stacks topple toward one another in a shared world, allowing
# their islands to merge. Awake articulations use Newton's display palette,
# with a shared color for every tree in the same constraint island. Sleeping
# islands are gray.
#
# Command: python -m newton.examples mujoco_sleeping
#
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMuJoCo

STACK_COUNT = 4
ARTICULATIONS_PER_STACK = 5
STACK_SPACING = 2.5
ANT_BASE_HEIGHT = 1.0
ANT_VERTICAL_SPACING = 1.25
ANT_HORIZONTAL_OFFSET = 0.08
ANT_TILT = math.radians(4.0)
INACTIVE_COLOR = (0.32, 0.32, 0.32)

# Newton's standard shape palette (Paul Tol's Bright scheme).
ISLAND_PALETTE = (
    (68.0 / 255.0, 119.0 / 255.0, 170.0 / 255.0),
    (102.0 / 255.0, 204.0 / 255.0, 238.0 / 255.0),
    (34.0 / 255.0, 136.0 / 255.0, 51.0 / 255.0),
    (204.0 / 255.0, 187.0 / 255.0, 68.0 / 255.0),
    (238.0 / 255.0, 102.0 / 255.0, 119.0 / 255.0),
    (170.0 / 255.0, 51.0 / 255.0, 119.0 / 255.0),
    (238.0 / 255.0, 153.0 / 255.0, 51.0 / 255.0),
    (0.0, 153.0 / 255.0, 136.0 / 255.0),
)


@wp.kernel(enable_backward=False)
def update_shape_colors(
    shape_body: wp.array[wp.int32],
    body_tree: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    tree_asleep: wp.array2d[wp.int32],
    tree_island: wp.array2d[wp.int32],
    palette: wp.array[wp.vec3],
    inactive_color: wp.vec3,
    default_color: wp.array[wp.vec3],
    shape_color: wp.array[wp.vec3],
):
    """Color shapes by their live MuJoCo constraint island and sleep state."""
    shape_id = wp.tid()
    body_id = shape_body[shape_id]

    if body_id < 0:
        shape_color[shape_id] = default_color[shape_id]
        return

    tree_id = body_tree[body_id]
    world_id = body_world[body_id]
    if tree_id < 0 or world_id < 0:
        shape_color[shape_id] = default_color[shape_id]
    elif tree_asleep[world_id, tree_id] >= 0:
        shape_color[shape_id] = inactive_color
    else:
        tree_capacity = tree_asleep.shape[1]
        active_island = tree_island[world_id, tree_id]
        if active_island < 0:
            # An unconstrained awake tree is its own one-tree island.
            active_island = tree_capacity + tree_id
        island_key = world_id * (2 * tree_capacity) + active_island
        shape_color[shape_id] = palette[island_key % palette.shape[0]]


def build_articulation(asset_name: str) -> newton.ModelBuilder:
    """Build an actuated articulation whose passive tree is allowed to sleep."""
    articulation = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(articulation)
    articulation.default_shape_cfg.gap = 0.0

    root_body = len(articulation.body_mass)
    articulation.add_mjcf(
        newton.examples.get_asset(asset_name),
        ignore_names=["floor", "ground"],
        collapse_fixed_joints=False,
        parse_sites=False,
    )

    # MuJoCo prevents actuated trees from sleeping by default. This passive
    # example sends zero controls, so explicitly opt the root tree in.
    articulation.custom_attributes["mujoco:sleep_policy"].values[root_body] = SolverMuJoCo.SleepPolicy.ALLOWED
    return articulation


def build_stack() -> newton.ModelBuilder:
    """Build one clean vertical stack of five ants."""
    ant = build_articulation("nv_ant.xml")

    stack = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(stack)
    for ant_id in range(ARTICULATIONS_PER_STACK):
        height = ANT_BASE_HEIGHT + ANT_VERTICAL_SPACING * ant_id
        topple_level = max(0, ant_id - (ARTICULATIONS_PER_STACK - 3))
        offset = ANT_HORIZONTAL_OFFSET * topple_level
        tilt = ANT_TILT if topple_level > 0 else 0.0
        stack.add_builder(
            ant,
            xform=wp.transform(
                (offset, 0.0, height),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), tilt),
            ),
            label_prefix=f"ant_{ant_id}_",
        )
    return stack


class Example:
    def __init__(self, viewer, args):
        """Build the articulation stacks, sleeping solver, and island-color mapping."""
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        if not wp.get_device().is_cuda:
            raise RuntimeError("The MuJoCo sleeping example requires a CUDA device.")

        stack_count = int(args.stack_count)
        if stack_count <= 0:
            raise ValueError(f"stack_count must be positive, got {stack_count}.")
        stack = build_stack()

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_ground_plane()

        # Keep every stack in one world so contacts can merge their islands.
        side_length = math.ceil(math.sqrt(stack_count))
        row_count = math.ceil(stack_count / side_length)
        builder.begin_world(label="ant_stacks")
        for stack_id in range(stack_count):
            column = stack_id % side_length
            row = stack_id // side_length
            stack_x = (column - 0.5 * (side_length - 1)) * STACK_SPACING
            stack_y = (row - 0.5 * (row_count - 1)) * STACK_SPACING
            yaw = math.atan2(-stack_y, -stack_x) if stack_x != 0.0 or stack_y != 0.0 else 0.0
            builder.add_builder(
                stack,
                xform=wp.transform(
                    (stack_x, stack_y, 0.0),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw),
                ),
                label_prefix=f"stack_{stack_id}_",
            )
        builder.end_world()
        self.model = builder.finalize()

        self.solver = SolverMuJoCo(
            self.model,
            enable_sleeping=True,
            sleep_tolerance=float(args.sleep_tolerance),
            iterations=10,
            ls_iterations=10,
            jacobian="sparse",
            njmax=4096,
            nconmax=2048,
            update_data_interval=0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.default_shape_color = wp.clone(self.model.shape_color)
        self.island_palette = wp.array(ISLAND_PALETTE, dtype=wp.vec3, device=self.model.device)
        self.inactive_color = wp.vec3(*INACTIVE_COLOR)
        self.body_tree = self._build_body_tree_mapping()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(8.0, -10.0, 6.5), pitch=-27.0, yaw=128.0)

        self.capture()

    def _build_body_tree_mapping(self) -> wp.array[wp.int32]:
        """Map Newton body indices to MuJoCo tree indices for visualization."""
        body_tree = np.full(self.model.body_count, -1, dtype=np.int32)
        mujoco_to_newton = self.solver.mjc_body_to_newton.numpy()
        mujoco_body_tree = self.solver.mjw_model.body_treeid.numpy()
        for world_body_map in mujoco_to_newton:
            for mujoco_body, newton_body in enumerate(world_body_map):
                if newton_body >= 0:
                    body_tree[newton_body] = mujoco_body_tree[mujoco_body]

        self.body_tree_np = body_tree
        return wp.array(body_tree, dtype=wp.int32, device=self.model.device)

    def capture(self):
        """Capture the sleeping simulation step in a CUDA graph."""
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        """Advance the passive articulation stacks by one rendered frame."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance the stacks by one rendered frame."""
        wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def _update_shape_colors(self):
        """Update model display colors from MuJoCo's live island state."""
        wp.launch(
            update_shape_colors,
            dim=self.model.shape_count,
            inputs=[
                self.model.shape_body,
                self.body_tree,
                self.model.body_world,
                self.solver.mjw_data.tree_asleep,
                self.solver.mjw_data.tree_island,
                self.island_palette,
                self.inactive_color,
                self.default_shape_color,
            ],
            outputs=[self.model.shape_color],
            device=self.model.device,
        )

    def render(self):
        """Render active islands in color and sleeping islands in gray."""
        self._update_shape_colors()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify every dynamic shape color matches its tree's island state."""
        self._update_shape_colors()

        shape_body = self.model.shape_body.numpy()
        body_world = self.model.body_world.numpy()
        shape_color = self.model.shape_color.numpy()
        tree_asleep = self.solver.mjw_data.tree_asleep.numpy()
        tree_island = self.solver.mjw_data.tree_island.numpy()
        tree_capacity = tree_asleep.shape[1]
        palette = np.asarray(ISLAND_PALETTE, dtype=np.float32)

        for shape_id, body_id in enumerate(shape_body):
            if body_id < 0:
                continue
            tree_id = self.body_tree_np[body_id]
            world_id = body_world[body_id]
            if tree_id < 0 or world_id < 0:
                raise AssertionError(f"Dynamic shape {shape_id} has no MuJoCo tree/world mapping.")

            if tree_asleep[world_id, tree_id] >= 0:
                expected = np.asarray(INACTIVE_COLOR, dtype=np.float32)
            else:
                active_island = tree_island[world_id, tree_id]
                if active_island < 0:
                    active_island = tree_capacity + tree_id
                island_key = world_id * (2 * tree_capacity) + active_island
                expected = palette[island_key % len(palette)]
            np.testing.assert_allclose(shape_color[shape_id], expected, rtol=0.0, atol=1.0e-6)

    @staticmethod
    def create_parser():
        """Create command-line arguments for stack count and sleep sensitivity."""
        parser = newton.examples.create_parser()
        parser.add_argument("--stack-count", type=int, default=STACK_COUNT, help="Number of articulation stacks.")
        parser.add_argument(
            "--sleep-tolerance",
            type=float,
            default=0.05,
            help="Scaled velocity threshold below which an island can sleep.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
