# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# RigidBodySim: Generic Kamino rigid body simulator for RL
#
# Consolidates common simulation infrastructure (model building, solver
# setup, state/control tensor wiring, contact aggregation, selective
# resets, CUDA graphs) into a single reusable class.  Any robot RL
# example can use this instead of duplicating ~300 lines of boilerplate.
###########################################################################

from __future__ import annotations

# Python
import glob
import os
import threading

# Thirdparty
import torch  # noqa: TID253
import warp as wp

# Kamino
import newton
from newton._src.solvers.kamino._src.core.bodies import convert_body_com_to_origin
from newton._src.solvers.kamino._src.core.control import ControlKamino
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.core.state import StateKamino
from newton._src.solvers.kamino._src.core.types import transformf, vec6f
from newton._src.solvers.kamino._src.geometry import CollisionDetector
from newton._src.solvers.kamino._src.geometry.aggregation import ContactAggregation
from newton._src.solvers.kamino._src.solver_kamino_impl import SolverKaminoImpl
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.sim import Simulator
from newton._src.solvers.kamino._src.utils.viewer import Color3, ViewerConfig
from newton._src.viewer import ViewerGL


class SimulatorFromNewton:
    """Kamino :class:`Simulator`-like wrapper initialized from a Newton :class:`~newton.Model`.

    Mirrors the core API of the Kamino ``Simulator`` class but accepts an
    already-finalized :class:`newton.Model` instead of a ``ModelBuilderKamino``.
    Internally uses :meth:`ModelKamino.from_newton` to obtain Kamino-native
    model, state, control, and solver objects.
    """

    def __init__(
        self,
        newton_model: newton.Model,
        config: Simulator.Config | None = None,
        device: wp.DeviceLike = None,
    ):
        self._device = wp.get_device(device)

        if config is None:
            config = Simulator.Config()
        self._config = config

        # Create Kamino model from Newton model
        self._model: ModelKamino = ModelKamino.from_newton(newton_model)
        if isinstance(config.dt, float):
            self._model.time.set_uniform_timestep(config.dt)

        # Allocate state and control
        self._state_p: StateKamino = self._model.state(device=self._device)
        self._state_n: StateKamino = self._model.state(device=self._device)
        self._control: ControlKamino = self._model.control(device=self._device)

        # Collision detection
        self._collision_detector = CollisionDetector(
            model=self._model,
            config=config.collision_detector,
        )
        self._contacts = self._collision_detector.contacts

        # Solver
        self._solver = SolverKaminoImpl(
            model=self._model,
            contacts=self._contacts,
            config=config.solver,
        )

        # Initialize state
        self._solver.reset(state_out=self._state_n)
        self._state_p.copy_from(self._state_n)

    @property
    def model(self) -> ModelKamino:
        return self._model

    @property
    def state(self) -> StateKamino:
        """Current (next-step) state."""
        return self._state_n

    @property
    def state_previous(self) -> StateKamino:
        """Previous-step state."""
        return self._state_p

    @property
    def control(self) -> ControlKamino:
        return self._control

    @property
    def contacts(self):
        return self._contacts

    @property
    def collision_detector(self) -> CollisionDetector:
        return self._collision_detector

    @property
    def solver(self) -> SolverKaminoImpl:
        return self._solver

    def step(self):
        """Advance the simulation by one timestep.

        ``q_i`` is kept in COM-frame throughout (matching ``q_i_0`` and
        the internal solver).  The COM→body-frame conversion is done
        only for rendering via :meth:`RigidBodySim.render`.
        """
        self._state_p.copy_from(self._state_n)
        self._solver.step(
            state_in=self._state_p,
            state_out=self._state_n,
            control=self._control,
            contacts=self._contacts,
            detector=self._collision_detector,
        )

    def reset(self, **kwargs):
        """Reset the simulation state.

        Keyword arguments are forwarded to :meth:`SolverKaminoImpl.reset`
        (e.g. ``world_mask``, ``joint_q``, ``joint_u``, ``base_q``, ``base_u``).
        """
        self._solver.reset(state_out=self._state_n, **kwargs)
        self._state_p.copy_from(self._state_n)


class RigidBodySim:
    """Generic Kamino rigid body simulator for RL.

    Features:
        * USD model loading via ``newton.ModelBuilder.add_usd``
        * ``ModelKamino.from_newton`` for Kamino-native state/control layout
        * Configurable solver settings with sensible RL defaults
        * Zero-copy PyTorch views of state, control and contact arrays
        * Automatic extraction of actuated joint metadata
        * Selective per-world reset infrastructure (world mask + deferred buffers)
        * Optional CUDA graph capture for step and reset
        * Optional Newton ViewerGL

    Args:
        usd_model_path: Full filesystem path to the USD model file.
        num_worlds: Number of parallel simulation worlds.
        sim_dt: Physics timestep in seconds.
        device: Warp device (e.g. ``"cuda:0"``).  ``None`` → preferred device.
        headless: If ``True``, skip viewer creation.
        body_pose_offset: Optional ``(x, y, z, qx, qy, qz, qw)`` tuple to
            offset every body's initial pose (e.g. to place the robot above
            the ground plane).
        add_ground: Add a ground-plane to each world.
        enable_gravity: Enable gravity in every world.
        settings: Simulator settings.  ``None`` uses ``default_settings(sim_dt)``.
        use_cuda_graph: Capture CUDA graphs for step and reset (requires
            CUDA device with memory pool enabled).
        render_config: Viewer appearance settings.  ``None`` uses defaults.
    """

    def __init__(
        self,
        usd_model_path: str,
        num_worlds: int = 1,
        sim_dt: float = 0.01,
        device: wp.DeviceLike = None,
        headless: bool = False,
        body_pose_offset: tuple | None = None,
        add_ground: bool = True,
        enable_gravity: bool = True,
        settings: Simulator.Config | None = None,
        use_cuda_graph: bool = False,
        record_video: bool = False,
        video_folder: str | None = None,
        async_save: bool = True,
        max_contacts_per_pair: int | None = None,
        render_config: ViewerConfig | None = None,
        collapse_fixed_joints: bool = False,
    ):
        # ----- Device setup -----
        self._device = wp.get_device(device)
        self._torch_device: str = "cuda" if self._device.is_cuda else "cpu"
        self._use_cuda_graph = use_cuda_graph
        self._sim_dt = sim_dt

        # ----- Video recording -----
        self._record_video = record_video
        self._video_folder = video_folder or "./frames"
        self._async_save = async_save
        self._frame_buffer = None
        self._img_idx = 0
        if self._record_video:
            os.makedirs(self._video_folder, exist_ok=True)

        # Resolve settings (subclass override via default_settings)
        if settings is None:
            settings = self.default_settings(sim_dt)

        # ----- Build Newton model (needed for ViewerGL) -----
        msg.notif("Constructing builder from imported USD ...")
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
        robot_builder.default_shape_cfg.margin = 0.0
        robot_builder.default_shape_cfg.gap = 0.0

        robot_builder.add_usd(
            usd_model_path,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
            collapse_fixed_joints=collapse_fixed_joints,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        # Create the multi-world model by duplicating the single-robot
        # builder for the specified number of worlds
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(num_worlds):
            builder.add_world(robot_builder)

        # Add a global ground plane applied to all worlds
        builder.add_ground_plane()

        # Create the model from the builder
        self._newton_model = builder.finalize(skip_validation_joints=True)

        # TODO remove after fixing bug
        self._newton_model.shape_margin.fill_(0.0)
        self._newton_model.shape_gap.fill_(1e-5)

        if enable_gravity:
            self._newton_model.set_gravity((0.0, 0.0, -9.81))

        # ----- Create Kamino simulator from Newton model -----
        msg.notif("Building Kamino simulator ...")

        # Cap per-pair contact count to limit Delassus matrix size
        if max_contacts_per_pair is not None:
            settings.collision_detector.max_contacts_per_pair = max_contacts_per_pair

        self.sim = SimulatorFromNewton(
            newton_model=self._newton_model,
            config=settings,
            device=self._device,
        )
        self.model: ModelKamino = self.sim.model
        msg.info(f"Model size: {self.sim.model.size}")
        msg.info(f"Contacts capacity: {self.sim.contacts.model_max_contacts_host}")

        # Apply body pose offset to initial body poses (affects all resets).
        # Kamino q_i stores COM positions, so we offset the COM directly.
        if body_pose_offset is not None:
            offset_z = body_pose_offset[2]
            q_i_0 = self.sim.model.bodies.q_i_0
            q_i_0_np = q_i_0.numpy().copy()
            q_i_0_np[:, 2] += offset_z
            q_i_0.assign(q_i_0_np)
            # Re-initialize state from the offset initial poses
            self.sim.reset()

        # ----- Wire RL interface (zero-copy tensors) -----
        self._make_rl_interface()

        # ----- Extract metadata -----
        self._extract_metadata()

        # ----- Viewer -----
        self.viewer: ViewerGL | None = None
        self._newton_state: newton.State | None = None
        self._render_config = render_config or ViewerConfig()
        if not headless:
            msg.notif("Creating the 3D viewer ...")
            self.viewer = ViewerGL()
            self.viewer.set_model(self._newton_model)
            # Newton state used only for rendering (body_q synced from Kamino each frame)
            self._newton_state = self._newton_model.state()
            self._apply_render_config(self._render_config)

        # ----- CUDA graphs -----
        self._reset_graph = None
        self._step_graph = None
        self._capture_graphs()

        # ----- Warm-up (compiles Warp kernels) -----
        msg.notif("Warming up simulator ...")
        self.step()
        self.reset()

    # ------------------------------------------------------------------
    # Viewer appearance
    # ------------------------------------------------------------------

    def _apply_render_config(self, cfg: ViewerConfig):
        """Apply render configuration to the viewer."""
        viewer = self.viewer
        renderer = viewer.renderer

        # Shape colors (robot only)
        if cfg.robot_color is not None:
            model = self._newton_model
            shape_body = model.shape_body.numpy()
            color_overrides: dict[int, Color3] = {}
            for s in range(model.shape_count):
                if int(shape_body[s]) >= 0:
                    color_overrides[s] = cfg.robot_color
            if color_overrides:
                viewer.update_shape_colors(color_overrides)

        # Lighting settings
        if cfg.diffuse_scale is not None:
            renderer.diffuse_scale = cfg.diffuse_scale
        if cfg.specular_scale is not None:
            renderer.specular_scale = cfg.specular_scale
        if cfg.shadow_radius is not None:
            renderer.shadow_radius = cfg.shadow_radius
        if cfg.shadow_extents is not None:
            renderer.shadow_extents = cfg.shadow_extents
        if cfg.spotlight_enabled is not None:
            renderer.spotlight_enabled = cfg.spotlight_enabled

        # Sky color (renderer.sky_upper = "Sky Color" in viewer GUI)
        if cfg.sky_color is not None:
            renderer.sky_upper = cfg.sky_color

        # Directional light color
        if cfg.light_color is not None:
            renderer._light_color = cfg.light_color

        # Background brightness — scales ground color (renderer.sky_lower) and ground plane shapes
        if cfg.background_brightness_scale is not None:
            s = cfg.background_brightness_scale
            renderer.sky_lower = tuple(min(c * s, 1.0) for c in renderer.sky_lower)
            # Also brighten ground plane shape colors
            model = self._newton_model
            shape_body = model.shape_body.numpy()
            ground_colors: dict[int, Color3] = {}
            for s_idx in range(model.shape_count):
                if int(shape_body[s_idx]) < 0:
                    cur = viewer.model_shape_color.numpy()[int(viewer._shape_to_slot[s_idx])]
                    ground_colors[s_idx] = tuple(min(float(c) * s, 1.0) for c in cur)
            if ground_colors:
                viewer.update_shape_colors(ground_colors)

    # ------------------------------------------------------------------
    # RL interface wiring
    # ------------------------------------------------------------------

    def _make_rl_interface(self):
        """Create zero-copy PyTorch views of simulator state, control and contact arrays."""
        nw = self.sim.model.size.num_worlds
        njd = self.sim.model.size.max_of_num_joint_dofs
        nb = self.sim.model.size.max_of_num_bodies

        # State tensors (read-only views into simulator)
        self._q_j = wp.to_torch(self.sim.state.q_j).reshape(nw, njd)
        self._dq_j = wp.to_torch(self.sim.state.dq_j).reshape(nw, njd)
        self._q_i = wp.to_torch(self.sim.state.q_i).reshape(nw, nb, 7)
        self._u_i = wp.to_torch(self.sim.state.u_i).reshape(nw, nb, 6)

        # Control tensors (writable views)
        self._q_j_ref = wp.to_torch(self.sim.control.q_j_ref).reshape(nw, njd)
        self._dq_j_ref = wp.to_torch(self.sim.control.dq_j_ref).reshape(nw, njd)
        self._tau_j_ref = wp.to_torch(self.sim.control.tau_j_ref).reshape(nw, njd)

        # World mask for selective resets
        self._world_mask_wp = wp.zeros((nw,), dtype=wp.int32, device=self._device)
        self._world_mask = wp.to_torch(self._world_mask_wp)

        # Reset buffers
        self._reset_base_q_wp = wp.zeros(nw, dtype=transformf, device=self._device)
        self._reset_base_u_wp = wp.zeros(nw, dtype=vec6f, device=self._device)
        self._reset_q_j = torch.zeros((nw, njd), device=self._torch_device)
        self._reset_dq_j = torch.zeros((nw, njd), device=self._torch_device)
        self._reset_base_q = wp.to_torch(self._reset_base_q_wp).reshape(nw, 7)
        self._reset_base_u = wp.to_torch(self._reset_base_u_wp).reshape(nw, 6)

        # Reset flags
        self._update_q_j = False
        self._update_dq_j = False
        self._update_base_q = False
        self._update_base_u = False

        # Contact aggregation
        self._contact_aggregation = ContactAggregation(
            model=self.sim.model, contacts=self.sim.contacts, device=self._device
        )
        self._contact_flags = wp.to_torch(self._contact_aggregation.body_contact_flag).reshape(nw, nb)
        self._ground_contact_flags = wp.to_torch(self._contact_aggregation.body_static_contact_flag).reshape(nw, nb)
        self._net_contact_forces = wp.to_torch(self._contact_aggregation.body_net_force).reshape(nw, nb, 3)
        self._body_pair_contact_flag: torch.Tensor | None = None  # Set via set_body_pair_contact_filter()

        # Default joint positions (cloned from initial state)
        self._default_q_j = self._q_j.clone()

        # Environment origins for multi-env offsets
        self._env_origins = torch.zeros((nw, 3), device=self._torch_device)

        # External wrenches (zero-copy view)
        self._w_e_i = wp.to_torch(self.sim.solver.data.bodies.w_e_i).reshape(nw, nb, 6)

        # Body masses (zero-copy view)
        self._mass = wp.to_torch(self.sim.model.bodies.m_i).reshape(nw, nb)

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    def _extract_metadata(self):
        """Extract joint/body names, actuated DOF indices, and joint limits from the Kamino model."""
        max_joints = self.sim.model.size.max_of_num_joints
        max_bodies = self.sim.model.size.max_of_num_bodies

        # Read per-joint metadata from the Kamino model (first world only)
        joint_labels = [lbl.rsplit("/", 1)[-1] for lbl in self.sim.model.joints.label[:max_joints]]
        joint_num_dofs = wp.to_torch(self.sim.model.joints.num_dofs)[:max_joints].tolist()
        joint_act_type = wp.to_torch(self.sim.model.joints.act_type)[:max_joints].tolist()
        joint_q_j_min = wp.to_torch(self.sim.model.joints.q_j_min)
        joint_q_j_max = wp.to_torch(self.sim.model.joints.q_j_max)

        # Joint names and actuated indices
        self._joint_names: list[str] = []
        self._actuated_joint_names: list[str] = []
        self._actuated_dof_indices: list[int] = []
        dof_offset = 0
        for j in range(max_joints):
            ndofs = int(joint_num_dofs[j])
            self._joint_names.append(joint_labels[j])
            if int(joint_act_type[j]) > 0:  # act_type > PASSIVE means actuated
                self._actuated_joint_names.append(joint_labels[j])
                for dof_idx in range(ndofs):
                    self._actuated_dof_indices.append(dof_offset + dof_idx)
            dof_offset += ndofs

        self._actuated_dof_indices_tensor = torch.tensor(
            self._actuated_dof_indices, device=self._torch_device, dtype=torch.long
        )

        msg.info(f"Actuated joints ({self.num_actuated}): {self._actuated_joint_names}")

        # Body names
        self._body_names: list[str] = [lbl.rsplit("/", 1)[-1] for lbl in self.sim.model.bodies.label[:max_bodies]]

        # Joint limits (per-DOF, first world only)
        num_dofs_total = self.sim.model.size.max_of_num_joint_dofs
        self._joint_limits: list[list[float]] = []
        for d in range(num_dofs_total):
            lower = float(joint_q_j_min[d])
            upper = float(joint_q_j_max[d])
            self._joint_limits.append([lower, upper])

    # ------------------------------------------------------------------
    # CUDA graph capture
    # ------------------------------------------------------------------

    def _capture_graphs(self):
        """Capture CUDA graphs for step and reset if requested and available."""
        if not self._use_cuda_graph:
            return
        if not (self._device.is_cuda and wp.is_mempool_enabled(self._device)):
            msg.warning("CUDA graphs requested but not available (need CUDA device with mempool). Using kernels.")
            return

        msg.notif("Capturing CUDA graphs ...")
        with wp.ScopedCapture(device=self._device) as reset_capture:
            self._reset_worlds()
        self._reset_graph = reset_capture.graph

        with wp.ScopedCapture(device=self._device) as step_capture:
            self.sim.step()
            self._contact_aggregation.compute()
        self._step_graph = step_capture.graph

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def step(self):
        """Execute a single physics step (simulator + contact aggregation).

        Flushes any pending resets via :meth:`apply_resets` first, then runs
        the physics step.  Uses CUDA graph replay if available.
        """
        self.apply_resets()
        if self._step_graph:
            wp.capture_launch(self._step_graph)
        else:
            self.sim.step()
            self._contact_aggregation.compute()

    def reset(self):
        """Full reset of all worlds to initial state."""
        self._world_mask.fill_(1)
        self._reset_worlds()
        self._world_mask.zero_()

    def apply_resets(self):
        """Apply pending selective resets staged via :meth:`set_dof` / :meth:`set_root`.

        Applies resets for worlds marked in :attr:`world_mask`, then clears
        the mask and all update flags.  Call this before :meth:`step` when
        using deferred resets.
        """
        if self._reset_graph:
            wp.capture_launch(self._reset_graph)
        else:
            self._reset_worlds()
        self._world_mask.zero_()
        self._update_q_j = False
        self._update_dq_j = False
        self._update_base_q = False
        self._update_base_u = False

    def _reset_worlds(self):
        """Reset selected worlds based on world_mask."""
        self.sim.reset(
            world_mask=self._world_mask_wp,
            joint_q=wp.from_torch(self._reset_q_j.view(-1)) if self._update_q_j else None,
            joint_u=wp.from_torch(self._reset_dq_j.view(-1)) if self._update_dq_j else None,
            base_q=wp.from_torch(self._reset_base_q.view(-1, 7)) if self._update_base_q else None,
            base_u=wp.from_torch(self._reset_base_u.view(-1, 6)) if self._update_base_u else None,
        )

    def render(self):
        """Render the current frame if viewer exists."""
        if self.viewer is not None:
            self.viewer.begin_frame(self.time)
            # Kamino q_i is COM-frame; ViewerGL expects body-frame-origin poses.
            convert_body_com_to_origin(
                body_com=self.sim.model.bodies.i_r_com_i,
                body_q_com=self.sim.state.q_i,
                body_q=self._newton_state.body_q,
            )
            self.viewer.log_state(self._newton_state)
            self.viewer.end_frame()
            if self._record_video:
                self._capture_frame()

    def _capture_frame(self):
        """Capture and save the current rendered frame as a PNG.

        If ``render_width``/``render_height`` in the render config differ from
        the viewer's native resolution, the renderer is temporarily resized to
        the target resolution, re-rendered, captured, then restored — giving
        true high-res capture without affecting the display window.
        """
        try:
            # Thirdparty
            from PIL import Image
        except ImportError:
            msg.warning("PIL not installed. Install with: pip install pillow")
            return

        renderer = self.viewer.renderer
        target_w = self._render_config.render_width
        target_h = self._render_config.render_height
        native_w = renderer._screen_width
        native_h = renderer._screen_height
        needs_hires = target_w != native_w or target_h != native_h

        if needs_hires:
            # Resize FBOs to target resolution and re-render the scene
            renderer.resize(target_w, target_h)
            renderer.render(self.viewer.camera, self.viewer.objects, self.viewer.lines)
            # Invalidate PBO and cached buffer so they get reallocated at new size
            self.viewer._pbo = None
            self.viewer._wp_pbo = None
            self._frame_buffer = None

        frame = self.viewer.get_frame(target_image=self._frame_buffer)
        if self._frame_buffer is None:
            self._frame_buffer = frame

        if needs_hires:
            # Restore native (window) resolution and invalidate PBO again
            renderer.resize()
            self.viewer._pbo = None
            self.viewer._wp_pbo = None

        frame_np = frame.numpy()
        image = Image.fromarray(frame_np, mode="RGB")
        filename = os.path.join(self._video_folder, f"{self._img_idx:05d}.png")

        if self._async_save:
            threading.Thread(target=image.save, args=(filename,), daemon=False).start()
        else:
            image.save(filename)

        self._img_idx += 1

    def generate_video(self, output_filename: str = "recording.mp4", fps: int = 60, keep_frames: bool = True) -> bool:
        """Generate MP4 video from recorded PNG frames using imageio-ffmpeg.

        Args:
            output_filename: Name of output video file.
            fps: Frames per second for video.
            keep_frames: If ``True``, keep PNG frames after video creation.
        """
        try:
            # Thirdparty
            import imageio_ffmpeg as ffmpeg  # noqa: PLC0415
        except ImportError:
            msg.warning("imageio-ffmpeg not installed. Install with: pip install imageio-ffmpeg")
            return False
        try:
            # Thirdparty
            from PIL import Image
        except ImportError:
            msg.warning("PIL not installed. Install with: pip install pillow")
            return False
        # Thirdparty
        import numpy as np  # noqa: PLC0415

        frame_files = sorted(glob.glob(os.path.join(self._video_folder, "*.png")))
        if not frame_files:
            msg.warning(f"No PNG frames found in {self._video_folder}")
            return False

        # Read first frame to get dimensions; ensure even for libx264 yuv420p
        first_img = Image.open(frame_files[0])
        w, h = first_img.size
        # libx264 with yuv420p requires even width and height
        even_w = w if w % 2 == 0 else w + 1
        even_h = h if h % 2 == 0 else h + 1
        needs_pad = even_w != w or even_h != h
        size = (even_w, even_h)

        msg.info(f"Generating video from {len(frame_files)} frames at {even_w}x{even_h}...")
        try:
            writer = ffmpeg.write_frames(
                output_filename,
                size=size,
                fps=fps,
                codec="libx264",
                macro_block_size=1,
                quality=5,
            )
            writer.send(None)
            for frame_path in frame_files:
                img = Image.open(frame_path)
                frame = np.array(img)
                if needs_pad:
                    padded = np.zeros((even_h, even_w, frame.shape[2]), dtype=frame.dtype)
                    padded[:h, :w] = frame
                    frame = padded
                writer.send(frame)
            writer.close()
            msg.info(f"Video generated: {output_filename}")

            if not keep_frames:
                for frame_path in frame_files:
                    os.remove(frame_path)
                msg.info("Frames deleted")

            return True
        except Exception as e:
            msg.warning(f"Failed to generate video: {e}")
            return False

    @property
    def time(self) -> float:
        """Current simulation time."""
        return getattr(self, "_sim_time", 0.0)

    def is_running(self) -> bool:
        """Check if the viewer is still running (always ``True`` in headless mode)."""
        if self.viewer is None:
            return True
        return self.viewer.is_running()

    # ------------------------------------------------------------------
    # Deferred reset staging
    # ------------------------------------------------------------------

    def set_dof(
        self,
        dof_positions: torch.Tensor | None = None,
        dof_velocities: torch.Tensor | None = None,
        env_ids: torch.Tensor | list[int] | None = None,
    ):
        """Stage joint state for deferred reset.

        The actual reset happens on the next call to :meth:`apply_resets`.

        Args:
            dof_positions: Joint positions ``(len(env_ids), num_joint_dofs)``.
            dof_velocities: Joint velocities ``(len(env_ids), num_joint_dofs)``.
            env_ids: Which worlds to reset.  ``None`` resets all.
        """
        if env_ids is None:
            self._world_mask.fill_(1)
            ids = slice(None)
        else:
            self._world_mask[env_ids] = 1
            ids = env_ids

        if dof_positions is not None:
            self._update_q_j = True
            self._reset_q_j[ids] = dof_positions
        if dof_velocities is not None:
            self._update_dq_j = True
            self._reset_dq_j[ids] = dof_velocities

    def set_root(
        self,
        root_positions: torch.Tensor | None = None,
        root_orientations: torch.Tensor | None = None,
        root_linear_velocities: torch.Tensor | None = None,
        root_angular_velocities: torch.Tensor | None = None,
        env_ids: torch.Tensor | list[int] | None = None,
    ):
        """Stage root body state for deferred reset.

        The actual reset happens on the next call to :meth:`apply_resets`.

        Args:
            root_positions: Root positions ``(len(env_ids), 3)``.
            root_orientations: Root orientations ``(len(env_ids), 4)`` (quaternion).
            root_linear_velocities: Root linear velocities ``(len(env_ids), 3)``.
            root_angular_velocities: Root angular velocities ``(len(env_ids), 3)``.
            env_ids: Which worlds to reset.  ``None`` resets all.
        """
        if env_ids is None:
            self._world_mask.fill_(1)
            ids = slice(None)
        else:
            self._world_mask[env_ids] = 1
            ids = env_ids

        if root_positions is not None or root_orientations is not None:
            self._update_base_q = True
            # Copy current state as baseline
            self._reset_base_q[ids] = self._q_i[ids, 0, :7]
            if root_positions is not None:
                self._reset_base_q[ids, :3] = root_positions
            if root_orientations is not None:
                self._reset_base_q[ids, 3:] = root_orientations

        if root_linear_velocities is not None or root_angular_velocities is not None:
            self._update_base_u = True
            self._reset_base_u[ids] = self._u_i[ids, 0, :6]
            if root_linear_velocities is not None:
                self._reset_base_u[ids, :3] = root_linear_velocities
            if root_angular_velocities is not None:
                self._reset_base_u[ids, 3:] = root_angular_velocities

    # ------------------------------------------------------------------
    # State properties (zero-copy torch views)
    # ------------------------------------------------------------------

    @property
    def q_j(self) -> torch.Tensor:
        """Joint positions ``(num_worlds, num_joint_coords)``."""
        return self._q_j

    @property
    def dq_j(self) -> torch.Tensor:
        """Joint velocities ``(num_worlds, num_joint_dofs)``."""
        return self._dq_j

    @property
    def q_i(self) -> torch.Tensor:
        """Body poses ``(num_worlds, num_bodies, 7)`` — position + quaternion."""
        return self._q_i

    @property
    def u_i(self) -> torch.Tensor:
        """Body twists ``(num_worlds, num_bodies, 6)`` — linear + angular velocity."""
        return self._u_i

    # ------------------------------------------------------------------
    # Control properties (zero-copy torch views)
    # ------------------------------------------------------------------

    @property
    def q_j_ref(self) -> torch.Tensor:
        """Joint position reference ``(num_worlds, num_joint_coords)`` for implicit PD."""
        return self._q_j_ref

    @property
    def dq_j_ref(self) -> torch.Tensor:
        """Joint velocity reference ``(num_worlds, num_joint_dofs)`` for implicit PD."""
        return self._dq_j_ref

    @property
    def tau_j_ref(self) -> torch.Tensor:
        """Joint torque reference ``(num_worlds, num_joint_dofs)`` for feed-forward control."""
        return self._tau_j_ref

    # ------------------------------------------------------------------
    # Contact properties
    # ------------------------------------------------------------------

    @property
    def contact_flags(self) -> torch.Tensor:
        """Per-body contact flags ``(num_worlds, num_bodies)``."""
        return self._contact_flags

    @property
    def ground_contact_flags(self) -> torch.Tensor:
        """Per-body ground contact flags ``(num_worlds, num_bodies)``."""
        return self._ground_contact_flags

    @property
    def net_contact_forces(self) -> torch.Tensor:
        """Net contact forces ``(num_worlds, num_bodies, 3)``."""
        return self._net_contact_forces

    @property
    def body_pair_contact_flag(self) -> torch.Tensor:
        """Per-world body-pair contact flag ``(num_worlds,)``."""
        return self._body_pair_contact_flag

    def set_body_pair_contact_filter(self, body_a_name: str, body_b_name: str) -> None:
        """Configure detection of contacts between two named bodies.

        Must be called after construction. The detection runs outside the
        CUDA graph via :meth:`compute_body_pair_contacts`.

        Args:
            body_a_name: Name of the first body.
            body_b_name: Name of the second body.
        """
        a_idx = self.find_body_index(body_a_name)
        b_idx = self.find_body_index(body_b_name)
        self._contact_aggregation.set_body_pair_filter(a_idx, b_idx)
        self._body_pair_contact_flag = wp.to_torch(self._contact_aggregation.body_pair_contact_flag)

    def compute_body_pair_contacts(self) -> None:
        """Run body-pair contact detection (call after physics step)."""
        self._contact_aggregation.compute_body_pair_contact()

    # ------------------------------------------------------------------
    # Metadata properties
    # ------------------------------------------------------------------

    @property
    def num_worlds(self) -> int:
        return self.sim.model.size.num_worlds

    @property
    def num_joint_coords(self) -> int:
        return self.sim.model.size.max_of_num_joint_coords

    @property
    def num_joint_dofs(self) -> int:
        return self.sim.model.size.max_of_num_joint_dofs

    @property
    def num_bodies(self) -> int:
        return self.sim.model.size.max_of_num_bodies

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def body_names(self) -> list[str]:
        return self._body_names

    @property
    def actuated_joint_names(self) -> list[str]:
        return self._actuated_joint_names

    @property
    def actuated_dof_indices(self) -> list[int]:
        return self._actuated_dof_indices

    @property
    def actuated_dof_indices_tensor(self) -> torch.Tensor:
        """Actuated DOF indices as a ``torch.long`` tensor on the simulation device."""
        return self._actuated_dof_indices_tensor

    @property
    def num_actuated(self) -> int:
        return len(self._actuated_dof_indices)

    @property
    def env_origins(self) -> torch.Tensor:
        """Environment origins ``(num_worlds, 3)``."""
        return self._env_origins

    @property
    def external_wrenches(self) -> torch.Tensor:
        """External wrenches ``(num_worlds, num_bodies, 6)``."""
        return self._w_e_i

    @property
    def body_masses(self) -> torch.Tensor:
        """Body masses ``(num_worlds, num_bodies)``."""
        return self._mass

    @property
    def default_q_j(self) -> torch.Tensor:
        """Default joint positions ``(num_worlds, num_joint_coords)`` cloned at init."""
        return self._default_q_j

    @property
    def joint_limits(self) -> list[list[float]]:
        """Per-joint ``[lower, upper]`` limits."""
        return self._joint_limits

    @property
    def torch_device(self) -> str:
        """Torch device string (``"cuda"`` or ``"cpu"``)."""
        return self._torch_device

    @property
    def device(self):
        """Warp device."""
        return self._device

    @property
    def sim_dt(self) -> float:
        return self._sim_dt

    @property
    def world_mask(self) -> torch.Tensor:
        """World mask ``(num_worlds,)`` int32 for selective resets."""
        return self._world_mask

    # ------------------------------------------------------------------
    # Name lookup helpers
    # ------------------------------------------------------------------

    def find_body_index(self, name: str) -> int:
        """Return the index of the body with the given *name*.

        Raises ``ValueError`` if not found.
        """
        try:
            return self._body_names.index(name)
        except ValueError:
            raise ValueError(f"Body '{name}' not found. Available: {self._body_names}") from None

    def find_body_indices(self, names: list[str]) -> list[int]:
        """Return indices for a list of body *names*."""
        return [self.find_body_index(n) for n in names]

    # ------------------------------------------------------------------
    # Default solver settings
    # ------------------------------------------------------------------

    @staticmethod
    def default_settings(sim_dt: float = 0.01) -> Simulator.Config:
        """Return sensible default solver settings for RL."""
        settings = Simulator.Config()
        settings.dt = sim_dt
        settings.solver.sparse_jacobian = True
        settings.solver.use_fk_solver = False
        settings.solver.use_collision_detector = True
        settings.collision_detector.pipeline = "unified"
        settings.collision_detector.max_contacts_per_pair = 8
        settings.solver.integrator = "moreau"
        settings.solver.constraints.alpha = 0.1
        settings.solver.padmm.primal_tolerance = 1e-4
        settings.solver.padmm.dual_tolerance = 1e-4
        settings.solver.padmm.compl_tolerance = 1e-4
        settings.solver.padmm.max_iterations = 200
        settings.solver.padmm.eta = 1e-5
        settings.solver.padmm.rho_0 = 0.05
        settings.solver.padmm.use_acceleration = True
        settings.solver.padmm.warmstart_mode = "containers"
        settings.solver.padmm.contact_warmstart_method = "geom_pair_net_force"
        settings.solver.collect_solver_info = False
        settings.solver.compute_solution_metrics = False
        settings.solver.padmm.use_graph_conditionals = False
        return settings
