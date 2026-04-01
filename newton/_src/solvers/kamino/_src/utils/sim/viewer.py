# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The customized debug viewer of Kamino"""

# Python
import glob
import os
import threading
from typing import ClassVar

# Thirdparty
import warp as wp

from ......geometry.types import GeoType
from ......viewer import ViewerGL
from ...core.builder import ModelBuilderKamino
from ...core.geometry import GeometryDescriptor
from ...core.types import vec3f
from ...core.world import WorldDescriptor
from ...geometry.contacts import ContactMode
from ...utils import logger as msg
from .simulator import Simulator

###
# Kernels
###


@wp.kernel
def compute_contact_box_transforms(
    # Kamino contact data
    position_A: wp.array(dtype=wp.vec3),  # Contact position on body A
    position_B: wp.array(dtype=wp.vec3),  # Contact position on body B
    frame: wp.array(dtype=wp.quatf),  # Contact frames
    mode: wp.array(dtype=wp.int32),  # Contact modes
    wid: wp.array(dtype=wp.int32),
    num_contacts: int,
    world_spacing: wp.vec3,
    box_size: wp.vec3,  # Box dimensions
    # Output buffers
    transforms: wp.array(dtype=wp.transform),
    scales: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
):
    """
    Compute transforms, scales, and colors for contact frame boxes.
    """
    i = wp.tid()

    # Hide contacts beyond the active count or with INACTIVE mode
    contact_mode = mode[i]
    if i >= num_contacts or contact_mode == wp.int32(ContactMode.INACTIVE):
        scales[i] = wp.vec3(0.0, 0.0, 0.0)
        colors[i] = wp.vec3(0.0, 0.0, 0.0)
        transforms[i] = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        return

    # Contact position - we could also use the midpoint?
    contact_pos = position_B[i]

    # Apply world spacing
    world_id = float(wid[i])
    contact_pos = contact_pos + world_spacing * world_id

    # Contact frame rotation
    q = frame[i]

    # Set transform
    transforms[i] = wp.transform(contact_pos, q)

    # Set scale
    scales[i] = box_size

    # Set color based on contact mode
    if contact_mode == wp.int32(ContactMode.OPENING):
        # White
        colors[i] = wp.vec3(1.0, 1.0, 1.0)
    elif contact_mode == wp.int32(ContactMode.STICKING):
        # Black
        colors[i] = wp.vec3(0.1, 0.1, 0.1)
    elif contact_mode == wp.int32(ContactMode.SLIDING):
        # Blue
        colors[i] = wp.vec3(0.404, 0.647, 0.953)
    else:
        # Unknown mode: Gray
        colors[i] = wp.vec3(0.5, 0.5, 0.5)


@wp.kernel
def compute_contact_force_arrows(
    # Kamino contact data
    position_A: wp.array(dtype=wp.vec3),
    position_B: wp.array(dtype=wp.vec3),
    frame: wp.array(dtype=wp.quatf),  # Contact frames
    reaction: wp.array(dtype=wp.vec3),  # Contact forces in respective local contact frame
    mode: wp.array(dtype=wp.int32),  # Contact modes
    wid: wp.array(dtype=wp.int32),
    num_contacts: int,
    world_spacing: wp.vec3,
    force_scale: float,
    force_threshold: float,  # Minimum force to display
    # Output buffers
    line_starts: wp.array(dtype=wp.vec3),
    line_ends: wp.array(dtype=wp.vec3),
    line_colors: wp.array(dtype=wp.vec3),
    line_widths: wp.array(dtype=float),
):
    """
    Compute line segments for visualizing contact forces as arrows.
    """
    i = wp.tid()

    if i >= num_contacts:
        return

    # Skip inactive contacts
    if mode[i] == wp.int32(ContactMode.INACTIVE):
        line_starts[i] = wp.vec3(0.0, 0.0, 0.0)
        line_ends[i] = wp.vec3(0.0, 0.0, 0.0)
        line_widths[i] = 0.0
        return

    # Contact position - we could also use the midpoint?
    contact_pos = position_B[i]

    # Apply world spacing
    world_id = float(wid[i])
    contact_pos = contact_pos + world_spacing * world_id

    # Transform force from contact frame to world frame
    # reaction is in contact frame, need to rotate by frame quaternion
    q = frame[i]
    C = wp.quat_to_matrix(q)
    f_world = C * reaction[i]

    # Compute force magnitude
    f_mag = wp.length(f_world)

    # Only render if force is above threshold
    if f_mag < force_threshold:
        line_starts[i] = wp.vec3(0.0, 0.0, 0.0)
        line_ends[i] = wp.vec3(0.0, 0.0, 0.0)
        line_widths[i] = 0.0
        return

    # linear - Nonlinear scaling # todo make this be an option?
    scaled_length = force_scale * f_mag
    # scaled_length = force_scale * wp.sqrt(f_mag)

    # Force direction
    f_dir = f_world / f_mag

    # Arrow from contact point along force direction
    line_starts[i] = contact_pos
    line_ends[i] = contact_pos + f_dir * scaled_length

    # Magenta color for forces
    line_colors[i] = wp.vec3(1.0, 0.0, 1.0)

    # Line width proportional to force magnitude but clipped, requires modification in viewer_gl to work properly
    # We could also use this to actually visualize something meaningful
    line_widths[i] = wp.clamp(1.0 + 0.1 * f_mag, 1.0, 5.0)


###
# Interfaces
###


class ViewerKamino(ViewerGL):
    """
    A customized debug viewer for Kamino.
    """

    # Define a static set of colors for different bodies
    body_colors: ClassVar[list[wp.array]] = [
        wp.array([wp.vec3(0.9, 0.1, 0.3)], dtype=wp.vec3),  # Crimson Red
        wp.array([wp.vec3(0.1, 0.7, 0.9)], dtype=wp.vec3),  # Cyan Blue
        wp.array([wp.vec3(1.0, 0.5, 0.0)], dtype=wp.vec3),  # Orange
        wp.array([wp.vec3(0.6, 0.2, 0.8)], dtype=wp.vec3),  # Purple
        wp.array([wp.vec3(0.2, 0.8, 0.2)], dtype=wp.vec3),  # Green
        wp.array([wp.vec3(0.8, 0.8, 0.2)], dtype=wp.vec3),  # Yellow
        wp.array([wp.vec3(0.8, 0.2, 0.8)], dtype=wp.vec3),  # Magenta
        wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3),  # Gray
    ]

    # Define a static world spacing offset for multiple worlds
    world_spacing: ClassVar[vec3f] = vec3f(-2.0, 0.0, 0.0)

    def __init__(
        self,
        builder: ModelBuilderKamino,
        simulator: Simulator,
        width: int = 1920,
        height: int = 1080,
        vsync: bool = False,
        headless: bool = False,
        show_contacts: bool = False,
        record_video: bool = False,
        video_folder: str | None = None,
        skip_img_idx: int = 0,
        async_save: bool = False,
    ):
        """
        Initialize the Kamino viewer.

        Args:
            builder: Model builder.
            simulator: The simulator instance to visualize.
            width: Window width in pixels.
            height: Window height in pixels.
            vsync: Enable vertical sync.
            headless: Run without displaying a window.
            show_contacts: Enable contact point visualization (default: False).
            record_video: Enable frame recording to disk.
            video_folder: Directory to save recorded frames (default: "./frames").
            skip_img_idx: Number of initial frames to skip before recording.
            async_save: Save frames asynchronously in background threads.
        """
        # Initialize the base viewer
        super().__init__(width=width, height=height, vsync=vsync, headless=headless)

        # Cache references to the simulator
        self._simulator = simulator

        # Declare and initialize geometry info cache
        self._worlds: list[WorldDescriptor] = builder.worlds
        self._geometry: list[GeometryDescriptor] = builder.geoms

        # Initialize video recording settings
        self._record_video = record_video
        self._video_folder = video_folder or "./frames"
        self._async_save = async_save
        self._skip_img_idx = skip_img_idx
        self._img_idx = 0
        self._frame_buffer = None

        # Contact visualization settings
        self._show_contacts = show_contacts

        if self._record_video:
            os.makedirs(self._video_folder, exist_ok=True)

    def render_geometry(self, body_poses: wp.array, geom: GeometryDescriptor, scope: str):
        # TODO: Fix this
        bid = geom.body + self._worlds[geom.wid].bodies_idx_offset if geom.body >= 0 else -1

        # Handle the case of static geometry (bid < 0)
        if bid < 0:
            body_transform = wp.transform_identity()
        else:
            body_transform = wp.transform(*body_poses[bid])

        # Retrieve the geometry ID as a float
        wid = float(geom.wid)

        # Apply world spacing based on world ID
        offset_transform = wp.transform(self.world_spacing * wid, wp.quat_identity())

        # Combine body and offset transforms
        geom_transform = wp.transform_multiply(body_transform, geom.offset)
        geom_transform = wp.transform_multiply(offset_transform, geom_transform)

        # Choose color based on body ID
        color = self.body_colors[geom.body % len(self.body_colors)]

        # Shape params are already in Newton convention (half-extents, half-heights)
        params = geom.shape.params

        # Update the geometry data
        self.log_shapes(
            name=f"/world_{geom.wid}/body_{geom.body}/{scope}/{geom.gid}-{geom.name}",
            geo_type=geom.shape.type,
            geo_scale=params,
            xforms=wp.array([geom_transform], dtype=wp.transform),
            geo_is_solid=geom.shape.is_solid,
            colors=color,
            geo_src=geom.shape.data,
        )

    def render_frame(self, stop_recording: bool = False):
        # Begin a new frame
        self.begin_frame(self.time)

        # Extract body poses from the kamino simulator
        body_poses = self._simulator.state.q_i.numpy()

        # Render each collision geom
        for geom in self._geometry:
            if geom.shape.type == GeoType.NONE:
                continue
            self.render_geometry(body_poses, geom, scope="geoms")

        # Render contacts if they exist and visualization is enabled
        if hasattr(self._simulator, "contacts") and self._simulator.contacts is not None:
            self.render_contacts_kamino(self._simulator.contacts)

        # End the new frame
        self.end_frame()

        # Capture frame if recording is enabled and not stopped
        if self._record_video and not stop_recording:
            # todo : think about if we should continue to step the _img_idx even when not recording
            self._capture_frame()

    def render_contacts_kamino(self, contacts):
        """
        Render contact points, frames, and forces for contacts.

        Visualizations include:
        - Small oriented boxes showing contact frame by mode
        - Force arrows showing contact force magnitude and direction
        """
        if not self._show_contacts:
            # Hide all contact visualizations
            if hasattr(self, "_contact_box_mesh_created"):
                self.log_instances("/contact_boxes", "/contact_box_mesh", None, None, None, materials=None, hidden=True)
            self.log_lines("/contact_forces", None, None, None)
            return

        # Get number of active contacts
        num_contacts = contacts.model_active_contacts.numpy()[0]
        max_contacts = contacts.model_max_contacts_host

        if False:  # Debug: Always print contact info
            print(f"[VIEWER] Frame {getattr(self, '_frame', 0)}: num_contacts={num_contacts} (max={max_contacts})")

            # Print all contact slots
            modes = contacts.mode.numpy()[:max_contacts]
            positions = contacts.position_B.numpy()[:max_contacts]
            velocities = contacts.velocity.numpy()[:max_contacts]
            reactions = contacts.reaction.numpy()[:max_contacts]

            for i in range(max_contacts):
                active = "ACTIVE" if i < num_contacts else "STALE"
                print(
                    f"  [{active}] Contact[{i}]: mode={modes[i]} (INACTIVE={ContactMode.INACTIVE}), "
                    f"pos={positions[i]}, vel={velocities[i]}, reaction={reactions[i]}"
                )

            self._frame = getattr(self, "_frame", 0) + 1

        # ======================================================================
        # Render Contact Frame Boxes
        # ======================================================================

        # Allocate buffers for box transforms
        if not hasattr(self, "_contact_box_transforms"):
            self._contact_box_transforms = wp.zeros(max_contacts, dtype=wp.transform, device=self.device)
            self._contact_box_scales = wp.zeros(max_contacts, dtype=wp.vec3, device=self.device)
            self._contact_box_colors = wp.zeros(max_contacts, dtype=wp.vec3, device=self.device)

        # Render boxes as instanced meshes
        if not hasattr(self, "_contact_box_mesh_created"):
            # Unit box mesh
            points, indices_wp = self._create_box_mesh_simple(1.0, 1.0, 1.0)
            self.log_mesh(
                "/contact_box_mesh",
                points,
                indices_wp,
                normals=None,
                hidden=True,
            )
            self._contact_box_mesh_created = True

        # Log instances of the box mesh
        if num_contacts > 0:
            # small scaled unit box to show frame orientation
            box_size = wp.vec3(
                0.025, 0.025, 0.025
            )  # a little bit flat would look better? todo should we have like a viewer config somewhere?

            # Compute box transforms, scales, and colors
            wp.launch(
                kernel=compute_contact_box_transforms,
                dim=max_contacts,
                inputs=[
                    contacts.position_A,
                    contacts.position_B,
                    contacts.frame,
                    contacts.mode,
                    contacts.wid,
                    num_contacts,
                    self.world_spacing,
                    box_size,
                ],
                outputs=[
                    self._contact_box_transforms,
                    self._contact_box_scales,
                    self._contact_box_colors,
                ],
                device=self.device,
            )

            # Always render all max_contacts instances, not just active ones
            # Inactive ones will have zero scale from the kernel
            xforms = self._contact_box_transforms
            scales = self._contact_box_scales
            colors = self._contact_box_colors
            self.log_instances(
                "/contact_boxes",
                "/contact_box_mesh",
                xforms,
                scales,
                colors,
                materials=None,
                hidden=False,
            )
        else:
            # Hide instances when no contacts
            if hasattr(self, "_contact_box_mesh_created"):
                self.log_instances(
                    "/contact_boxes",
                    "/contact_box_mesh",
                    None,
                    None,
                    None,
                    materials=None,
                    hidden=True,
                )

        # ======================================================================
        # Render Contact Force Arrow
        # ======================================================================

        # Allocate buffers for force arrows
        if not hasattr(self, "_contact_force_starts"):
            self._contact_force_starts = wp.zeros(max_contacts, dtype=wp.vec3, device=self.device)
            self._contact_force_ends = wp.zeros(max_contacts, dtype=wp.vec3, device=self.device)
            self._contact_force_colors = wp.zeros(max_contacts, dtype=wp.vec3, device=self.device)
            self._contact_force_widths = wp.zeros(max_contacts, dtype=float, device=self.device)

        # Compute force arrows
        wp.launch(
            kernel=compute_contact_force_arrows,
            dim=max_contacts,
            inputs=[
                contacts.position_A,
                contacts.position_B,
                contacts.frame,
                contacts.reaction,
                contacts.mode,
                contacts.wid,
                num_contacts,
                self.world_spacing,
                0.05,  # force_scale # todo move to a cfg file?
                1e-4,  # force_threshold # todo move to a cfg file?
            ],
            outputs=[
                self._contact_force_starts,
                self._contact_force_ends,
                self._contact_force_colors,
                self._contact_force_widths,
            ],
            device=self.device,
        )

        # Render force arrows as lines
        if num_contacts > 0:
            self.log_lines(
                "/contact_forces",
                self._contact_force_starts[:num_contacts],
                self._contact_force_ends[:num_contacts],
                self._contact_force_colors[:num_contacts],
                width=3.0,  # todo this assumes we fix the viewer_gl line width issue
            )
        else:
            self.log_lines("/contact_forces", None, None, None)

    def set_camera_lookat(self, pos: wp.vec3, target: wp.vec3):
        """
        Set the camera position and orient it to face a specific target.

        Args:
            pos: The camera position.
            target: The point the camera should look at.
        """
        # Calculate the direction vector from camera to target
        dir = wp.normalize(target - pos)

        # Calculate camera angles
        yaw = wp.degrees(wp.atan2(dir[1], dir[0]))
        pitch = wp.degrees(wp.asin(dir[2]))

        # Call basic set camera method
        self.set_camera(pos, pitch, yaw)

    def _create_box_mesh_simple(self, sx, sy, sz):
        """
        # todo where should this function go, is it already implemented somewhere else?
        Helper to create a simple box mesh for contact visualization using warp.
        Returns (vertices, indices) as warp arrays.
        """
        # Create vertex array (8 corners of box)
        verts = wp.array(
            [
                wp.vec3(-0.5 * sx, -0.5 * sy, -0.5 * sz),  # 0
                wp.vec3(0.5 * sx, -0.5 * sy, -0.5 * sz),  # 1
                wp.vec3(0.5 * sx, 0.5 * sy, -0.5 * sz),  # 2
                wp.vec3(-0.5 * sx, 0.5 * sy, -0.5 * sz),  # 3
                wp.vec3(-0.5 * sx, -0.5 * sy, 0.5 * sz),  # 4
                wp.vec3(0.5 * sx, -0.5 * sy, 0.5 * sz),  # 5
                wp.vec3(0.5 * sx, 0.5 * sy, 0.5 * sz),  # 6
                wp.vec3(-0.5 * sx, 0.5 * sy, 0.5 * sz),  # 7
            ],
            dtype=wp.vec3,
            device=self.device,
        )

        # Create index array (12 triangles, flattened)
        indices = wp.array(
            [
                # Bottom face
                0,
                2,
                1,
                0,
                3,
                2,
                # Top face
                4,
                5,
                6,
                4,
                6,
                7,
                # Front face
                0,
                1,
                5,
                0,
                5,
                4,
                # Back face
                3,
                7,
                6,
                3,
                6,
                2,
                # Left face
                0,
                4,
                7,
                0,
                7,
                3,
                # Right face
                1,
                2,
                6,
                1,
                6,
                5,
            ],
            dtype=wp.int32,
            device=self.device,
        )

        return verts, indices

    def _capture_frame(self):
        """
        Capture and save a single frame from the viewer.

        This method retrieves the current rendered frame, converts it to a PIL Image,
        and saves it as a PNG file.
        """
        # Attempt to import PIL, which is required for image saving
        try:
            from PIL import Image
        except ImportError:
            msg.warning("PIL not installed. Frames cannot be saved as images.")
            msg.info("Install with: pip install pillow")
            return False

        # Only capture and save if we've reached the skip threshold
        if self._img_idx >= self._skip_img_idx:
            # Get frame from viewer as GPU array (height, width, 3) uint8
            frame = self.get_frame(target_image=self._frame_buffer)

            # Cache buffer for reuse to minimize allocations
            if self._frame_buffer is None:
                self._frame_buffer = frame

            # Convert to numpy on CPU and PIL
            frame_np = frame.numpy()
            image = Image.fromarray(frame_np, mode="RGB")

            # Generate filename with zero-padded frame number # todo : 05d is currently hardcoded
            filename = os.path.join(self._video_folder, f"{self._img_idx - self._skip_img_idx:05d}.png")

            # Save either asynchronously or synchronously
            if self._async_save:
                # Use non-daemon thread to save in background
                # Each image has its own copy, so thread safety is maintained
                threading.Thread(
                    target=image.save,
                    args=(filename,),
                    daemon=False,  # make sure the thread completes even if main program exits todo can be challenged
                ).start()
            else:
                # Synchronous save - blocks until complete
                image.save(filename)

        self._img_idx += 1

    def generate_video(self, output_filename: str = "recording.mp4", fps: int = 60, keep_frames: bool = True) -> bool:
        """
        Generate MP4 video from recorded png frames using imageio-ffmpeg.

        Args:
            output_filename: Name of output video file (default: "recording.mp4")
            fps: Frames per second for video (default: 60)
            keep_frames: If True, keep png frames after video creation; if False, delete them (default: True)
        """
        # Try to import imageio-ffmpeg (optional dependency)
        try:
            import imageio_ffmpeg as ffmpeg  # noqa: PLC0415
        except ImportError:
            msg.warning("imageio-ffmpeg not installed. Frames saved but video not generated.")
            msg.info("Install with: pip install imageio-ffmpeg")
            return False
        # Try to import PIL (optional dependency for image loading)
        try:
            from PIL import Image
        except ImportError:
            msg.warning("PIL not installed. Frames saved but video not generated.")
            msg.info("Install with: pip install pillow")
            return False
        import numpy as np  # noqa: PLC0415

        # Check if we have frames to process
        if not self._record_video or self._img_idx <= self._skip_img_idx:
            msg.warning("No frames recorded, cannot generate video")
            return False

        # Get sorted list of frame files
        frame_files = sorted(glob.glob(os.path.join(self._video_folder, "*.png")))

        if not frame_files:
            msg.warning(f"No png frames found in {self._video_folder}")
            return False

        msg.info(f"Generating video from {len(frame_files)} frames...")
        try:
            # Use imageio-ffmpeg to write video
            writer = ffmpeg.write_frames(
                output_filename,
                size=(self.renderer._screen_width, self.renderer._screen_height),
                fps=fps,
                codec="libx264",
                macro_block_size=8,
                quality=5,  # set to default quality
            )
            writer.send(None)  # Initialize the writer

            # Read each frame and send each frame from and to disk
            for frame_path in frame_files:
                img = Image.open(frame_path)
                frame_array = np.array(img)
                writer.send(frame_array)

            writer.close()
            msg.info(f"Video generated successfully: {output_filename}")

            if not keep_frames:
                msg.info("Deleting png frames...")
                for frame_path in frame_files:
                    os.remove(frame_path)
                msg.info("Frames deleted")

            return True

        except Exception as e:
            msg.warning(f"Failed to generate video: {e}")
            return False
