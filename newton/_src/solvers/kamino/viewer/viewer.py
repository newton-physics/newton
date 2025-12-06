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

"""The customized debug viewer of Kamino"""

import glob
import os
import threading
from typing import ClassVar

import warp as wp
from PIL import Image

from ....viewer import ViewerGL
from ..core.builder import ModelBuilder
from ..core.geometry import CollisionGeometryDescriptor, GeometryDescriptor
from ..core.shapes import ShapeType
from ..core.types import vec3f
from ..core.world import WorldDescriptor
from ..simulation.simulator import Simulator
from ..utils import logger as msg

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
        builder: ModelBuilder,
        simulator: Simulator,
        width: int = 1920,
        height: int = 1080,
        vsync: bool = False,
        headless: bool = False,
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
        self._collision_geometry: list[CollisionGeometryDescriptor] = builder.collision_geoms
        self._physical_geometry: list[GeometryDescriptor] = builder.physical_geoms

        # Initialize video recording settings
        self._record_video = record_video
        self._video_folder = video_folder or "./frames"
        self._async_save = async_save
        self._skip_img_idx = skip_img_idx
        self._img_idx = 0
        self._frame_buffer = None

        if self._record_video:
            os.makedirs(self._video_folder, exist_ok=True)

    def render_geometry(self, body_poses: wp.array, geom: GeometryDescriptor, scope: str):
        # TODO: Fix this
        bid = geom.bid + self._worlds[geom.wid].bodies_idx_offset if geom.bid >= 0 else -1

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
        color = self.body_colors[geom.bid % len(self.body_colors)]

        # Convert shape parameters to Newton format w/ half-extents
        params = geom.shape.params
        if geom.shape.type == ShapeType.CYLINDER:
            params = (params[0], 0.5 * params[1])
        elif geom.shape.type == ShapeType.CONE:
            params = (params[0], 0.5 * params[1])
        elif geom.shape.type == ShapeType.CAPSULE:
            params = (params[0], 0.5 * params[1])
        elif geom.shape.type == ShapeType.BOX:
            params = (0.5 * params[0], 0.5 * params[1], 0.5 * params[2])

        # Update the geometry data
        self.log_shapes(
            name=f"/world_{geom.wid}/body_{geom.bid}/{scope}/{geom.gid}-{geom.name}",
            geo_type=geom.shape.type.to_newton(),
            geo_scale=params,
            xforms=wp.array([geom_transform], dtype=wp.transform),
            geo_is_solid=geom.shape.is_solid,
            colors=color,
            geo_src=geom.shape.data,
        )

    def render_frame(self, stop_recording: bool = False):
        # Begin a new frame
        self.begin_frame(self._simulator.time)

        # Extract body poses from the kamino simulator
        body_poses = self._simulator.state.q_i.numpy()

        # Render each collision geom
        for cgeom in self._collision_geometry:
            if cgeom.shape.type == ShapeType.EMPTY:
                continue
            self.render_geometry(body_poses, cgeom, scope="collision")

        # Render each physical geom
        for pgeom in self._physical_geometry:
            if pgeom.shape.type == ShapeType.EMPTY:
                continue
            self.render_geometry(body_poses, pgeom, scope="physical")

        # End the new frame
        self.end_frame()

        # Capture frame if recording is enabled and not stopped
        if self._record_video and not stop_recording:
            # todo : think about if we should continue to step the _img_idx even when not recording
            self._capture_frame()

    def _capture_frame(self):
        """
        Capture and save a single frame from the viewer.

        This method retrieves the current rendered frame, converts it to a PIL Image,
        and saves it as a PNG file.
        """
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
