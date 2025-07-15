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

from __future__ import annotations

from collections import defaultdict

import numpy as np
import warp as wp
from warp.render import OpenGLRenderer, UsdRenderer
from warp.render.utils import solidify_mesh, tab10_color_map

import newton


@wp.kernel
def compute_contact_points(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    # outputs
    contact_pos0: wp.array(dtype=wp.vec3),
    contact_pos1: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        contact_pos0[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        contact_pos1[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        return
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        contact_pos0[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        contact_pos1[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        return

    body_a = shape_body[shape_a]
    body_b = shape_body[shape_b]
    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    if body_a >= 0:
        X_wb_a = body_q[body_a]
    if body_b >= 0:
        X_wb_b = body_q[body_b]

    contact_pos0[tid] = wp.transform_point(X_wb_a, contact_point0[tid])
    contact_pos1[tid] = wp.transform_point(X_wb_b, contact_point1[tid])


def CreateSimRenderer(renderer):
    """A factory function to create a simulation renderer class."""

    class SimRenderer(renderer):
        use_unique_colors = True

        def __init__(
            self,
            model: newton.Model | None = None,
            path: str = "No path specified",
            scaling: float = 1.0,
            fps: int = 60,
            up_axis: newton.AxisType | None = None,
            show_joints: bool = False,
            show_particles: bool = True,
            **render_kwargs,
        ):
            """
            Initializes the simulation renderer.
            Args:
                model (newton.Model, optional): The simulation model to render. Defaults to None.
                path (str, optional): The path for the rendered output (e.g., filename for USD, window title for OpenGL).
                scaling (float, optional): Scaling factor for the rendered output. Defaults to 1.0.
                fps (int, optional): Frames per second for the rendered output. Defaults to 60.
                up_axis (newton.AxisType, optional): The up-axis for the scene. If not provided, it's inferred from the model, or defaults to "Z" if no model is given. Defaults to None.
                show_joints (bool, optional): Whether to visualize joints. Defaults to False.
                show_particles (bool, optional): Whether to visualize particles. Defaults to True.
                **render_kwargs: Additional keyword arguments for the underlying renderer.
            """
            if up_axis is None:
                if model:
                    up_axis = model.up_axis
                else:
                    up_axis = newton.Axis.Z
            up_axis = newton.Axis.from_any(up_axis)
            super().__init__(path, scaling=scaling, fps=fps, up_axis=str(up_axis), **render_kwargs)
            self.scaling = scaling
            self.cam_axis = up_axis.value
            self.show_joints = show_joints
            self.show_particles = show_particles
            self._instance_key_count = {}
            if model:
                self.populate(model)

            self._contact_points0 = None
            self._contact_points1 = None

        def populate(self, model: newton.Model):
            """
            Populates the renderer with objects from the simulation model.
            This method sets up the rendering scene by creating visual representations
            for all the bodies, shapes, and other components of the simulation model.
            Args:
                model (newton.Model): The simulation model containing the objects to populate.
            """
            self.skip_rendering = False

            self.model = model
            self.num_envs = model.num_envs
            self.body_names = []

            bodies_per_env = model.body_count // self.num_envs
            self.body_env = []
            self.body_names = self.populate_bodies(
                self.model.body_key, bodies_per_env=bodies_per_env, body_env=self.body_env
            )

            # create rigid shape children
            if self.model.shape_count:
                # mapping from hash of geometry to shape ID
                self.geo_shape = {}

                self.instance_count = 0

                self.body_name = {}  # mapping from body name to its body ID
                self.body_shapes = defaultdict(list)  # mapping from body index to its shape IDs

                self.instance_count = self.populate_shapes(
                    self.body_names,
                    self.geo_shape,
                    model.shape_body.numpy(),
                    model.shape_geo_src,
                    model.shape_geo.type.numpy(),
                    model.shape_geo.scale.numpy(),
                    model.shape_geo.thickness.numpy(),
                    model.shape_geo.is_solid.numpy(),
                    model.shape_transform.numpy(),
                    model.shape_flags.numpy(),
                    self.model.shape_key,
                    instance_count=self.instance_count,
                    use_unique_colors=self.use_unique_colors,
                )

                if self.show_joints and model.joint_count:
                    self.instance_count = self.populate_joints(
                        self.body_names,
                        model.joint_type.numpy(),
                        model.joint_axis.numpy(),
                        model.joint_qd_start.numpy(),
                        model.joint_dof_dim.numpy(),
                        model.joint_parent.numpy(),
                        model.joint_child.numpy(),
                        model.joint_X_p.numpy(),
                        model.shape_collision_radius.numpy(),
                        model.body_shapes,
                        instance_count=self.instance_count,
                    )

            if hasattr(self, "complete_setup"):
                self.complete_setup()

        def populate_bodies(self, body_name_arr: list, bodies_per_env: int = -1, body_env: list | None = None) -> list:
            """
            Populates the renderer with body objects.

            Args:
                body_name_arr (list): List of body names from the model.
                bodies_per_env (int, optional): Number of bodies per environment. If -1, all bodies belong to same environment. Defaults to -1.
                body_env (list, optional): List to store environment IDs for each body. Defaults to None.

            Returns:
                list: List of generated unique body names in the format "body_{index}_{name}".
            """
            body_names = []
            body_count = len(body_name_arr)

            if body_env is not None:
                body_env.clear()
                env_id = 0

            for b in range(body_count):
                body_name = f"body_{b}_{body_name_arr[b].replace(' ', '_')}"
                body_names.append(body_name)
                self.register_body(body_name)
                if body_env is not None and bodies_per_env > 0:
                    if b > 0 and b % bodies_per_env == 0:
                        env_id += 1
                    body_env.append(env_id)

            return body_names

        def populate_shapes(
            self,
            body_names: list,
            geo_shape: dict,
            shape_body: np.ndarray,
            shape_geo_src: list,
            shape_geo_type: np.ndarray,
            shape_geo_scale: np.ndarray,
            shape_geo_thickness: np.ndarray,
            shape_geo_is_solid: np.ndarray,
            shape_transform: np.ndarray,
            shape_flags: np.ndarray,
            shape_key: list,
            instance_count: int = 0,
            use_unique_colors: bool = True,
        ) -> int:
            """
            Populates the renderer with shapes for rigid bodies.
            Args:
                body_names (list): List of body names.
                geo_shape (dict): A dictionary to cache geometry shapes.
                shape_body (numpy.ndarray): Maps shape index to body index.
                shape_geo_src (list): Source geometry for each shape.
                shape_geo_type (numpy.ndarray): Type of each shape's geometry.
                shape_geo_scale (numpy.ndarray): Scale of each shape's geometry.
                shape_geo_thickness (numpy.ndarray): Thickness of each shape's geometry.
                shape_geo_is_solid (numpy.ndarray): Solid flag for each shape's geometry.
                shape_transform (numpy.ndarray): Local transform of each shape.
                shape_flags (numpy.ndarray): Visibility and other flags for each shape.
                shape_key (list): List of shape names.
                instance_count (int, optional): Initial instance count. Defaults to 0.
                use_unique_colors (bool, optional): Whether to assign unique colors to shapes. Defaults to True.
            Returns:
                int: The updated instance count after adding shapes.
            """
            p = np.zeros(3, dtype=np.float32)
            q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            color = (1.0, 1.0, 1.0)
            shape_count = len(shape_body)
            # loop over shapes
            for s in range(shape_count):
                scale = np.ones(3, dtype=np.float32)
                geo_type = shape_geo_type[s]
                geo_scale = [float(v) for v in shape_geo_scale[s]]
                geo_thickness = float(shape_geo_thickness[s])
                geo_is_solid = bool(shape_geo_is_solid[s])
                geo_src = shape_geo_src[s]
                name = shape_key[s]
                count = self._instance_key_count.get(name, 0)
                if count > 0:
                    self._instance_key_count[name] += 1
                    # ensure unique name for the shape instance
                    name = f"{name}_{count + 1}"
                else:
                    self._instance_key_count[name] = 1
                add_shape_instance = True

                # shape transform in body frame
                body = int(shape_body[s])
                if body >= 0 and body < len(body_names):
                    body = body_names[body]
                else:
                    body = None

                if use_unique_colors and body is not None:
                    color = self.get_new_color(instance_count)

                # shape transform in body frame
                X_bs = wp.transform_expand(shape_transform[s])
                # check whether we can instance an already created shape with the same geometry
                geo_hash = hash((int(geo_type), geo_src, *geo_scale, geo_thickness, geo_is_solid))
                if geo_hash in geo_shape:
                    shape = geo_shape[geo_hash]
                else:
                    if geo_type == newton.GEO_PLANE:
                        # plane mesh
                        width = geo_scale[0] if geo_scale[0] > 0.0 else 100.0
                        length = geo_scale[1] if geo_scale[1] > 0.0 else 100.0

                        if name == "ground_plane":
                            normal = wp.quat_rotate(X_bs.q, wp.vec3(0.0, 0.0, 1.0))
                            offset = wp.dot(normal, X_bs.p)
                            shape = self.render_ground(plane=[*normal, offset])
                            add_shape_instance = False
                        else:
                            shape = self.render_plane(
                                name, p, q, width, length, color, parent_body=body, is_template=True
                            )

                    elif geo_type == newton.GEO_SPHERE:
                        shape = self.render_sphere(
                            name, p, q, geo_scale[0], parent_body=body, is_template=True, color=color
                        )

                    elif geo_type == newton.GEO_CAPSULE:
                        shape = self.render_capsule(
                            name, p, q, geo_scale[0], geo_scale[1], parent_body=body, is_template=True, color=color
                        )

                    elif geo_type == newton.GEO_CYLINDER:
                        shape = self.render_cylinder(
                            name, p, q, geo_scale[0], geo_scale[1], parent_body=body, is_template=True, color=color
                        )

                    elif geo_type == newton.GEO_CONE:
                        shape = self.render_cone(
                            name, p, q, geo_scale[0], geo_scale[1], parent_body=body, is_template=True, color=color
                        )

                    elif geo_type == newton.GEO_BOX:
                        shape = self.render_box(name, p, q, geo_scale, parent_body=body, is_template=True, color=color)

                    elif geo_type == newton.GEO_MESH:
                        if not geo_is_solid:
                            faces, vertices = solidify_mesh(geo_src.indices, geo_src.vertices, geo_thickness)
                        else:
                            faces, vertices = geo_src.indices, geo_src.vertices

                        shape = self.render_mesh(
                            name,
                            vertices,
                            faces,
                            pos=p,
                            rot=q,
                            scale=geo_scale,
                            colors=color,
                            parent_body=body,
                            is_template=True,
                        )
                        scale = np.asarray(geo_scale, dtype=np.float32)

                    elif geo_type == newton.GEO_SDF:
                        continue

                    geo_shape[geo_hash] = shape

                if add_shape_instance and shape_flags[s] & int(newton.geometry.SHAPE_FLAG_VISIBLE):
                    # TODO support dynamic visibility
                    q_shape = X_bs.q
                    if geo_type in (newton.GEO_CAPSULE, newton.GEO_CYLINDER, newton.GEO_CONE):
                        q_shape = X_bs.q * wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -wp.pi / 2.0)
                    self.add_shape_instance(name, shape, body, X_bs.p, q_shape, scale, custom_index=s, visible=True)
                instance_count += 1
            return instance_count

        def populate_joints(
            self,
            body_names: list,
            joint_type: np.ndarray,
            joint_axis: np.ndarray,
            joint_qd_start: np.ndarray,
            joint_dof_dim: np.ndarray,
            joint_parent: np.ndarray,
            joint_child: np.ndarray,
            joint_tf: np.ndarray,
            shape_collision_radius: np.ndarray,
            body_shapes: defaultdict,
            instance_count: int = 0,
        ) -> int:
            """
            Populates the renderer with joint visualizations.
            Args:
                body_names (list): List of body names.
                joint_type (numpy.ndarray): Type of each joint.
                joint_axis (numpy.ndarray): Axis of each joint.
                joint_qd_start (numpy.ndarray): Start index for joint dofs.
                joint_dof_dim (numpy.ndarray): Dimensions of joint dofs (linear and angular).
                joint_parent (numpy.ndarray): Parent body index for each joint.
                joint_child (numpy.ndarray): Child body index for each joint.
                joint_tf (numpy.ndarray): Transform of each joint.
                shape_collision_radius (numpy.ndarray): Collision radius of shapes, used for scaling joint arrows.
                body_shapes (list): List of shapes attached to each body.
                instance_count (int, optional): Initial instance count. Defaults to 0.
            Returns:
                int: The updated instance count after adding joint visualizations.
            """
            y_axis = wp.vec3(0.0, 1.0, 0.0)
            color = (1.0, 0.0, 1.0)

            shape = self.render_arrow(
                "joint_arrow",
                None,
                None,
                base_radius=0.01,
                base_height=0.4,
                cap_radius=0.02,
                cap_height=0.1,
                parent_body=None,
                is_template=True,
                color=color,
            )
            for i, t in enumerate(joint_type):
                if t not in {
                    newton.JOINT_REVOLUTE,
                    # newton.JOINT_PRISMATIC,
                    newton.JOINT_D6,
                }:
                    continue
                tf = joint_tf[i]
                body = int(joint_parent[i])
                if body >= 0 and body < len(body_names):
                    body = body_names[body]
                else:
                    body = None
                # if body == -1:
                #     continue
                num_linear_axes = int(joint_dof_dim[i][0])
                num_angular_axes = int(joint_dof_dim[i][1])

                # find a good scale for the arrow based on the average radius
                # of the shapes attached to the joint child body
                scale = np.ones(3, dtype=np.float32)
                child = int(joint_child[i])
                if child >= 0:
                    radii = []
                    bs = body_shapes.get(child, [])
                    for s in bs:
                        radii.append(shape_collision_radius[s])
                    if len(radii) > 0:
                        scale *= np.mean(radii) * 2.0

                for a in range(num_linear_axes, num_linear_axes + num_angular_axes):
                    index = joint_qd_start[i] + a
                    axis = joint_axis[index]
                    if np.linalg.norm(axis) < 1e-6:
                        continue
                    p = wp.vec3(tf[:3])
                    q = wp.quat(tf[3:])
                    # compute rotation between axis and y
                    axis = axis / np.linalg.norm(axis)
                    q = q * wp.quat_between_vectors(wp.vec3(axis), y_axis)
                    name = f"joint_{i}_{a}"
                    self.add_shape_instance(name, shape, body, p, q, scale, color1=color, color2=color)
                    instance_count += 1
            return instance_count

        def get_new_color(self, instance_count: int) -> tuple:
            """Gets a new color from a predefined color map.
            This method provides a new color based on the current instance count,
            cycling through a predefined color map to ensure variety.
            Returns:
                tuple: A tuple representing the new color (e.g., in RGB format).
            """
            return tab10_color_map(instance_count)

        def render(self, state: newton.State):
            """
            Updates the renderer with the given simulation state.
            Args:
                state (newton.State): The simulation state to render.
            """
            if self.skip_rendering:
                return

            if self.model.particle_count:
                self.render_particles_and_springs(
                    particle_q=state.particle_q.numpy(),
                    particle_radius=self.model.particle_radius.numpy(),
                    tri_indices=self.model.tri_indices.numpy() if self.model.tri_count else None,
                    spring_indices=self.model.spring_indices.numpy() if self.model.spring_count else None,
                )

            # render muscles
            if self.model.muscle_count:
                self.render_muscles(
                    body_q=state.body_q.numpy(),
                    muscle_start=self.model.muscle_start.numpy(),
                    muscle_links=self.model.muscle_bodies.numpy(),
                    muscle_points=self.model.muscle_points.numpy(),
                    muscle_activation=self.model.muscle_activation.numpy(),
                )

            # update bodies
            if self.model.body_count:
                self.update_body_transforms(state.body_q)

        def render_particles_and_springs(
            self,
            particle_q: np.ndarray,
            particle_radius: np.ndarray,
            tri_indices: np.ndarray | None = None,
            spring_indices: np.ndarray | None = None,
        ):
            """Renders particles, mesh surface, and springs.
            Args:
                particle_q (numpy.ndarray): Array of particle positions.
                particle_radius (float): Radius of the particles.
                tri_indices (numpy.ndarray, optional): Triangle indices for the surface mesh. Defaults to None.
                spring_indices (numpy.ndarray, optional): Spring indices. Defaults to None.
            """
            # render particles
            if self.show_particles:
                self.render_points("particles", particle_q, radius=particle_radius, colors=(0.8, 0.3, 0.2))

            # render tris
            if tri_indices is not None:
                self.render_mesh(
                    "surface",
                    particle_q,
                    tri_indices.flatten(),
                    colors=(0.75, 0.25, 0.0),
                )

            # render springs
            if spring_indices is not None:
                self.render_line_list("springs", particle_q, spring_indices.flatten(), (0.25, 0.5, 0.25), 0.02)

        def render_muscles(
            self,
            body_q: np.ndarray,
            muscle_start: np.ndarray,
            muscle_links: np.ndarray,
            muscle_points: np.ndarray,
            muscle_activation: np.ndarray,
        ):
            """Renders muscles as line strips.
            Args:
                body_q (numpy.ndarray): Array of body transformations.
                muscle_start (numpy.ndarray): Start indices for muscles in other muscle arrays.
                muscle_links (numpy.ndarray): Body indices for each muscle point.
                muscle_points (numpy.ndarray): Local positions of muscle attachment points.
                muscle_activation (numpy.ndarray): Activation level for each muscle, used for color.
            """
            muscle_count = (len(muscle_start) - 1) if muscle_start is not None else 0
            for m in range(muscle_count):
                start = int(muscle_start[m])
                end = int(muscle_start[m + 1])

                points = []

                for w in range(start, end):
                    link = muscle_links[w]
                    point = muscle_points[w]

                    X_sc = wp.transform_expand(body_q[link][0])

                    points.append(wp.transform_point(X_sc, point).tolist())

                self.render_line_strip(
                    name=f"muscle_{m}",
                    vertices=points,
                    radius=0.0075,
                    color=(muscle_activation[m], 0.2, 0.5),
                )

        def render_contacts(
            self,
            body_q: wp.array,
            contacts: newton.Contacts,
            contact_point_radius: float = 1e-3,
        ):
            """
            Render contact points between rigid bodies.
            Args:
                body_q (wp.array): Array of body transformations.
                contacts (newton.Contacts): The contacts to render.
                contact_point_radius (float, optional): The radius of the contact points.
            """
            if self._contact_points0 is None or len(self._contact_points0) < contacts.rigid_contact_max:
                self._contact_points0 = wp.array(
                    np.zeros((contacts.rigid_contact_max, 3)), dtype=wp.vec3, device=self.model.device
                )
                self._contact_points1 = wp.array(
                    np.zeros((contacts.rigid_contact_max, 3)), dtype=wp.vec3, device=self.model.device
                )

            wp.launch(
                kernel=compute_contact_points,
                dim=contacts.rigid_contact_max,
                inputs=[
                    body_q,
                    self.model.shape_body,
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                ],
                outputs=[
                    self._contact_points0,
                    self._contact_points1,
                ],
                device=self.model.device,
            )

            self.render_points(
                "contact_points0",
                self._contact_points0,
                radius=contact_point_radius * self.scaling,
                colors=(1.0, 0.5, 0.0),
            )
            self.render_points(
                "contact_points1",
                self._contact_points1,
                radius=contact_point_radius * self.scaling,
                colors=(0.0, 0.5, 1.0),
            )

    return SimRenderer


class SimRendererUsd(CreateSimRenderer(renderer=UsdRenderer)):
    """
    USD renderer for Newton Physics simulations.

    This renderer exports simulation data to USD (Universal Scene Description)
    format, which can be visualized in Omniverse or other USD-compatible viewers.

    This renderer supports rendering a Newton simulation as a time-sampled animation of USD prims through the render_update_stage method. This method requires a source stage, a path_body_map, a path_body_relative_transform, and builder_results.

    Args:
        model (newton.Model): The Newton physics model to render.
        path (str): Output path for the USD file.
        scaling (float, optional): Scaling factor for the rendered objects.
            Defaults to 1.0.
        fps (int, optional): Frames per second for the animation. Defaults to 60.
        up_axis (newton.AxisType, optional): Up axis for the scene. If None,
            uses model's up axis.
        show_rigid_contact_points (bool, optional): Whether to show contact
            points. Defaults to False.
        contact_points_radius (float, optional): Radius of contact point
            spheres. Defaults to 1e-3.
        show_joints (bool, optional): Whether to show joint visualizations.
            Defaults to False.
        **render_kwargs: Additional arguments passed to the underlying
            UsdRenderer.

    Example:
        .. code-block:: python

            import newton

            model = newton.Model()  # your model setup
            renderer = newton.utils.SimRendererUsd(model, "output.usd", scaling=2.0)
            # In your simulation loop:
            renderer.begin_frame(time)
            renderer.render(state)
            renderer.end_frame()
            renderer.save()  # Save the USD file
    """

    def __init__(
        self,
        model: newton.Model,
        stage: str | Usd.Stage,
        source_stage: str | Usd.Stage | None = None,
        scaling: float = 1.0,
        fps: int = 60,
        up_axis: newton.AxisType | None = None,
        show_joints: bool = False,
        path_body_map: dict | None = None,
        path_body_relative_transform: dict | None = None,
        builder_results: dict | None = None,
        **render_kwargs,
    ):
        """
        Construct a SimRendererUsd object.

        Args:
            model (newton.Model): The Newton physics model to render.
            stage (str | Usd.Stage): The USD stage to render to. This is the output stage.
            source_stage (str | Usd.Stage, optional): The USD stage to use as a source for the output stage.
            scaling (float, optional): Scaling factor for the rendered objects. Defaults to 1.0.
            fps (int, optional): Frames per second for the animation. Defaults to 60.
            up_axis (newton.AxisType, optional): Up axis for the scene. If None, uses model's up axis. Defaults to None.
            show_joints (bool, optional): Whether to show joint visualizations.  Defaults to False.
            path_body_map (dict, optional): A dictionary mapping prim paths to body IDs.
            path_body_relative_transform (dict, optional): A dictionary mapping prim paths to relative transformations.
            builder_results (dict, optional): A dictionary containing builder results.
            **render_kwargs: Additional arguments passed to the underlying UsdRenderer.
        """
        if source_stage:
            if path_body_map is None:
                raise ValueError("path_body_map must be set if you are providing a source_stage")
            if path_body_relative_transform is None:
                raise ValueError("path_body_relative_transform must be set if you are providing a source_stage")
            if builder_results is None:
                raise ValueError("builder_results must be set if you are providing a source_stage")

        self.source_stage = source_stage

        if not self.source_stage:
            super().__init__(
                model=model,
                path=stage,
                scaling=scaling,
                fps=fps,
                up_axis=up_axis,
                show_joints=show_joints,
                **render_kwargs,
            )
        else:
            self.stage = self._create_output_stage(self.source_stage, stage)
            self.fps = fps
            self.scaling = scaling
            self.up_axis = up_axis
            self.path_body_map = path_body_map
            self.path_body_relative_transform = path_body_relative_transform
            self.builder_results = builder_results
            self._prepare_output_stage()
            self._precompute_parents_xform_inverses()

    def render_update_stage(self, state: newton.State):
        if not self.source_stage:
            raise ValueError("source_stage must be set before calling render_update_stage")

        self._update_usd_stage(state)

    def _update_usd_stage(self, state: newton.State):
        """
        Render transforms of USD prims as time-sampled animation in USD.

        Args:
            state (newton.State): The simulation state to render.
            sim_time (float): The current simulation time.
        """
        from pxr import Sdf  # noqa: PLC0415

        body_q = state.body_q.numpy()
        with Sdf.ChangeBlock():
            for prim_path, body_id in self.path_body_map.items():
                full_xform = body_q[body_id]
                # TODO: do this once in __init__
                # TODO: sanity check this with Eric Heiden
                # Take relative xform into account
                rel_xform = self.path_body_relative_transform.get(prim_path)
                if rel_xform:
                    full_xform = wp.mul(full_xform, rel_xform)

                full_xform = self._apply_parents_inverse_xform(full_xform, prim_path)
                self._update_usd_prim_xform(prim_path, full_xform)

    def _apply_parents_inverse_xform(self, full_xform: wp.transform, prim_path: str) -> wp.transform:
        """
        Transformation in Warp sim consists of translation and pure rotation: trnslt and quat.
        Transformations of bodies are stored in body_q in simulation state.
        For sim_usd, trnslt is computed directly from PhysicsUtils by the function GetRigidBodyTransformation
        in parseUtils.cpp:
            const GfMatrix4d mat = UsdGeomXformable(bodyPrim).ComputeLocalToWorldTransform(UsdTimeCode::Default());
            const GfTransform tr(mat);
            const GfVec3d pos = tr.GetTranslation();
            const GfQuatd rot = tr.GetRotation().GetQuat();
        In import_nvusd, we set trnslt = pos and quat = fromgfquat(rot), where fromgfquat has the following logic:
        wp.normalize(wp.quat(*gfquat.imaginary, gfquat.real)).

        For trnslt, we have:
            warp_trnslt = xform.ComputeLocalToWorldTransform().GetTranslation().
        But in USD space:
            xform.ComputeLocalToworldTransform() = xform.GetLocalTransform() * xform.ComputeParentToWorldTransform()
            Prim_LTW_USD = Prim_Local_USD * Parent_LTW_USD,
        or in Warp space, we work with transpose and arrive at:
            Prim_LTW = Parent_LTW * Prim_Local
            warp_trnslt = p_Rot * prim_trnslt + p_trnslt,
            i.e.
            prim_trnslt = p_inv_Rot * (warp_trnslt - p_trnslt).

        For rotation, we have:
            rot = tr.GetRotation().GetQuat();
            warp_quat = wp.normalize(wp.quat(rot)).
        However, rot is already normalized, so we don't actually need to renormalize it.
        So in Warp space,
            warp_Rot = p_Rot * diag(1/s_x, 1/s_y, 1/s_z) * prim_Rot, so
            prim_Rot = wp.inv(p_Rot * diag(1/s_x, 1/s_y, 1/s_z)) * warp_Rot

        Both p_inv_Rot and wp.inv(p_Rot * diag(1/s_x, 1/s_y, 1/s_z)) do not change during sim, so they are computed in __init__.
        """
        from pxr import Sdf  # noqa: PLC0415

        current_prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))
        parent_path = str(current_prim.GetParent().GetPath())

        if parent_path in self.builder_results["path_body_map"]:
            return

        parent_translate = self.parent_translates[parent_path]
        parent_inv_Rot = self.parent_inv_Rs[parent_path]
        parent_inv_Rot_n = self.parent_inv_Rns[parent_path]

        warp_translate = wp.transform_get_translation(full_xform)
        warp_quat = wp.transform_get_rotation(full_xform)

        prim_translate = parent_inv_Rot * (warp_translate - parent_translate)
        prim_quat = parent_inv_Rot_n * warp_quat

        return wp.transform(prim_translate, prim_quat)

    def _update_usd_prim_xform(self, prim_path: str, warp_xform: wp.transform):
        from pxr import Gf, Sdf, UsdGeom  # noqa: PLC0415

        prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))

        pos = tuple(map(float, warp_xform[0:3]))
        rot = tuple(map(float, warp_xform[3:7]))

        xform = UsdGeom.Xform(prim)
        xform_ops = xform.GetOrderedXformOps()

        if pos is not None:
            xform_ops[0].Set(Gf.Vec3f(pos[0], pos[1], pos[2]), self.time)
        if rot is not None:
            xform_ops[1].Set(Gf.Quatf(rot[3], rot[0], rot[1], rot[2]), self.time)

    # TODO: if _compute_parents_inverses turns to be too slow, then we should consider using a UsdGeomXformCache as described here:
    # https://openusd.org/release/api/class_usd_geom_imageable.html#a4313664fa692f724da56cc254bce70fc
    def _compute_parents_inverses(self, prim_path: str, time: Usd.TimeCode) -> tuple[wp.vec3, wp.mat33, wp.quat]:
        from pxr import Gf, Sdf, UsdGeom  # noqa: PLC0415

        prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))
        xform = UsdGeom.Xform(prim)

        parent_world = Gf.Matrix4f(xform.ComputeParentToWorldTransform(time))
        Rpw = wp.mat33(parent_world.ExtractRotationMatrix().GetTranspose())  # Rot_parent_world
        (_, _, s, _, translate_parent_world, _) = parent_world.Factor()

        transpose_Rpwn = wp.mat33(
            Rpw[0, 0] / s[0],
            Rpw[1, 0] / s[0],
            Rpw[2, 0] / s[0],
            Rpw[0, 1] / s[1],
            Rpw[1, 1] / s[1],
            Rpw[2, 1] / s[1],
            Rpw[0, 2] / s[2],
            Rpw[1, 2] / s[2],
            Rpw[2, 2] / s[2],
        )  # Rot_parent_world_normalized
        inv_Rpwn = wp.quat_from_matrix(transpose_Rpwn)  # b/c Rpwn is a pure rotation
        inv_Rpw = wp.inverse(Rpw)

        return translate_parent_world, inv_Rpw, inv_Rpwn

    def _precompute_parents_xform_inverses(self):
        """
        Convention: prefix c is for **current** prim.
        Prefix p is for **parent** prim.
        """

        from pxr import Sdf, Usd  # noqa: PLC0415

        if self.path_body_map is None:
            raise ValueError("self.path_body_map must be set before calling _precompute_parents_xform_inverses")

        self.parent_translates = {}
        self.parent_inv_Rs = {}
        self.parent_inv_Rns = {}

        with wp.ScopedTimer("prep_parents_xform"):
            time = Usd.TimeCode.Default()
            for prim_path in self.path_body_map.keys():
                current_prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))
                parent_path = str(current_prim.GetParent().GetPath())

                if parent_path not in self.parent_translates:
                    (
                        self.parent_translates[parent_path],
                        self.parent_inv_Rs[parent_path],
                        self.parent_inv_Rns[parent_path],
                    ) = self._compute_parents_inverses(prim_path, time)

    def _prepare_output_stage(self):
        """
        Set USD parameters on the output stage to match the simulation settings.

        Must be called after _apply_solver_attributes!"""
        from pxr import Sdf  # noqa: PLC0415

        if self.path_body_map is None:
            raise ValueError("self.path_body_map must be set before calling _prepare_output_stage")

        self.stage.SetStartTimeCode(0.0)
        self.stage.SetEndTimeCode(0.0)
        # NB: this is now coming from warp:fps, but timeCodesPerSecond is a good source too
        self.stage.SetTimeCodesPerSecond(self.fps)

        for prim_path in self.path_body_map.keys():
            prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))
            SimRendererUsd._xform_to_tqs(prim)

    def _create_output_stage(self, source_stage: str | Usd.Stage, output_stage: str | Usd.Stage) -> Usd.Stage:
        from pxr import Usd  # noqa: PLC0415

        if isinstance(output_stage, str):
            source_stage = Usd.Stage.Open(source_stage, Usd.Stage.LoadAll)
            flattened = source_stage.Flatten()
            stage = Usd.Stage.Open(flattened.identifier)
            exported = stage.ExportToString()

            output_stage = Usd.Stage.CreateNew(output_stage)
            output_stage.GetRootLayer().ImportFromString(exported)
            return output_stage
        elif isinstance(output_stage, Usd.Stage):
            return output_stage
        else:
            raise ValueError("output_stage must be a string or a Usd.Stage")

    @staticmethod
    def _xform_to_tqs(prim: Usd.Prim, time: Usd.TimeCode | None = None):
        """
        Update the transformation stack of a primitive to translate/orient/scale format.

        The original transformation stack is assumed to be a rigid transformation.
        """
        from pxr import Gf, Usd, UsdGeom  # noqa: PLC0415

        if time is None:
            time = Usd.TimeCode.Default()

        _tqs_op_order = [UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.TypeScale]
        _tqs_op_precision = [
            UsdGeom.XformOp.PrecisionFloat,
            UsdGeom.XformOp.PrecisionFloat,
            UsdGeom.XformOp.PrecisionFloat,
        ]

        xform = UsdGeom.Xform(prim)
        xform_ops = xform.GetOrderedXformOps()

        # if the order, type, and precision of the transformation is already in our canonical form, then there's no need to change anything.
        if _tqs_op_order == [op.GetOpType() for op in xform_ops] and _tqs_op_precision == [
            op.GetPrecision() for op in xform_ops
        ]:
            return

        # this assumes no skewing
        # NB: the rotation coming from Factor is the result of solving an eigenvalue problem. We found wrong answer with non-identity scaling.
        m_lcl = xform.GetLocalTransformation(time)
        (_, _, scale, _, translation, _) = m_lcl.Factor()

        t = Gf.Vec3f(translation)
        q = Gf.Quatf(m_lcl.ExtractRotationQuat())
        s = Gf.Vec3f(scale)

        # need to reset the transform
        for op in xform_ops:
            attr = op.GetAttr()
            prim.RemoveProperty(attr.GetName())

        xform.ClearXformOpOrder()
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(t)
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(q)
        xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(s)


class SimRendererOpenGL(CreateSimRenderer(renderer=OpenGLRenderer)):
    """
    Real-time OpenGL renderer for Newton Physics simulations.

    This renderer provides real-time visualization of physics simulations using
    OpenGL, with interactive camera controls and various rendering options.

    Args:
        model (newton.Model): The Newton physics model to render.
        path (str): Window title for the OpenGL window.
        scaling (float, optional): Scaling factor for the rendered objects.
            Defaults to 1.0.
        fps (int, optional): Target frames per second. Defaults to 60.
        up_axis (newton.AxisType, optional): Up axis for the scene. If None, uses model's up axis. Defaults to None.
        show_rigid_contact_points (bool, optional): Whether to show contact
            points. Defaults to False.
        contact_points_radius (float, optional): Radius of contact point
            spheres. Defaults to 1e-3.
        show_joints (bool, optional): Whether to show joint visualizations.
            Defaults to False.
        **render_kwargs: Additional arguments passed to the underlying
            OpenGLRenderer.

    Example:
        .. code-block:: python

            import newton

            model = newton.Model()  # your model setup
            renderer = newton.utils.SimRendererOpenGL(model, "Newton Simulator")
            # In your simulation loop:
            renderer.begin_frame(time)
            renderer.render(state)
            renderer.end_frame()

    Note:
        Keyboard shortcuts available during rendering:

        - W, A, S, D (or arrow keys) + mouse: FPS-style camera movement
        - X: Toggle wireframe rendering
        - B: Toggle backface culling
        - C: Toggle coordinate system axes
        - G: Toggle ground grid
        - T: Toggle depth rendering
        - I: Toggle info text
        - SPACE: Pause/continue simulation
        - TAB: Skip rendering (background simulation)
    """

    pass


SimRenderer = SimRendererUsd
