from abc import abstractmethod

import numpy as np
import warp as wp

import newton


class ViewerBase:
    def __init__(self, model):
        self.time = 0.0

        self.model = model
        self.model_changed = True

        self.device = model.device

        # map from shape hash -> Instances
        self.shape_instances = {}
        self.joint_instances = None

        # cache for geometry created via log_shapes()
        # maps from geometry hash -> mesh path
        self._geometry_cache: dict[str, str] = {}

    def _populate(self, model):
        self._populate_shapes()
        # self._populate_joints()

    def begin_frame(self, time):
        self.time = time

    def log_model(self, state):
        """Render the Newton model."""

        if self.model is None:
            return

        # compute shape transforms and render
        for shapes in self.shape_instances.values():
            shapes.update(state)
            self.log_instances(
                shapes.name,
                shapes.mesh,
                shapes.world_xforms,
                shapes.scales if self.model_changed else None,
                shapes.colors if self.model_changed else None,
                shapes.materials if self.model_changed else None,
            )

        # # update joints
        # if self.joint_instances:
        #     self.joint_instances.update(state)
        #     self.log_instances(
        #         "joints",
        #         self.joint_instances.mesh,
        #         self.joint_instances.xforms,
        #         self.joint_instances.scales,
        #         self.joint_instances.colors,
        #         self.joint_instances.materials,
        #     )

        self.model_changed = False

    def log_contacts(self, name, contacts, state):
        """
        Computes the world-space positions of contact points for rendering.
        Args:
            body_q (wp.array): Array of body transformations.
            contacts (newton.Contacts): The contacts to render.
        """
        if self._contact_points0 is None or len(self._contact_points0) < contacts.rigid_contact_max:
            self._contact_points0 = wp.array(
                np.zeros((contacts.rigid_contact_max, 3)), dtype=wp.vec3, device=self.model.device
            )
            self._contact_points1 = wp.array(
                np.zeros((contacts.rigid_contact_max, 3)), dtype=wp.vec3, device=self.model.device
            )

        from newton.utils.render import compute_contact_points  # noqa: PLC0415

        wp.launch(
            kernel=compute_contact_points,
            dim=contacts.rigid_contact_max,
            inputs=[
                state.body_q,
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

    def log_shapes(
        self,
        name: str,
        geo_type: int,
        geo_scale,
        xforms,
        colors=None,
        materials=None,
        geo_thickness: float = 0.0,
        geo_is_solid: bool = True,
    ):
        """
        Convenience helper to create/cache a mesh of a given geometry and
        render a batch of instances with the provided transforms/colors/materials.

        Args:
            name: Instance path/name (e.g., "/world/spheres").
            geo_type: newton.GEO_* constant.
            geo_scale: Geometry scale parameters:
                - Sphere: float radius
                - Capsule/Cylinder/Cone: (radius, height)
                - Plane: (width, length) or float for both
                - Box: (x_extent, y_extent, z_extent) or float for all
            xforms: wp.array(dtype=wp.transform) of instance transforms
            colors: wp.array(dtype=wp.vec3) or None (broadcasted if length 1)
            materials: wp.array(dtype=wp.vec4) or None (broadcasted if length 1)
            thickness: Optional thickness (used for hashing consistency)
            is_solid: If False, can be used for wire/solid hashing parity
        """

        # normalize geo_scale to a list for hashing + mesh creation
        def _as_float_list(value):
            if isinstance(value, (tuple, list, np.ndarray)):
                return [float(v) for v in value]
            else:
                return [float(value)]

        geo_scale = _as_float_list(geo_scale)

        # ensure mesh exists (shared with populate path)
        mesh_path = self._populate_geometry(
            int(geo_type),
            tuple(geo_scale),
            float(geo_thickness),
            bool(geo_is_solid),
        )

        # prepare instance properties
        num_instances = len(xforms)

        # scales default to ones
        scales = wp.array([wp.vec3(1.0, 1.0, 1.0)] * num_instances, dtype=wp.vec3, device=self.device)

        # broadcast helpers
        def _ensure_vec3_array(arr, default):
            if arr is None:
                return wp.array([default] * num_instances, dtype=wp.vec3, device=self.device)
            if len(arr) == 1 and num_instances > 1:
                return wp.array([arr[0]] * num_instances, dtype=wp.vec3, device=self.device)
            return arr

        def _ensure_vec4_array(arr, default):
            if arr is None:
                return wp.array([default] * num_instances, dtype=wp.vec4, device=self.device)
            if len(arr) == 1 and num_instances > 1:
                return wp.array([arr[0]] * num_instances, dtype=wp.vec4, device=self.device)
            return arr

        # defaults
        default_color = wp.vec3(0.3, 0.8, 0.9)
        default_material = wp.vec4(0.0, 0.7, 0.0, 0.0)

        # planes default to checkerboard and mid-gray if not overridden
        if geo_type == newton.GEO_PLANE:
            default_color = wp.vec3(0.125, 0.125, 0.25)
            default_material = wp.vec4(0.5, 0.7, 1.0, 0.0)

        colors = _ensure_vec3_array(colors, default_color)
        materials = _ensure_vec4_array(materials, default_material)

        # finally, log the instances
        self.log_instances(name, mesh_path, xforms, scales, colors, materials)

    def log_geo(
        self,
        name,
        geo_type: int,
        geo_scale: tuple[float, ...],
        geo_thickness: float,
        geo_is_solid: bool,
        geo_src=None,
    ):
        """
        Create a primitive mesh and upload it via log_mesh.

        Expects mesh generators to return interleaved vertices [x, y, z, nx, ny, nz, u, v]
        and an index buffer. Slices them into separate arrays and forwards to log_mesh.
        """

        # GEO_MESH handled by provided source geometry
        if geo_type == newton.GEO_MESH:
            if geo_src is None:
                raise ValueError("log_geo requires geo_src for GEO_MESH")

            # resolve points/indices from source, solidify if requested
            from warp.render.utils import solidify_mesh  # noqa: PLC0415

            if not geo_is_solid:
                indices, points = solidify_mesh(geo_src.indices, geo_src.vertices, geo_thickness)
            else:
                indices, points = geo_src.indices, geo_src.vertices

            # prepare warp arrays; synthesize normals/uvs
            points = wp.array(points, dtype=wp.vec3, device=self.device)
            indices = wp.array(indices, dtype=wp.uint32, device=self.device)
            normals = None
            uvs = None

            if geo_src._normals is not None:
                normals = wp.array(geo_src._normals, dtype=wp.vec3, device=self.device)

            if geo_src._uvs is not None:
                uvs = wp.array(geo_src._uvs, dtype=wp.vec2, device=self.device)

            self.log_mesh(name, points, indices, normals, uvs)
            return

        # Generate vertices/indices for supported primitive types
        if geo_type == newton.GEO_PLANE:
            from .mesh import create_plane_mesh  # noqa: PLC0415

            # Handle "infinite" planes encoded with non-positive scales
            width = geo_scale[0] if geo_scale and geo_scale[0] > 0.0 else 100.0
            length = geo_scale[1] if len(geo_scale) > 1 and geo_scale[1] > 0.0 else 100.0
            vertices, indices = create_plane_mesh(width, length)
        elif geo_type == newton.GEO_SPHERE:
            from .mesh import create_sphere_mesh  # noqa: PLC0415

            radius = geo_scale[0]
            vertices, indices = create_sphere_mesh(radius)
        elif geo_type == newton.GEO_CAPSULE:
            from .mesh import create_capsule_mesh  # noqa: PLC0415

            radius, height = geo_scale[:2]
            vertices, indices = create_capsule_mesh(radius, height, up_axis=2)
        elif geo_type == newton.GEO_CYLINDER:
            from .mesh import create_cylinder_mesh  # noqa: PLC0415

            radius, height = geo_scale[:2]
            vertices, indices = create_cylinder_mesh(radius, height, up_axis=2)
        elif geo_type == newton.GEO_CONE:
            from .mesh import create_cone_mesh  # noqa: PLC0415

            radius, height = geo_scale[:2]
            vertices, indices = create_cone_mesh(radius, height, up_axis=2)
        elif geo_type == newton.GEO_BOX:
            from .mesh import create_box_mesh  # noqa: PLC0415

            if len(geo_scale) == 1:
                ext = (geo_scale[0],) * 3
            else:
                ext = tuple(geo_scale[:3])
            vertices, indices = create_box_mesh(ext)
        else:
            raise ValueError(f"log_geo does not support geo_type={geo_type}")

        # Convert to Warp arrays and forward to log_mesh
        points = wp.array(vertices[:, 0:3], dtype=wp.vec3, device=self.device)
        normals = wp.array(vertices[:, 3:6], dtype=wp.vec3, device=self.device)
        uvs = wp.array(vertices[:, 6:8], dtype=wp.vec2, device=self.device)
        indices = wp.array(indices, dtype=wp.uint32, device=self.device)

        self.log_mesh(name, points, indices, normals, uvs)

    @abstractmethod
    def log_mesh(self, name, points: wp.array, indices: wp.array, normals: wp.array = None, uvs: wp.array = None):
        pass

    @abstractmethod
    def log_instances(self, name, mesh, xforms, scales, colors, materials):
        pass

    @abstractmethod
    def log_lines(self, name, state):
        pass

    @abstractmethod
    def log_points(self, name, state):
        pass

    @abstractmethod
    def log_array(self, name, array):
        pass

    @abstractmethod
    def log_scalar(self, name, value):
        pass

    @abstractmethod
    def end_frame(self):
        pass

    @abstractmethod
    def close(self):
        pass

    # handles a batch of mesh instances attached to bodies in the Newton Model
    class Instances:
        def __init__(self, name, mesh, device):
            self.name = name
            self.mesh = mesh
            self.device = device

            self.parents = []
            self.xforms = []
            self.scales = []
            self.colors = []
            self.materials = []

            self.world_xforms = None

        def add(self, parent, xform, scale, color, material):
            # add an instance of the geometry to the batch
            self.parents.append(parent)
            self.xforms.append(xform)
            self.scales.append(scale)
            self.colors.append(color)
            self.materials.append(material)

        def finalize(self):
            # convert to warp arrays
            self.parents = wp.array(self.parents, dtype=int, device=self.device)
            self.xforms = wp.array(self.xforms, dtype=wp.transform, device=self.device)
            self.scales = wp.array(self.scales, dtype=wp.vec3, device=self.device)
            self.colors = wp.array(self.colors, dtype=wp.vec3, device=self.device)
            self.materials = wp.array(self.materials, dtype=wp.vec4, device=self.device)

            self.world_xforms = wp.zeros_like(self.xforms)

        def update(self, state):
            from .kernels import update_shape_xforms  # noqa: PLC0415

            wp.launch(
                kernel=update_shape_xforms,
                dim=len(self.xforms),
                inputs=[self.xforms, self.parents, state.body_q],
                outputs=[self.world_xforms],
                device=self.device,
            )

    # returns a unique (non-stable) identifier for a geometry configuration
    def _hash_geometry(
        self,
        geo_type: int,
        geo_scale,
        thickness: float,
        is_solid: bool,
        geo_src=None,
    ) -> int:
        return hash((int(geo_type), geo_src, *geo_scale, float(thickness), bool(is_solid)))

    def _populate_geometry(
        self,
        geo_type: int,
        geo_scale,
        thickness: float,
        is_solid: bool,
        geo_src=None,
    ) -> str:
        """Ensure a geometry mesh exists and return its mesh path.

        Computes a stable hash from the parameters; creates and caches the mesh path if needed.
        """

        # normalize
        if isinstance(geo_scale, (list, tuple, np.ndarray)):
            scale_list = [float(v) for v in geo_scale]
        else:
            scale_list = [float(geo_scale)]

        # include geo_src in hash to match model-driven batching
        geo_hash = self._hash_geometry(
            int(geo_type),
            tuple(scale_list),
            float(thickness),
            bool(is_solid),
            geo_src,
        )

        if geo_hash in self._geometry_cache:
            return self._geometry_cache[geo_hash]

        base_name = {
            newton.GEO_PLANE: "plane",
            newton.GEO_SPHERE: "sphere",
            newton.GEO_CAPSULE: "capsule",
            newton.GEO_CYLINDER: "cylinder",
            newton.GEO_CONE: "cone",
            newton.GEO_BOX: "box",
            newton.GEO_MESH: "mesh",
        }.get(geo_type)

        if base_name is None:
            raise ValueError(f"Unsupported geo_type for ensure_geometry: {geo_type}")

        mesh_path = f"/geometry/{base_name}_{len(self._geometry_cache)}"
        self.log_geo(
            mesh_path,
            int(geo_type),
            tuple(scale_list),
            float(thickness),
            bool(is_solid),
            geo_src=geo_src if geo_type == newton.GEO_MESH else None,
        )
        self._geometry_cache[geo_hash] = mesh_path
        return mesh_path

    # creates meshes and instances for each shape in the Model
    def _populate_shapes(self):
        # convert to NumPy
        shape_body = self.model.shape_body.numpy()
        shape_geo_src = self.model.shape_geo_src
        shape_geo_type = self.model.shape_geo.type.numpy()
        shape_geo_scale = self.model.shape_geo.scale.numpy()
        shape_geo_thickness = self.model.shape_geo.thickness.numpy()
        shape_geo_is_solid = self.model.shape_geo.is_solid.numpy()
        shape_transform = self.model.shape_transform.numpy()
        shape_flags = self.model.shape_flags.numpy()
        shape_count = len(shape_body)

        # loop over shapes
        for s in range(shape_count):
            # skip invisible
            if (shape_flags[s] & int(newton.geometry.SHAPE_FLAG_VISIBLE)) == 0:
                continue

            geo_type = shape_geo_type[s]
            geo_scale = [float(v) for v in shape_geo_scale[s]]
            geo_thickness = float(shape_geo_thickness[s])
            geo_is_solid = bool(shape_geo_is_solid[s])
            geo_src = shape_geo_src[s]

            # skip unsupported
            if geo_type == newton.GEO_SDF:
                continue

            # check whether we can instance an already created shape with the same geometry
            geo_hash = self._hash_geometry(
                int(geo_type),
                tuple(geo_scale),
                float(geo_thickness),
                bool(geo_is_solid),
                geo_src,
            )

            if geo_hash in self.shape_instances:
                batch = self.shape_instances[geo_hash]
            else:
                # ensure geometry exists and get mesh path
                mesh_name = self._populate_geometry(
                    int(geo_type),
                    tuple(geo_scale),
                    float(geo_thickness),
                    bool(geo_is_solid),
                    geo_src=geo_src if geo_type == newton.GEO_MESH else None,
                )

                # add instances
                shape_name = f"/model/shapes/shape_{len(self.shape_instances)}"
                batch = ViewerBase.Instances(shape_name, mesh_name, self.model.device)

                self.shape_instances[geo_hash] = batch

            parent = shape_body[s]
            xform = wp.transform_expand(shape_transform[s])
            scale = np.array([1.0, 1.0, 1.0])
            color = wp.vec3(self._shape_color_map(s))
            material = wp.vec4(0.5, 0.0, 0.0, 0.0)  # roughness, metallic, checker, unused

            if geo_type == newton.GEO_MESH:
                scale = np.asarray(geo_scale, dtype=np.float32)

                if geo_src._color is not None:
                    color = wp.vec3(geo_src._color[0:3])

            # plane appearance: checkerboard + gray
            if geo_type == newton.GEO_PLANE:
                # color = wp.vec3(0.5, 0.5, 0.5)
                # # enable checkerboard in material's z component
                # try:
                #     material[2] = 1.0
                # except Exception:
                #     material = wp.vec4(material[0], material[1], 1.0, material[3])

                color = wp.vec3(0.125, 0.125, 0.15)
                material = wp.vec4(0.5, 0.5, 1.0, 0.0)

            # add render instance
            batch.add(parent, xform, scale, color, material)

        # upload all batches to the GPU
        for batch in self.shape_instances.values():
            batch.finalize()

    def populate_joints(self) -> int:
        # convert to NumPy
        joint_type = self.model.joint_type.numpy()
        joint_axis = self.model.joint_axis.numpy()
        joint_qd_start = self.model.joint_qd_start.numpy()
        joint_dof_dim = self.model.joint_dof_dim.numpy()
        joint_parent = self.model.joint_parent.numpy()
        joint_child = self.model.joint_child.numpy()
        joint_transform = self.model.joint_transform.numpy()
        shape_collision_radius = self.model.shape_collision_radius.numpy()
        body_shapes = self.model.body_shapes

        y_axis = wp.vec3(0.0, 1.0, 0.0)
        color = (1.0, 0.0, 1.0)

        self.joint_instances = Instances("joints", self.model.device)

        for i, t in enumerate(joint_type):
            if t not in {
                newton.JOINT_REVOLUTE,
                # newton.JOINT_PRISMATIC,
                newton.JOINT_D6,
            }:
                continue
            tf = joint_transform[i]
            body = int(joint_parent[i])

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

                self.joint_instances.add(body, p, q, scale, color)

    @staticmethod
    def _shape_color_map(i: int) -> list[float]:
        # Paul Tol - Bright 9
        colors = [
            [68, 119, 170],  # blue
            [102, 204, 238],  # cyan
            [34, 136, 51],  # green
            [204, 187, 68],  # yellow
            [238, 102, 119],  # red
            [170, 51, 119],  # magenta
            [187, 187, 187],  # grey
            [238, 153, 51],  # orange
            [0, 153, 136],  # teal
        ]

        num_colors = len(colors)
        return [c / 255.0 for c in colors[i % num_colors]]
