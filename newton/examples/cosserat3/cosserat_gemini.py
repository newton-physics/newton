
import os
import ctypes
import numpy as np
import warp as wp
import newton
import newton.examples

# Load DLLs
unity_ref_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../unity_ref"))
defkit_path = os.path.join(unity_ref_path, "DefKit.dll")
defkit_adv_path = os.path.join(unity_ref_path, "DefKitAdv.dll")

# Ensure DLLs can find dependencies (if any)
os.environ["PATH"] += os.pathsep + unity_ref_path

try:
    defkit = ctypes.CDLL(defkit_path)
    defkit_adv = ctypes.CDLL(defkit_adv_path)
except OSError as e:
    print(f"Error loading DLLs from {unity_ref_path}: {e}")
    raise

# Define types
c_float_p = ctypes.POINTER(ctypes.c_float)

# Function signatures
# void PredictPositions_native(float dt, float damping, int pointsCount, btVector3* positions, btVector3* predictedPositions, btVector3* velocities, btVector3* forces, float* invMasses, btVector3* gravity)
defkit.PredictPositions_native.argtypes = [
    ctypes.c_float, ctypes.c_float, ctypes.c_int,
    c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_float_p
]

# void Integrate_native(float dt, int pointsCount, btVector3* positions, btVector3* predictedPositions, btVector3* velocities, float* invMasses)
defkit.Integrate_native.argtypes = [
    ctypes.c_float, ctypes.c_int,
    c_float_p, c_float_p, c_float_p, c_float_p
]

# void PredictRotationsPBD(float dt, float damping, int pointsCount, Quaternion* rotPtr, Quaternion* predRotPtr, Vector4* angVelPtr, Vector4* torques, float* quatInvMasses)
defkit.PredictRotationsPBD.argtypes = [
    ctypes.c_float, ctypes.c_float, ctypes.c_int,
    c_float_p, c_float_p, c_float_p, c_float_p, c_float_p
]

# void IntegrateRotationsPBD(float dt, int pointsCount, Quaternion* rotPtr, Quaternion* predRotPtr, Quaternion* prevRotPtr, Vector4* angVelPtr, float* quatInvMasses)
defkit.IntegrateRotationsPBD.argtypes = [
    ctypes.c_float, ctypes.c_int,
    c_float_p, c_float_p, c_float_p, c_float_p, c_float_p
]

# void ProjectElasticRodConstraints(int pointsCount, Vector4* positions, Quaternion* orientations, float* invMasses, float* quatInvMasses, Vector4* intrinsicBend, Vector4* intrinsicBendKs, float* restLengths, float stretchAndShearKs, float bendAndTwistKs)
defkit_adv.ProjectElasticRodConstraints.argtypes = [
    ctypes.c_int,
    c_float_p, c_float_p, c_float_p, c_float_p,
    c_float_p, c_float_p, c_float_p,
    ctypes.c_float, ctypes.c_float
]


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.dt = self.frame_dt / self.sim_substeps
        
        self.viewer = viewer
        
        # Rod parameters
        self.num_particles = 20
        self.rod_length = 5.0
        self.segment_length = self.rod_length / (self.num_particles - 1)
        self.radius = 0.1
        
        # Physics parameters
        self.damping = 0.01
        self.gravity = np.array([0.0, -9.81, 0.0, 0.0], dtype=np.float32) # Vector4 for gravity to match pointer type if needed, or just 3
        # PredictPositions_native takes btVector3* gravity. btVector3 is 4 floats.
        
        self.stretch_shear_ks = 1000.0
        self.bend_twist_ks = 1000.0
        self.constraint_iterations = 5

        # Initialize data
        self._init_data()
        
        # Newton model for visualization
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        
        # Add particles for visualization
        for i in range(self.num_particles):
            pos = self.positions[i]
            mass = 1.0 if self.inv_masses[i] > 0 else 0.0
            builder.add_particle(
                pos=(float(pos[0]), float(pos[1]), float(pos[2])),
                vel=(0.0, 0.0, 0.0),
                mass=mass,
                radius=self.radius,
            )
            
        self.newton_model = builder.finalize()
        self.state = self.newton_model.state()
        
        self.viewer.set_model(self.newton_model)
        self.viewer.show_particles = True
        
        # Visualization buffers
        self.show_directors = True
        self.director_scale = 0.3
        num_director_lines = self.num_particles * 3
        self.director_line_starts = wp.zeros(num_director_lines, dtype=wp.vec3, device=self.newton_model.device)
        self.director_line_ends = wp.zeros(num_director_lines, dtype=wp.vec3, device=self.newton_model.device)
        self.director_line_colors = wp.zeros(num_director_lines, dtype=wp.vec3, device=self.newton_model.device)

    def _init_data(self):
        N = self.num_particles
        
        # Positions (x, y, z, w)
        self.positions = np.zeros((N, 4), dtype=np.float32)
        self.predicted_positions = np.zeros((N, 4), dtype=np.float32)
        self.velocities = np.zeros((N, 4), dtype=np.float32)
        self.forces = np.zeros((N, 4), dtype=np.float32)
        
        # Orientations (x, y, z, w)
        self.orientations = np.zeros((N, 4), dtype=np.float32)
        self.predicted_orientations = np.zeros((N, 4), dtype=np.float32)
        self.prev_orientations = np.zeros((N, 4), dtype=np.float32)
        self.angular_velocities = np.zeros((N, 4), dtype=np.float32)
        self.torques = np.zeros((N, 4), dtype=np.float32)
        
        # Masses
        self.inv_masses = np.ones(N, dtype=np.float32)
        self.inv_masses[0] = 0.0 # Pin first particle
        
        self.quat_inv_masses = np.ones(N, dtype=np.float32)
        self.quat_inv_masses[0] = 0.0 # Pin first quaternion
        
        # Rod properties
        self.rest_lengths = np.full(N, self.segment_length, dtype=np.float32) # Last one unused but array size matches
        self.intrinsic_bend = np.zeros((N, 4), dtype=np.float32) # Darboux vectors
        self.intrinsic_bend[:, 3] = 1.0 # Set w=1 for identity quaternion
        
        self.intrinsic_bend_ks = np.ones((N, 4), dtype=np.float32) # Stiffnesses
        self.intrinsic_bend_ks *= 1.0 # Base stiffness multiplier
        
        # Initialize positions (horizontal rod)
        for i in range(N):
            self.positions[i] = [i * self.segment_length, 5.0, 0.0, 0.0]
            self.orientations[i] = [0.0, 0.0, 0.0, 1.0] # Identity quaternion
            
        # Pointers
        self.ptr_positions = self.positions.ctypes.data_as(c_float_p)
        self.ptr_pred_positions = self.predicted_positions.ctypes.data_as(c_float_p)
        self.ptr_velocities = self.velocities.ctypes.data_as(c_float_p)
        self.ptr_forces = self.forces.ctypes.data_as(c_float_p)
        self.ptr_inv_masses = self.inv_masses.ctypes.data_as(c_float_p)
        self.ptr_gravity = self.gravity.ctypes.data_as(c_float_p)
        
        self.ptr_orientations = self.orientations.ctypes.data_as(c_float_p)
        self.ptr_pred_orientations = self.predicted_orientations.ctypes.data_as(c_float_p)
        self.ptr_prev_orientations = self.prev_orientations.ctypes.data_as(c_float_p)
        self.ptr_ang_vel = self.angular_velocities.ctypes.data_as(c_float_p)
        self.ptr_torques = self.torques.ctypes.data_as(c_float_p)
        self.ptr_quat_inv_masses = self.quat_inv_masses.ctypes.data_as(c_float_p)
        
        self.ptr_rest_lengths = self.rest_lengths.ctypes.data_as(c_float_p)
        self.ptr_intrinsic_bend = self.intrinsic_bend.ctypes.data_as(c_float_p)
        self.ptr_intrinsic_bend_ks = self.intrinsic_bend_ks.ctypes.data_as(c_float_p)

    def simulate(self):
        for _ in range(self.sim_substeps):
            # Predict positions
            defkit.PredictPositions_native(
                self.dt, self.damping, self.num_particles,
                self.ptr_positions, self.ptr_pred_positions,
                self.ptr_velocities, self.ptr_forces, self.ptr_inv_masses,
                self.ptr_gravity
            )
            
            # Predict rotations
            defkit.PredictRotationsPBD(
                self.dt, self.damping, self.num_particles,
                self.ptr_orientations, self.ptr_pred_orientations,
                self.ptr_ang_vel, self.ptr_torques, self.ptr_quat_inv_masses
            )
            
            # Solve constraints
            for _ in range(self.constraint_iterations):
                defkit_adv.ProjectElasticRodConstraints(
                    self.num_particles,
                    self.ptr_pred_positions,
                    self.ptr_pred_orientations,
                    self.ptr_inv_masses,
                    self.ptr_quat_inv_masses,
                    self.ptr_intrinsic_bend,
                    self.ptr_intrinsic_bend_ks,
                    self.ptr_rest_lengths,
                    self.stretch_shear_ks,
                    self.bend_twist_ks
                )
            
            # Integrate positions
            defkit.Integrate_native(
                self.dt, self.num_particles,
                self.ptr_positions, self.ptr_pred_positions,
                self.ptr_velocities, self.ptr_inv_masses
            )
            
            # Integrate rotations
            defkit.IntegrateRotationsPBD(
                self.dt, self.num_particles,
                self.ptr_orientations, self.ptr_pred_orientations,
                self.ptr_prev_orientations,
                self.ptr_ang_vel, self.ptr_quat_inv_masses
            )

    def _sync_state(self):
        # Sync positions to Newton state
        positions_wp = wp.from_numpy(self.positions[:, :3], dtype=wp.vec3, device=self.newton_model.device)
        self.state.particle_q = positions_wp

    def _update_viz(self):
        if not self.show_directors:
            self.viewer.log_lines("/directors", None, None, None)
            return

        starts = []
        ends = []
        colors = []
        
        for i in range(self.num_particles):
            pos = self.positions[i]
            q = self.orientations[i] # x, y, z, w
            
            # Rotate basis vectors
            # q * v * q_inv
            def rotate_vector(v, q):
                # q = [x, y, z, w]
                # v = [x, y, z]
                qx, qy, qz, qw = q
                vx, vy, vz = v
                
                # Quaternion mult v * q_inv
                # v = [vx, vy, vz, 0]
                # q_inv = [-qx, -qy, -qz, qw]
                
                # Easier: use matrix conversion or formula
                # Using formula: v + 2*cross(q_xyz, cross(q_xyz, v) + q_w*v)
                q_xyz = np.array([qx, qy, qz])
                uv = np.cross(q_xyz, v)
                uuv = np.cross(q_xyz, uv)
                return v + 2.0 * (qw * uv + uuv)

            x_axis = rotate_vector(np.array([1.0, 0.0, 0.0]), q)
            y_axis = rotate_vector(np.array([0.0, 1.0, 0.0]), q)
            z_axis = rotate_vector(np.array([0.0, 0.0, 1.0]), q)
            
            p = pos[:3]
            s = self.director_scale
            
            starts.extend([p, p, p])
            ends.extend([p + x_axis * s, p + y_axis * s, p + z_axis * s])
            colors.extend([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            
        self.director_line_starts = wp.array(starts, dtype=wp.vec3, device=self.newton_model.device)
        self.director_line_ends = wp.array(ends, dtype=wp.vec3, device=self.newton_model.device)
        self.director_line_colors = wp.array(colors, dtype=wp.vec3, device=self.newton_model.device)
        
        self.viewer.log_lines(
            "/directors",
            self.director_line_starts,
            self.director_line_ends,
            self.director_line_colors,
        )

    def step(self):
        self.simulate()
        self._sync_state()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self._update_viz()
        self.viewer.end_frame()

    def test_final(self):
        # Check if rod moved
        initial_tip_pos = np.array([(self.num_particles - 1) * self.segment_length, 5.0, 0.0])
        # The tip is the last particle
        current_tip_pos = self.positions[-1, :3]
        
        # Calculate displacement
        displacement = np.linalg.norm(current_tip_pos - initial_tip_pos)
        print(f"Tip displacement: {displacement}")
        
        # It should have moved due to gravity
        if displacement < 0.01:
            print("Warning: Rod did not move significantly.")
        else:
            print("Rod simulation active.")

if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
