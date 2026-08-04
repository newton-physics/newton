# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contrast a settled LIMX cloth patch with persistent VF/EE contact churn."""

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils

_REST_POSITIONS = np.asarray(
    [
        [0.12893946, 0.39048225, 2.0031788],
        [0.13945262, 0.3922484, 1.984915],
        [0.12698215, 0.38766122, 1.9893966],
        [0.114696175, 0.3837394, 1.9917175],
        [0.12569286, 0.38243312, 2.0224073],
        [0.117002465, 0.37853378, 2.0207462],
        [0.108881965, 0.38132584, 2.0091412],
        [0.10813384, 0.37899995, 2.0145013],
        [0.10760347, 0.3760879, 2.0197096],
        [0.13218673, 0.35047674, 2.06132],
        [0.14335933, 0.3648014, 2.0541975],
        [0.13106397, 0.36376834, 2.0494084],
        [0.12073193, 0.37347484, 2.0319738],
        [0.110214844, 0.36816543, 2.0337465],
        [0.11660771, 0.3604623, 2.0464995],
        [0.11818884, 0.35917664, 2.0486183],
        [0.11978593, 0.35786128, 2.0507135],
        [0.094880655, 0.37811053, 1.9931738],
        [0.09491957, 0.3776486, 2.0091686],
        [0.13419606, 0.39136535, 1.9940469],
        [0.1332174, 0.3899548, 1.9871558],
        [0.1279608, 0.38907164, 1.9962877],
        [0.13226762, 0.38660377, 1.9811225],
        [0.14170887, 0.3890618, 1.9796045],
        [0.12083917, 0.38570035, 1.990557],
        [0.121817805, 0.38711077, 1.9974481],
        [0.14739615, 0.39146394, 1.9786631],
        [0.12731618, 0.38645768, 2.012793],
        [0.12297095, 0.38450795, 2.0119627],
        [0.12134768, 0.38048357, 2.021577],
        [0.12603238, 0.3843102, 1.9833632],
        [0.11988939, 0.3823493, 1.9845237],
        [0.11256814, 0.37876683, 2.017624],
        [0.11294221, 0.37992972, 2.0149438],
        [0.11230296, 0.37731075, 2.020228],
        [0.1189107, 0.3859039, 2.00616],
        [0.11178906, 0.3825326, 2.0004294],
        [0.14055158, 0.35288453, 2.0616477],
        [0.15230191, 0.3618815, 2.0587666],
        [0.14613792, 0.3600471, 2.0580862],
        [0.13721165, 0.36428493, 2.0518029],
        [0.13162534, 0.35712248, 2.0553641],
        [0.13777304, 0.35763907, 2.0577588],
        [0.114167705, 0.37478137, 2.0258417],
        [0.11886721, 0.37600434, 2.02636],
        [0.108909145, 0.37212664, 2.026728],
        [0.11547338, 0.3708201, 2.03286],
        [0.12359755, 0.36671257, 2.0431907],
        [0.12565964, 0.37321877, 2.0359278],
        [0.118669815, 0.36696857, 2.0392365],
        [0.12462639, 0.36147243, 2.0490134],
        [0.12542495, 0.3608148, 2.050061],
        [0.13058737, 0.37296277, 2.039882],
        [0.13697335, 0.36888212, 2.0470397],
        [0.12321239, 0.37795395, 2.0271907],
        [0.11341126, 0.3643139, 2.040123],
        [0.12383585, 0.36211538, 2.0479538],
        [0.12598634, 0.35416913, 2.0560167],
        [0.103459515, 0.3675673, 2.033456],
        [0.10923018, 0.35475385, 2.0536606],
        [0.110028744, 0.3540963, 2.0547082],
        [0.10952916, 0.35971898, 2.046412],
        [0.10873862, 0.360362, 2.0453525],
        [0.10215382, 0.3715285, 2.0264378],
        [0.098875694, 0.3736779, 2.0212662],
        [0.104013175, 0.37821168, 1.9864606],
        [0.11392093, 0.38102615, 1.9857323],
        [0.1012615, 0.37686813, 2.014439],
        [0.09253375, 0.37445825, 2.0159957],
        [0.10478842, 0.380925, 1.9924456],
        [0.10152671, 0.37832427, 2.0118349],
        [0.10190077, 0.37948722, 2.0091548],
        [0.10188131, 0.37971812, 2.0011575],
        [0.09490012, 0.3778795, 2.001171],
    ],
    dtype=np.float32,
)

_INITIAL_POSITIONS = np.asarray(
    [
        [-0.26310834, 0.29979458, 0.47466478],
        [-0.24774468, 0.30006373, 0.4613137],
        [-0.26109567, 0.29584563, 0.4613649],
        [-0.27262774, 0.28958717, 0.46107113],
        [-0.26491302, 0.28625396, 0.49085477],
        [-0.2736657, 0.28327075, 0.48852438],
        [-0.28019977, 0.2884729, 0.47702956],
        [-0.28181216, 0.28699148, 0.48250923],
        [-0.2832737, 0.28462481, 0.4878451],
        [-0.26333976, 0.26223463, 0.5200665],
        [-0.25240332, 0.25361565, 0.5114253],
        [-0.26307222, 0.2609649, 0.50962085],
        [-0.2708853, 0.27505034, 0.49687612],
        [-0.2818237, 0.27049577, 0.49592796],
        [-0.27697015, 0.2578068, 0.5047782],
        [-0.27587909, 0.25695232, 0.5073714],
        [-0.27492678, 0.2565696, 0.5101335],
        [-0.28883561, 0.27693802, 0.46189243],
        [-0.293645, 0.2833777, 0.4757085],
        [-0.2566229, 0.30210242, 0.46665713],
        [-0.2545332, 0.29831332, 0.46098986],
        [-0.26268962, 0.2986905, 0.46765375],
        [-0.25295386, 0.2918315, 0.46301052],
        [-0.24685536, 0.29812494, 0.4675583],
        [-0.26696607, 0.29291508, 0.46109232],
        [-0.26811182, 0.29500747, 0.46779093],
        [-0.2428201, 0.30227482, 0.46990582],
        [-0.26429403, 0.293684, 0.48322684],
        [-0.268897, 0.2923918, 0.48237902],
        [-0.26969248, 0.28548527, 0.49035355],
        [-0.25877213, 0.28945535, 0.4598956],
        [-0.2645843, 0.2871898, 0.45789203],
        [-0.27775976, 0.28502223, 0.4855093],
        [-0.27741712, 0.28479236, 0.48257512],
        [-0.27842873, 0.28435442, 0.48836577],
        [-0.2689761, 0.2900368, 0.47555324],
        [-0.27550313, 0.28642902, 0.46930343],
        [-0.25539577, 0.2586687, 0.5205524],
        [-0.24507385, 0.24834268, 0.5167958],
        [-0.2511471, 0.25049743, 0.51733196],
        [-0.25725442, 0.25811943, 0.51099485],
        [-0.26512048, 0.25467253, 0.51563954],
        [-0.25922975, 0.2535411, 0.5184318],
        [-0.27740344, 0.2800835, 0.4930953],
        [-0.27265197, 0.28066403, 0.49429238],
        [-0.28254113, 0.27743664, 0.49166122],
        [-0.2762388, 0.27256158, 0.49603608],
        [-0.268895, 0.26263916, 0.50152963],
        [-0.26581368, 0.27140838, 0.49790153],
        [-0.2739228, 0.26639235, 0.50074553],
        [-0.26926428, 0.25843146, 0.50819683],
        [-0.26926994, 0.25942355, 0.50928384],
        [-0.26083815, 0.26751387, 0.4983182],
        [-0.25656775, 0.26063183, 0.5049023],
        [-0.26775628, 0.28093883, 0.49449536],
        [-0.2793222, 0.26397374, 0.50006634],
        [-0.26977694, 0.2588182, 0.5068716],
        [-0.27049613, 0.2573233, 0.5178865],
        [-0.2885648, 0.27109557, 0.4959847],
        [-0.28577828, 0.2585475, 0.5123142],
        [-0.28524473, 0.2594921, 0.51330996],
        [-0.28401646, 0.2577903, 0.50372607],
        [-0.28456894, 0.25835794, 0.5024889],
        [-0.2891312, 0.27858847, 0.49283946],
        [-0.2917518, 0.28108254, 0.4874563],
        [-0.27851883, 0.278415, 0.45745882],
        [-0.27021345, 0.2845245, 0.45755264],
        [-0.28895274, 0.28602695, 0.48197806],
        [-0.29781052, 0.28368932, 0.4824446],
        [-0.28108972, 0.28365293, 0.46058533],
        [-0.28801635, 0.2861787, 0.4790943],
        [-0.28708062, 0.28642052, 0.47627172],
        [-0.28565764, 0.2846776, 0.46859184],
        [-0.29152393, 0.28046265, 0.4685457],
    ],
    dtype=np.float32,
)

_TRIANGLE_INDICES = np.asarray(
    [
        [19, 20, 21],
        [24, 25, 21],
        [27, 28, 29],
        [24, 30, 31],
        [5, 33, 32],
        [34, 5, 32],
        [33, 28, 35],
        [35, 25, 36],
        [22, 30, 20],
        [40, 41, 42],
        [43, 44, 34],
        [37, 39, 42],
        [45, 46, 43],
        [47, 48, 49],
        [11, 50, 51],
        [52, 40, 53],
        [44, 54, 29],
        [49, 46, 55],
        [11, 56, 50],
        [54, 48, 52],
        [41, 51, 57],
        [52, 47, 56],
        [63, 58, 45],
        [67, 68, 64],
        [66, 65, 69],
        [70, 18, 67],
        [71, 72, 73],
        [69, 72, 36],
        [71, 18, 70],
        [0, 19, 21],
        [19, 1, 20],
        [21, 20, 2],
        [22, 1, 23],
        [2, 24, 21],
        [24, 3, 25],
        [21, 25, 0],
        [26, 23, 1],
        [4, 27, 29],
        [27, 0, 28],
        [29, 28, 5],
        [3, 24, 31],
        [24, 2, 30],
        [6, 7, 33],
        [32, 33, 7],
        [8, 32, 7],
        [32, 8, 34],
        [6, 33, 35],
        [33, 5, 28],
        [35, 28, 0],
        [6, 35, 36],
        [35, 0, 25],
        [36, 25, 3],
        [1, 22, 20],
        [20, 30, 2],
        [38, 10, 39],
        [10, 40, 42],
        [40, 11, 41],
        [42, 41, 9],
        [8, 43, 34],
        [43, 12, 44],
        [34, 44, 5],
        [9, 37, 42],
        [42, 39, 10],
        [8, 45, 43],
        [45, 13, 46],
        [43, 46, 12],
        [14, 47, 49],
        [47, 52, 48],
        [49, 48, 12],
        [50, 16, 51],
        [16, 50, 15],
        [52, 11, 40],
        [53, 40, 10],
        [5, 44, 29],
        [44, 12, 54],
        [29, 54, 4],
        [14, 49, 55],
        [49, 12, 46],
        [55, 46, 13],
        [14, 15, 56],
        [50, 56, 15],
        [4, 54, 52],
        [54, 12, 48],
        [9, 41, 57],
        [41, 11, 51],
        [57, 51, 16],
        [11, 52, 56],
        [56, 47, 14],
        [15, 60, 16],
        [60, 15, 59],
        [15, 61, 59],
        [14, 61, 15],
        [61, 14, 62],
        [64, 63, 8],
        [14, 55, 62],
        [8, 63, 45],
        [45, 58, 13],
        [3, 31, 66],
        [8, 67, 64],
        [67, 18, 68],
        [3, 66, 69],
        [69, 65, 17],
        [7, 67, 8],
        [67, 7, 70],
        [18, 71, 73],
        [71, 6, 72],
        [73, 72, 17],
        [3, 69, 36],
        [69, 17, 72],
        [36, 72, 6],
        [6, 71, 7],
        [70, 7, 71],
    ],
    dtype=np.int32,
)

_MASSES = np.asarray(
    [
        1.56898896e-05,
        9.97820825e-06,
        9.50548019e-06,
        1.87359401e-05,
        8.39182212e-06,
        1.10210203e-05,
        1.91413583e-05,
        9.87133990e-06,
        1.68506322e-05,
        1.92682601e-05,
        9.97050211e-06,
        1.57281702e-05,
        1.29759355e-05,
        1.25288752e-05,
        1.29794043e-05,
        9.53243762e-06,
        1.25055931e-05,
        2.30241276e-05,
        1.63389232e-05,
        7.41268923e-06,
        1.47394221e-05,
        1.43912548e-05,
        1.65390084e-05,
        1.51953336e-05,
        1.37770603e-05,
        1.99796596e-05,
        5.98305178e-06,
        7.57859425e-06,
        1.96771543e-05,
        1.20935583e-05,
        1.41252349e-05,
        1.29819546e-05,
        4.34460753e-06,
        1.49753396e-05,
        5.98674342e-06,
        2.50996e-05,
        2.61313999e-05,
        1.64709036e-05,
        8.43471207e-06,
        1.28046995e-05,
        1.71262509e-05,
        1.66352766e-05,
        1.72014152e-05,
        1.16769561e-05,
        9.04864464e-06,
        1.53579876e-05,
        1.43852249e-05,
        1.39572385e-05,
        1.33136782e-05,
        1.61950211e-05,
        2.76859328e-06,
        8.94477125e-06,
        2.34159161e-05,
        4.25621829e-06,
        1.32362284e-05,
        1.56298902e-05,
        1.18523985e-05,
        1.76414687e-05,
        1.48134413e-05,
        1.32007463e-05,
        1.40761013e-05,
        1.33709646e-05,
        9.29530324e-06,
        1.68548577e-05,
        1.78155096e-05,
        2.03451073e-05,
        1.62992892e-05,
        1.33793783e-05,
        1.60139189e-05,
        2.32461753e-05,
        4.17378396e-06,
        1.28072288e-05,
        2.1794016e-05,
        1.66415484e-05,
    ],
    dtype=np.float32,
)

_BOUNDARY_INDICES = np.asarray(
    [
        0,
        1,
        4,
        9,
        10,
        13,
        16,
        17,
        18,
        19,
        22,
        23,
        26,
        27,
        30,
        31,
        37,
        38,
        39,
        52,
        53,
        55,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        68,
        73,
    ],
    dtype=np.int32,
)

for _patch_array in (
    _REST_POSITIONS,
    _INITIAL_POSITIONS,
    _TRIANGLE_INDICES,
    _MASSES,
    _BOUNDARY_INDICES,
):
    _patch_array.setflags(write=False)


class _PatchSimulation:
    def __init__(
        self,
        rest_positions: np.ndarray,
        initial_positions: np.ndarray,
        triangle_indices: np.ndarray,
        masses: np.ndarray,
        boundary_indices: np.ndarray,
        translation: tuple[float, float, float],
        enable_self_collision: bool,
        device,
    ):
        expected_shapes = ((74, 3), (74, 3), (112, 3), (74,), (34,))
        actual_shapes = (
            rest_positions.shape,
            initial_positions.shape,
            triangle_indices.shape,
            masses.shape,
            boundary_indices.shape,
        )
        if actual_shapes != expected_shapes:
            raise ValueError(f"LIMX EE chatter patch data has shapes {actual_shapes}, expected {expected_shapes}")

        translated_rest_positions = rest_positions + np.asarray(translation, dtype=np.float32)
        translated_initial_positions = initial_positions + np.asarray(translation, dtype=np.float32)
        particle_count = len(rest_positions)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_particles(
            pos=[wp.vec3(*position) for position in translated_rest_positions],
            vel=[wp.vec3(0.0)] * particle_count,
            mass=masses.tolist(),
            radius=[0.003] * particle_count,
        )
        builder.add_triangles(triangle_indices[:, 0], triangle_indices[:, 1], triangle_indices[:, 2])
        self.model = builder.finalize(device=device)
        self.model.set_gravity((0.0, 0.0, 0.0))

        edge_rows = newton.utils.MeshAdjacency(triangle_indices).edge_indices
        interior_edge_rows = edge_rows[edge_rows[:, 1] >= 0]
        dihedral_indices = interior_edge_rows[:, [2, 3, 0, 1]]
        constraints = [
            newton.solvers.ConstraintTriangleElastic(
                triangle_indices=triangle_indices,
                inverse_rest_matrices=self.model.tri_poses.numpy(),
                rest_areas=self.model.tri_areas.numpy(),
                stiffnesses=[wp.vec3(1.0e4, 1.0e4, 1.0e3)] * len(triangle_indices),
                particle_count=particle_count,
                device=self.model.device,
            ),
            newton.solvers.ConstraintDihedralBending(
                dihedral_indices=dihedral_indices,
                rest_positions=translated_rest_positions,
                stiffness=1.0e-5,
                particle_count=particle_count,
                device=self.model.device,
            ),
            newton.solvers.ConstraintAnchor(
                indices=boundary_indices,
                targets=[wp.vec3(*position) for position in translated_initial_positions[boundary_indices]],
                stiffnesses=[1.0e6] * len(boundary_indices),
                particle_count=particle_count,
                device=self.model.device,
            ),
        ]
        self.self_collision = None
        if enable_self_collision:
            self.self_collision = newton.solvers.ConstraintSelfCollision(
                self.model,
                thickness=0.006,
                stiffness=None,
                max_contacts=4096,
                stiffness_factors=(0.5, 0.1, 1.5),
            )
        self.solver = newton.solvers.SolverLIMX(
            self.model,
            constraints,
            nonlinear_iterations=1,
            linear_iterations=50,
            velocity_damping=1.0,
            dynamic_operator=self.self_collision,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.state_0.particle_q.assign(translated_initial_positions)
        self.state_0.particle_qd.zero_()
        self.render_indices = wp.array(
            triangle_indices.reshape(-1),
            dtype=wp.int32,
            device=self.model.device,
        )


class Example:
    def __init__(self, viewer, args):
        device = wp.get_device()
        if not device.is_cuda:
            raise RuntimeError("cloth_limx_ee_chatter requires a CUDA device")

        self.viewer = viewer
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt
        self.sim_time = 0.0
        self.patch_vertex_count = len(_REST_POSITIONS)
        self.patch_triangle_count = len(_TRIANGLE_INDICES)
        self.boundary_vertex_count = len(_BOUNDARY_INDICES)
        self.interior_indices = np.setdiff1d(
            np.arange(self.patch_vertex_count, dtype=np.int32),
            _BOUNDARY_INDICES,
        )

        self.control_patch = _PatchSimulation(
            _REST_POSITIONS,
            _INITIAL_POSITIONS,
            _TRIANGLE_INDICES,
            _MASSES,
            _BOUNDARY_INDICES,
            (-0.06, 0.0, 0.0),
            False,
            device,
        )
        self.collision_patch = _PatchSimulation(
            _REST_POSITIONS,
            _INITIAL_POSITIONS,
            _TRIANGLE_INDICES,
            _MASSES,
            _BOUNDARY_INDICES,
            (0.06, 0.0, 0.0),
            True,
            device,
        )

        self.viewer.set_camera(wp.vec3(-0.02, -0.02, 0.56), -10.0, 129.0)
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for patch in (self.control_patch, self.collision_patch):
            patch.state_0.clear_forces()
            patch.solver.step(patch.state_0, patch.state_1, patch.control, None, self.sim_dt)
            patch.state_0.assign(patch.state_1)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        for name, patch in (("control", self.control_patch), ("collision", self.collision_patch)):
            positions = patch.state_0.particle_q.numpy()
            velocities = patch.state_0.particle_qd.numpy()
            if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
                raise AssertionError(f"LIMX EE chatter {name} patch contains non-finite values")

    def test_final(self):
        self.test_post_step()
        if self.sim_time <= 0.0:
            raise AssertionError("LIMX EE chatter scene did not advance")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_mesh(
            "/control_no_collision",
            self.control_patch.state_0.particle_q,
            self.control_patch.render_indices,
            color=(0.25, 0.55, 0.95),
            backface_culling=False,
        )
        self.viewer.log_mesh(
            "/vf_ee_collision",
            self.collision_patch.state_0.particle_q,
            self.collision_patch.render_indices,
            color=(0.95, 0.42, 0.16),
            backface_culling=False,
        )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=1400, render_fps=100)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
