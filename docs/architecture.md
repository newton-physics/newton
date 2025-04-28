# Newton Physics - Architecture & Design

## Table of Contents

- [Newton Physics - Architecture \& Design](#newton-physics---architecture--design)
    - [Table of Contents](#table-of-contents)
    - [Abstract](#abstract)
    - [Requirements](#requirements)
    - [Goals and Non-Goals](#goals-and-non-goals)
    - [General Principles](#general-principles)
    - [Repository Structure](#repository-structure)
        - [Development Processes](#development-processes)
            - [Tooling](#tooling)
            - [Testing](#testing)
            - [CI/CD Pipelines](#cicd-pipelines)
                - [GPU Runners for GitHub](#gpu-runners-for-github)
            - [Mirroring](#mirroring)
            - [Branches](#branches)
            - [Publishing](#publishing)
            - [Versioning](#versioning)
            - [Dependencies](#dependencies)
    - [Core Data Structures and Responsibilities](#core-data-structures-and-responsibilities)
    - [Class Definitions](#class-definitions)
        - [`newton.Model`](#newtonmodel)
        - [`newton.State`](#newtonstate)
        - [`newton.Control`](#newtoncontrol)
        - [`newton.Contact`](#newtoncontact)
        - [`newton.Tensors`](#newtontensors)
    - [Example Workflow](#example-workflow)
    - [Implementing Custom Solvers](#implementing-custom-solvers)
    - [Implementing Custom Collision Detection](#implementing-custom-collision-detection)
    - [Handling MuJoCo Warp as a Runtime Dependency](#handling-mujoco-warp-as-a-runtime-dependency)
    - [Kinematics \& Dynamics Utilities](#kinematics--dynamics-utilities)
    - [Isaac Lab Integration](#isaac-lab-integration)
        - [Exposing Custom Attributes via Tensor API](#exposing-custom-attributes-via-tensor-api)
            - [Example: Heat Exchange Simulation](#example-heat-exchange-simulation)
            - [Example: Stribeck Friction Model](#example-stribeck-friction-model)
            - [Example: Mapping USD Properties to Newton](#example-mapping-usd-properties-to-newton)
            - [Example: Multi-DOF Joints](#example-multi-dof-joints)
            - [UsdPhysics to Newton Mapping](#usdphysics-to-newton-mapping)
        - [omni.newton](#omninewton)
    - [Other Considerations](#other-considerations)
        - [Coupling Multiple Solvers](#coupling-multiple-solvers)
        - [MuJoCo Warp Integration](#mujoco-warp-integration)
        - [Maximal-Coordinate Solver Support](#maximal-coordinate-solver-support)
    - [Open Questions](#open-questions)
        - [Batched vs Indexed](#batched-vs-indexed)
        - [How will Newton be made available to Isaac Lab?](#how-will-newton-be-made-available-to-isaac-lab)
        - [Where will the Newton repository be hosted?](#where-will-the-newton-repository-be-hosted)

## Abstract

Newton is a GPU-accelerated physics simulation engine built upon NVIDIA Warp, specifically targeting roboticists and simulation researchers. It extends and generalizes Warp's existing `warp.sim` module, integrating MuJoCo Warp (DeepMind) as a primary backend. Newton emphasizes GPU-based computation, differentiability, and user-defined extensibility, facilitating rapid iteration and scalable robotics simulation.

In this document, we propose a foundational architecture and design for Newton by defining guiding principles, core data structures and APIs, and code structure.

Related Documents

* [Newton Tech Blog](https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/)  
* [Warp Sim documentation](https://nvidia.github.io/warp/modules/sim.html)  
* [Warp Contributing guidelines](https://nvidia.github.io/warp/modules/contribution_guide.html)  
* [Isaac Lab Tensor API documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/index.html)  
* [MuJoCo Warp GitHub Repository](https://github.com/google-deepmind/mujoco_warp)  
* [MuJoCo Warp Roadmap](https://github.com/orgs/google-deepmind/projects/5)  

## Requirements

* GPU-accelerated simulation built upon NVIDIA Warp  
* Integration with MuJoCo Warp  
* Integration with Isaac Lab's Tensor API  
* Enable differentiability and interoperability with DL and RL frameworks   
* Enable extensibility through custom solvers and physics models  
* Support for maximal-coordinate solvers (e.g., Disney Research)   
* Handling closed-loop systems and constraints

## Goals and Non-Goals

Newton aims to define a *useful abstraction* for roboticists and simulation engineers to develop high-performance, GPU-accelerated simulators for robot learning. While Newton is intended to be a flexible developer framework that allows users to implement custom dynamics using built-in primitives (e.g., collision detection, solvers, model representations), ***it is not intended to wrap or homogenize every possible physics engine or domain-specific library*** which would be impossible given the huge variance amongst simulation engines and physical models.

Examples of functionality that is not expected to be part of the core library:

* Specific actuator models (e.g., motor or muscle models)  
* Specific sensor models (e.g., cameras or force-torque sensing)  
* Specific robotic manipulator or end-effector models (like suction grippers)

Instead, Newton’s architecture encourages the community to implement these components on top of its robust GPU-based framework in Isaac Lab or other learning environments. This approach ensures a lightweight framework while preserving maximum flexibility for domain experts to design and extend. 

## General Principles

The following are general principles underlying the design of the Newton architecture that have proved effective in the development of differentiable Python-based simulators using Warp:

1. **Separation of Physical Modeling and Numerical Method**  
   Newton aims to cleanly separate the physical model from the numerical method. To achieve this the description of the physical system is split into (`Model, State)` objects (holding time-invariant and time-varying quantities respectively). The computation of forward dynamics is abstracted into a `Solver` object that advances the simulation forward in time. In instances where additional solver-specific parameters (e.g.: regularization values) are required the application / user code may assign additional array data onto the `Model` to be consumed directly by the solver. This separation of concerns is important to allow for checkpointing and differentiability where previous states must be stored and replayed during back-propagation. 

2. **Minimal Dependencies and Ease of Deployment**  
   To ensure broad platform compatibility and reduce the size of build-configuration matrices, Newton (and Warp) restrict dependencies on third party libraries. As with Warp, the only *required* external Python package is NumPy. Native code dependencies on CUDA and underlying geometry types are handled by Warp through static linking and `ctypes` bindings. **Unlike Warp, the Newton repository will not contain CUDA C++ source files.** This ensures that Newton can run in an architecture agnostic fashion wherever Warp is supported. Third-party developers may expose native C++/CUDA code to Newton, but such integration should occur via custom solver or collision modules, **not** by modifying the Newton core. Additional optional dependencies (e.g.: USD, MuJoCo), should be handled through the Python \[extras\] mechanism.

3. **Behavior Compatibility but not API Compatibility with MuJoCo**  
   Newton aims to provide broad out-of-the-box behavior and file format compatibility with MuJoCo, however it does not aim to duplicate or expose all features using identical APIs. Engine-specific functionality should be accessible to developers who wish to bypass official APIs through introspection on the Solver objects.

## Repository Structure

Newton's repository structure clearly delineates responsibilities:

```
docs/
newton/
├── collision/            # GPU-accelerated collision-detection kernels
├── solvers/
│   ├── solver_base.py    # Base solver abstract class
│   ├── solver_mujoco.py  # MuJoCo-based solver backend
│   ├── solver_vbd.py     # VBD solver (soft bodies, cloth)
│   └── solver_xpbd.py    # XPBD solver (maximal coords, constraints)
├── fem/                  # Optional FEM module (warp.fem integration)
├── core/
│   ├── model.py
│   ├── state.py
│   ├── control.py
│   ├── kinematics.py     # eval_fk(), eval_ik(), Jacobians, mass matrix 
│   └── utils.py
├── examples/
└── tests/
```

### Development Processes

Please see the Warp Contributing guide for best practices around code contributions: [https://nvidia.github.io/warp/modules/contribution\_guide.html](https://nvidia.github.io/warp/modules/contribution_guide.html).

#### Tooling

* Similar to Warp and mujoco\_warp, Newton will use [Ruff](https://github.com/astral-sh/ruff) for code formatting and linting checks and support [pre-commit](https://pre-commit.com/) hooks.  
* Documentation will use [Sphinx](https://www.sphinx-doc.org/en/master/) and a company-agnostic theme like Furo (i.e. no use of [nvidia-sphinx-theme](https://pypi.org/project/nvidia-sphinx-theme/)).  
* Docstrings will follow the [Google-style convention](https://google.github.io/styleguide/pyguide.html). Public API functions should be expected to have docstrings with correct type annotations and usage examples.  
* Benchmarks will use [Airspeed Velocity](https://github.com/airspeed-velocity/asv).  
* Code coverage will be measured using [coveragepy](https://github.com/nedbat/coveragepy).

#### Testing

Tests will use the [unittest](https://docs.python.org/3/library/unittest.html) unit testing framework to minimize dependencies and to leverage existing tooling and practices adopted for Warp. It’s possible that future maintainers will want to transition to using [pytest](https://docs.pytest.org/en/stable/) like [MJWarp](https://github.com/google-deepmind/mujoco_warp?tab=readme-ov-file#installing-for-development), but the initial repository will use unittest to get rapid development going.

#### CI/CD Pipelines

Pipelines will be tested against the nightly (development) versions of Warp to identify regressions quickly. Workflows in GitHub will be used to check pull requests against a test suite on [GitHub-hosted runners](https://docs.github.com/en/actions/using-github-hosted-runners/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources) (CPU only).

##### GPU Runners for GitHub

Program management should work with the [NVIDIA GitHub Action Runners](https://docs.gha-runners.nvidia.com/) team to evaluate the feasibility of enabling the testing of pull requests in the Newton GitHub repository using NVIDIA’s self-hosted runners. If this is not possible due to the security risks involved (a real possibility given Newton’s neutral ownership), then use of the [paid GPU runners on GitHub](https://docs.github.com/en/actions/using-github-hosted-runners/using-larger-runners/about-larger-runners#specifications-for-gpu-larger-runners) might be required.

The Newton repository may also contain a top-level .gitlab-ci.yml file in order to also leverage Omniverse GitLab runners at NVIDIA.

#### Mirroring 

To enable CI/CD on internal Gitlab runners we should adopt a [pull mirroring](https://docs.gitlab.com/user/project/repository/mirror/pull/) approach where we have an internal Gitlab fork that pulls regularly from GitHub to run CI/CD. This is similar to the strategy taken by Warp where changes are immediately pushed to Github. A discussion with OV Security is needed to evaluate the feasibility of this approach (probably similar to how Google DeepMind tests MJWarp on internal CI/CD hardware).

#### Branches

The branching model should be similar to that of the [Python project](https://devguide.python.org/developer-workflow/development-cycle/).

* The main repository should only consist of the **main** branch consisting of development for the next feature release and **release branches** created for feature releases.  
* A developer seeking to contribute a new feature must fork the repository and create a pull request into the Newton repository targeting the **main** branch.  
* When a new feature version is about to undergo release testing, a new feature branch is created from the main branch, allowing development to continue on the main branch while the release candidate is stabilized on the feature branch.  
* After the initial release of a new minor version, the feature branch continues to be used for additional patch versions by cherry-picking bug fixes, documentation updates, example updates, etc. from the main branch.

#### Publishing

(PyPI ownership of newton-physics TBD. Initially this will be controlled by NVIDIA employees as we’re standing up the initial repository, but the maintainers of the newton-physics package will need to be expanded to outside maintainers.)

#### Versioning

Newton will use a MAJOR.MINOR.PATCH versioning scheme, but otherwise does not follow the prescriptions of [Semantic Versioning](https://semver.org/).

* New **major** versions: For strongly incompatible changes.  
* New **minor** versions: Standard feature releases.  
* New **patch** versions: Bugfix releases.

#### Dependencies

Newton’s primary dependencies are:

* NVIDIA Warp \- required  
* NumPy \- required  
* MuJoCo Warp \- optional  
* usd-core \- optional

NVIDIA Warp and NumPy provides API stability across minor releases (with the exception of occasional minor deprecation after a period of notification, see [Versioning](https://github.com/nvidia/warp?tab=readme-ov-file#versioning)). We expect Newton to work *everywhere that Warp is supported*.

Note that some optional dependencies, e.g.: MuJoCo Warp are not controlled by NVIDIA and may require version locking to ensure compatibility.

## Core Data Structures and Responsibilities

| Component | Responsibility | Type |
| :---- | :---- | ----- |
| `newton.Model` | Definition of simulation elements (bodies, joints, geometry, etc) | Input |
| `newton.State` | Time-varying variables (positions `qpos`, velocities `qvel`) | Input/Output |
| `newton.Control` | Actuator inputs (joint & body forces) | Input |
| `newton.Contact` | Per-step collision information | Input/Output |
| `newton.solvers.*` | Advance simulation forward (`state_in` -> `state_out`),  | Compute |

## Class Definitions

### `newton.Model`

Represents the *time-invariant* quantities of the simulation environment (bodies, joints, etc.). 

```py
class newton.Model:
    def __init__(self):
        self.joint_count = 0
        self.body_count = 0

        self.body_masses = wp.array(...)
        self.joint_parents = wp.array(...)

# Users can directly extend model at runtime:
model = builder.finalize()
model.user_array = wp.array(...)
```

Changes from [Warp.Sim.Model](https://nvidia.github.io/warp/modules/sim.html#warp.sim.Model):

* Explicit handling of multi-DOF and free-floating joints.  
* Accurate mapping of joint frames and quaternion conventions to align with MuJoCo Warp.  
* Add support for activating/deactivating individual bodies / joints to allow for user-controlled environment variations / dynamic object insertion.   
* Update joint drives to be more compatible with MuJoCo \- perhaps having a separate definition for joint actuators to indicate which joint (axes) are actuated and through which mode  
* Add support for sensors (probably need an abstract newton.Sensor class to handle force, position, gyro sensors, etc.)   
* Add support for [sites](https://mujoco.readthedocs.io/en/stable/overview.html#site) (from Mujoco; sites are empty geoms attached to bodies with no mass/collision to attach sensors, tendons, or use as query points for RL envs)  
  **Note:** the classification of Model data as time-invariant does not preclude domain randomization \- users can still modify Model parameters at runtime, however when doing so they must explicitly manage versioning of these quantities themselves for differentiability.

### `newton.State`

Holds the *time-varying* dynamic state variables (positions and velocities) that describe the system configuration at a point in time, e.g.:

```py
class newton.State:
    def __init__(self, model: newton.Model):
        self.qpos = wp.zeros(model.joint_count, dtype=wp.float32)
        self.qvel = wp.zeros(model.joint_count, dtype=wp.float32)
# Users can directly extend state:
state = model.state()
state.user_param_1 = wp.zeros(model.body_count, dtype=wp.vec3)
state.user_param_2 = wp.zeros(model.body_count, dtype=wp.float)
```

Changes from [Warp.Sim.State](https://nvidia.github.io/warp/modules/sim.html#warp.sim.State):

* Removed transient data such as `contacts` from state; **contacts** are now explicitly passed to the solver each step.

### `newton.Control`

Encapsulates external control inputs, cleanly separating actuation from physical state:

```py
class newton.Control:
    def __init__(self, model: newton.Model):
# Forces & torques        
self.joint_f = wp.zeros(model.joint_count, dtype=wp.float32)
self.body_f = wp.zeros((model.body_count, 6), dtype=wp.float32)  	 
self.joint_ctrl = wp.zeros(model.joint_count, dtype=wp.float32)
```

Changes from [Warp.Sim.Control](https://nvidia.github.io/warp/modules/sim.html#warp.sim.Control):

* Add `body_f` to represent external forces in maximal coordinates.

### `newton.Contact`

Represents per-step collision information, explicitly GPU-resident:

```py
class newton.Contact:
dist: wp.array(dtype=wp.float32)
pos: wp.array(dtype=wp.vec3f)
frame: wp.array(dtype=wp.mat33f)
dim: wp.array(dtype=wp.int32)
geom: wp.array(dtype=wp.vec2i)
```

Notes:

* Contact information should include *only* geometric data, not parameters that are specific to individual solvers, e.g.: friction, solimp, etc.  
* Solvers should have the opportunity (inside step() or otherwise) to augment the geometric data coming from the collision detector, e.g.: combining friction coefficients based on the relative surface attributes of the shapes.  
* Made this regular Python class not a `wp.struct` to allow users to extend attributes at runtime (similar to State / Model objects)

Changes from [Warp.Sim](https://nvidia.github.io/warp/modules/sim.html#warp.sim.collide):

* Warp.Sim stores contact data inside the Model which is inconsistent with the general principle of only storing time-invariant data, for Newton we will move these into a separate reusable structure.

Open questions:

* How to report contact forces efficiently?

### `newton.Tensors`

> [!NOTE]
> This section still in development, not clear if this kind of API belongs in Newton or Isaac Lab.

For robot learning it is desirable to have batched access to simulation data. In this section we propose a standalone module that enables users to create custom views onto simulation data.

Question: should we move core functionality related to view creation into a lower-level Newton module that would enable the engine to be easily integrated into other RL frameworks, e.g.:

```py
class newton.Tensors:
def __init__():
	...
# view onto state data 
@staticmethod
def create_model_view(model: Model, property: string, indices: wp.array) -> wp.indexed_array
return wp.indexedArray(model.get_model_attr(property), indices) 

# view onto state data
@staticmethod
def create_state_view(model: Model, state: State, property: string, indices: wp.array) -> wp.indexed_array
return wp.indexedArray(model.get_state_attr(property), indices)

```

---

## Example Workflow

```py
import warp as wp
import newton

# 1. Build the model from MJCF
builder = newton.ModelBuilder()
builder.import_mjcf("assets/scene.xml")
model = builder.finalize()

# 2. Create simulation state and control objects
state_in = newton.State(model)
state_out = newton.State(model)
ctrl = newton.Control(model)

# 3. Allocate a GPU array for contacts
max_contacts = 64*1024
contacts = newton.collide.allocate(max_contacts)

# 4. Create the MuJoCo-based solver
solver = newton.solvers.MJCSolver(model)

# 5. Run the simulation loop
for i in range(100):
    # Generate contacts
    newton.collide(model, state_in, contacts)

    # Set control input (e.g. torque on joint 0)
    ctrl.joint_f[0] = 15.0

    # Step the solver
    solver.step(state_in, state_out, ctrl, contacts, dt=1.0 / 60.0)

    # Swap input/output states
    state_in, state_out = state_out, state_in
```

## Implementing Custom Solvers

To implement a new solver in Newton, inherit from `newton.solvers.SolverBase`:

```py
class MyCustomSolver(newton.solvers.SolverBase):
    @override
    def step(self, state_in, state_out, ctrl, contacts, dt):
        # Implement custom update logic using Warp kernels
        # e.g., read state_in.qpos, qvel, ctrl.joint_f, etc.
	 ...
```

* **Stateless design**: For checkpointing and differentiability, solver implementations should not hold persistent data across time steps, although this decision can be made on a per-solver basis.  
* **Differentiable**: Because the solver is purely a function of `(state_in, ctrl, contacts, dt) → state_out`, it can be integrated into an autodiff workflow, e.g.:wrapped in `torch.autograd.Function`,  
* **Extended data**: If specific solver functionality needs additional arrays, they may be stored attached as custom fields on `Model` or `State` objects.

Although not strictly necessary, users may also choose to inherit from Model classes. This can be useful when solvers require specific additional data, users can then override the `Model.state()`, and `Model.control()`factory methods to return the augmented objects:

```py
class MyCustomModel(newton.Model):

    @override
    def state(): 
        
        # create base state
        state = super().state()

        # add our own properties
        state.custom_state_1 = wp.array(...)
        state.custom_state_2 = wp.array(...)

        return state

    @override
    def control():
        
        # create base state
        ctrl = super().control()

        # add our own properties
        ctrl.custom_control_1 = wp.array(...)
        ctrl.custom_control_2 = wp.array(...)

        return ctrl
```

## Implementing Custom Collision Detection

To define custom collision detection users need to provide a method mapping from a (model, state) pair to an array of contact points described by the `newton.Contact` type:

```py

# define custom collision detection method
def my_collide(model: newton.Model, 
               state: newton.State, 
               contacts: newton.Contact):
   pass
```

Notes:

* Users have freedom to perform collision detection per-step, or to cache the result of collision detection across multiple steps (in the case of substepping) as contacts are passed explicitly as inputs to the solver step method.

* Users may also wish to implement specific pair-wise functionality for custom collision types, or new support functions for additional convex shapes. These fine grained modifications may need lower level interfaces. We do not explore these further in this document.

---

## Handling MuJoCo Warp as a Runtime Dependency

Newton can expose a MuJoCo-based solver backend while avoiding a hard dependency by **deferring the import**:

```py
class MujocoSolver(newton.solvers.SolverBase):  # Located in solver_mujoco.py
    def __init__(self, model):
        super().__init__(model)
        try:
            from mujoco_warp import MJWarpSim
        except ImportError:
            raise ImportError("MuJoCo Warp backend not installed. Install with `pip install mujoco-warp`.")
        self.sim = MJWarpSim(model.mjcf_repr)

    def step(self, state_in, state_out, dt):
        self.sim.set_state(state_in.qpos, state_in.qvel)
        self.sim.step(state_in.ctrl, dt)
        state_out.qpos, state_out.qvel = self.sim.get_state()
```

* Users need not install `mujoco-warp` unless they require MuJoCo-based solver functionality.  
* The solver is inserted at runtime, preventing circular library dependencies.

## Kinematics & Dynamics Utilities

Newton provides standard GPU-based routines for forward/inverse kinematics and rigid-body dynamics. These functions reside in `newton/core/kinematics.py`, leveraging Warp primitives for parallel computations.

```py
def eval_fk(model: newton.Model, state: newton.State) -> wp.array:
    """
    Computes the forward kinematics (body transforms) for all rigid bodies
    in the model, returning an array of shape [model.body_count].
    Each entry might store a transform or (pos, rot).
    """
    pass

def eval_ik(model: newton.Model, target_poses: wp.array, initial_state: newton.State) -> wp.array:
    """
    Computes inverse kinematics for specified bodies (target_poses).
    Returns updated joint positions that minimize the error
    between the body pose(s) and each desired target pose.
    """
    pass

def jacobian(model: newton.Model, state: newton.State, body_idx: int) -> wp.array:
    """
    Computes the 6xDOF Jacobian for body_idx w.r.t. the model's joint angles.
    This can be used for manipulator control tasks (Jacobian-based IK,
    force control, etc.).
    """
    pass

def mass_matrix(model: newton.Model, state: newton.State) -> wp.array:
    """
    Computes the joint-space mass matrix of the articulated system,
    typically of shape [model.joint_count, model.joint_count].
    Useful for inverse dynamics, advanced control, or
    derivative-based analyses.
    """
    pass
```

* These routines support integration into autodiff pipelines.  
* Advanced versions may support partial joints, loop constraints, or symbolic acceleration.

## Isaac Lab Integration

Ideally users developing simulators should be able to implement the Newton interfaces directly and have them work in Isaac Lab. To achieve this there are a few open questions:

* How does data from Isaac Lab (e.g.: MJCF specific properties) flow through the Tensor API into the Newton Model and State objects. Preferably this would be automatic to some degree to avoid writing a lot of glue code from USD-\>Tensor-\>Newton.

* Users would need a way to “select” the solver applied to a specific scene (and presumably map any solver specific parameters to the corresponding constructor). This USD-\>Newton mapping needs to be fleshed out likely in a separate doc.

* For differentiability gradients would need to also be available through the Tensor API, this is a secondary concern though

### Exposing Custom Attributes via Tensor API

One of the goals for Newton is to allow users to easily extend our models, state, and solvers. This raises the question of how to expose new runtime simulation data into Isaac Lab where it can be used for reinforcement learning.

Currently the Tensor API exposes a *fixed* set of data based on pre-defined accessor functions, e.g.: [ArticulationView.get\_dof\_positions()](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_dof_positions): 

```py
import omni.physics.tensors as tensors
sim_view = tensors.create_simulation_view("warp")
articulation_view = sim_view.create_articulation_view("/World/Franka_*") # 
dof_positions = articulation_view.get_dof_positions() # Get the DOF position 
```

For Newton we encourage users to extend the simulator Model, State, and Solver classes. To access this data we propose a set of simple key/value accessors to lookup custom data:

```py
# get/set attributes on the Newton Model object
SimulationView.get_model_attr(key: string) -> wp.array
SimulationView.set_model_attr(key: string, value: wp.array)

# get/set attributes on the Newton State object
SimulationView.get_state_attr(key: string, indices: Optional) -> wp.array
SimulationView.set_state_attr(key: string, indices: Optional, value: wp.array)

# get/set attributes on the Newton Solver object
SimulationView.get_solver_attr(key: string) -> wp.array
SimulationView.set_solver_attr(key: string, value: Any)
```

This mirrors the dynamic nature of USD where Prims provide the ability to query and set custom attributes via. the Prim.GetAttribute() / Prim.SetAttribute() methods.

##### Example: Heat Exchange Simulation

As a first example we consider a custom simulator that computes heat exchange between particles and tracks a temperature value for each particle in the simulation such that the state is extended as follows:

```py
class newton.State:
    def __init__(self) -> None:
        self.particle_q: wp.array
        self.particle_qd: wp.array
        self.particle_f: wp.array	
	 # user adds additional `temperature` array per-particle
        self.particle_t: wp.array
```

This leverages the dynamic nature of Python programs, which allow users to define additional attributes at runtime on existing objects.

Inside Lab, users can access the additional particle data as follows:

```py
import omni.physics.tensors as tensors
sim_view = tensors.create_simulation_view("warp")
print(sim_view.get_state_attr("particle_t"))
```

##### Example: Stribeck Friction Model

Users may also extend the `Model` class by defining additional parameters that are interpreted by specific `Solver` implementations.

For example, users could implement a solver that supports Stribeck friction that introduces an additional “lubrication” parameter per-shape:

```py
class newton.Model:
    def __init__(self):
...
self.shape_transform : wp.array
self.shape_body : wp.array
self.shape_visible : wp.array
self.shape_materials : wp.array
self.shape_geo : wp.array
self.shape_geo_src : wp.array

# define additional shape Stribeck parameters
self.shape_lubrication : wp.array
```

Users may then define (and potentially randomize) these parameters through key/value setters and getters:

```py
import omni.physics.tensors as tensors
sim_view = tensors.create_simulation_view("warp")
sim_view.set_model_attr("shape_lubrication", wp.array(rng.random(num_shapes))
```

##### Example: Mapping USD Properties to Newton

The mechanisms above can be used to automatically map USD attributes into the Solver. For example, Isaac Lab may choose to look for `newton:attribute` properties on specific USD prims and automatically set these to the solver.

For example, the following could be used to automatically map USD attributes to Newton:

##### Example: Multi-DOF Joints

TODO: example showing how users might handle things like multi-dof joints inside Lab even though these concepts aren’t natively supported by UsdPhysics

##### UsdPhysics to Newton Mapping

* UsdPhysicsScene: The USD Physics Schema only defines gravity direction and gravity magnitude attributes on the primitive. However this is the best element to gather settings affecting the entire simulation and default values for the simulated elements.  
  Attributes defined on other physical primitives (e.g. `contact_thickness`) should be valid on PhysicsScene primitives to act as default values.

*Generic simulation attributes*

| Name | Type | Newton Mapping |
| :---- | :---- | :---- |
| newton:joint\_attach\_kd | float | Model.joint\_attach\_kd |
| newton:joint\_attach\_ke | float | Model.joint\_attach\_ke |
| newton:soft\_contact\_kd | float | Model.soft\_contact\_kd |
| newton:soft\_contact\_ke | float | Model.soft\_contact\_ke |
| newton:collide\_on\_substeps | bool | Simulation.collide\_on\_substeps |
| newton:fps | uint | Simulation.fps |
| newton:solver | string: euler/xpbd/vbd | Simulation.solver\_type |
| newton:substeps | uint | Simulation.substeps |

*Generic solver attributes, and attributes specific to the Semi-Implicit Euler and XPBD solvers*

| Name | Type | Recognized by |
| :---- | :---- | :---- |
| newton:solver:iterations | uint | SolverBase.iterations |
| newton:solver:angular\_damping | float | SemiImplicitSolver.angular\_damping, XPBDSolver.angular\_damping |
| newton:solver:friction\_smoothing | float | SemiImplicitSolver.friction\_smoothing, XPBDSolver.enable\_restitution |
| newton:solver:soft\_body\_relaxation | float | XPBDSolver.soft\_body\_relaxation |
| newton:solver:soft\_contact\_relaxation | float | XPBDSolver.soft\_contact\_relaxation |
| newton:solver:joint\_linear\_relaxation | float | XPBDSolver.joint\_linear\_relaxation |
| newton:solver:joint\_angular\_relaxation | float | XPBDSolver.joint\_angular\_relaxation |
| newton:solver:rigid\_contact\_relaxation | float | XPBDSolver.rigid\_contact\_relaxation |
| newton:solver:rigid\_contact\_con\_weighting | bool | XPBDSolver.rigid\_contact\_con\_weighting |

Example:

```
def PhysicsScene "physicsScene"
{
    vector3f physics:gravityDirection = (0, 0, -1)
    float physics:gravityMagnitude = 9.81
    bool newton:collide_on_substeps = 1
    uint newton:fps = 60
    string newton:solver = "xpbd"
    uint newton:solver:iterations = 10
    float newton:joint_attach_kd = 20
    float newton:joint_attach_ke = 1600
    uint newton:substeps = 20
    float newton:armature = 0.1
    float newton:contact_ke = 1e-3
    float newton:contact_kd = 1e-1
    float newton:contact_kf = 0.0
    float newton:collapse_fixed_joints = 0
}
```

* UsdPhysicsJoint: joint primitives recognize the following attributes

| Name | Type | Description |
| :---- | :---- | :---- |
| newton:joint\_limit\_ke | float | joint position limit stiffness |
| newton:joint\_limit\_kd | float | joint position limit damping |

* UsdPhysicsRigidBody: Primitives with the API’s schema applied are the simulated *bodies* in Newton.  
* UsdPhysicsCollisionAPI: Primitives with the API’s schema applied are collision shapes in Newton, and the following attributes are be recognized

| Name | Type | Description |
| :---- | :---- | :---- |
| newton:contact\_ke | float | contact elastic stiffness |
| newton:contact\_kd | float | contact damping stiffness |
| newton:contact\_kf | float | contact friction stiffness |
| newton:contact\_ka | float | contact adhesion distance |
| newton:contact\_thickness | float | thickness to use for collision handling |

* UsdPhysicsMaterialAPI: Attributes defined by the API map to shape attributes according to the table below

| Name | Type | Mapping |
| :---- | :---- | :---- |
| physics:density | float | shape density in Model |
| physics:restitution | float | shape restitution in Model |
| physics:staticFriction | float | n/a |
| physics:dynamicFriction | float | shape mu in Model |

```py
from pxr import Usd

def apply_newton_attrs(prim, simulation_view):
	for attr in prim.GetAttributes():
    	name = attr.GetName()
    		if name.startswith("newton:"):
        	param_name = name.split("newton:")[1]
        	simulation_view.set_solver_attr(param_name, attr.Get())
```

### omni.newton

To expose Newton into Omniverse and ultimately Isaac Lab / Sim we propose to publish a `omni.newton` extension that includes the Newton Python package and any additional integration code required (similar to how `omni.warp` exposes Warp in Omniverse). The extension code should live in the Isaac Sim repo. and be maintained by the Isaac team.

## Other Considerations

### Coupling Multiple Solvers

While not part of the MVP, Newton supports manual solver coupling via user-orchestrated techniques (e.g., impulse exchange or co-simulation). Future versions will explore APIs for automatic solver coupling.

* **Multi-Model and State:** if multiple solvers operate on the same scene, it is necessary to define which parts of the scene each solver should handle. For instance, both VDB and FEM solvers can handle cloth and softbodies, but one may wish to simulate all cloths with VDB, and all softbodies with FEM; each solver should still see the non-simulated objects as (partially) kinematic colliders. To achieve this without requiring new *Model* and *State* APIs, each solver can get assigned its own *Model* and *State* objects, where the kinematic parts are references to the arrays defined on the *Model* associated to the primary solver for each element. Solvers can enrich their *Model/State* with custom attributes without risk of conflicting with other solvers. Collision detection is run on each *Model/State* pair before stepping the corresponding solver, ignoring kinematic object pairs. Note that if solvers are known to only simulate distinct kinds of objects (e.g. rigid-bodies/deformables), then using a single Model remains possible.  
* **Finer-granularity solver stages:** Stable coupling will require defining more stages for the integration of a single timestep. Typically, coupled solvers may be iterated over multiple times until they reach an agreement about moment exchange before timestep finalization operations, such as advection and/or projection, are performed. A first strategy could be to decompose `step()` into `predict()`\+ `accept().`Typically, for a fixed-point loop differentiable simulation will only need to back-propagate through the last `predict()` operation, but an additional \`state\_tentative\` may be required.

### MuJoCo Warp Integration

Detailed recommendations for joint representation and frame conventions between Newton and MuJoCo Warp:

* **Multi-DOF Joints**: Warp's articulated body solver typically requires a single joint per body; MuJoCo can define multiple "stacked" joints in one body. To represent this consistently in Newton, combine them into one multi-axis joint (universal or compound) whenever possible.

* **Free Joints (Floating Bases)**: MuJoCo's `type="free"` is a 6-DOF root link. Newton should treat these as a `JOINT_FREE` or an equivalent 6-DOF representation.

* **Frame Conventions**: MuJoCo anchor positions and axes are local to each body. Newton uses a parent-child anchor transform model. Carefully compute each anchor transform from MuJoCo’s definitions for consistent joint axes.

* **Quaternion Ordering**: MuJoCo quaternions are in (w,x,y,z); Warp uses (x,y,z,w). Newton’s MJCF importer should reorder them on import.

* **Joint Limits & Damping**: By default, MuJoCo applies friction/damping at the joint. Newton must mirror these by setting limit stiffness/damping in the Model for each joint.

* **Inertia & Armature**: Use the actual inertia specified in MJCF instead of recomputing from geometry. Also reflect MuJoCo’s per-joint `armature` param into Newton’s joint data.

* **Closed-Loop / Additional Constraints**: If the MJCF model has equality constraints (loops, gears, tendons), Newton may not handle them under Featherstone’s tree solver. For future expansions, store them for XPBD or a custom constraint solver.

* **MJCF Parsing Enhancements**: Pass `ignore_inertial_definitions=False`, `collapse_fixed_joints=True`, etc., to ensure more faithful MuJoCo replication and more efficient models.

* Architectural support for MuJoCo-specific constraints (tendons, gears).

### Maximal-Coordinate Solver Support

* Explicit mapping between maximal and reduced coordinate representations.  
* Utility functions for coordinate conversions.  
* Consistency in collision and constraint representations.  
* Clear documentation and API design supporting additional constraint specifications (e.g., loop closures).

---

## Open Questions

### Batched vs Indexed

There is a question about whether to ‘bake’ batching into the Model / State objects directly, i.e.: to directly expose the concept of environments on the physics engine itself. Engines like PhysX and Warp.Sim do not do this explicitly, but rather index environments into large unstructured groups of objects / articulations. Quick summary of trade-offs:

* Batching simplifies implementations by assuming homogenous groups of objects  
* Batching can lead to better performance (sharing single values across all environments)  
* Indexing can allow describing heterogeneous environments e.g.: environments with different numbers of objects or objects that are shared (e.g.: terrain) across environments

Batched interfaces can also be provided at a higher level (e.g.: TensorAPI).

### How will Newton be made available to Isaac Lab?

It would be ideal if we can avoid creating and publishing omni.newton and omni.mujoco\_warp extensions, whose only purpose is to get the respective libraries made available to the IsaacSim Python interpreter.

### Where will the Newton repository be hosted?

The plan is to use [https://github.com/Newton-Physics](https://github.com/Newton-Physics) as the name of the GitHub organization inside the GitHub organization, the repository for Newton will be either *newton* (preferred) or *newton-physics* (potentially less disruptive if repository is moved into another GitHub organization).

Newton will be published as a Python package on PyPi with the final name TBD.
