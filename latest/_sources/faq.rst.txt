Frequently Asked Questions (FAQ)
================================

This FAQ addresses general questions about Newton. For technical questions and answers, please refer to `GitHub Discussions <https://github.com/newton-physics/newton/discussions>`_.

What is Newton?
----------------

Newton is an open-source, GPU-accelerated, extensible, and differentiable physics engine for robotics built on `NVIDIA Warp <https://github.com/NVIDIA/warp>`_, initially focused on high-performance robot learning and `MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>`_ integration.

Newton supports OpenUSD, URDF, and MJCF asset formats and provides multiple solver backends within a unified architecture for simulation, learning, and extensibility.

Newton is a `Linux Foundation <https://www.linuxfoundation.org/>`_ project initiated by `Disney Research <https://www.disneyresearch.com/>`_, `Google DeepMind <https://deepmind.google/>`_, and `NVIDIA <https://www.nvidia.com/>`_. The project is community-built and maintained under the permissive `Apache-2.0 license <https://github.com/newton-physics/newton/blob/main/LICENSE.md>`_.

What is the difference between Warp and Newton?
-----------------------------------------------

`Warp <https://github.com/NVIDIA/warp>`_ is a Python framework for writing high‑performance, differentiable GPU kernels for physics simulation and spatial computing. Newton is a full physics engine built on Warp that adds high‑level simulation APIs, interchangeable solvers, and asset I/O for robotics.

What is the difference between Warp.sim and Newton?
---------------------------------------------------

Warp.sim is the predecessor to Newton, developed by NVIDIA as a module in Warp, and will be deprecated as of Warp 1.10. See the `announcement <https://github.com/NVIDIA/warp/discussions/735>`_ and :doc:`migration` to Newton.

Does Newton support coupling of solvers for multiphysics or co‑simulation?
--------------------------------------------------------------------------

Yes, Newton is explicitly designed to be extensible with multiple solver implementations for rich multiphysics scenarios. Newton provides APIs for users to implement coupling between solvers, and we have successfully demonstrated one-way coupling in examples such as cloth manipulation by a robotic arm and a quadruped walking through non-rigid terrain. Two-way coupling and implicit coupling between select solvers are on the Newton roadmap.

Does Newton support MuJoCo simulation?
--------------------------------------

Newton leverages `MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>`_ as a key solver, which is a reimplementation of MuJoCo in Warp for GPU acceleration, developed and maintained by Google DeepMind and NVIDIA.

Newton can import assets in MJCF, URDF, and OpenUSD formats, making it compatible with MuJoCo at both asset and solver levels.

Is Newton exposed and accessible in Isaac Lab and Isaac Sim?
------------------------------------------------------------

Yes, an experimental Newton integration is available in Isaac Lab and under active development. Initial training environments include quadruped and biped locomotion, and basic manipulation. Read more on `the integration <https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html>`_.

Newton integration with Isaac Sim as a physics backend is under development.

Is Newton a standalone framework?
-----------------------------------------

Yes, Newton and its modern Python API can be used as a standalone simulation framework. See the :doc:`api/newton` or the `Quickstart Guide <https://github.com/newton-physics/newton?tab=readme-ov-file#quickstart>`_ for more information.

Does Newton provide visualization capabilities?
-----------------------------------------------

Newton provides basic visualization for debugging purposes. Read more in the :doc:`guide/visualization` Guide.

For rich real‑time graphics, users commonly pair Newton with Isaac Lab, which provides advanced rendering. Users can also export simulation outputs to a time-sampled USD that can be visualized, for example, in `NVIDIA Omniverse <https://www.nvidia.com/en-us/omniverse/>`_ or `Isaac Sim <https://developer.nvidia.com/isaac/sim>`_.

How can I contribute to Newton?
-------------------------------

Newton welcomes community contributions. Please see the `Contribution Guide <https://github.com/newton-physics/newton/blob/main/CONTRIBUTING.md>`_ and :doc:`guide/development` Guide for more information.

Can I use Newton to develop my own custom solver?
-------------------------------------------------

Yes, Newton is designed to be highly extensible, supporting custom solvers, integrators, and numerical methods within its modular architecture.

What is PhysX?
--------------

`PhysX <https://github.com/NVIDIAGameWorks/PhysX>`_ is an open-source, multi-physics SDK that provides scalable simulation across CPUs and GPUs, widely used for industrial digital-twin simulation in Omniverse, and robotics simulation in IsaacSim and IsaacLab.

It features a unified simulation framework for reduced-coordinate articulations and rigid bodies, deformable bodies and cloth (FEM), fluids/particles (PBD), vehicle dynamics, and character controllers.

Will Newton replace PhysX?
--------------------------

No, the two engines serve different primary goals: Newton targets robot learning and extensible multiphysics with differentiability, while PhysX focuses on industrial digital-twin simulation as a mature, multi-platform real‑time physics SDK that is actively maintained and updated.

Isaac Lab's experimental Newton integration does not support PhysX, but Isaac Lab plans to continue supporting PhysX as a simulation backend.

Can PhysX work in Newton?
-------------------------

No. However, different approaches for supporting PhysX as a Newton solver option are under consideration.
