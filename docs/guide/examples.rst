.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Examples
========

Newton ships runnable examples that demonstrate focused simulation workflows
across rigid bodies, robotics, cloth, cables, MPM, sensors, differentiable
simulation, and other areas. Use the examples to browse concrete scenes,
compare solver configurations, or start from a working script.

Install the examples extra before running the packaged examples:

.. code-block:: console

    pip install "newton[examples]"

When working from a source checkout, use ``uv run`` from the repository root:

.. code-block:: console

    uv run -m newton.examples --list

Network and Asset Downloads
---------------------------

Some examples download external assets the first time they run. Newton pins
these asset repositories to known revisions and caches downloads locally, so
later runs can reuse the cached files. Many of these assets come from the
`newton-assets <https://github.com/newton-physics/newton-assets>`__ repository.
Some examples use other pinned external asset repositories for robot models,
meshes, or policies.

Examples that need external assets require network access the first time they
run. After the assets are cached, later runs can reuse them offline.

Browse Examples
---------------

List the available examples from the command line:

.. code-block:: console

    python -m newton.examples --list

Launch the default example browser:

.. code-block:: console

    python -m newton.examples

The GL viewer includes an examples side panel for browsing examples by
category. Selecting an entry switches to that example, and the viewer can reset
the current example while preserving the relevant command-line options.

Run an Example
--------------

Run any listed short name with ``python -m newton.examples <name>``. For
example:

.. code-block:: console

    python -m newton.examples basic_pendulum
    python -m newton.examples basic_shapes
    python -m newton.examples robot_cartpole

Use ``--help`` after a short name to show the options for that example,
including common options and any example-specific flags:

.. code-block:: console

    python -m newton.examples robot_cartpole --help

The examples share common command-line options:

.. list-table::
   :header-rows: 1
   :widths: 24 54 22

   * - Option
     - Description
     - Example
   * - ``--viewer``
     - Select the viewer: ``gl``, ``usd``, ``rtx``, ``rerun``, ``viser``, or
       ``null``.
     - ``--viewer usd``
   * - ``--device``
     - Select the Warp device.
     - ``--device cuda:0``
   * - ``--num-frames``
     - Set the number of frames for finite runs and file output.
     - ``--num-frames 500``
   * - ``--output-path``
     - Set the output path for file-based viewers.
     - ``--output-path output.usd``
   * - ``--test``
     - Run example validation checks when available.
     - ``--viewer null --test``

For example:

.. code-block:: console

    python -m newton.examples basic_viewer --viewer usd --output-path my_output.usd
    python -m newton.examples basic_urdf --device cuda:0
    python -m newton.examples basic_viewer --viewer gl --num-frames 500 --device cpu

Example Gallery
---------------

The command-line list is the canonical catalog. The galleries below show the
examples that have preview images in the docs.

Basic and Visualization
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 33 33 33
   :class: gallery

   * - .. image:: ../images/examples/example_basic_pendulum.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_pendulum.py
          :alt: basic_pendulum
     - .. image:: ../images/examples/example_basic_urdf.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_urdf.py
          :alt: basic_urdf
     - .. image:: ../images/examples/example_basic_viewer.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_viewer.py
          :alt: basic_viewer
   * - ``python -m newton.examples basic_pendulum``
     - ``python -m newton.examples basic_urdf``
     - ``python -m newton.examples basic_viewer``
   * - .. image:: ../images/examples/example_basic_shapes.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_shapes.py
          :alt: basic_shapes
     - .. image:: ../images/examples/example_basic_joints.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_joints.py
          :alt: basic_joints
     - .. image:: ../images/examples/example_basic_conveyor.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_conveyor.py
          :alt: basic_conveyor
   * - ``python -m newton.examples basic_shapes``
     - ``python -m newton.examples basic_joints``
     - ``python -m newton.examples basic_conveyor``
   * - .. image:: ../images/examples/example_basic_heightfield.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_heightfield.py
          :alt: basic_heightfield
     - .. image:: ../images/examples/example_recording.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_recording.py
          :alt: recording
     - .. image:: ../images/examples/example_replay_viewer.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_replay_viewer.py
          :alt: replay_viewer
   * - ``python -m newton.examples basic_heightfield``
     - ``python -m newton.examples recording``
     - ``python -m newton.examples replay_viewer``
   * - .. image:: ../images/examples/example_basic_plotting.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_plotting.py
          :alt: basic_plotting
     -
     -
   * - ``python -m newton.examples basic_plotting``
     -
     -

Robotics
^^^^^^^^

.. list-table::
   :widths: 33 33 33
   :class: gallery

   * - .. image:: ../images/examples/example_robot_cartpole.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_cartpole.py
          :alt: robot_cartpole
     - .. image:: ../images/examples/example_robot_g1.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_g1.py
          :alt: robot_g1
     - .. image:: ../images/examples/example_robot_h1.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_h1.py
          :alt: robot_h1
   * - ``python -m newton.examples robot_cartpole``
     - ``python -m newton.examples robot_g1``
     - ``python -m newton.examples robot_h1``
   * - .. image:: ../images/examples/example_robot_anymal_d.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_anymal_d.py
          :alt: robot_anymal_d
     - .. image:: ../images/examples/example_robot_anymal_c_walk.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_anymal_c_walk.py
          :alt: robot_anymal_c_walk
     -
   * - ``python -m newton.examples robot_anymal_d``
     - ``python -m newton.examples robot_anymal_c_walk``
     -
   * - .. image:: ../images/examples/example_robot_policy.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_policy.py
          :alt: robot_policy
     - .. image:: ../images/examples/example_robot_ur10.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_ur10.py
          :alt: robot_ur10
     - .. image:: ../images/examples/example_robot_panda_hydro.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_panda_hydro.py
          :alt: robot_panda_hydro
   * - ``python -m newton.examples robot_policy``
     - ``python -m newton.examples robot_ur10``
     - ``python -m newton.examples robot_panda_hydro``
   * - .. image:: ../images/examples/example_robot_allegro_hand.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_allegro_hand.py
          :alt: robot_allegro_hand
     -
     -
   * - ``python -m newton.examples robot_allegro_hand``
     -
     -

Cables
^^^^^^

.. list-table::
   :widths: 33 33 33
   :class: gallery

   * - .. image:: ../images/examples/example_cable_twist.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/cable/example_cable_twist.py
          :alt: cable_twist
     - .. image:: ../images/examples/example_cable_y_junction.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/cable/example_cable_y_junction.py
          :alt: cable_y_junction
     - .. image:: ../images/examples/example_cable_bundle_hysteresis.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/cable/example_cable_bundle_hysteresis.py
          :alt: cable_bundle_hysteresis
   * - ``python -m newton.examples cable_twist``
     - ``python -m newton.examples cable_y_junction``
     - ``python -m newton.examples cable_bundle_hysteresis``
   * - .. image:: ../images/examples/example_cable_pile.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/cable/example_cable_pile.py
          :alt: cable_pile
     - .. image:: ../images/examples/example_cable_cross_slide_table.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/cable/example_cable_cross_slide_table.py
          :alt: cable_cross_slide_table
     -
   * - ``python -m newton.examples cable_pile``
     - ``python -m newton.examples cable_cross_slide_table``
     -

Cloth
^^^^^

.. list-table::
   :widths: 33 33 33
   :class: gallery

   * - .. image:: ../images/examples/example_cloth_bending.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_bending.py
          :alt: cloth_bending
     - .. image:: ../images/examples/example_cloth_hanging.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_hanging.py
          :alt: cloth_hanging
     - .. image:: ../images/examples/example_cloth_style3d.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_style3d.py
          :alt: cloth_style3d
   * - ``python -m newton.examples cloth_bending``
     - ``python -m newton.examples cloth_hanging``
     - ``python -m newton.examples cloth_style3d``
   * - .. image:: ../images/examples/example_cloth_h1.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_h1.py
          :alt: cloth_h1
     - .. image:: ../images/examples/example_cloth_twist.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_twist.py
          :alt: cloth_twist
     - .. image:: ../images/examples/example_cloth_franka.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_franka.py
          :alt: cloth_franka
   * - ``python -m newton.examples cloth_h1``
     - ``python -m newton.examples cloth_twist``
     - ``python -m newton.examples cloth_franka``
   * - .. image:: ../images/examples/example_cloth_rollers.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_rollers.py
          :alt: cloth_rollers
     - .. image:: ../images/examples/example_cloth_poker_cards.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_poker_cards.py
          :alt: cloth_poker_cards
     -
   * - ``python -m newton.examples cloth_rollers``
     - ``python -m newton.examples cloth_poker_cards``
     -

Inverse Kinematics
^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 33 33 33
   :class: gallery

   * - .. image:: ../images/examples/example_ik_franka.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/ik/example_ik_franka.py
          :alt: ik_franka
     - .. image:: ../images/examples/example_ik_h1.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/ik/example_ik_h1.py
          :alt: ik_h1
     - .. image:: ../images/examples/example_ik_custom.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/ik/example_ik_custom.py
          :alt: ik_custom
   * - ``python -m newton.examples ik_franka``
     - ``python -m newton.examples ik_h1``
     - ``python -m newton.examples ik_custom``
   * - .. image:: ../images/examples/example_ik_cube_stacking.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/ik/example_ik_cube_stacking.py
          :alt: ik_cube_stacking
     -
     -
   * - ``python -m newton.examples ik_cube_stacking``
     -
     -

Material Point Method
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 33 33 33
   :class: gallery

   * - .. image:: ../images/examples/example_mpm_granular.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_granular.py
          :alt: mpm_granular
     - .. image:: ../images/examples/example_mpm_anymal.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_anymal.py
          :alt: mpm_anymal
     - .. image:: ../images/examples/example_mpm_twoway_coupling.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_twoway_coupling.py
          :alt: mpm_twoway_coupling
   * - ``python -m newton.examples mpm_granular``
     - ``python -m newton.examples mpm_anymal``
     - ``python -m newton.examples mpm_twoway_coupling``
   * - .. image:: ../images/examples/example_mpm_grain_rendering.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_grain_rendering.py
          :alt: mpm_grain_rendering
     - .. image:: ../images/examples/example_mpm_multi_material.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_multi_material.py
          :alt: mpm_multi_material
     - .. image:: ../images/examples/example_mpm_viscous.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_viscous.py
          :alt: mpm_viscous
   * - ``python -m newton.examples mpm_grain_rendering``
     - ``python -m newton.examples mpm_multi_material``
     - ``python -m newton.examples mpm_viscous``
   * - .. image:: ../images/examples/example_mpm_beam_twist.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_beam_twist.py
          :alt: mpm_beam_twist
     - .. image:: ../images/examples/example_mpm_snow_ball.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_snow_ball.py
          :alt: mpm_snow_ball
     -
   * - ``python -m newton.examples mpm_beam_twist``
     - ``python -m newton.examples mpm_snow_ball``
     -

Sensors
^^^^^^^

.. list-table::
   :widths: 33 33 33
   :class: gallery

   * - .. image:: ../images/examples/example_sensor_contact.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/sensors/example_sensor_contact.py
          :alt: sensor_contact
     - .. image:: ../images/examples/example_sensor_tiled_camera.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/sensors/example_sensor_tiled_camera.py
          :alt: sensor_tiled_camera
     - .. image:: ../images/examples/example_sensor_imu.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/sensors/example_sensor_imu.py
          :alt: sensor_imu
   * - ``python -m newton.examples sensor_contact``
     - ``python -m newton.examples sensor_tiled_camera``
     - ``python -m newton.examples sensor_imu``

Selection
^^^^^^^^^

.. list-table::
   :widths: 33 33 33
   :class: gallery

   * - .. image:: ../images/examples/example_selection_cartpole.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/selection/example_selection_cartpole.py
          :alt: selection_cartpole
     - .. image:: ../images/examples/example_selection_materials.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/selection/example_selection_materials.py
          :alt: selection_materials
     - .. image:: ../images/examples/example_selection_articulations.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/selection/example_selection_articulations.py
          :alt: selection_articulations
   * - ``python -m newton.examples selection_cartpole``
     - ``python -m newton.examples selection_materials``
     - ``python -m newton.examples selection_articulations``
   * - .. image:: ../images/examples/example_selection_multiple.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/selection/example_selection_multiple.py
          :alt: selection_multiple
     -
     -
   * - ``python -m newton.examples selection_multiple``
     -
     -

Differentiable Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 33 33 33
   :class: gallery

   * - .. image:: ../images/examples/example_diffsim_ball.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_ball.py
          :alt: diffsim_ball
     - .. image:: ../images/examples/example_diffsim_cloth.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_cloth.py
          :alt: diffsim_cloth
     - .. image:: ../images/examples/example_diffsim_drone.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_drone.py
          :alt: diffsim_drone
   * - ``python -m newton.examples diffsim_ball``
     - ``python -m newton.examples diffsim_cloth``
     - ``python -m newton.examples diffsim_drone``
   * - .. image:: ../images/examples/example_diffsim_spring_cage.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_spring_cage.py
          :alt: diffsim_spring_cage
     - .. image:: ../images/examples/example_diffsim_soft_body.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_soft_body.py
          :alt: diffsim_soft_body
     - .. image:: ../images/examples/example_diffsim_bear.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_bear.py
          :alt: diffsim_bear
   * - ``python -m newton.examples diffsim_spring_cage``
     - ``python -m newton.examples diffsim_soft_body``
     - ``python -m newton.examples diffsim_bear``

Multiphysics
^^^^^^^^^^^^

.. list-table::
   :widths: 33 33 33
   :class: gallery

   * - .. image:: ../images/examples/example_softbody_gift.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/multiphysics/example_softbody_gift.py
          :alt: softbody_gift
     - .. image:: ../images/examples/example_softbody_dropping_to_cloth.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/multiphysics/example_softbody_dropping_to_cloth.py
          :alt: softbody_dropping_to_cloth
     - .. image:: ../images/examples/example_rigid_soft_contact.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/multiphysics/example_rigid_soft_contact.py
          :alt: rigid_soft_contact
   * - ``python -m newton.examples softbody_gift``
     - ``python -m newton.examples softbody_dropping_to_cloth``
     - ``python -m newton.examples rigid_soft_contact``

Contacts
^^^^^^^^

.. list-table::
   :widths: 33 33 33
   :class: gallery

   * - .. image:: ../images/examples/example_nut_bolt_hydro.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/contacts/example_nut_bolt_hydro.py
          :alt: nut_bolt_hydro
     - .. image:: ../images/examples/example_nut_bolt_sdf.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/contacts/example_nut_bolt_sdf.py
          :alt: nut_bolt_sdf
     - .. image:: ../images/examples/example_brick_stacking.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/contacts/example_brick_stacking.py
          :alt: brick_stacking
   * - ``python -m newton.examples nut_bolt_hydro``
     - ``python -m newton.examples nut_bolt_sdf``
     - ``python -m newton.examples brick_stacking``
   * - .. image:: ../images/examples/example_pyramid.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/contacts/example_pyramid.py
          :alt: pyramid
     - .. image:: ../images/examples/example_contacts_rj45_plug.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/contacts/example_contacts_rj45_plug.py
          :alt: contacts_rj45_plug
     -
   * - ``python -m newton.examples pyramid``
     - ``python -m newton.examples contacts_rj45_plug``
     -

Soft Bodies
^^^^^^^^^^^

.. list-table::
   :widths: 33 33 33
   :class: gallery

   * - .. image:: ../images/examples/example_softbody_hanging.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/softbody/example_softbody_hanging.py
          :alt: softbody_hanging
     - .. image:: ../images/examples/example_softbody_franka.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/softbody/example_softbody_franka.py
          :alt: softbody_franka
     -
   * - ``python -m newton.examples softbody_hanging``
     - ``python -m newton.examples softbody_franka``
     -

Kamino
^^^^^^

.. list-table::
   :widths: 33 33 33
   :class: gallery

   * - .. image:: ../images/examples/example_kamino_basic_dr_testmech.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/kamino/example_kamino_basic_dr_testmech.py
          :alt: kamino_basic_dr_testmech
     - .. image:: ../images/examples/example_kamino_basic_fourbar.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/kamino/example_kamino_basic_fourbar.py
          :alt: kamino_basic_fourbar
     - .. image:: ../images/examples/example_kamino_basic_heterogeneous.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/kamino/example_kamino_basic_heterogeneous.py
          :alt: kamino_basic_heterogeneous
   * - ``python -m newton.examples kamino_basic_dr_testmech``
     - ``python -m newton.examples kamino_basic_fourbar``
     - ``python -m newton.examples kamino_basic_heterogeneous``
   * - .. image:: ../images/examples/example_kamino_robot_anymal_d.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/kamino/example_kamino_robot_anymal_d.py
          :alt: kamino_robot_anymal_d
     - .. image:: ../images/examples/example_kamino_robot_dr_legs.jpg
          :target: https://github.com/newton-physics/newton/blob/main/newton/examples/kamino/example_kamino_robot_dr_legs.py
          :alt: kamino_robot_dr_legs
     -
   * - ``python -m newton.examples kamino_robot_anymal_d``
     - ``python -m newton.examples kamino_robot_dr_legs``
     -
