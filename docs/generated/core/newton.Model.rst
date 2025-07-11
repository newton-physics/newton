newton.Model
============

.. currentmodule:: newton

.. autoclass:: Model
   :members:
   :inherited-members:
   :member-order: bysource
   
   

   
   .. rubric:: Methods

   .. autosummary::
   
      ~Model.__init__
      ~Model.add_attribute
      ~Model.collide
      ~Model.control
      ~Model.get_attribute_frequency
      ~Model.state
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~Model.requires_grad
      ~Model.num_envs
      ~Model.particle_q
      ~Model.particle_qd
      ~Model.particle_mass
      ~Model.particle_inv_mass
      ~Model.particle_radius
      ~Model.particle_max_radius
      ~Model.particle_ke
      ~Model.particle_kd
      ~Model.particle_kf
      ~Model.particle_mu
      ~Model.particle_cohesion
      ~Model.particle_adhesion
      ~Model.particle_grid
      ~Model.particle_flags
      ~Model.particle_max_velocity
      ~Model.shape_key
      ~Model.shape_transform
      ~Model.shape_body
      ~Model.shape_flags
      ~Model.body_shapes
      ~Model.shape_materials
      ~Model.shape_geo
      ~Model.shape_geo_src
      ~Model.shape_collision_group
      ~Model.shape_collision_group_map
      ~Model.shape_collision_filter_pairs
      ~Model.shape_collision_radius
      ~Model.shape_contact_pairs
      ~Model.shape_contact_pair_count
      ~Model.spring_indices
      ~Model.spring_rest_length
      ~Model.spring_stiffness
      ~Model.spring_damping
      ~Model.spring_control
      ~Model.spring_constraint_lambdas
      ~Model.tri_indices
      ~Model.tri_poses
      ~Model.tri_activations
      ~Model.tri_materials
      ~Model.tri_areas
      ~Model.edge_indices
      ~Model.edge_rest_angle
      ~Model.edge_rest_length
      ~Model.edge_bending_properties
      ~Model.edge_constraint_lambdas
      ~Model.tet_indices
      ~Model.tet_poses
      ~Model.tet_activations
      ~Model.tet_materials
      ~Model.muscle_start
      ~Model.muscle_params
      ~Model.muscle_bodies
      ~Model.muscle_points
      ~Model.muscle_activations
      ~Model.body_q
      ~Model.body_qd
      ~Model.body_com
      ~Model.body_inertia
      ~Model.body_inv_inertia
      ~Model.body_mass
      ~Model.body_inv_mass
      ~Model.body_key
      ~Model.joint_q
      ~Model.joint_qd
      ~Model.joint_f
      ~Model.joint_target
      ~Model.joint_type
      ~Model.joint_parent
      ~Model.joint_child
      ~Model.joint_ancestor
      ~Model.joint_X_p
      ~Model.joint_X_c
      ~Model.joint_axis
      ~Model.joint_armature
      ~Model.joint_target_ke
      ~Model.joint_target_kd
      ~Model.joint_effort_limit
      ~Model.joint_velocity_limit
      ~Model.joint_friction
      ~Model.joint_dof_dim
      ~Model.joint_dof_mode
      ~Model.joint_enabled
      ~Model.joint_limit_lower
      ~Model.joint_limit_upper
      ~Model.joint_limit_ke
      ~Model.joint_limit_kd
      ~Model.joint_twist_lower
      ~Model.joint_twist_upper
      ~Model.joint_q_start
      ~Model.joint_qd_start
      ~Model.joint_key
      ~Model.articulation_start
      ~Model.articulation_key
      ~Model.soft_contact_ke
      ~Model.soft_contact_kd
      ~Model.soft_contact_kf
      ~Model.soft_contact_mu
      ~Model.soft_contact_restitution
      ~Model.rigid_contact_max
      ~Model.rigid_contact_torsional_friction
      ~Model.rigid_contact_rolling_friction
      ~Model.up_vector
      ~Model.up_axis
      ~Model.gravity
      ~Model.particle_count
      ~Model.body_count
      ~Model.shape_count
      ~Model.joint_count
      ~Model.tri_count
      ~Model.tet_count
      ~Model.edge_count
      ~Model.spring_count
      ~Model.muscle_count
      ~Model.articulation_count
      ~Model.joint_dof_count
      ~Model.joint_coord_count
      ~Model.particle_color_groups
      ~Model.particle_colors
      ~Model.device
      ~Model.attribute_frequency
   
   