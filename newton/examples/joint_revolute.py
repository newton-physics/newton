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

###########################################################################
# Example Sim Quadruped
#
# Shows how to set up a simulation of a rigid-body quadruped articulation
# from a URDF using the newton.ModelBuilder().
# Note this example does not include a trained policy.
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.collision
import newton.utils

from newton.core import (
	JOINT_MODE_FORCE,
	JOINT_MODE_TARGET_POSITION,
	JOINT_MODE_TARGET_VELOCITY
)

@wp.kernel
def apply_jointForce(jointForceArray: wp.array(dtype=float)):
	tid = wp.tid()
	print(tid)
	jointForceArray[tid] = 2.0

@wp.kernel
def print_jointPos(jointPosArray: wp.array(dtype=float)):
	tid = wp.tid()
	print(jointPosArray[tid])

@wp.kernel
def print_bodyq(bodyq: wp.array(dtype=wp.transformf)):
	tid = wp.tid()
	wp.printf("bodyq %f, %f, %f, %f, %f, %f, %f\n",
		bodyq[tid].p.x, bodyq[tid].p.y, bodyq[tid].p.z,
		bodyq[tid].q.x, bodyq[tid].q.y, bodyq[tid].q.z, bodyq[tid].q.w)

@wp.kernel
def set_jointSpeed(driveSpeedArray: wp.array(dtype=float)):
	tid = wp.tid()
	driveSpeedArray[tid] = 2.0

def computeStartPos(originPos: wp.vec3, x: int, deltaX: float, y: int, deltaY: float, z: int, deltaZ: float) -> wp.vec:
	offset = wp.vec3(x*deltaX, y*deltaY, z*deltaZ)
	startPos = originPos + offset
	return startPos

class Example:
	def __init__(self, stage_path="example_pendulum.usd", num_envs=1):

		use_xpbd = False
		use_mujoco = False
		
		self.sim_time = 0.0
		fps = 60
		self.frame_dt = 1.0 / fps

		self.sim_substeps = 10
		self.sim_dt = self.frame_dt / self.sim_substeps

		self.num_envs = num_envs

		contact_mu=1.0
		contact_restitution = 0.001

		# Cuboid properties
		box_mass = 1.0
		box_extents = wp.vec3(0.2, 0.2, 0.2)
		box_halfextents = box_extents*0.5
		box_inertia = wp.mat33((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
		box_rot = wp.quat_identity()
		box_com = wp.vec3(0.0, 0.0, 0.0)
 
		# Separation of boxes along x,z axes
		delta_X= 1.0
		delta_Y = 1.0
		delta_Z = 1.0

		rbody_builder = newton.ModelBuilder(
			up_axis=newton.Axis.Y,
			gravity = 10)

		nbBodies = 0

		for x in range(self.num_envs):
			for y in range(1):
				for z in range(self.num_envs): 

					# compute the start pos of body at x,z
					start_pos = computeStartPos(wp.vec3(0.0, 0.0, 0.0), x, delta_X, y, delta_Y, z, delta_Z)
					start_pose = wp.transform(start_pos, wp.quat_identity())

					# Create a body instance
					body_id= rbody_builder.add_body(
						xform=start_pose, armature=0.0, 
						com=box_com, I_m=box_inertia, mass=box_mass)

					# Create a box shape for the body
					rbody_builder.add_shape_box(
						body=body_id, 
						hx=box_halfextents.x, hy=box_halfextents.y, hz=box_halfextents.z)

					# Create a revolute joint
					joint_id = rbody_builder.add_joint_revolute(
						parent=-1, child=body_id, 
						parent_xform=wp.transform(start_pos + wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()),
						child_xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()),
						axis=wp.vec3(0.0, 0.0, 1.0),
						target=math.pi/2,
						armature=0.0,
						mode=JOINT_MODE_TARGET_VELOCITY,
						# limit_lower=-math.pi,
						# limit_upper=math.pi,
						limit_ke=0.0,
						limit_kd=0.0,
						target_ke=2000.0,
						target_kd=500.0)

					nbBodies += 1
		
		print(nbBodies)
		np.set_printoptions(suppress=True)

		# finalize model
		self.model = rbody_builder.finalize()
		self.model.ground = False

		if use_xpbd:
			self.solver = newton.solvers.XPBDSolver(self.model)
		else:
			self.solver = newton.solvers.MuJoCoSolver(
				self.model,
				use_mujoco=use_mujoco,
				solver="newton",
				integrator="euler",
				iterations=20)
			

		if stage_path:
			self.renderer = newton.utils.SimRendererOpenGL(self.model, stage_path)
		else:
			self.renderer = None

		self.state_0 = self.model.state()
		self.state_1 = self.model.state()
		self.control = self.model.control()

		# simulate() allocates memory via a clone, so we can't use graph capture if the device does not support mempools
		self.use_cuda_graph = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device()) and not use_mujoco
		if self.use_cuda_graph:
			with wp.ScopedCapture() as capture:
				self.simulate()
			self.graph = capture.graph
		else:
			self.graph = None

	def simulate(self):
		for _ in range(self.sim_substeps):
			self.state_0.clear_forces()
			self.solver.step(self.model, self.state_0, self.state_1, self.control, None, self.sim_dt)
			self.state_0, self.state_1 = self.state_1, self.state_0


	def step(self):
		with wp.ScopedTimer("step"):
			if self.use_cuda_graph:
				wp.capture_launch(self.graph)
			else:
				self.simulate()
		self.sim_time += self.frame_dt

	def render(self):
		if self.renderer is None:
			return

		with wp.ScopedTimer("render"):
			self.renderer.begin_frame(self.sim_time)
			self.renderer.render(self.state_0)
			self.renderer.end_frame()


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
	parser.add_argument(
		"--stage_path",
		type=lambda x: None if x == "None" else str(x),
		default="example_pendulum.usd",
		help="Path to the output USD file.",
	)
	parser.add_argument("--num_frames", type=int, default=20000, help="Total number of frames.")
	parser.add_argument("--num_envs", type=int, default=1, help="Total number of simulated environments.")

	args = parser.parse_known_args()[0]

	with wp.ScopedDevice(args.device):
		example = Example(stage_path=args.stage_path, num_envs=args.num_envs)

		for _ in range(args.num_frames):
			example.step()
			example.render()
			# wp.launch(
			# 	kernel=print_jointPos,
			# 	dim=example.state_0.joint_qd.shape[0],
			# 	inputs=(example.state_0.joint_qd,),
			# 	device=wp.get_device()
			# )
			# wp.launch(
			# 	kernel=print_bodyq,
			# 	dim=example.state_0.body_q.shape[0],
			# 	inputs=(example.state_0.body_q,),
			# 	device=wp.get_device()
			# )


		if example.renderer:
			example.renderer.save()
