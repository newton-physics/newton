#ifndef __RIGIDBODY_H__
#define __RIGIDBODY_H__

#include <vector>
#include "Common/Common.h"
#include "RigidBodyGeometry.h"
#include "Utils/VolumeIntegration.h"


namespace PBD
{
	/** This class encapsulates the state of a rigid body.
	 */
	class RigidBody
	{
		private:
			/** mass */
			Real m_mass;
			/** inverse mass */
			Real m_invMass;
			/** center of mass */
			Vector3r m_x;
			Vector3r m_lastX;
			Vector3r m_oldX;
			Vector3r m_x0;
			/** center of mass velocity */
			Vector3r m_v;
			Vector3r m_v0;
			/** acceleration (by external forces) */
			Vector3r m_a;

			/** Inertia tensor in the principal axis system: \n
			* After the main axis transformation the inertia tensor is a diagonal matrix.
			* So only three values are required to store the inertia tensor. These values
			* are constant over time.
			*/
			Vector3r m_inertiaTensor;
			/** 3x3 matrix, inertia tensor in world space */
			Matrix3r m_inertiaTensorW;
			/** Inverse inertia tensor in body space */
			Vector3r m_inertiaTensorInverse;
			/** 3x3 matrix, inverse of the inertia tensor in world space */
			Matrix3r m_inertiaTensorInverseW;
			/** Quaternion that describes the rotation of the body in world space */
			Quaternionr m_q;
			Quaternionr m_lastQ;
			Quaternionr m_oldQ;
			Quaternionr m_q0;
			/** Quaternion representing the rotation of the main axis transformation
			that is performed to get a diagonal inertia tensor */
			Quaternionr m_q_mat;
			/** Quaternion representing the initial rotation of the geometry */
			Quaternionr m_q_initial;
			/** difference of the initial translation and the translation of the main axis transformation */
			Vector3r m_x0_mat;
			/** rotationMatrix = 3x3 matrix. 
			* Important for the transformation from world in body space and vice versa.
			* When using quaternions the rotation matrix is computed out of the quaternion.
			*/
			Matrix3r m_rot;
			/** Angular velocity, defines rotation axis and velocity (magnitude of the vector) */
			Vector3r m_omega;
			Vector3r m_omega0;
			/** external torque */
			Vector3r m_torque;

			Real m_restitutionCoeff;
			Real m_frictionCoeff;

			RigidBodyGeometry m_geometry;

			// transformation required to transform a point to local space or vice vera
			Matrix3r m_transformation_R;
			Vector3r m_transformation_v1;
			Vector3r m_transformation_v2;
			Vector3r m_transformation_R_X_v1;
			
		public:
			RigidBody(void) 
			{
			}

			~RigidBody(void)
			{
			}

			void initBody(const Real mass, const Vector3r &x, 
				const Vector3r &inertiaTensor, const Quaternionr &rotation, 
				const VertexData &vertices, const Utilities::IndexedFaceMesh &mesh, 
				const Vector3r &scale = Vector3r(1.0, 1.0, 1.0))
			{
				setMass(mass);
				m_x = x; 
				m_x0 = x;
				m_lastX = x;
				m_oldX = x;
				m_v.setZero();
				m_v0.setZero();
				m_a.setZero();

				setInertiaTensor(inertiaTensor);
				m_q = rotation;
				m_q0 = rotation;
				m_lastQ = rotation;
				m_oldQ = rotation;
				m_rot = m_q.matrix();
				m_q_mat = Quaternionr(1.0, 0.0, 0.0, 0.0);
				m_q_initial = Quaternionr(1.0, 0.0, 0.0, 0.0);
				m_x0_mat.setZero();
				rotationUpdated();
				m_omega.setZero();
				m_omega0.setZero();
				m_torque.setZero();

				m_restitutionCoeff = static_cast<Real>(0.6);
				m_frictionCoeff = static_cast<Real>(0.2);

				getGeometry().initMesh(vertices.size(), mesh.numFaces(), &vertices.getPosition(0), mesh.getFaces().data(), mesh.getUVIndices(), mesh.getUVs(), scale, mesh.getFlatShading());
				getGeometry().updateMeshTransformation(getPosition(), getRotationMatrix());
			}

			void initBody(const Real density, const Vector3r &x, const Quaternionr &rotation,
				const VertexData &vertices, const Utilities::IndexedFaceMesh &mesh, const Vector3r &scale = Vector3r(1.0, 1.0, 1.0))
			{
				m_mass = 1.0;
				m_inertiaTensor = Vector3r(1.0, 1.0, 1.0);
				m_x = x;
				m_x0 = x;
				m_lastX = x;
				m_oldX = x;
				m_v.setZero();
				m_v0.setZero();
				m_a.setZero();

				m_q = rotation;
				m_q0 = rotation;
				m_lastQ = rotation;
				m_oldQ = rotation;
				m_rot = m_q.matrix();
				rotationUpdated();
				m_omega.setZero();
				m_omega0.setZero();
				m_torque.setZero();

				m_restitutionCoeff = static_cast<Real>(0.6);
				m_frictionCoeff = static_cast<Real>(0.2);

				getGeometry().initMesh(vertices.size(), mesh.numFaces(), &vertices.getPosition(0), mesh.getFaces().data(), mesh.getUVIndices(), mesh.getUVs(), scale, mesh.getFlatShading());
				determineMassProperties(density);
				getGeometry().updateMeshTransformation(getPosition(), getRotationMatrix());
			}

			void reset()
			{
				getPosition() = getPosition0();
				getOldPosition() = getPosition0();
				getLastPosition() = getPosition0();

				getRotation() = getRotation0();
				getOldRotation() = getRotation0();
				getLastRotation() = getRotation0();

				getVelocity() = getVelocity0();
				getAngularVelocity() = getAngularVelocity0();

				getAcceleration().setZero();
				getTorque().setZero();

				rotationUpdated();
			}

			void updateInverseTransformation()
			{
				// remove the rotation of the main axis transformation that is performed
				// to get a diagonal inertia tensor since the distance function is 
				// evaluated in local coordinates
				//
				// transformation world to local:
				// p_local = R_initial^T ( R_MAT R^T (p_world - x) - x_initial + x_MAT)
				// 
				// transformation local to world:
				// p_world = R R_MAT^T (R_initial p_local + x_initial - x_MAT) + x
				//
				m_transformation_R = (getRotationInitial().inverse() * getRotationMAT() * getRotation().inverse()).matrix();
				m_transformation_v1 = -getRotationInitial().inverse().matrix() * getPositionInitial_MAT();
				m_transformation_v2 = (getRotation()*getRotationMAT().inverse()).matrix() * getPositionInitial_MAT() + getPosition();
				m_transformation_R_X_v1 = -m_transformation_R * getPosition() + m_transformation_v1;
			}

			void rotationUpdated()
			{
				if (m_mass != 0.0)
				{
					m_rot = m_q.matrix();
					updateInertiaW();
					updateInverseTransformation();
				}
			}

			void updateInertiaW()
			{
				if (m_mass != 0.0)
				{
					m_inertiaTensorW = m_rot * m_inertiaTensor.asDiagonal() * m_rot.transpose();
					m_inertiaTensorInverseW = m_rot * m_inertiaTensorInverse.asDiagonal() * m_rot.transpose();
				}
			}

			/** Determine mass and inertia tensor of the given geometry.
			 */
			void determineMassProperties(const Real density)
			{
				// apply initial rotation
				VertexData &vd = m_geometry.getVertexDataLocal();
				
				Utilities::VolumeIntegration vi(m_geometry.getVertexDataLocal().size(), m_geometry.getMesh().numFaces(), &m_geometry.getVertexDataLocal().getPosition(0), m_geometry.getMesh().getFaces().data());
				vi.compute_inertia_tensor(density);

				// Diagonalize Inertia Tensor
				Eigen::SelfAdjointEigenSolver<Matrix3r> es(vi.getInertia());
				Vector3r inertiaTensor = es.eigenvalues();
				Matrix3r R = es.eigenvectors();

				setMass(vi.getMass());
				setInertiaTensor(inertiaTensor);

				if (R.determinant() < 0.0)
					R = -R;

				for (unsigned int i = 0; i < vd.size(); i++)
					vd.getPosition(i) = m_rot * vd.getPosition(i) + m_x0;

				Vector3r x_MAT = vi.getCenterOfMass();
				R = m_rot * R;
				x_MAT = m_rot * x_MAT + m_x0;

				// rotate vertices back				
				for (unsigned int i = 0; i < vd.size(); i++)
					vd.getPosition(i) = R.transpose() * (vd.getPosition(i) - x_MAT);

				// set rotation
				Quaternionr qR = Quaternionr(R);
				qR.normalize();
				m_q_mat = qR;
				m_q_initial = m_q0;
				m_x0_mat = m_x0 - x_MAT;

				m_q0 = qR;
				m_q = m_q0;
				m_lastQ = m_q0;
				m_oldQ = m_q0;
				rotationUpdated();

				// set translation
				m_x0 = x_MAT;
				m_x = m_x0;
				m_lastX = m_x0;
				m_oldX = m_x0;
				updateInverseTransformation();
			}

			const Matrix3r &getTransformationR() { return m_transformation_R;  }
			const Vector3r &getTransformationV1() { return m_transformation_v1; }
			const Vector3r &getTransformationV2() { return m_transformation_v2; }
			const Vector3r &getTransformationRXV1() { return m_transformation_R_X_v1; }

			FORCE_INLINE Real &getMass()
			{
				return m_mass;
			}

			FORCE_INLINE const Real &getMass() const
			{
				return m_mass;
			}

			FORCE_INLINE void setMass(const Real &value)
			{
				m_mass = value;
				if (m_mass != 0.0)
					m_invMass = static_cast<Real>(1.0) / m_mass;
				else
					m_invMass = 0.0;
			}

			FORCE_INLINE const Real &getInvMass() const
			{
				return m_invMass;
			}

			FORCE_INLINE Vector3r &getPosition()
			{
				return m_x;
			}

			FORCE_INLINE const Vector3r &getPosition() const 
			{
				return m_x;
			}

			FORCE_INLINE void setPosition(const Vector3r &pos)
			{
				m_x = pos;
			}

			FORCE_INLINE Vector3r &getLastPosition()
			{
				return m_lastX;
			}

			FORCE_INLINE const Vector3r &getLastPosition() const
			{
				return m_lastX;
			}

			FORCE_INLINE void setLastPosition(const Vector3r &pos)
			{
				m_lastX = pos;
			}

			FORCE_INLINE Vector3r &getOldPosition()
			{
				return m_oldX;
			}

			FORCE_INLINE const Vector3r &getOldPosition() const
			{
				return m_oldX;
			}

			FORCE_INLINE void setOldPosition(const Vector3r &pos)
			{
				m_oldX = pos;
			}

			FORCE_INLINE Vector3r &getPosition0()
			{
				return m_x0;
			}

			FORCE_INLINE const Vector3r &getPosition0() const
			{
				return m_x0;
			}

			FORCE_INLINE void setPosition0(const Vector3r &pos)
			{
				m_x0 = pos;
			}

			FORCE_INLINE Vector3r &getPositionInitial_MAT()
			{
				return m_x0_mat;
			}

			FORCE_INLINE const Vector3r &getPositionInitial_MAT() const
			{
				return m_x0_mat;
			}

			FORCE_INLINE void setPositionInitial_MAT(const Vector3r &pos)
			{
				m_x0_mat = pos;
			}

			FORCE_INLINE Vector3r &getVelocity()
			{
				return m_v;
			}

			FORCE_INLINE const Vector3r &getVelocity() const
			{
				return m_v;
			}

			FORCE_INLINE void setVelocity(const Vector3r &value)
			{
				m_v = value;
			}			

			FORCE_INLINE Vector3r &getVelocity0()
			{
				return m_v0;
			}

			FORCE_INLINE const Vector3r &getVelocity0() const
			{
				return m_v0;
			}

			FORCE_INLINE void setVelocity0(const Vector3r &value)
			{
				m_v0 = value;
			}

			FORCE_INLINE Vector3r &getAcceleration()
			{
				return m_a;
			}

			FORCE_INLINE const Vector3r &getAcceleration() const 
			{
				return m_a;
			}

			FORCE_INLINE void setAcceleration(const Vector3r &accel)
			{
				m_a = accel;
			}

			FORCE_INLINE const Vector3r &getInertiaTensor() const
			{
				return m_inertiaTensor;
			}

			FORCE_INLINE void setInertiaTensor(const Vector3r &value)
			{
				m_inertiaTensor = value;
				m_inertiaTensorInverse = Vector3r(static_cast<Real>(1.0) / value[0], static_cast<Real>(1.0) / value[1], static_cast<Real>(1.0) / value[2]);
			}

			FORCE_INLINE Matrix3r& getInertiaTensorW()
			{
				return m_inertiaTensorW;
			}

			FORCE_INLINE const Matrix3r& getInertiaTensorW() const
			{
				return m_inertiaTensorW;
			}

			FORCE_INLINE const Vector3r &getInertiaTensorInverse() const
			{
				return m_inertiaTensorInverse;
			}

			FORCE_INLINE Matrix3r &getInertiaTensorInverseW()
			{
				return m_inertiaTensorInverseW;
			}

			FORCE_INLINE const Matrix3r &getInertiaTensorInverseW() const
			{
				return m_inertiaTensorInverseW;
			}

			FORCE_INLINE void setInertiaTensorInverseW(const Matrix3r &value)
			{
				m_inertiaTensorInverseW = value;
			}

			FORCE_INLINE Quaternionr &getRotation()
			{
				return m_q;
			}

			FORCE_INLINE const Quaternionr &getRotation() const
			{
				return m_q;
			}

			FORCE_INLINE void setRotation(const Quaternionr &value)
			{
				m_q = value;
			}

			FORCE_INLINE Quaternionr &getLastRotation()
			{
				return m_lastQ;
			}

			FORCE_INLINE const Quaternionr &getLastRotation() const
			{
				return m_lastQ;
			}

			FORCE_INLINE void setLastRotation(const Quaternionr &value)
			{
				m_lastQ = value;
			}

			FORCE_INLINE Quaternionr &getOldRotation()
			{
				return m_oldQ;
			}

			FORCE_INLINE const Quaternionr &getOldRotation() const
			{
				return m_oldQ;
			}

			FORCE_INLINE void setOldRotation(const Quaternionr &value)
			{
				m_oldQ = value;
			}

			FORCE_INLINE Quaternionr &getRotation0()
			{
				return m_q0;
			}

			FORCE_INLINE const Quaternionr &getRotation0() const
			{
				return m_q0;
			}

			FORCE_INLINE void setRotation0(const Quaternionr &value)
			{
				m_q0 = value;
			}

			FORCE_INLINE Quaternionr &getRotationMAT()
			{
				return m_q_mat;
			}

			FORCE_INLINE const Quaternionr &getRotationMAT() const
			{
				return m_q_mat;
			}

			FORCE_INLINE void setRotationMAT(const Quaternionr &value)
			{
				m_q_mat = value;
			}

			FORCE_INLINE Quaternionr &getRotationInitial()
			{
				return m_q_initial;
			}

			FORCE_INLINE const Quaternionr &getRotationInitial() const
			{
				return m_q_initial;
			}

			FORCE_INLINE void setRotationInitial(const Quaternionr &value)
			{
				m_q_initial = value;
			}

			FORCE_INLINE Matrix3r &getRotationMatrix()
			{
				return m_rot;
			}

			FORCE_INLINE const Matrix3r &getRotationMatrix() const
			{
				return m_rot;
			}

			FORCE_INLINE void setRotationMatrix(const Matrix3r &value)
			{
				m_rot = value;
			}

			FORCE_INLINE Vector3r &getAngularVelocity()
			{
				return m_omega;
			}

			FORCE_INLINE const Vector3r &getAngularVelocity() const
			{
				return m_omega;
			}

			FORCE_INLINE void setAngularVelocity(const Vector3r &value)
			{
				m_omega = value;
			}

			FORCE_INLINE Vector3r &getAngularVelocity0()
			{
				return m_omega0;
			}

			FORCE_INLINE const Vector3r &getAngularVelocity0() const
			{
				return m_omega0;
			}

			FORCE_INLINE void setAngularVelocity0(const Vector3r &value)
			{
				m_omega0 = value;
			}

			FORCE_INLINE Vector3r &getTorque()
			{
				return m_torque;
			}

			FORCE_INLINE const Vector3r &getTorque() const
			{
				return m_torque;
			}

			FORCE_INLINE void setTorque(const Vector3r &value)
			{
				m_torque = value;
			}

			FORCE_INLINE Real getRestitutionCoeff() const 
			{ 
				return m_restitutionCoeff; 
			}

			FORCE_INLINE void setRestitutionCoeff(Real val) 
			{ 
				m_restitutionCoeff = val; 
			}

			FORCE_INLINE Real getFrictionCoeff() const 
			{ 
				return m_frictionCoeff; 
			}

			FORCE_INLINE void setFrictionCoeff(Real val) 
			{ 
				m_frictionCoeff = val; 
			}

			RigidBodyGeometry& getGeometry()
			{
				return m_geometry;
			}
	};
}

#endif
