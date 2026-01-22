#ifndef __PARTICLEDATA_H__
#define __PARTICLEDATA_H__

#include <vector>
#include "Common/Common.h"


namespace PBD
{
	/** This class encapsulates the state of all vertices.
	* All parameters are stored in individual arrays.
	*/
	class VertexData
	{
	private:
		std::vector<Vector3r> m_x;

	public:
		FORCE_INLINE VertexData(void) :
			m_x()
		{
		}

		FORCE_INLINE ~VertexData(void)
		{
			m_x.clear();
		}

		FORCE_INLINE void addVertex(const Vector3r &vertex)
		{
			m_x.push_back(vertex);
		}

		FORCE_INLINE Vector3r &getPosition(const unsigned int i)
		{
			return m_x[i];
		}

		FORCE_INLINE const Vector3r &getPosition(const unsigned int i) const
		{
			return m_x[i];
		}

		FORCE_INLINE void setPosition(const unsigned int i, const Vector3r &pos)
		{
			m_x[i] = pos;
		}

		/** Resize the array containing the particle data.
		*/
		FORCE_INLINE void resize(const unsigned int newSize)
		{
			m_x.resize(newSize);
		}

		/** Reserve the array containing the particle data.
		*/
		FORCE_INLINE void reserve(const unsigned int newSize)
		{
			m_x.reserve(newSize);
		}

		/** Release the array containing the particle data.
		*/
		FORCE_INLINE void release()
		{
			m_x.clear();
		}

		/** Release the array containing the particle data.
		*/
		FORCE_INLINE unsigned int size() const
		{
			return (unsigned int)m_x.size();
		}

		FORCE_INLINE const std::vector<Vector3r>& getVertices() const
		{
			return m_x;
		}
	};

	/** This class encapsulates the state of all particles of a particle model.
	 * All parameters are stored in individual arrays.
	 */
	class ParticleData
	{
		private:
			// Mass
			// If the mass is zero, the particle is static
			std::vector<Real> m_masses;
			std::vector<Real> m_invMasses;

			// Dynamic state
			std::vector<Vector3r> m_x0;
			std::vector<Vector3r> m_x;
			std::vector<Vector3r> m_v;
			std::vector<Vector3r> m_a;
			std::vector<Vector3r> m_oldX;
			std::vector<Vector3r> m_lastX;

		public:
			FORCE_INLINE ParticleData(void)	:
				  m_masses(),
				  m_invMasses(),
				  m_x0(),
				  m_x(),
				  m_v(),
				  m_a(),
				  m_oldX(),
				  m_lastX()
			{
			}

			FORCE_INLINE ~ParticleData(void) 
			{
				m_masses.clear();
				m_invMasses.clear();
				m_x0.clear();
				m_x.clear();
				m_v.clear();
				m_a.clear();
				m_oldX.clear();
				m_lastX.clear();
			}

			FORCE_INLINE void addVertex(const Vector3r &vertex)
			{
				m_x0.push_back(vertex);
				m_x.push_back(vertex);
				m_oldX.push_back(vertex);
				m_lastX.push_back(vertex);
				m_masses.push_back(1.0);
				m_invMasses.push_back(1.0);
				m_v.push_back(Vector3r(0.0, 0.0, 0.0));
				m_a.push_back(Vector3r(0.0, 0.0, 0.0));
			}

			FORCE_INLINE Vector3r &getPosition(const unsigned int i)
			{
				return m_x[i];
			}

			FORCE_INLINE const Vector3r &getPosition(const unsigned int i) const 
			{
				return m_x[i];
			}

			FORCE_INLINE void setPosition(const unsigned int i, const Vector3r &pos)
			{
				m_x[i] = pos;
			}

			FORCE_INLINE Vector3r &getPosition0(const unsigned int i)
			{
				return m_x0[i];
			}

			FORCE_INLINE const Vector3r &getPosition0(const unsigned int i) const
			{
				return m_x0[i];
			}

			FORCE_INLINE void setPosition0(const unsigned int i, const Vector3r &pos)
			{
				m_x0[i] = pos;
			}

			FORCE_INLINE Vector3r &getLastPosition(const unsigned int i)
			{
				return m_lastX[i];
			}

			FORCE_INLINE const Vector3r &getLastPosition(const unsigned int i) const
			{
				return m_lastX[i];
			}

			FORCE_INLINE void setLastPosition(const unsigned int i, const Vector3r &pos)
			{
				m_lastX[i] = pos;
			}

			FORCE_INLINE Vector3r &getOldPosition(const unsigned int i)
			{
				return m_oldX[i];
			}

			FORCE_INLINE const Vector3r &getOldPosition(const unsigned int i) const
			{
				return m_oldX[i];
			}

			FORCE_INLINE void setOldPosition(const unsigned int i, const Vector3r &pos)
			{
				m_oldX[i] = pos;
			}
			
			FORCE_INLINE Vector3r &getVelocity(const unsigned int i)
			{
				return m_v[i];
			}

			FORCE_INLINE const Vector3r &getVelocity(const unsigned int i) const 
			{
				return m_v[i];
			}

			FORCE_INLINE void setVelocity(const unsigned int i, const Vector3r &vel)
			{
				m_v[i] = vel;
			}

			FORCE_INLINE Vector3r &getAcceleration(const unsigned int i)
			{
				return m_a[i];
			}

			FORCE_INLINE const Vector3r &getAcceleration(const unsigned int i) const 
			{
				return m_a[i];
			}

			FORCE_INLINE void setAcceleration(const unsigned int i, const Vector3r &accel)
			{
				m_a[i] = accel;
			}

			FORCE_INLINE const Real getMass(const unsigned int i) const
			{
				return m_masses[i];
			}

			FORCE_INLINE Real& getMass(const unsigned int i)
			{
				return m_masses[i];
			}

			FORCE_INLINE void setMass(const unsigned int i, const Real mass)
			{
				m_masses[i] = mass;
				if (mass != 0.0)
					m_invMasses[i] = static_cast<Real>(1.0) / mass;
				else
					m_invMasses[i] = 0.0;
			}

			FORCE_INLINE const Real getInvMass(const unsigned int i) const
			{
				return m_invMasses[i];
			}

			FORCE_INLINE const unsigned int getNumberOfParticles() const
			{
				return (unsigned int) m_x.size();
			}

			FORCE_INLINE const std::vector<Vector3r>& getVertices() const
			{
				return m_x;
			}

			/** Resize the array containing the particle data.
			 */
			FORCE_INLINE void resize(const unsigned int newSize)
			{
				m_masses.resize(newSize);
				m_invMasses.resize(newSize);
				m_x0.resize(newSize);
				m_x.resize(newSize);
				m_v.resize(newSize);
				m_a.resize(newSize);
				m_oldX.resize(newSize);
				m_lastX.resize(newSize);
			}

			/** Reserve the array containing the particle data.
			 */
			FORCE_INLINE void reserve(const unsigned int newSize)
			{
				m_masses.reserve(newSize);
				m_invMasses.reserve(newSize);
				m_x0.reserve(newSize);
				m_x.reserve(newSize);
				m_v.reserve(newSize);
				m_a.reserve(newSize);
				m_oldX.reserve(newSize);
				m_lastX.reserve(newSize);
			}

			/** Release the array containing the particle data.
			 */
			FORCE_INLINE void release()
			{
				m_masses.clear();
				m_invMasses.clear();
				m_x0.clear();
				m_x.clear();
				m_v.clear();
				m_a.clear();
				m_oldX.clear();
				m_lastX.clear();
			}

			/** Release the array containing the particle data.
			 */
			FORCE_INLINE unsigned int size() const 
			{
				return (unsigned int) m_x.size();
			}
	};

	/** This class encapsulates the state of all orientations of a quaternion model.
	* All parameters are stored in individual arrays.
	*/
	class OrientationData
	{
	private:
		// Mass
		// If the mass is zero, the particle is static
		std::vector<Real> m_masses;
		std::vector<Real> m_invMasses;

		// Dynamic state
		std::vector<Quaternionr> m_q0;
		std::vector<Quaternionr> m_q;
		std::vector<Vector3r> m_omega;
		std::vector<Vector3r> m_alpha;
		std::vector<Quaternionr> m_oldQ;
		std::vector<Quaternionr> m_lastQ;

	public:
		FORCE_INLINE OrientationData(void) :
			m_masses(),
			m_invMasses(),
			m_q0(),
			m_q(),
			m_omega(),
			m_alpha(),
			m_oldQ(),
			m_lastQ()
		{
		}

		FORCE_INLINE ~OrientationData(void)
		{
			m_masses.clear();
			m_invMasses.clear();
			m_q0.clear();
			m_q.clear();
			m_omega.clear();
			m_alpha.clear();
			m_oldQ.clear();
			m_lastQ.clear();
		}

		FORCE_INLINE void addQuaternion(const Quaternionr &vertex)
		{
			m_q0.push_back(vertex);
			m_q.push_back(vertex);
			m_oldQ.push_back(vertex);
			m_lastQ.push_back(vertex);
			m_masses.push_back(1.0);
			m_invMasses.push_back(1.0);
			m_omega.push_back(Vector3r(0.0, 0.0, 0.0));
			m_alpha.push_back(Vector3r(0.0, 0.0, 0.0));
		}

		FORCE_INLINE Quaternionr &getQuaternion(const unsigned int i)
		{
			return m_q[i];
		}

		FORCE_INLINE const Quaternionr &getQuaternion(const unsigned int i) const
		{
			return m_q[i];
		}

		FORCE_INLINE void setQuaternion(const unsigned int i, const Quaternionr &pos)
		{
			m_q[i] = pos;
		}

		FORCE_INLINE Quaternionr &getQuaternion0(const unsigned int i)
		{
			return m_q0[i];
		}

		FORCE_INLINE const Quaternionr &getQuaternion0(const unsigned int i) const
		{
			return m_q0[i];
		}

		FORCE_INLINE void setQuaternion0(const unsigned int i, const Quaternionr &pos)
		{
			m_q0[i] = pos;
		}

		FORCE_INLINE Quaternionr &getLastQuaternion(const unsigned int i)
		{
			return m_lastQ[i];
		}

		FORCE_INLINE const Quaternionr &getLastQuaternion(const unsigned int i) const
		{
			return m_lastQ[i];
		}

		FORCE_INLINE void setLastQuaternion(const unsigned int i, const Quaternionr &pos)
		{
			m_lastQ[i] = pos;
		}

		FORCE_INLINE Quaternionr &getOldQuaternion(const unsigned int i)
		{
			return m_oldQ[i];
		}

		FORCE_INLINE const Quaternionr &getOldQuaternion(const unsigned int i) const
		{
			return m_oldQ[i];
		}

		FORCE_INLINE void setOldQuaternion(const unsigned int i, const Quaternionr &pos)
		{
			m_oldQ[i] = pos;
		}

		FORCE_INLINE Vector3r &getVelocity(const unsigned int i)
		{
			return m_omega[i];
		}

		FORCE_INLINE const Vector3r &getVelocity(const unsigned int i) const
		{
			return m_omega[i];
		}

		FORCE_INLINE void setVelocity(const unsigned int i, const Vector3r &vel)
		{
			m_omega[i] = vel;
		}

		FORCE_INLINE Vector3r &getAcceleration(const unsigned int i)
		{
			return m_alpha[i];
		}

		FORCE_INLINE const Vector3r &getAcceleration(const unsigned int i) const
		{
			return m_alpha[i];
		}

		FORCE_INLINE void setAcceleration(const unsigned int i, const Vector3r &accel)
		{
			m_alpha[i] = accel;
		}

		FORCE_INLINE const Real getMass(const unsigned int i) const
		{
			return m_masses[i];
		}

		FORCE_INLINE Real& getMass(const unsigned int i)
		{
			return m_masses[i];
		}

		FORCE_INLINE void setMass(const unsigned int i, const Real mass)
		{
			m_masses[i] = mass;
			if (mass != 0.0)
				m_invMasses[i] = static_cast<Real>(1.0) / mass;
			else
				m_invMasses[i] = 0.0;
		}

		FORCE_INLINE const Real getInvMass(const unsigned int i) const
		{
			return m_invMasses[i];
		}

		FORCE_INLINE const unsigned int getNumberOfQuaternions() const
		{
			return (unsigned int)m_q.size();
		}

		/** Resize the array containing the particle data.
		*/
		FORCE_INLINE void resize(const unsigned int newSize)
		{
			m_masses.resize(newSize);
			m_invMasses.resize(newSize);
			m_q0.resize(newSize);
			m_q.resize(newSize);
			m_omega.resize(newSize);
			m_alpha.resize(newSize);
			m_oldQ.resize(newSize);
			m_lastQ.resize(newSize);
		}

		/** Reserve the array containing the particle data.
		*/
		FORCE_INLINE void reserve(const unsigned int newSize)
		{
			m_masses.reserve(newSize);
			m_invMasses.reserve(newSize);
			m_q0.reserve(newSize);
			m_q.reserve(newSize);
			m_omega.reserve(newSize);
			m_alpha.reserve(newSize);
			m_oldQ.reserve(newSize);
			m_lastQ.reserve(newSize);
		}

		/** Release the array containing the particle data.
		*/
		FORCE_INLINE void release()
		{
			m_masses.clear();
			m_invMasses.clear();
			m_q0.clear();
			m_q.clear();
			m_omega.clear();
			m_alpha.clear();
			m_oldQ.clear();
			m_lastQ.clear();
		}

		/** Release the array containing the particle data.
		*/
		FORCE_INLINE unsigned int size() const
		{
			return (unsigned int)m_q.size();
		}
	};
}

#endif
