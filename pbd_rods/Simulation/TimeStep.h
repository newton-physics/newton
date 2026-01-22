#ifndef __TimeStep_h__
#define __TimeStep_h__

#include "Common/Common.h"
#include "SimulationModel.h"
#include "CollisionDetection.h"
#include "ParameterObject.h"

namespace PBD
{
	/** \brief Base class for the simulation methods. 
	*/
	class TimeStep : public GenParam::ParameterObject
	{
	protected:
		CollisionDetection *m_collisionDetection;

		/** Clear accelerations and add gravitation.
		*/
		void clearAccelerations(SimulationModel &model);

		virtual void initParameters();

		static void contactCallbackFunction(const unsigned int contactType,
			const unsigned int bodyIndex1, const unsigned int bodyIndex2,
			const Vector3r &cp1, const Vector3r &cp2,
			const Vector3r &normal, const Real dist,
			const Real restitutionCoeff, const Real frictionCoeff, void *userData);

		static void solidContactCallbackFunction(const unsigned int contactType,
			const unsigned int bodyIndex1, const unsigned int bodyIndex2,
			const unsigned int tetIndex, const Vector3r &bary,
			const Vector3r &cp1, const Vector3r &cp2,
			const Vector3r &normal, const Real dist,
			const Real restitutionCoeff, const Real frictionCoeff, void *userData);

	public:
		TimeStep();
		virtual ~TimeStep(void);

		virtual void step(SimulationModel &model) = 0;
		virtual void reset();

		virtual void init();

		void setCollisionDetection(SimulationModel &model, CollisionDetection *cd);
		CollisionDetection *getCollisionDetection();
	};
}

#endif
