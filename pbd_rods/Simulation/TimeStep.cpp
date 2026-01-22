#include "TimeStep.h"
#include "TimeManager.h"
#include "Simulation.h"


using namespace PBD;
using namespace std;
using namespace GenParam;

TimeStep::TimeStep()
{
}

TimeStep::~TimeStep(void)
{
}

void TimeStep::init()
{
	initParameters();
}

void TimeStep::initParameters()
{
	ParameterObject::initParameters();
}

void TimeStep::clearAccelerations(SimulationModel &model)
{
	//////////////////////////////////////////////////////////////////////////
	// rigid body model
	//////////////////////////////////////////////////////////////////////////

	SimulationModel::RigidBodyVector &rb = model.getRigidBodies();
	Simulation *sim = Simulation::getCurrent();
	const Vector3r grav(sim->getVecValue<Real>(Simulation::GRAVITATION));
	for (size_t i = 0; i < rb.size(); i++)
	{
		// Clear accelerations of dynamic particles
		if (rb[i]->getMass() != 0.0)
		{
			Vector3r &a = rb[i]->getAcceleration();
			a = grav;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// particle model
	//////////////////////////////////////////////////////////////////////////

	ParticleData &pd = model.getParticles();
	const unsigned int count = pd.size();
	for (unsigned int i = 0; i < count; i++)
	{
		// Clear accelerations of dynamic particles
		if (pd.getMass(i) != 0.0)
		{
			Vector3r &a = pd.getAcceleration(i);
			a = grav;
		}
	}
}

void TimeStep::reset()
{
}

void TimeStep::setCollisionDetection(SimulationModel &model, CollisionDetection *cd)
{
	m_collisionDetection = cd;
	m_collisionDetection->setContactCallback(contactCallbackFunction, &model);
	m_collisionDetection->setSolidContactCallback(solidContactCallbackFunction, &model);
}

CollisionDetection *TimeStep::getCollisionDetection()
{
	return m_collisionDetection;
}

void TimeStep::contactCallbackFunction(const unsigned int contactType, const unsigned int bodyIndex1, const unsigned int bodyIndex2,
	const Vector3r &cp1, const Vector3r &cp2,
	const Vector3r &normal, const Real dist,
	const Real restitutionCoeff, const Real frictionCoeff, void *userData)
{
	SimulationModel *model = (SimulationModel*)userData;
	if (contactType == CollisionDetection::RigidBodyContactType)
		model->addRigidBodyContactConstraint(bodyIndex1, bodyIndex2, cp1, cp2, normal, dist, restitutionCoeff, frictionCoeff);
	else if (contactType == CollisionDetection::ParticleRigidBodyContactType)
		model->addParticleRigidBodyContactConstraint(bodyIndex1, bodyIndex2, cp1, cp2, normal, dist, restitutionCoeff, frictionCoeff);
}

void TimeStep::solidContactCallbackFunction(const unsigned int contactType, const unsigned int bodyIndex1, const unsigned int bodyIndex2,
	const unsigned int tetIndex, const Vector3r &bary,
	const Vector3r &cp1, const Vector3r &cp2,
	const Vector3r &normal, const Real dist,
	const Real restitutionCoeff, const Real frictionCoeff, void *userData)
{
	SimulationModel *model = (SimulationModel*)userData;
	if (contactType == CollisionDetection::ParticleSolidContactType)
		model->addParticleSolidContactConstraint(bodyIndex1, bodyIndex2, tetIndex, bary, cp1, cp2, normal, dist, restitutionCoeff, frictionCoeff);
}