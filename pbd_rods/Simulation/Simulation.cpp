#include "Simulation.h"
#include "TimeManager.h"
#include "Utils/Timing.h"
#include "TimeStep.h"
#include "TimeStepController.h"

using namespace PBD;
using namespace std;
using namespace GenParam;

Simulation* Simulation::current = nullptr;
int Simulation::GRAVITATION = -1;

Simulation::Simulation () 
{
	m_gravitation = Vector3r(0.0, -9.81, 0.0);

	m_timeStep = nullptr;
	m_model = nullptr;
}

Simulation::~Simulation () 
{
	delete m_timeStep;
	delete TimeManager::getCurrent();

	current = nullptr;
}

Simulation* Simulation::getCurrent ()
{
	if (current == nullptr)
	{
		current = new Simulation ();
		current->init();
	}
	return current;
}

void Simulation::setCurrent (Simulation* tm)
{
	current = tm;
}

bool Simulation::hasCurrent()
{
	return (current != nullptr);
}

void Simulation::init()
{
	initParameters();
	
	m_timeStep = new TimeStepController();
	m_timeStep->init();
	TimeManager::getCurrent()->setTimeStepSize(static_cast<Real>(0.005));
}

void Simulation::initParameters()
{
	ParameterObject::initParameters();

 	GRAVITATION = createVectorParameter("gravitation", "Gravitation", 3u, m_gravitation.data());
 	setGroup(GRAVITATION, "Simulation|General");
 	setDescription(GRAVITATION, "Vector to define the gravitational acceleration.");
}

void Simulation::reset()
{
	m_model->reset();
	if (m_timeStep)
		m_timeStep->reset();

	TimeManager::getCurrent()->setTime(static_cast<Real>(0.0));
}

