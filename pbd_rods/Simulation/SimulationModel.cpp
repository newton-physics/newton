#include "SimulationModel.h"
#include "PositionBasedDynamics/PositionBasedRigidBodyDynamics.h"
#include "Constraints.h"

using namespace PBD;
using namespace GenParam;

int SimulationModel::CLOTH_SIMULATION_METHOD = -1;
int SimulationModel::ENUM_CLOTHSIM_NONE = -1;
int SimulationModel::ENUM_CLOTHSIM_DISTANCE_CONSTRAINTS = -1;
int SimulationModel::ENUM_CLOTHSIM_FEM_PBD = -1;
int SimulationModel::ENUM_CLOTHSIM_SBD = -1;
int SimulationModel::ENUM_CLOTHSIM_DISTANCE_CONSTRAINTS_XPBD = -1;
int SimulationModel::CLOTH_BENDING_METHOD = -1;
int SimulationModel::ENUM_CLOTH_BENDING_NONE = -1;
int SimulationModel::ENUM_CLOTH_BENDING_DIHEDRAL_ANGLE = -1;
int SimulationModel::ENUM_CLOTH_BENDING_ISOMETRIC = -1;
int SimulationModel::ENUM_CLOTH_BENDING_ISOMETRIX_XPBD = -1;
int SimulationModel::CLOTH_STIFFNESS = -1;
int SimulationModel::CLOTH_STIFFNESS_XX = -1;
int SimulationModel::CLOTH_STIFFNESS_YY = -1;
int SimulationModel::CLOTH_STIFFNESS_XY = -1;
int SimulationModel::CLOTH_POISSON_RATIO_XY = -1;
int SimulationModel::CLOTH_POISSON_RATIO_YX = -1;
int SimulationModel::CLOTH_BENDING_STIFFNESS = -1;
int SimulationModel::CLOTH_NORMALIZE_STRETCH = -1;
int SimulationModel::CLOTH_NORMALIZE_SHEAR = -1;

int SimulationModel::SOLID_SIMULATION_METHOD = -1;
int SimulationModel::ENUM_SOLIDSIM_NONE = -1;
int SimulationModel::ENUM_SOLIDSIM_DISTANCE_CONSTRAINTS = -1;
int SimulationModel::ENUM_SOLIDSIM_FEM_PBD = -1;
int SimulationModel::ENUM_SOLIDSIM_FEM_XPBD = -1;
int SimulationModel::ENUM_SOLIDSIM_SBD = -1;
int SimulationModel::ENUM_SOLIDSIM_SHAPE_MATCHING = -1;
int SimulationModel::ENUM_SOLIDSIM_DISTANCE_CONSTRAINTS_XPBD = -1;
int SimulationModel::SOLID_STIFFNESS = -1;
int SimulationModel::SOLID_POISSON_RATIO = -1;
int SimulationModel::SOLID_VOLUME_STIFFNESS = -1;
int SimulationModel::SOLID_NORMALIZE_STRETCH = -1;
int SimulationModel::SOLID_NORMALIZE_SHEAR = -1;

int SimulationModel::ROD_STRETCHING_STIFFNESS = -1;
int SimulationModel::ROD_SHEARING_STIFFNESS_X = -1;
int SimulationModel::ROD_SHEARING_STIFFNESS_Y = -1;
int SimulationModel::ROD_BENDING_STIFFNESS_X = -1;
int SimulationModel::ROD_BENDING_STIFFNESS_Y = -1;
int SimulationModel::ROD_TWISTING_STIFFNESS = -1;

int SimulationModel::CONTACT_STIFFNESS_RB = -1;
int SimulationModel::CONTACT_STIFFNESS_PARTICLE_RB = -1;


SimulationModel::SimulationModel()
{
	m_contactStiffnessRigidBody = 1.0;
	m_contactStiffnessParticleRigidBody = 100.0;

	m_clothSimulationMethod = 2;
	m_clothBendingMethod = 2;
	m_cloth_stiffness = static_cast<Real>(1.0);
	m_cloth_bendingStiffness = static_cast<Real>(0.01);
	m_cloth_xxStiffness = static_cast<Real>(1.0);
	m_cloth_yyStiffness = static_cast<Real>(1.0);
	m_cloth_xyStiffness = static_cast<Real>(1.0);
	m_cloth_xyPoissonRatio = static_cast<Real>(0.3);
	m_cloth_yxPoissonRatio = static_cast<Real>(0.3);
	m_cloth_normalizeShear = false;
	m_cloth_normalizeStretch = false;

	m_solidSimulationMethod = 2;
	m_solid_stiffness = static_cast<Real>(1.0);
	m_solid_poissonRatio = static_cast<Real>(0.3);
	m_solid_volumeStiffness = static_cast<Real>(1.0);
	m_solid_normalizeShear = false;
	m_solid_normalizeStretch = false;

	m_rod_stretchingStiffness = static_cast<Real>(1.0);
	m_rod_shearingStiffnessX = static_cast<Real>(1.0);
	m_rod_shearingStiffnessY = static_cast<Real>(1.0);
	m_rod_bendingStiffnessX = static_cast<Real>(0.5);
	m_rod_bendingStiffnessY = static_cast<Real>(0.5);
	m_rod_twistingStiffness = static_cast<Real>(0.5);

	m_groupsInitialized = false;

	m_rigidBodyContactConstraints.reserve(10000);
	m_particleRigidBodyContactConstraints.reserve(10000);
	m_particleSolidContactConstraints.reserve(10000);

	m_clothSimMethodChanged = nullptr;
	m_solidSimMethodChanged = nullptr;
}

SimulationModel::~SimulationModel(void)
{
	cleanup();
}

void SimulationModel::init()
{
	initParameters();
}

void SimulationModel::cleanup()
{
	resetContacts();
	for (unsigned int i = 0; i < m_rigidBodies.size(); i++)
		delete m_rigidBodies[i];
	m_rigidBodies.clear();
	for (unsigned int i = 0; i < m_triangleModels.size(); i++)
		delete m_triangleModels[i];
	m_triangleModels.clear();
	for (unsigned int i = 0; i < m_tetModels.size(); i++)
		delete m_tetModels[i];
	m_tetModels.clear();
	for (unsigned int i = 0; i < m_lineModels.size(); i++)
		delete m_lineModels[i];
	m_lineModels.clear();
	for (unsigned int i = 0; i < m_constraints.size(); i++)
		delete m_constraints[i];
	m_constraints.clear();
	m_particles.release();
	m_orientations.release();
	m_groupsInitialized = false;
}

void SimulationModel::initParameters()
{
	ParameterObject::initParameters();

	CLOTH_SIMULATION_METHOD = createEnumParameter("clothSimulationMethod", "Simulation method", std::bind(&SimulationModel::getClothSimulationMethod, this), std::bind(static_cast<void (SimulationModel::*)(const int)>(&SimulationModel::setClothSimulationMethod), this, std::placeholders::_1));
	setGroup(CLOTH_SIMULATION_METHOD, "Cloth|General");
	setDescription(CLOTH_SIMULATION_METHOD, "Cloth simulation method");
	EnumParameter* enumParam = static_cast<EnumParameter*>(getParameter(CLOTH_SIMULATION_METHOD));
	enumParam->addEnumValue("None", ENUM_CLOTHSIM_NONE);
	enumParam->addEnumValue("Distance constraints", ENUM_CLOTHSIM_DISTANCE_CONSTRAINTS);
	enumParam->addEnumValue("FEM based PBD", ENUM_CLOTHSIM_FEM_PBD);
	enumParam->addEnumValue("Strain based dynamics", ENUM_CLOTHSIM_SBD);
	enumParam->addEnumValue("XPBD distance constraints", ENUM_CLOTHSIM_DISTANCE_CONSTRAINTS_XPBD);

	CLOTH_BENDING_METHOD = createEnumParameter("clothBendingMethod", "Bending method", std::bind(&SimulationModel::getClothBendingMethod, this), std::bind(static_cast<void (SimulationModel::*)(const int)>(&SimulationModel::setClothBendingMethod), this, std::placeholders::_1));
	setGroup(CLOTH_BENDING_METHOD, "Cloth|General");
	setDescription(CLOTH_BENDING_METHOD, "Cloth bending method");
	enumParam = static_cast<EnumParameter*>(getParameter(CLOTH_BENDING_METHOD));
	enumParam->addEnumValue("None", ENUM_CLOTH_BENDING_NONE);
	enumParam->addEnumValue("Dihedral angle", ENUM_CLOTH_BENDING_DIHEDRAL_ANGLE);
	enumParam->addEnumValue("Isometric bending", ENUM_CLOTH_BENDING_ISOMETRIC);
	enumParam->addEnumValue("Isometric bending (XPBD)", ENUM_CLOTH_BENDING_ISOMETRIX_XPBD);

	CLOTH_STIFFNESS = createNumericParameter<Real>("cloth_stiffness", "Distance constraint stiffness", std::bind(&SimulationModel::getClothStiffness, this), std::bind(static_cast<void (SimulationModel::*)(const Real)>(&SimulationModel::setClothStiffness), this, std::placeholders::_1));
	setGroup(CLOTH_STIFFNESS, "Cloth|Distance constraints");
	setDescription(CLOTH_STIFFNESS, "Distance constraint stiffness");
	static_cast<NumericParameter<Real>*>(getParameter(CLOTH_STIFFNESS))->setMinValue(0.0);

	CLOTH_STIFFNESS_XX = createNumericParameter<Real>("cloth_xxStiffness", "Youngs modulus XX", std::bind(&SimulationModel::getClothStiffnessXX, this), std::bind(static_cast<void (SimulationModel::*)(const Real)>(&SimulationModel::setClothStiffnessXX), this, std::placeholders::_1));
	setGroup(CLOTH_STIFFNESS_XX, "Cloth|FEM");
	setDescription(CLOTH_STIFFNESS_XX, "XX stiffness of orthotropic cloth models.");
	static_cast<NumericParameter<Real>*>(getParameter(CLOTH_STIFFNESS_XX))->setMinValue(0.0);

	CLOTH_STIFFNESS_YY = createNumericParameter<Real>("cloth_yyStiffness", "Youngs modulus YY", std::bind(&SimulationModel::getClothStiffnessYY, this), std::bind(static_cast<void (SimulationModel::*)(const Real)>(&SimulationModel::setClothStiffnessYY), this, std::placeholders::_1));
	setGroup(CLOTH_STIFFNESS_YY, "Cloth|FEM");
	setDescription(CLOTH_STIFFNESS_YY, "YY stiffness of orthotropic cloth models.");
	static_cast<NumericParameter<Real>*>(getParameter(CLOTH_STIFFNESS_YY))->setMinValue(0.0);

	CLOTH_STIFFNESS_XY = createNumericParameter<Real>("cloth_xyStiffness", "Youngs modulus XY", std::bind(&SimulationModel::getClothStiffnessXY, this), std::bind(static_cast<void (SimulationModel::*)(const Real)>(&SimulationModel::setClothStiffnessXY), this, std::placeholders::_1));
	setGroup(CLOTH_STIFFNESS_XY, "Cloth|FEM");
	setDescription(CLOTH_STIFFNESS_XY, "XY stiffness of orthotropic cloth models.");
	static_cast<NumericParameter<Real>*>(getParameter(CLOTH_STIFFNESS_XY))->setMinValue(0.0);

	CLOTH_POISSON_RATIO_XY = createNumericParameter<Real>("cloth_xyPoissonRatio", "Poisson ratio XY", std::bind(&SimulationModel::getClothPoissonRatioXY, this), std::bind(static_cast<void (SimulationModel::*)(const Real)>(&SimulationModel::setClothPoissonRatioXY), this, std::placeholders::_1));
	setGroup(CLOTH_POISSON_RATIO_XY, "Cloth|FEM");
	setDescription(CLOTH_POISSON_RATIO_XY, "XY Poisson ratio of orthotropic cloth models.");
	static_cast<NumericParameter<Real>*>(getParameter(CLOTH_POISSON_RATIO_XY))->setMinValue(0.0);

	CLOTH_POISSON_RATIO_YX = createNumericParameter<Real>("cloth_yxPoissonRatio", "Poisson ratio YX", std::bind(&SimulationModel::getClothPoissonRatioYX, this), std::bind(static_cast<void (SimulationModel::*)(const Real)>(&SimulationModel::setClothPoissonRatioYX), this, std::placeholders::_1));
	setGroup(CLOTH_POISSON_RATIO_YX, "Cloth|FEM");
	setDescription(CLOTH_POISSON_RATIO_YX, "YX Poisson ratio of orthotropic cloth models.");
	static_cast<NumericParameter<Real>*>(getParameter(CLOTH_POISSON_RATIO_YX))->setMinValue(0.0);

	CLOTH_BENDING_STIFFNESS = createNumericParameter<Real>("cloth_bendingStiffness", "Bending stiffness", std::bind(&SimulationModel::getClothBendingStiffness, this), std::bind(static_cast<void (SimulationModel::*)(const Real)>(&SimulationModel::setClothBendingStiffness), this, std::placeholders::_1));
	setGroup(CLOTH_BENDING_STIFFNESS, "Cloth|Bending");
	setDescription(CLOTH_BENDING_STIFFNESS, "Bending stiffness of cloth models.");
	static_cast<NumericParameter<Real>*>(getParameter(CLOTH_BENDING_STIFFNESS))->setMinValue(0.0);

	CLOTH_NORMALIZE_STRETCH = createBoolParameter("cloth_normalizeStretch", "Normalize stretch", std::bind(&SimulationModel::getClothNormalizeStretch, this), std::bind(static_cast<void (SimulationModel::*)(const bool)>(&SimulationModel::setClothNormalizeStretch), this, std::placeholders::_1));
	setGroup(CLOTH_NORMALIZE_STRETCH, "Cloth|Strain Based Dynamics");
	setDescription(CLOTH_NORMALIZE_STRETCH, "Normalize stretch (strain based dynamics)");

	CLOTH_NORMALIZE_SHEAR = createBoolParameter("cloth_normalizeShear", "Normalize shear", std::bind(&SimulationModel::getClothNormalizeShear, this), std::bind(static_cast<void (SimulationModel::*)(const bool)>(&SimulationModel::setClothNormalizeShear), this, std::placeholders::_1));
	setGroup(CLOTH_NORMALIZE_SHEAR, "Cloth|Strain Based Dynamics");
	setDescription(CLOTH_NORMALIZE_SHEAR, "Normalize shear (strain based dynamics)");

	SOLID_SIMULATION_METHOD = createEnumParameter("solidSimulationMethod", "Simulation method", std::bind(&SimulationModel::getSolidSimulationMethod, this), std::bind(static_cast<void (SimulationModel::*)(const int)>(&SimulationModel::setSolidSimulationMethod), this, std::placeholders::_1));
	setGroup(SOLID_SIMULATION_METHOD, "Solid|General");
	setDescription(SOLID_SIMULATION_METHOD, "Solid simulation method");
	enumParam = static_cast<EnumParameter*>(getParameter(SOLID_SIMULATION_METHOD));
	enumParam->addEnumValue("None", ENUM_SOLIDSIM_NONE);
	enumParam->addEnumValue("Distance constraints", ENUM_SOLIDSIM_DISTANCE_CONSTRAINTS);
	enumParam->addEnumValue("FEM based PBD", ENUM_SOLIDSIM_FEM_PBD);
	enumParam->addEnumValue("FEM based XPBD", ENUM_SOLIDSIM_FEM_XPBD);
	enumParam->addEnumValue("Strain based dynamics (no inversion handling)", ENUM_SOLIDSIM_SBD);
	enumParam->addEnumValue("Shape Matching (no inversion handling)", ENUM_SOLIDSIM_SHAPE_MATCHING);
	enumParam->addEnumValue("XPBD distance constraints", ENUM_SOLIDSIM_DISTANCE_CONSTRAINTS_XPBD);

	SOLID_STIFFNESS = createNumericParameter<Real>("solid_stiffness", "Stiffness/Youngs modulus", std::bind(&SimulationModel::getSolidStiffness, this), std::bind(static_cast<void (SimulationModel::*)(const Real)>(&SimulationModel::setSolidStiffness), this, std::placeholders::_1));
	setGroup(SOLID_STIFFNESS, "Solid|General");
	setDescription(SOLID_STIFFNESS, "Stiffness/Young's modulus of solid models.");
	static_cast<NumericParameter<Real>*>(getParameter(SOLID_STIFFNESS))->setMinValue(0.0);

	SOLID_POISSON_RATIO = createNumericParameter<Real>("solid_poissonRatio", "Poisson ratio", std::bind(&SimulationModel::getSolidPoissonRatio, this), std::bind(static_cast<void (SimulationModel::*)(const Real)>(&SimulationModel::setSolidPoissonRatio), this, std::placeholders::_1));
	setGroup(SOLID_POISSON_RATIO, "Solid|FEM");
	setDescription(SOLID_POISSON_RATIO, "Poisson ratio of solid models.");
	static_cast<NumericParameter<Real>*>(getParameter(SOLID_POISSON_RATIO))->setMinValue(0.0);

	SOLID_VOLUME_STIFFNESS = createNumericParameter<Real>("solid_volumeStiffness", "Volume stiffness", std::bind(&SimulationModel::getSolidVolumeStiffness, this), std::bind(static_cast<void (SimulationModel::*)(const Real)>(&SimulationModel::setSolidVolumeStiffness), this, std::placeholders::_1));
	setGroup(SOLID_VOLUME_STIFFNESS, "Solid|Volume constraints");
	setDescription(SOLID_VOLUME_STIFFNESS, "Volume stiffness of solid models.");
	static_cast<NumericParameter<Real>*>(getParameter(SOLID_VOLUME_STIFFNESS))->setMinValue(0.0);

	SOLID_NORMALIZE_STRETCH = createBoolParameter("solid_normalizeStretch", "Normalize stretch", std::bind(&SimulationModel::getSolidNormalizeStretch, this), std::bind(static_cast<void (SimulationModel::*)(const bool)>(&SimulationModel::setSolidNormalizeStretch), this, std::placeholders::_1));
	setGroup(SOLID_NORMALIZE_STRETCH, "Solid|Strain Based Dynamics");
	setDescription(SOLID_NORMALIZE_STRETCH, "Normalize stretch (strain based dynamics)");

	SOLID_NORMALIZE_SHEAR = createBoolParameter("solid_normalizeShear", "Normalize shear", std::bind(&SimulationModel::getSolidNormalizeShear, this), std::bind(static_cast<void (SimulationModel::*)(const bool)>(&SimulationModel::setSolidNormalizeShear), this, std::placeholders::_1));
	setGroup(SOLID_NORMALIZE_SHEAR, "Solid|Strain Based Dynamics");
	setDescription(SOLID_NORMALIZE_SHEAR, "Normalize shear (strain based dynamics)");

	ROD_STRETCHING_STIFFNESS = createNumericParameter<Real>("rod_stretchingStiffness", "Stretching stiffness/Youngs modulus", std::bind(&SimulationModel::getRodStretchingStiffness, this), std::bind(static_cast<void (SimulationModel::*)(const Real)>(&SimulationModel::setRodStretchingStiffness), this, std::placeholders::_1));
	setGroup(ROD_STRETCHING_STIFFNESS, "Elastic Rod|Parameters");
	setDescription(ROD_STRETCHING_STIFFNESS, "Stretching stiffness/Youngs modulus of elastic rod models.");
	static_cast<NumericParameter<Real>*>(getParameter(ROD_STRETCHING_STIFFNESS))->setMinValue(0.0);

	ROD_SHEARING_STIFFNESS_X = createNumericParameter<Real>("rod_shearingStiffnessX", "Shearing stiffness x", std::bind(&SimulationModel::getRodShearingStiffnessX, this), std::bind(static_cast<void (SimulationModel::*)(const Real)>(&SimulationModel::setRodShearingStiffnessX), this, std::placeholders::_1));
	setGroup(ROD_SHEARING_STIFFNESS_X, "Elastic Rod|Parameters");
	setDescription(ROD_SHEARING_STIFFNESS_X, "Shearing stiffness of elastic rod models.");
	static_cast<NumericParameter<Real>*>(getParameter(ROD_SHEARING_STIFFNESS_X))->setMinValue(0.0);

	ROD_SHEARING_STIFFNESS_Y = createNumericParameter<Real>("rod_shearingStiffnessY", "Shearing stiffness y", std::bind(&SimulationModel::getRodShearingStiffnessY, this), std::bind(static_cast<void (SimulationModel::*)(const Real)>(&SimulationModel::setRodShearingStiffnessY), this, std::placeholders::_1));
	setGroup(ROD_SHEARING_STIFFNESS_Y, "Elastic Rod|Parameters");
	setDescription(ROD_SHEARING_STIFFNESS_Y, "Shearing stiffness of elastic rod models.");
	static_cast<NumericParameter<Real>*>(getParameter(ROD_SHEARING_STIFFNESS_Y))->setMinValue(0.0);

	ROD_BENDING_STIFFNESS_X  = createNumericParameter<Real>("rod_bendingStiffnessX", "Bending stiffness x", std::bind(&SimulationModel::getRodBendingStiffnessX, this), std::bind(static_cast<void (SimulationModel::*)(const Real)>(&SimulationModel::setRodBendingStiffnessX), this, std::placeholders::_1));
	setGroup(ROD_BENDING_STIFFNESS_X, "Elastic Rod|Parameters");
	setDescription(ROD_BENDING_STIFFNESS_X, "Bending stiffness of elastic rod models.");
	static_cast<NumericParameter<Real>*>(getParameter(ROD_BENDING_STIFFNESS_X))->setMinValue(0.0);

	ROD_BENDING_STIFFNESS_Y = createNumericParameter<Real>("rod_bendingStiffnessY", "Bending stiffness y", std::bind(&SimulationModel::getRodBendingStiffnessY, this), std::bind(static_cast<void (SimulationModel::*)(const Real)>(&SimulationModel::setRodBendingStiffnessY), this, std::placeholders::_1));
	setGroup(ROD_BENDING_STIFFNESS_Y, "Elastic Rod|Parameters");
	setDescription(ROD_BENDING_STIFFNESS_Y, "Bending stiffness of elastic rod models.");
	static_cast<NumericParameter<Real>*>(getParameter(ROD_BENDING_STIFFNESS_Y))->setMinValue(0.0);

	ROD_TWISTING_STIFFNESS = createNumericParameter<Real>("rod_twistingStiffness", "Twisting stiffness", std::bind(&SimulationModel::getRodTwistingStiffness, this), std::bind(static_cast<void (SimulationModel::*)(const Real)>(&SimulationModel::setRodTwistingStiffness), this, std::placeholders::_1));
	setGroup(ROD_TWISTING_STIFFNESS, "Elastic Rod|Parameters");
	setDescription(ROD_TWISTING_STIFFNESS, "Twisting stiffness of elastic rod models.");
	static_cast<NumericParameter<Real>*>(getParameter(ROD_TWISTING_STIFFNESS))->setMinValue(0.0);

	CONTACT_STIFFNESS_RB = createNumericParameter<Real>("contactStiffnessRigidBody", "Contact stiffness RB", std::bind(&SimulationModel::getContactStiffnessRigidBody, this), std::bind(static_cast<void (SimulationModel::*)(const Real)>(&SimulationModel::setContactStiffnessRigidBody), this, std::placeholders::_1));
	setGroup(CONTACT_STIFFNESS_RB, "Simulation|Contact");
	setDescription(CONTACT_STIFFNESS_RB, "Stiffness coefficient for rigid-rigid contact resolution.");
	static_cast<NumericParameter<Real>*>(getParameter(CONTACT_STIFFNESS_RB))->setMinValue(0.0);

	CONTACT_STIFFNESS_PARTICLE_RB = createNumericParameter<Real>("contactStiffnessParticleRigidBody", "Contact stiffness Particle-RB", std::bind(&SimulationModel::getContactStiffnessParticleRigidBody, this), std::bind(static_cast<void (SimulationModel::*)(const Real)>(&SimulationModel::setContactStiffnessParticleRigidBody), this, std::placeholders::_1));
	setGroup(CONTACT_STIFFNESS_PARTICLE_RB, "Simulation|Contact");
	setDescription(CONTACT_STIFFNESS_PARTICLE_RB, "Stiffness coefficient for particle-rigid contact resolution.");
	static_cast<NumericParameter<Real>*>(getParameter(CONTACT_STIFFNESS_PARTICLE_RB))->setMinValue(0.0);
}

void SimulationModel::reset()
{
	resetContacts();

	// rigid bodies
	for (size_t i = 0; i < m_rigidBodies.size(); i++)
	{
		m_rigidBodies[i]->reset();
		m_rigidBodies[i]->getGeometry().updateMeshTransformation(m_rigidBodies[i]->getPosition(), m_rigidBodies[i]->getRotationMatrix());
	}

	// particles
	for (unsigned int i = 0; i < m_particles.size(); i++)
	{
		const Vector3r& x0 = m_particles.getPosition0(i);
		m_particles.getPosition(i) = x0;
		m_particles.getLastPosition(i) = m_particles.getPosition(i);
		m_particles.getOldPosition(i) = m_particles.getPosition(i);
		m_particles.getVelocity(i).setZero();
		m_particles.getAcceleration(i).setZero();
	}

	// orientations
	for(unsigned int i = 0; i < m_orientations.size(); i++)
	{
		const Quaternionr& q0 = m_orientations.getQuaternion0(i);
		m_orientations.getQuaternion(i) = q0;
		m_orientations.getLastQuaternion(i) = q0;
		m_orientations.getOldQuaternion(i) = q0;
		m_orientations.getVelocity(i).setZero();
		m_orientations.getAcceleration(i).setZero();
	}

	updateConstraints();
}

SimulationModel::RigidBodyVector & SimulationModel::getRigidBodies()
{
	return m_rigidBodies;
}

ParticleData & SimulationModel::getParticles()
{
	return m_particles;
}

OrientationData & SimulationModel::getOrientations()
{
	return m_orientations;
}

SimulationModel::TriangleModelVector & SimulationModel::getTriangleModels()
{
	return m_triangleModels;
}

SimulationModel::TetModelVector & SimulationModel::getTetModels()
{
	return m_tetModels;
}

SimulationModel::LineModelVector & SimulationModel::getLineModels()
{
	return m_lineModels;
}

SimulationModel::ConstraintVector & SimulationModel::getConstraints()
{
	return m_constraints;
}

SimulationModel::RigidBodyContactConstraintVector & SimulationModel::getRigidBodyContactConstraints()
{
	return m_rigidBodyContactConstraints;
}

SimulationModel::ParticleRigidBodyContactConstraintVector & SimulationModel::getParticleRigidBodyContactConstraints()
{
	return m_particleRigidBodyContactConstraints;
}

SimulationModel::ParticleSolidContactConstraintVector & SimulationModel::getParticleSolidContactConstraints()
{
	return m_particleSolidContactConstraints;
}

SimulationModel::ConstraintGroupVector & SimulationModel::getConstraintGroups()
{
	return m_constraintGroups;
}

void SimulationModel::updateConstraints()
{
	for (unsigned int i = 0; i < m_constraints.size(); i++)
		m_constraints[i]->updateConstraint(*this);
}


bool SimulationModel::addBallJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos)
{
	BallJoint *bj = new BallJoint();
	const bool res = bj->initConstraint(*this, rbIndex1, rbIndex2, pos);
	if (res)
	{
		m_constraints.push_back(bj);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addBallOnLineJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &dir)
{
	BallOnLineJoint *bj = new BallOnLineJoint();
	const bool res = bj->initConstraint(*this, rbIndex1, rbIndex2, pos, dir);
	if (res)
	{
		m_constraints.push_back(bj);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addHingeJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis)
{
	HingeJoint *hj = new HingeJoint();
	const bool res = hj->initConstraint(*this, rbIndex1, rbIndex2, pos, axis);
	if (res)
	{
		m_constraints.push_back(hj);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addUniversalJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis1, const Vector3r &axis2)
{
	UniversalJoint *uj = new UniversalJoint();
	const bool res = uj->initConstraint(*this, rbIndex1, rbIndex2, pos, axis1, axis2);
	if (res)
	{
		m_constraints.push_back(uj);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addSliderJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &axis)
{
	SliderJoint *joint = new SliderJoint();
	const bool res = joint->initConstraint(*this, rbIndex1, rbIndex2, axis);
	if (res)
	{
		m_constraints.push_back(joint);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addTargetPositionMotorSliderJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &axis)
{
	TargetPositionMotorSliderJoint *joint = new TargetPositionMotorSliderJoint();
	const bool res = joint->initConstraint(*this, rbIndex1, rbIndex2, axis);
	if (res)
	{
		m_constraints.push_back(joint);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addTargetVelocityMotorSliderJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &axis)
{
	TargetVelocityMotorSliderJoint *joint = new TargetVelocityMotorSliderJoint();
	const bool res = joint->initConstraint(*this, rbIndex1, rbIndex2, axis);
	if (res)
	{
		m_constraints.push_back(joint);
		m_groupsInitialized = false;
	}
	return res;
}


bool SimulationModel::addTargetAngleMotorHingeJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis)
{
	TargetAngleMotorHingeJoint *hj = new TargetAngleMotorHingeJoint();
	const bool res = hj->initConstraint(*this, rbIndex1, rbIndex2, pos, axis);
	if (res)
	{
		m_constraints.push_back(hj);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addTargetVelocityMotorHingeJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis)
{
	TargetVelocityMotorHingeJoint *hj = new TargetVelocityMotorHingeJoint();
	const bool res = hj->initConstraint(*this, rbIndex1, rbIndex2, pos, axis);
	if (res)
	{
		m_constraints.push_back(hj);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addDamperJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &axis, const Real stiffness)
{
	DamperJoint *joint = new DamperJoint();
	const bool res = joint->initConstraint(*this, rbIndex1, rbIndex2, axis, stiffness);
	if (res)
	{
		m_constraints.push_back(joint);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addRigidBodyParticleBallJoint(const unsigned int rbIndex, const unsigned int particleIndex)
{
	RigidBodyParticleBallJoint *bj = new RigidBodyParticleBallJoint();
	const bool res = bj->initConstraint(*this, rbIndex, particleIndex);
	if (res)
	{
		m_constraints.push_back(bj);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addRigidBodySpring(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos1, const Vector3r &pos2, const Real stiffness)
{
	RigidBodySpring *s = new RigidBodySpring();
	const bool res = s->initConstraint(*this, rbIndex1, rbIndex2, pos1, pos2, stiffness);
	if (res)
	{
		m_constraints.push_back(s);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addDistanceJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos1, const Vector3r &pos2)
{
	DistanceJoint *j = new DistanceJoint();
	const bool res = j->initConstraint(*this, rbIndex1, rbIndex2, pos1, pos2);
	if (res)
	{
		m_constraints.push_back(j);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addRigidBodyContactConstraint(const unsigned int rbIndex1, const unsigned int rbIndex2, 
	const Vector3r &cp1, const Vector3r &cp2, 
	const Vector3r &normal, const Real dist,
	const Real restitutionCoeff, const Real frictionCoeff)
{
	m_rigidBodyContactConstraints.emplace_back(RigidBodyContactConstraint());
	RigidBodyContactConstraint &cc = m_rigidBodyContactConstraints.back();
	const bool res = cc.initConstraint(*this, rbIndex1, rbIndex2, cp1, cp2, normal, dist, restitutionCoeff, m_contactStiffnessRigidBody, frictionCoeff);
	if (!res)
		m_rigidBodyContactConstraints.pop_back();
	return res;
}

 bool SimulationModel::addParticleRigidBodyContactConstraint(const unsigned int particleIndex, const unsigned int rbIndex, 
 	const Vector3r &cp1, const Vector3r &cp2, 
 	const Vector3r &normal, const Real dist,
 	const Real restitutionCoeff, const Real frictionCoeff)
{
	m_particleRigidBodyContactConstraints.emplace_back(ParticleRigidBodyContactConstraint());
 	ParticleRigidBodyContactConstraint &cc = m_particleRigidBodyContactConstraints.back();
 	const bool res = cc.initConstraint(*this, particleIndex, rbIndex, cp1, cp2, normal, dist, restitutionCoeff, m_contactStiffnessParticleRigidBody, frictionCoeff);
 	if (!res)
 		m_particleRigidBodyContactConstraints.pop_back();
 	return res;
}

bool SimulationModel::addParticleSolidContactConstraint(const unsigned int particleIndex, const unsigned int solidIndex, 
	const unsigned int tetIndex, const Vector3r &bary,
 	const Vector3r &cp1, const Vector3r &cp2, 
 	const Vector3r &normal, const Real dist,
 	const Real restitutionCoeff, const Real frictionCoeff)
{
	m_particleSolidContactConstraints.emplace_back(ParticleTetContactConstraint());
 	ParticleTetContactConstraint &cc = m_particleSolidContactConstraints.back();
 	const bool res = cc.initConstraint(*this, particleIndex, solidIndex, tetIndex, bary, cp1, cp2, normal, dist, frictionCoeff);
 	if (!res)
		m_particleSolidContactConstraints.pop_back();
 	return res;
}

bool SimulationModel::addDistanceConstraint(const unsigned int particle1, const unsigned int particle2, const Real stiffness)
{
	DistanceConstraint *c = new DistanceConstraint();
	const bool res = c->initConstraint(*this, particle1, particle2, stiffness);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addDistanceConstraint_XPBD(const unsigned int particle1, const unsigned int particle2, const Real stiffness)
{
	DistanceConstraint_XPBD* c = new DistanceConstraint_XPBD();
	const bool res = c->initConstraint(*this, particle1, particle2, stiffness);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addDihedralConstraint(const unsigned int particle1, const unsigned int particle2, 
											const unsigned int particle3, const unsigned int particle4, const Real stiffness)
{
	DihedralConstraint *c = new DihedralConstraint();
	const bool res = c->initConstraint(*this, particle1, particle2, particle3, particle4, stiffness);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addIsometricBendingConstraint(const unsigned int particle1, const unsigned int particle2,
													const unsigned int particle3, const unsigned int particle4, const Real stiffness)
{
	IsometricBendingConstraint *c = new IsometricBendingConstraint();
	const bool res = c->initConstraint(*this, particle1, particle2, particle3, particle4, stiffness);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addIsometricBendingConstraint_XPBD(const unsigned int particle1, const unsigned int particle2,
														const unsigned int particle3, const unsigned int particle4, const Real stiffness)
{
	IsometricBendingConstraint_XPBD* c = new IsometricBendingConstraint_XPBD();
	const bool res = c->initConstraint(*this, particle1, particle2, particle3, particle4, stiffness);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addFEMTriangleConstraint(const unsigned int particle1, const unsigned int particle2,
	const unsigned int particle3, const Real xxStiffness, const Real yyStiffness, const Real xyStiffness,
	const Real xyPoissonRatio, const Real yxPoissonRatio)
{
	FEMTriangleConstraint *c = new FEMTriangleConstraint();
	const bool res = c->initConstraint(*this, particle1, particle2, particle3, xxStiffness, 
		yyStiffness, xyStiffness, xyPoissonRatio, yxPoissonRatio);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addStrainTriangleConstraint(const unsigned int particle1, const unsigned int particle2,
	const unsigned int particle3, const Real xxStiffness, const Real yyStiffness, const Real xyStiffness,
	const bool normalizeStretch, const bool normalizeShear)
{
	StrainTriangleConstraint *c = new StrainTriangleConstraint();
	const bool res = c->initConstraint(*this, particle1, particle2, particle3, xxStiffness, 
		yyStiffness, xyStiffness, normalizeStretch, normalizeShear);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addVolumeConstraint(const unsigned int particle1, const unsigned int particle2,
										const unsigned int particle3, const unsigned int particle4, const Real stiffness)
{
	VolumeConstraint *c = new VolumeConstraint();
	const bool res = c->initConstraint(*this, particle1, particle2, particle3, particle4, stiffness);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addVolumeConstraint_XPBD(const unsigned int particle1, const unsigned int particle2,
	const unsigned int particle3, const unsigned int particle4, const Real stiffness)
{
	VolumeConstraint_XPBD* c = new VolumeConstraint_XPBD();
	const bool res = c->initConstraint(*this, particle1, particle2, particle3, particle4, stiffness);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addFEMTetConstraint(const unsigned int particle1, const unsigned int particle2,
										const unsigned int particle3, const unsigned int particle4, 
										const Real stiffness, const Real poissonRatio)
{
	FEMTetConstraint *c = new FEMTetConstraint();
	const bool res = c->initConstraint(*this, particle1, particle2, particle3, particle4, stiffness, poissonRatio);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addFEMTetConstraint_XPBD(const unsigned int particle1, const unsigned int particle2,
										const unsigned int particle3, const unsigned int particle4, 
										const Real stiffness, const Real poissonRatio)
{
	XPBD_FEMTetConstraint *c = new XPBD_FEMTetConstraint();
	const bool res = c->initConstraint(*this, particle1, particle2, particle3, particle4, stiffness, poissonRatio);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addStrainTetConstraint(const unsigned int particle1, const unsigned int particle2,
										const unsigned int particle3, const unsigned int particle4,
										const Real stretchStiffness, const Real shearStiffness, 
										const bool normalizeStretch, const bool normalizeShear)
{
	StrainTetConstraint *c = new StrainTetConstraint();
	const bool res = c->initConstraint(*this, particle1, particle2, particle3, particle4, stretchStiffness, shearStiffness, 
		normalizeStretch, normalizeShear);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addShapeMatchingConstraint(const unsigned int numberOfParticles, const unsigned int particleIndices[], const unsigned int numClusters[], const Real stiffness)
{
	ShapeMatchingConstraint *c = new ShapeMatchingConstraint(numberOfParticles);
	const bool res = c->initConstraint(*this, particleIndices, numClusters, stiffness);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addStretchShearConstraint(const unsigned int particle1, const unsigned int particle2, 
	const unsigned int quaternion1, const Real stretchingStiffness,
	const Real shearingStiffness1, const Real shearingStiffness2)
{
	StretchShearConstraint *c = new StretchShearConstraint();
	const bool res = c->initConstraint(*this, particle1, particle2, quaternion1, stretchingStiffness, shearingStiffness1, shearingStiffness2);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addBendTwistConstraint(const unsigned int quaternion1, 
	const unsigned int quaternion2, const Real twistingStiffness,
	const Real bendingStiffness1, const Real bendingStiffness2)
{
	BendTwistConstraint *c = new BendTwistConstraint();
	const bool res = c->initConstraint(*this, quaternion1, quaternion2, twistingStiffness, bendingStiffness1, bendingStiffness2);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

bool PBD::SimulationModel::addStretchBendingTwistingConstraint(
	const unsigned int rbIndex1,
	const unsigned int rbIndex2,
	const Vector3r &pos,
	const Real averageRadius,
	const Real averageSegmentLength,
	const Real youngsModulus,
	const Real torsionModulus)
{
	StretchBendingTwistingConstraint *c = new StretchBendingTwistingConstraint();
	const bool res = c->initConstraint(*this, rbIndex1, rbIndex2, pos,
		averageRadius, averageSegmentLength, youngsModulus, torsionModulus);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

bool PBD::SimulationModel::addDirectPositionBasedSolverForStiffRodsConstraint(
	const std::vector<std::pair<unsigned int, unsigned int>> & jointSegmentIndices,
	const std::vector<Vector3r> &jointPositions,
	const std::vector<Real> &averageRadii,
	const std::vector<Real> &averageSegmentLengths,
	const std::vector<Real> &youngsModuli,
	const std::vector<Real> &torsionModuli
	)
{
	DirectPositionBasedSolverForStiffRodsConstraint *c = new DirectPositionBasedSolverForStiffRodsConstraint();
	const bool res = c->initConstraint(*this, jointSegmentIndices, jointPositions,
		averageRadii, averageSegmentLengths, youngsModuli, torsionModuli);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

void SimulationModel::addTriangleModel(
	const unsigned int nPoints, 
	const unsigned int nFaces, 
	Vector3r *points,
	unsigned int* indices, 
	const TriangleModel::ParticleMesh::UVIndices& uvIndices, 
	const TriangleModel::ParticleMesh::UVs& uvs)
{
	TriangleModel *triModel = new TriangleModel();
	m_triangleModels.push_back(triModel);

	unsigned int startIndex = m_particles.size();
	m_particles.reserve(startIndex + nPoints);

	for (unsigned int i = 0; i < nPoints; i++)
		m_particles.addVertex(points[i]);

	triModel->initMesh(nPoints, nFaces, startIndex, indices, uvIndices, uvs);

	// Update normals
	triModel->updateMeshNormals(m_particles);
}

void SimulationModel::addRegularTriangleModel(const int width, const int height,
	const Vector3r& translation,
	const Matrix3r& rotation,
	const Vector2r& scale)
{
	TriangleModel::ParticleMesh::UVs uvs;
	uvs.resize(width * height);

	const Real dy = scale[1] / (Real)(height - 1);
	const Real dx = scale[0] / (Real)(width - 1);

	std::vector<Vector3r> points;
	points.resize(width * height);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			const Real y = (Real)dy * i;
			const Real x = (Real)dx * j;
			points[i * width + j] = rotation * Vector3r(x, y, 0.0) + translation;

			uvs[i * width + j][0] = x / scale[0];
			uvs[i * width + j][1] = y / scale[1];
		}
	}
	const int nIndices = 6 * (height - 1) * (width - 1);

	TriangleModel::ParticleMesh::UVIndices uvIndices;
	uvIndices.resize(nIndices);

	std::vector<unsigned int> indices;
	indices.resize(nIndices);
	int index = 0;
	for (int i = 0; i < height - 1; i++)
	{
		for (int j = 0; j < width - 1; j++)
		{
			int helper = 0;
			if (i % 2 == j % 2)
				helper = 1;

			indices[index] = i * width + j;
			indices[index + 1] = i * width + j + 1;
			indices[index + 2] = (i + 1) * width + j + helper;

			uvIndices[index] = i * width + j;
			uvIndices[index + 1] = i * width + j + 1;
			uvIndices[index + 2] = (i + 1) * width + j + helper;
			index += 3;

			indices[index] = (i + 1) * width + j + 1;
			indices[index + 1] = (i + 1) * width + j;
			indices[index + 2] = i * width + j + 1 - helper;

			uvIndices[index] = (i + 1) * width + j + 1;
			uvIndices[index + 1] = (i + 1) * width + j;
			uvIndices[index + 2] = i * width + j + 1 - helper;
			index += 3;
		}
	}

	const unsigned int nPoints = height * width;
	const unsigned int nFaces = nIndices / 3;
	const auto modelIndex = m_triangleModels.size();
	addTriangleModel(nPoints, nFaces, points.data(), indices.data(), uvIndices, uvs);
	const auto offset = m_triangleModels[modelIndex]->getIndexOffset();

	ParticleData& pd = getParticles();
	for (unsigned int i = offset; i < offset + m_triangleModels[modelIndex]->getParticleMesh().numVertices(); i++)
		pd.setMass(i, 1.0);
}

void SimulationModel::addTetModel(
	const unsigned int nPoints,
	const unsigned int nTets,
	Vector3r *points,
	unsigned int* indices)
{
	TetModel *tetModel = new TetModel();
	m_tetModels.push_back(tetModel);

	unsigned int startIndex = m_particles.size();
	m_particles.reserve(startIndex + nPoints);

	for (unsigned int i = 0; i < nPoints; i++)
		m_particles.addVertex(points[i]);

	tetModel->initMesh(nPoints, nTets, startIndex, indices);
}

void SimulationModel::addRegularTetModel(const int width, const int height, const int depth,
	const Vector3r& translation,
	const Matrix3r& rotation,
	const Vector3r& scale)
{
	std::vector<Vector3r> points;
	points.resize(width * height * depth);

	const Real dx = scale[0] / (Real)(width - 1);
	const Real dy = scale[1] / (Real)(height - 1);
	const Real dz = scale[2] / (Real)(depth - 1);

	// center in origin
	const Vector3r t = translation - 0.5*scale;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			for (int k = 0; k < depth; k++)
			{
				const Real x = (Real)dx * i;
				const Real y = (Real)dy * j;
				const Real z = (Real)dz * k;

				points[i * height * depth + j * depth + k] = rotation * Vector3r(x, y, z) + t;
			}
		}
	}

	std::vector<unsigned int> indices;
	indices.reserve(width * height * depth * 5);
	for (int i = 0; i < width - 1; i++)
	{
		for (int j = 0; j < height - 1; j++)
		{
			for (int k = 0; k < depth - 1; k++)
			{
				// For each block, the 8 corners are numerated as:
				//     4*-----*7
				//     /|    /|
				//    / |   / |
				//  5*-----*6 |
				//   | 0*--|--*3
				//   | /   | /
				//   |/    |/
				//  1*-----*2
				unsigned int p0 = i * height * depth + j * depth + k;
				unsigned int p1 = p0 + 1;
				unsigned int p3 = (i + 1) * height * depth + j * depth + k;
				unsigned int p2 = p3 + 1;
				unsigned int p7 = (i + 1) * height * depth + (j + 1) * depth + k;
				unsigned int p6 = p7 + 1;
				unsigned int p4 = i * height * depth + (j + 1) * depth + k;
				unsigned int p5 = p4 + 1;

				// Ensure that neighboring tetras are sharing faces
				if ((i + j + k) % 2 == 1)
				{
					indices.push_back(p2); indices.push_back(p1); indices.push_back(p6); indices.push_back(p3);
					indices.push_back(p6); indices.push_back(p3); indices.push_back(p4); indices.push_back(p7);
					indices.push_back(p4); indices.push_back(p1); indices.push_back(p6); indices.push_back(p5);
					indices.push_back(p3); indices.push_back(p1); indices.push_back(p4); indices.push_back(p0);
					indices.push_back(p6); indices.push_back(p1); indices.push_back(p4); indices.push_back(p3);
				}
				else
				{
					indices.push_back(p0); indices.push_back(p2); indices.push_back(p5); indices.push_back(p1);
					indices.push_back(p7); indices.push_back(p2); indices.push_back(p0); indices.push_back(p3);
					indices.push_back(p5); indices.push_back(p2); indices.push_back(p7); indices.push_back(p6);
					indices.push_back(p7); indices.push_back(p0); indices.push_back(p5); indices.push_back(p4);
					indices.push_back(p0); indices.push_back(p2); indices.push_back(p7); indices.push_back(p5);
				}
			}
		}
	}
	const auto modelIndex = m_tetModels.size();
	addTetModel(width * height * depth, (unsigned int)indices.size() / 4u, points.data(), indices.data());
	const auto offset = m_tetModels[modelIndex]->getIndexOffset();

	ParticleData& pd = getParticles();
	for (unsigned int i = offset; i < offset + m_tetModels[modelIndex]->getParticleMesh().numVertices(); i++)
	{
		pd.setMass(i, 1.0);
	}
}

void SimulationModel::addLineModel(
	const unsigned int nPoints,
	const unsigned int nQuaternions,
	Vector3r *points,
	Quaternionr *quaternions,
	unsigned int *indices,
	unsigned int *indicesQuaternions)
{
	LineModel *lineModel = new LineModel();
	m_lineModels.push_back(lineModel);

	unsigned int startIndex = m_particles.size();
	m_particles.reserve(startIndex + nPoints);

	for (unsigned int i = 0; i < nPoints; i++)
		m_particles.addVertex(points[i]);

	unsigned int startIndexOrientations = m_orientations.size();
	m_orientations.reserve(startIndexOrientations + nQuaternions);

	for (unsigned int i = 0; i < nQuaternions; i++)
		m_orientations.addQuaternion(quaternions[i]);

	lineModel->initMesh(nPoints, nQuaternions, startIndex, startIndexOrientations, indices, indicesQuaternions);
}

void SimulationModel::initConstraintGroups()
{
	if (m_groupsInitialized)
		return;

	const unsigned int numConstraints = (unsigned int) m_constraints.size();
	const unsigned int numParticles = (unsigned int) m_particles.size();
	const unsigned int numRigidBodies = (unsigned int) m_rigidBodies.size();
	const unsigned int numBodies = numParticles + numRigidBodies;
	m_constraintGroups.clear();

	// Maps in which group a particle is or 0 if not yet mapped
	std::vector<unsigned char*> mapping;

	for (unsigned int i = 0; i < numConstraints; i++)
	{
		Constraint *constraint = m_constraints[i];

		bool addToNewGroup = true;
		for (unsigned int j = 0; j < m_constraintGroups.size(); j++)
		{
			bool addToThisGroup = true;

			for (unsigned int k = 0; k < constraint->numberOfBodies(); k++)
			{
				if (mapping[j][constraint->m_bodies[k]] != 0)
				{
					addToThisGroup = false;
					break;
				}
			}

			if (addToThisGroup)
			{
				m_constraintGroups[j].push_back(i);

				for (unsigned int k = 0; k < constraint->numberOfBodies(); k++)
					mapping[j][constraint->m_bodies[k]] = 1;

				addToNewGroup = false;
				break;
			}
		}
		if (addToNewGroup)
		{
			mapping.push_back(new unsigned char[numBodies]);
			memset(mapping[mapping.size() - 1], 0, sizeof(unsigned char)*numBodies);
			m_constraintGroups.resize(m_constraintGroups.size() + 1);
			m_constraintGroups[m_constraintGroups.size()-1].push_back(i);
			for (unsigned int k = 0; k < constraint->numberOfBodies(); k++)
				mapping[m_constraintGroups.size() - 1][constraint->m_bodies[k]] = 1;
		}
	}

	for (unsigned int i = 0; i < mapping.size(); i++)
	{
		delete[] mapping[i];
	}
	mapping.clear();

	m_groupsInitialized = true;
}

void PBD::SimulationModel::setClothSimulationMethod(int val) 
{ 
	m_clothSimulationMethod = val;
	if (m_clothSimMethodChanged != nullptr)
		m_clothSimMethodChanged();
}

void PBD::SimulationModel::setClothBendingMethod(int val)
{
	m_clothBendingMethod = val;
	if (m_clothBendingMethodChanged != nullptr)
		m_clothBendingMethodChanged();
}

void PBD::SimulationModel::setSolidSimulationMethod(int val)
{
	m_solidSimulationMethod = val;
	if (m_solidSimMethodChanged != nullptr)
		m_solidSimMethodChanged();
}


void SimulationModel::resetContacts()
{
	m_rigidBodyContactConstraints.clear();
	m_particleRigidBodyContactConstraints.clear();
	m_particleSolidContactConstraints.clear();
}

void SimulationModel::addClothConstraints(const TriangleModel* tm, const unsigned int clothMethod, 
	const Real distanceStiffness, const Real xxStiffness, const Real yyStiffness, 
	const Real xyStiffness,	const Real xyPoissonRatio, const Real yxPoissonRatio, 
	const bool normalizeStretch, const bool normalizeShear)
{
	if (clothMethod == 1)
	{
		const unsigned int offset = tm->getIndexOffset();
		const unsigned int nEdges = tm->getParticleMesh().numEdges();
		const Utilities::IndexedFaceMesh::Edge* edges = tm->getParticleMesh().getEdges().data();
		for (unsigned int i = 0; i < nEdges; i++)
		{
			const unsigned int v1 = edges[i].m_vert[0] + offset;
			const unsigned int v2 = edges[i].m_vert[1] + offset;

			addDistanceConstraint(v1, v2, distanceStiffness);
		}
	}
	else if (clothMethod == 2)
	{
		const unsigned int offset = tm->getIndexOffset();
		const TriangleModel::ParticleMesh& mesh = tm->getParticleMesh();
		const unsigned int* tris = mesh.getFaces().data();
		const unsigned int nFaces = mesh.numFaces();
		for (unsigned int i = 0; i < nFaces; i++)
		{
			const unsigned int v1 = tris[3 * i] + offset;
			const unsigned int v2 = tris[3 * i + 1] + offset;
			const unsigned int v3 = tris[3 * i + 2] + offset;
			addFEMTriangleConstraint(v1, v2, v3, xxStiffness, yyStiffness, xyStiffness, xyPoissonRatio, yxPoissonRatio);
		}
	}
	else if (clothMethod == 3)
	{
		const unsigned int offset = tm->getIndexOffset();
		const TriangleModel::ParticleMesh& mesh = tm->getParticleMesh();
		const unsigned int* tris = mesh.getFaces().data();
		const unsigned int nFaces = mesh.numFaces();
		for (unsigned int i = 0; i < nFaces; i++)
		{
			const unsigned int v1 = tris[3 * i] + offset;
			const unsigned int v2 = tris[3 * i + 1] + offset;
			const unsigned int v3 = tris[3 * i + 2] + offset;
			addStrainTriangleConstraint(v1, v2, v3, xxStiffness, yyStiffness, xyStiffness, normalizeStretch, normalizeShear);
		}
	}
	else if (clothMethod == 4)
	{
		const unsigned int offset = tm->getIndexOffset();
		const unsigned int nEdges = tm->getParticleMesh().numEdges();
		const Utilities::IndexedFaceMesh::Edge* edges = tm->getParticleMesh().getEdges().data();
		for (unsigned int i = 0; i < nEdges; i++)
		{
			const unsigned int v1 = edges[i].m_vert[0] + offset;
			const unsigned int v2 = edges[i].m_vert[1] + offset;

			addDistanceConstraint_XPBD(v1, v2, distanceStiffness);
		}
	}
}

void SimulationModel::addBendingConstraints(const TriangleModel *tm, const unsigned int bendingMethod, const Real stiffness)
{
	if ((bendingMethod < 1) || (bendingMethod > 3))
		return;

	const unsigned int offset = tm->getIndexOffset();
	const TriangleModel::ParticleMesh& mesh = tm->getParticleMesh();
	unsigned int nEdges = mesh.numEdges();
	const TriangleModel::ParticleMesh::Edge* edges = mesh.getEdges().data();
	const unsigned int* tris = mesh.getFaces().data();
	for (unsigned int i = 0; i < nEdges; i++)
	{
		const int tri1 = edges[i].m_face[0];
		const int tri2 = edges[i].m_face[1];
		if ((tri1 != 0xffffffff) && (tri2 != 0xffffffff))
		{
			// Find the triangle points which do not lie on the axis
			const int axisPoint1 = edges[i].m_vert[0];
			const int axisPoint2 = edges[i].m_vert[1];
			int point1 = -1;
			int point2 = -1;
			for (int j = 0; j < 3; j++)
			{
				if ((tris[3 * tri1 + j] != axisPoint1) && (tris[3 * tri1 + j] != axisPoint2))
				{
					point1 = tris[3 * tri1 + j];
					break;
				}
			}
			for (int j = 0; j < 3; j++)
			{
				if ((tris[3 * tri2 + j] != axisPoint1) && (tris[3 * tri2 + j] != axisPoint2))
				{
					point2 = tris[3 * tri2 + j];
					break;
				}
			}
			if ((point1 != -1) && (point2 != -1))
			{
				const unsigned int vertex1 = point1 + offset;
				const unsigned int vertex2 = point2 + offset;
				const unsigned int vertex3 = edges[i].m_vert[0] + offset;
				const unsigned int vertex4 = edges[i].m_vert[1] + offset;
				if (bendingMethod == 1)
					addDihedralConstraint(vertex1, vertex2, vertex3, vertex4, stiffness);
				else if (bendingMethod == 2)
					addIsometricBendingConstraint(vertex1, vertex2, vertex3, vertex4, stiffness);
				else if (bendingMethod == 3)
				{
					addIsometricBendingConstraint_XPBD(vertex1, vertex2, vertex3, vertex4, stiffness);
				}
			}
		}
	}
}

void SimulationModel::addSolidConstraints(const TetModel* tm, const unsigned int solidMethod, const Real stiffness,
	const Real poissonRatio, const Real volumeStiffness,
	const bool normalizeStretch, const bool normalizeShear)
{
	const unsigned int nTets = tm->getParticleMesh().numTets();
	const unsigned int* tets = tm->getParticleMesh().getTets().data();
	const Utilities::IndexedTetMesh::VerticesTets& vTets = tm->getParticleMesh().getVertexTets();
	const unsigned int offset = tm->getIndexOffset();
	if (solidMethod == 1)
	{
		const unsigned int nEdges = tm->getParticleMesh().numEdges();
		const Utilities::IndexedTetMesh::Edge* edges = tm->getParticleMesh().getEdges().data();
		for (unsigned int i = 0; i < nEdges; i++)
		{
			const unsigned int v1 = edges[i].m_vert[0] + offset;
			const unsigned int v2 = edges[i].m_vert[1] + offset;

			addDistanceConstraint(v1, v2, stiffness);
		}

		for (unsigned int i = 0; i < nTets; i++)
		{
			const unsigned int v1 = tets[4 * i] + offset;
			const unsigned int v2 = tets[4 * i + 1] + offset;
			const unsigned int v3 = tets[4 * i + 2] + offset;
			const unsigned int v4 = tets[4 * i + 3] + offset;

			addVolumeConstraint(v1, v2, v3, v4, volumeStiffness);
		}
	}
	else if (solidMethod == 2)
	{
		const TetModel::ParticleMesh& mesh = tm->getParticleMesh();
		for (unsigned int i = 0; i < nTets; i++)
		{
			const unsigned int v1 = tets[4 * i] + offset;
			const unsigned int v2 = tets[4 * i + 1] + offset;
			const unsigned int v3 = tets[4 * i + 2] + offset;
			const unsigned int v4 = tets[4 * i + 3] + offset;

			addFEMTetConstraint(v1, v2, v3, v4, stiffness, poissonRatio);
		}
	}
	else if (solidMethod == 3)
	{
		const TetModel::ParticleMesh& mesh = tm->getParticleMesh();
		for (unsigned int i = 0; i < nTets; i++)
		{
			const unsigned int v1 = tets[4 * i] + offset;
			const unsigned int v2 = tets[4 * i + 1] + offset;
			const unsigned int v3 = tets[4 * i + 2] + offset;
			const unsigned int v4 = tets[4 * i + 3] + offset;

			addFEMTetConstraint_XPBD(v1, v2, v3, v4, stiffness, poissonRatio);
		}
	}
	else if (solidMethod == 4)
	{
		const TetModel::ParticleMesh& mesh = tm->getParticleMesh();
		for (unsigned int i = 0; i < nTets; i++)
		{
			const unsigned int v1 = tets[4 * i] + offset;
			const unsigned int v2 = tets[4 * i + 1] + offset;
			const unsigned int v3 = tets[4 * i + 2] + offset;
			const unsigned int v4 = tets[4 * i + 3] + offset;

			addStrainTetConstraint(v1, v2, v3, v4, stiffness, stiffness, normalizeStretch, normalizeStretch);
		}
	}
	else if (solidMethod == 5)
	{
		const TetModel::ParticleMesh& mesh = tm->getParticleMesh();
		for (unsigned int i = 0; i < nTets; i++)
		{
			const unsigned int v[4] = { tets[4 * i] + offset,
										tets[4 * i + 1] + offset,
										tets[4 * i + 2] + offset,
										tets[4 * i + 3] + offset };
			// Important: Divide position correction by the number of clusters 
			// which contain the vertex.
			const unsigned int nc[4] = { (unsigned int)vTets[v[0]-offset].size(), (unsigned int)vTets[v[1] - offset].size(), (unsigned int)vTets[v[2] - offset].size(), (unsigned int)vTets[v[3] - offset].size() };
			addShapeMatchingConstraint(4, v, nc, stiffness);
		}
	}
	else if (solidMethod == 6)
	{
		const unsigned int offset = tm->getIndexOffset();
		const unsigned int nEdges = tm->getParticleMesh().numEdges();
		const Utilities::IndexedTetMesh::Edge* edges = tm->getParticleMesh().getEdges().data();
		for (unsigned int i = 0; i < nEdges; i++)
		{
			const unsigned int v1 = edges[i].m_vert[0] + offset;
			const unsigned int v2 = edges[i].m_vert[1] + offset;

			addDistanceConstraint_XPBD(v1, v2, stiffness);
		}

		for (unsigned int i = 0; i < nTets; i++)
		{
			const unsigned int v1 = tets[4 * i] + offset;
			const unsigned int v2 = tets[4 * i + 1] + offset;
			const unsigned int v3 = tets[4 * i + 2] + offset;
			const unsigned int v4 = tets[4 * i + 3] + offset;

			addVolumeConstraint_XPBD(v1, v2, v3, v4, volumeStiffness);
		}
	}
}

void PBD::SimulationModel::setClothStiffness(Real val)
{
	m_cloth_stiffness = val;
	setConstraintValue<DistanceConstraint, Real, &DistanceConstraint::m_stiffness>(m_cloth_stiffness);
	setConstraintValue<DistanceConstraint_XPBD, Real, &DistanceConstraint_XPBD::m_stiffness>(m_cloth_stiffness);
}

void PBD::SimulationModel::setClothStiffnessXX(Real val)
{
	m_cloth_xxStiffness = val;
	setConstraintValue<FEMTriangleConstraint, Real, &FEMTriangleConstraint::m_xxStiffness>(val);
	setConstraintValue<StrainTriangleConstraint, Real, &StrainTriangleConstraint::m_xxStiffness>(val);
}

void PBD::SimulationModel::setClothStiffnessYY(Real val)
{
	m_cloth_yyStiffness = val;
	setConstraintValue<FEMTriangleConstraint, Real, &FEMTriangleConstraint::m_xxStiffness>(val);
	setConstraintValue<StrainTriangleConstraint, Real, &StrainTriangleConstraint::m_xxStiffness>(val);
}

void PBD::SimulationModel::setClothStiffnessXY(Real val)
{
	m_cloth_xyStiffness = val;
	setConstraintValue<FEMTriangleConstraint, Real, &FEMTriangleConstraint::m_xxStiffness>(val);
	setConstraintValue<StrainTriangleConstraint, Real, &StrainTriangleConstraint::m_xxStiffness>(val);
}

void PBD::SimulationModel::setClothPoissonRatioXY(Real val)
{
	m_cloth_xyPoissonRatio = val;
	setConstraintValue<FEMTriangleConstraint, Real, &FEMTriangleConstraint::m_xyPoissonRatio>(val);
}

void PBD::SimulationModel::setClothPoissonRatioYX(Real val)
{
	m_cloth_yxPoissonRatio = val;
	setConstraintValue<FEMTriangleConstraint, Real, &FEMTriangleConstraint::m_yxPoissonRatio>(val);
}

void PBD::SimulationModel::setClothBendingStiffness(Real val)
{
	m_cloth_bendingStiffness = val;
	setConstraintValue<DihedralConstraint, Real, &DihedralConstraint::m_stiffness>(val);
	setConstraintValue<IsometricBendingConstraint, Real, &IsometricBendingConstraint::m_stiffness>(val);
	setConstraintValue<IsometricBendingConstraint_XPBD, Real, &IsometricBendingConstraint_XPBD::m_stiffness>(val);
}

void PBD::SimulationModel::setClothNormalizeStretch(bool val)
{
	m_cloth_normalizeStretch = val;
	setConstraintValue<StrainTriangleConstraint, bool, &StrainTriangleConstraint::m_normalizeStretch>(val);
}

void PBD::SimulationModel::setClothNormalizeShear(bool val)
{
	m_cloth_normalizeShear = val;
	setConstraintValue<StrainTriangleConstraint, bool, &StrainTriangleConstraint::m_normalizeShear>(val);
}

void PBD::SimulationModel::setSolidStiffness(Real val)
{
	m_solid_stiffness = val;
	setConstraintValue<FEMTetConstraint, Real, &FEMTetConstraint::m_stiffness>(val);
	setConstraintValue<XPBD_FEMTetConstraint, Real, &XPBD_FEMTetConstraint::m_stiffness>(val);
	setConstraintValue<StrainTetConstraint, Real, &StrainTetConstraint::m_stretchStiffness>(val);
	setConstraintValue<StrainTetConstraint, Real, &StrainTetConstraint::m_shearStiffness>(val);
	setConstraintValue<DistanceConstraint, Real, &DistanceConstraint::m_stiffness>(val);
	setConstraintValue<DistanceConstraint_XPBD, Real, &DistanceConstraint_XPBD::m_stiffness>(val);
	setConstraintValue<ShapeMatchingConstraint, Real, &ShapeMatchingConstraint::m_stiffness>(val);
}

void PBD::SimulationModel::setSolidPoissonRatio(Real val)
{
	m_solid_poissonRatio = val;
	setConstraintValue<FEMTetConstraint, Real, &FEMTetConstraint::m_poissonRatio>(val);
	setConstraintValue<XPBD_FEMTetConstraint, Real, &XPBD_FEMTetConstraint::m_poissonRatio>(val);
}

void PBD::SimulationModel::setSolidVolumeStiffness(Real val)
{
	m_solid_volumeStiffness = val;
	setConstraintValue<VolumeConstraint, Real, &VolumeConstraint::m_stiffness>(val);
	setConstraintValue<VolumeConstraint_XPBD, Real, &VolumeConstraint_XPBD::m_stiffness>(val);
}

void PBD::SimulationModel::setSolidNormalizeStretch(bool val)
{
	m_solid_normalizeStretch = val;
	setConstraintValue<StrainTetConstraint, bool, &StrainTetConstraint::m_normalizeStretch>(val);
}

void PBD::SimulationModel::setSolidNormalizeShear(bool val)
{
	m_solid_normalizeShear = val;
	setConstraintValue<StrainTetConstraint, bool, &StrainTetConstraint::m_normalizeShear>(val);
}

void PBD::SimulationModel::setRodStretchingStiffness(Real val)
{
	m_rod_stretchingStiffness = val;
	setConstraintValue<StretchShearConstraint, Real, &StretchShearConstraint::m_stretchingStiffness>(val);
	setConstraintValue<DistanceConstraint, Real, &DistanceConstraint::m_stiffness>(val);
	setConstraintValue<DistanceConstraint_XPBD, Real, &DistanceConstraint_XPBD::m_stiffness>(val);
}

void PBD::SimulationModel::setRodShearingStiffnessX(Real val)
{
	m_rod_shearingStiffnessX = val;
	setConstraintValue<StretchShearConstraint, Real, &StretchShearConstraint::m_shearingStiffness1>(val);
}

void PBD::SimulationModel::setRodShearingStiffnessY(Real val)
{
	m_rod_shearingStiffnessY = val;
	setConstraintValue<StretchShearConstraint, Real, &StretchShearConstraint::m_shearingStiffness2>(val);
}

void PBD::SimulationModel::setRodBendingStiffnessX(Real val)
{
	m_rod_bendingStiffnessX = val;
	setConstraintValue<BendTwistConstraint, Real, &BendTwistConstraint::m_bendingStiffness1>(val);
}

void PBD::SimulationModel::setRodBendingStiffnessY(Real val)
{
	m_rod_bendingStiffnessY = val;
	setConstraintValue<BendTwistConstraint, Real, &BendTwistConstraint::m_bendingStiffness2>(val);
}

void PBD::SimulationModel::setRodTwistingStiffness(Real val)
{
	m_rod_twistingStiffness = val;
	setConstraintValue<BendTwistConstraint, Real, &BendTwistConstraint::m_twistingStiffness>(val);
}
