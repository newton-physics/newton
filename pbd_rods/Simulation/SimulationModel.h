#ifndef __SIMULATIONMODEL_H__
#define __SIMULATIONMODEL_H__

#include "Common/Common.h"
#include <vector>
#include "Simulation/RigidBody.h"
#include "Simulation/ParticleData.h"
#include "TriangleModel.h"
#include "TetModel.h"
#include "LineModel.h"
#include "ParameterObject.h"

namespace PBD 
{	
	class Constraint;

	class SimulationModel : public GenParam::ParameterObject
	{
		public:
			static int CLOTH_SIMULATION_METHOD;
			static int ENUM_CLOTHSIM_NONE;
			static int ENUM_CLOTHSIM_DISTANCE_CONSTRAINTS;
			static int ENUM_CLOTHSIM_FEM_PBD;
			static int ENUM_CLOTHSIM_SBD;
			static int ENUM_CLOTHSIM_DISTANCE_CONSTRAINTS_XPBD;

			static int CLOTH_STIFFNESS;
			static int CLOTH_STIFFNESS_XX;
			static int CLOTH_STIFFNESS_YY;
			static int CLOTH_STIFFNESS_XY;
			static int CLOTH_POISSON_RATIO_XY;
			static int CLOTH_POISSON_RATIO_YX;

			static int CLOTH_BENDING_METHOD;
			static int ENUM_CLOTH_BENDING_NONE;
			static int ENUM_CLOTH_BENDING_DIHEDRAL_ANGLE;
			static int ENUM_CLOTH_BENDING_ISOMETRIC;
			static int ENUM_CLOTH_BENDING_ISOMETRIX_XPBD;
			static int CLOTH_BENDING_STIFFNESS;
			static int CLOTH_NORMALIZE_STRETCH;
			static int CLOTH_NORMALIZE_SHEAR;

			static int SOLID_SIMULATION_METHOD;
			static int ENUM_SOLIDSIM_NONE;
			static int ENUM_SOLIDSIM_DISTANCE_CONSTRAINTS;
			static int ENUM_SOLIDSIM_FEM_PBD;
			static int ENUM_SOLIDSIM_FEM_XPBD;
			static int ENUM_SOLIDSIM_SBD;
			static int ENUM_SOLIDSIM_SHAPE_MATCHING;
			static int ENUM_SOLIDSIM_DISTANCE_CONSTRAINTS_XPBD;

			static int SOLID_STIFFNESS;
			static int SOLID_POISSON_RATIO;
			static int SOLID_VOLUME_STIFFNESS;
			static int SOLID_NORMALIZE_STRETCH;
			static int SOLID_NORMALIZE_SHEAR;

			static int ROD_STRETCHING_STIFFNESS;
			static int ROD_SHEARING_STIFFNESS_X;
			static int ROD_SHEARING_STIFFNESS_Y;
			static int ROD_BENDING_STIFFNESS_X;
			static int ROD_BENDING_STIFFNESS_Y;
			static int ROD_TWISTING_STIFFNESS;

			static int CONTACT_STIFFNESS_RB;
			static int CONTACT_STIFFNESS_PARTICLE_RB;

			SimulationModel();
			SimulationModel(const SimulationModel&) = delete;
			SimulationModel& operator=(const SimulationModel&) = delete;
			virtual ~SimulationModel();

			void init();
			virtual void initParameters();

			typedef std::vector<Constraint*> ConstraintVector;
			typedef std::vector<RigidBodyContactConstraint> RigidBodyContactConstraintVector;
			typedef std::vector<ParticleRigidBodyContactConstraint> ParticleRigidBodyContactConstraintVector;
			typedef std::vector<ParticleTetContactConstraint> ParticleSolidContactConstraintVector;
			typedef std::vector<RigidBody*> RigidBodyVector;
			typedef std::vector<TriangleModel*> TriangleModelVector;
			typedef std::vector<TetModel*> TetModelVector;
			typedef std::vector<LineModel*> LineModelVector;
			typedef std::vector<unsigned int> ConstraintGroup;
			typedef std::vector<ConstraintGroup> ConstraintGroupVector;


		protected:
			RigidBodyVector m_rigidBodies;
			TriangleModelVector m_triangleModels;
			TetModelVector m_tetModels;
			LineModelVector m_lineModels;
			ParticleData m_particles;
			OrientationData m_orientations;
			ConstraintVector m_constraints;
			RigidBodyContactConstraintVector m_rigidBodyContactConstraints;
			ParticleRigidBodyContactConstraintVector m_particleRigidBodyContactConstraints;
			ParticleSolidContactConstraintVector m_particleSolidContactConstraints;
			ConstraintGroupVector m_constraintGroups;

			int m_clothSimulationMethod;
			int m_clothBendingMethod;
			Real m_cloth_stiffness;
			Real m_cloth_bendingStiffness;
			Real m_cloth_xxStiffness;
			Real m_cloth_yyStiffness;
			Real m_cloth_xyStiffness;
			Real m_cloth_xyPoissonRatio;
			Real m_cloth_yxPoissonRatio;
			bool m_cloth_normalizeStretch;
			bool m_cloth_normalizeShear;

			int m_solidSimulationMethod;
			Real m_solid_stiffness;
			Real m_solid_poissonRatio;
			Real m_solid_volumeStiffness;
			bool m_solid_normalizeStretch;
			bool m_solid_normalizeShear;

			Real m_rod_stretchingStiffness;
			Real m_rod_shearingStiffnessX;
			Real m_rod_shearingStiffnessY;
			Real m_rod_bendingStiffnessX;
			Real m_rod_bendingStiffnessY;
			Real m_rod_twistingStiffness;

			Real m_contactStiffnessRigidBody;
			Real m_contactStiffnessParticleRigidBody;

			std::function<void()> m_clothSimMethodChanged;
			std::function<void()> m_clothBendingMethodChanged;
			std::function<void()> m_solidSimMethodChanged;

	public:
			virtual void reset();			
			virtual void cleanup();

			RigidBodyVector &getRigidBodies();
			ParticleData &getParticles();
			OrientationData &getOrientations();
			TriangleModelVector &getTriangleModels();
			TetModelVector &getTetModels();
			LineModelVector &getLineModels();
			ConstraintVector &getConstraints();
			RigidBodyContactConstraintVector &getRigidBodyContactConstraints();
			ParticleRigidBodyContactConstraintVector &getParticleRigidBodyContactConstraints();
			ParticleSolidContactConstraintVector &getParticleSolidContactConstraints();
			ConstraintGroupVector &getConstraintGroups();
			bool m_groupsInitialized;

			void resetContacts();

			void addTriangleModel(
				const unsigned int nPoints,
				const unsigned int nFaces,
				Vector3r *points,
				unsigned int* indices,
				const TriangleModel::ParticleMesh::UVIndices& uvIndices,
				const TriangleModel::ParticleMesh::UVs& uvs);
			void addRegularTriangleModel(const int width, const int height,
				const Vector3r& translation = Vector3r::Zero(), 
				const Matrix3r& rotation = Matrix3r::Identity(), 
				const Vector2r& scale = Vector2r::Ones());

			void addTetModel(
				const unsigned int nPoints, 
				const unsigned int nTets, 
				Vector3r *points,
				unsigned int* indices);
			void addRegularTetModel(const int width, const int height, const int depth,
				const Vector3r& translation = Vector3r::Zero(),
				const Matrix3r& rotation = Matrix3r::Identity(), 
				const Vector3r& scale = Vector3r::Ones());

			void addLineModel(
				const unsigned int nPoints,
				const unsigned int nQuaternions,
				Vector3r *points,
				Quaternionr *quaternions,
				unsigned int *indices,
				unsigned int *indicesQuaternions);

			void updateConstraints();
			void initConstraintGroups();

			bool addBallJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos);
			bool addBallOnLineJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &dir);
			bool addHingeJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
			bool addTargetAngleMotorHingeJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
			bool addTargetVelocityMotorHingeJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
			bool addUniversalJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis1, const Vector3r &axis2);
			bool addSliderJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &axis);
			bool addTargetPositionMotorSliderJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &axis);
			bool addTargetVelocityMotorSliderJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &axis);
			bool addRigidBodyParticleBallJoint(const unsigned int rbIndex, const unsigned int particleIndex);
			bool addRigidBodySpring(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos1, const Vector3r &pos2, const Real stiffness);
			bool addDistanceJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos1, const Vector3r &pos2);
			bool addDamperJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &axis, const Real stiffness);
			bool addRigidBodyContactConstraint(const unsigned int rbIndex1, const unsigned int rbIndex2, 
					const Vector3r &cp1, const Vector3r &cp2,	
					const Vector3r &normal, const Real dist, 
					const Real restitutionCoeff, const Real frictionCoeff);
			bool addParticleRigidBodyContactConstraint(const unsigned int particleIndex, const unsigned int rbIndex, 
					const Vector3r &cp1, const Vector3r &cp2, 
					const Vector3r &normal, const Real dist,
					const Real restitutionCoeff, const Real frictionCoeff);

			bool addParticleSolidContactConstraint(const unsigned int particleIndex, const unsigned int solidIndex,
				const unsigned int tetIndex, const Vector3r &bary,
				const Vector3r &cp1, const Vector3r &cp2,
				const Vector3r &normal, const Real dist,
				const Real restitutionCoeff, const Real frictionCoeff);

			bool addDistanceConstraint(const unsigned int particle1, const unsigned int particle2, const Real stiffness);
			bool addDistanceConstraint_XPBD(const unsigned int particle1, const unsigned int particle2, const Real stiffness);
			bool addDihedralConstraint(	const unsigned int particle1, const unsigned int particle2,
										const unsigned int particle3, const unsigned int particle4, const Real stiffness);
			bool addIsometricBendingConstraint(const unsigned int particle1, const unsigned int particle2,
										const unsigned int particle3, const unsigned int particle4, const Real stiffness);
			bool addIsometricBendingConstraint_XPBD(const unsigned int particle1, const unsigned int particle2,
										const unsigned int particle3, const unsigned int particle4, const Real stiffness);
			bool addFEMTriangleConstraint(const unsigned int particle1, const unsigned int particle2, const unsigned int particle3, 
				const Real xxStiffness, const Real yyStiffness, const Real xyStiffness,
				const Real xyPoissonRatio, const Real yxPoissonRatio);
			bool addStrainTriangleConstraint(const unsigned int particle1, const unsigned int particle2, 
				const unsigned int particle3, const Real xxStiffness, const Real yyStiffness, const Real xyStiffness,
				const bool normalizeStretch, const bool normalizeShear);
			bool addVolumeConstraint(const unsigned int particle1, const unsigned int particle2,
									const unsigned int particle3, const unsigned int particle4, const Real stiffness);
			bool addVolumeConstraint_XPBD(const unsigned int particle1, const unsigned int particle2,
									const unsigned int particle3, const unsigned int particle4, const Real stiffness);
			bool addFEMTetConstraint(const unsigned int particle1, const unsigned int particle2,
									const unsigned int particle3, const unsigned int particle4, 
									const Real stiffness, const Real poissonRatio);
			bool addFEMTetConstraint_XPBD(const unsigned int particle1, const unsigned int particle2,
									const unsigned int particle3, const unsigned int particle4, 
									const Real stiffness, const Real poissonRatio);
			bool addStrainTetConstraint(const unsigned int particle1, const unsigned int particle2,
									const unsigned int particle3, const unsigned int particle4, 
									const Real stretchStiffness, const Real shearStiffness,
									const bool normalizeStretch, const bool normalizeShear);
			bool addShapeMatchingConstraint(const unsigned int numberOfParticles, const unsigned int particleIndices[], const unsigned int numClusters[], const Real stiffness);
			bool addStretchShearConstraint(const unsigned int particle1, const unsigned int particle2, 
				const unsigned int quaternion1, const Real stretchingStiffness,
				const Real shearingStiffness1, const Real shearingStiffness2);
			bool addBendTwistConstraint(const unsigned int quaternion1, const unsigned int quaternion2, 
				const Real twistingStiffness, const Real bendingStiffness1, const Real bendingStiffness2);
			bool addStretchBendingTwistingConstraint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Real averageRadius, const Real averageSegmentLength, const Real youngsModulus, const Real torsionModulus);
			bool addDirectPositionBasedSolverForStiffRodsConstraint(const std::vector<std::pair<unsigned int, unsigned int>> & jointSegmentIndices, const std::vector<Vector3r> &jointPositions, const std::vector<Real> &averageRadii, const std::vector<Real> &averageSegmentLengths, const std::vector<Real> &youngsModuli, const std::vector<Real> &torsionModuli);

			Real getContactStiffnessRigidBody() const { return m_contactStiffnessRigidBody; }
			void setContactStiffnessRigidBody(Real val) { m_contactStiffnessRigidBody = val; }
			Real getContactStiffnessParticleRigidBody() const { return m_contactStiffnessParticleRigidBody; }
			void setContactStiffnessParticleRigidBody(Real val) { m_contactStiffnessParticleRigidBody = val; }
		
			void addClothConstraints(const TriangleModel* tm, const unsigned int clothMethod, 
				const Real distanceStiffness, const Real xxStiffness, const Real yyStiffness,
				const Real xyStiffness, const Real xyPoissonRatio, const Real yxPoissonRatio,
				const bool normalizeStretch, const bool normalizeShear);
			void addBendingConstraints(const TriangleModel* tm, const unsigned int bendingMethod, const Real stiffness);
			void addSolidConstraints(const TetModel* tm, const unsigned int solidMethod, const Real stiffness, 
				const Real poissonRatio, const Real volumeStiffness, 
				const bool normalizeStretch, const bool normalizeShear);

			template<typename ConstraintType, typename T, T ConstraintType::* MemPtr>
			void setConstraintValue(const T v)
			{
				for (auto i = 0; i < m_constraints.size(); i++)
				{
					ConstraintType* c = dynamic_cast<ConstraintType*>(m_constraints[i]);
					if (c != nullptr)
						c->*MemPtr = v;
				}
			}

			void setClothSimulationMethodChangedCallback(std::function<void()> const& callBackFct) { m_clothSimMethodChanged = callBackFct;	}
			void setClothBendingMethodChangedCallback(std::function<void()> const& callBackFct) { m_clothBendingMethodChanged = callBackFct; }
			void setSolidSimulationMethodChangedCallback(std::function<void()> const& callBackFct) { m_solidSimMethodChanged = callBackFct; }

			virtual int getClothSimulationMethod() { return m_clothSimulationMethod; }
			virtual void setClothSimulationMethod(const int val);
			virtual int getClothBendingMethod() { return m_clothBendingMethod; }
			virtual void setClothBendingMethod(const int val);
			virtual Real getClothStiffness() { return m_cloth_stiffness; }
			virtual void setClothStiffness(Real val);
			virtual Real getClothStiffnessXX() { return m_cloth_xxStiffness; }
			virtual void setClothStiffnessXX(Real val);
			virtual Real getClothStiffnessYY() { return m_cloth_yyStiffness; }
			virtual void setClothStiffnessYY(Real val);
			virtual Real getClothStiffnessXY() { return m_cloth_xyStiffness; }
			virtual void setClothStiffnessXY(Real val);
			virtual Real getClothPoissonRatioXY() { return m_cloth_xyPoissonRatio; }
			virtual void setClothPoissonRatioXY(Real val);
			virtual Real getClothPoissonRatioYX() { return m_cloth_yxPoissonRatio; }
			virtual void setClothPoissonRatioYX(Real val);
			virtual Real getClothBendingStiffness() { return m_cloth_bendingStiffness; }
			virtual void setClothBendingStiffness(Real val);
			virtual bool getClothNormalizeStretch() { return m_cloth_normalizeStretch; }
			virtual void setClothNormalizeStretch(bool val);
			virtual bool getClothNormalizeShear() { return m_cloth_normalizeShear; }
			virtual void setClothNormalizeShear(bool val);

			virtual int getSolidSimulationMethod() { return m_solidSimulationMethod; }
			virtual void setSolidSimulationMethod(const int val);
			virtual Real getSolidStiffness() { return m_solid_stiffness; }
			virtual void setSolidStiffness(Real val);
			virtual Real getSolidPoissonRatio() { return m_solid_poissonRatio; }
			virtual void setSolidPoissonRatio(Real val);
			virtual Real getSolidVolumeStiffness() { return m_solid_volumeStiffness; }
			virtual void setSolidVolumeStiffness(Real val);
			virtual bool getSolidNormalizeStretch() { return m_solid_normalizeStretch; }
			virtual void setSolidNormalizeStretch(bool val);
			virtual bool getSolidNormalizeShear() { return m_solid_normalizeShear; }
			virtual void setSolidNormalizeShear(bool val);

			virtual Real getRodStretchingStiffness() { return m_rod_stretchingStiffness; }
			virtual void setRodStretchingStiffness(Real val);
			virtual Real getRodShearingStiffnessX() { return m_rod_shearingStiffnessX; }
			virtual void setRodShearingStiffnessX(Real val);
			virtual Real getRodShearingStiffnessY() { return m_rod_shearingStiffnessY; }
			virtual void setRodShearingStiffnessY(Real val);
			virtual Real getRodBendingStiffnessX() { return m_rod_bendingStiffnessX; }
			virtual void setRodBendingStiffnessX(Real val);
			virtual Real getRodBendingStiffnessY() { return m_rod_bendingStiffnessY; }
			virtual void setRodBendingStiffnessY(Real val);
			virtual Real getRodTwistingStiffness() { return m_rod_twistingStiffness; }
			virtual void setRodTwistingStiffness(Real val);
	};
}

#endif