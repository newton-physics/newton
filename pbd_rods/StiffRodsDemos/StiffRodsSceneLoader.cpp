#include "StiffRodsSceneLoader.h"
#include "Utils/FileSystem.h"
#include "Utils/Logger.h"

using namespace PBD;
using namespace Utilities;

void StiffRodsSceneLoader::parseStiffRodsData(const nlohmann::json &child, SceneData &sceneData, const std::string &basePath)
{
	if (m_json.find("RodAnimators") != m_json.end())
		parseRodAnimators(m_json, "RodAnimators", sceneData);

	// parse cosserat joint data
	if (m_json.find("CosseratJoints") != m_json.end())
		parseStretchBendingTwistingConstraints(m_json, "CosseratJoints", sceneData);

	// parse tree data
	if (m_json.find("TreeModels") != m_json.end())
		parseTreeModels(m_json, "TreeModels", sceneData);

	// parse skinning mesh data
	if (m_json.find("SkinningMeshes") != m_json.end())
		parseSkinningMeshes(m_json, "SkinningMeshes", sceneData, basePath);

	if (m_json.find("CompactTreeModels") != m_json.end())
		parseCompactTreeModels(m_json, "CompactTreeModels", sceneData, basePath);

}

void StiffRodsSceneLoader::parseSkinningMeshes(const nlohmann::json &j, const std::string &key, SceneData & sceneData, const std::string &basePath)
{
	const nlohmann::json &child = j[key];
	{
		RodSceneData &rodSceneData = static_cast<RodSceneData&>(sceneData);
		rodSceneData.m_SkinningMeshes.reserve(child.size());

		for (auto &sMesh : child)
		{
			std::string meshFileName;
			if (readValue<std::string>(sMesh, "meshFile", meshFileName))
			{
				if (FileSystem::isRelativePath(meshFileName))
				{
					meshFileName = basePath + "/" + meshFileName;
				}

				rodSceneData.m_SkinningMeshes.resize(rodSceneData.m_SkinningMeshes.size() + 1);
				SkinningMeshData &smd = rodSceneData.m_SkinningMeshes.back();

				// id
				smd.m_Id = 0;
				readValue(sMesh, "id", smd.m_Id);

				// file name
				smd.m_MeshFile = meshFileName;

				// translation
				smd.m_X.setZero();
				readVector(sMesh, "translation", smd.m_X);

				// Orientation
				Vector3r axis;
				axis.setZero();
				Real angle = 0.0;
				if (readVector(sMesh, "rotationAxis", axis) &&
					readValue<Real>(sMesh, "rotationAngle", angle))
				{
					axis.normalize();
					smd.m_Q = Quaternionr(AngleAxisr(angle, axis));
				}
				else
					smd.m_Q = Quaternionr::Identity();

				// scaling
				smd.m_Scale = Vector3r(1.0, 1.0, 1.0);
				readVector(sMesh, "scale", smd.m_Scale);

				// vertex ids
				std::vector<unsigned int> &vertIDs(smd.m_VertexIDs);
				if (sMesh.find("skinnedVertsIds") != sMesh.end())
				{
					vertIDs.reserve(sMesh["skinnedVertsIds"].size());
					for (auto &skinnedVertID : sMesh["skinnedVertsIds"])
					{
						vertIDs.push_back(skinnedVertID.get<unsigned int>());
					}
				}

				// body ids
				std::vector<std::vector<unsigned int>> &rbIdsOfVerts(smd.m_BodyIDs);
				if (sMesh.find("rbIdsOfVerts") != sMesh.end())
				{
					rbIdsOfVerts.reserve(sMesh["rbIdsOfVerts"].size());
					for (auto &rbIDs : sMesh["rbIdsOfVerts"])
					{
						std::vector<unsigned int> ids;
						for (auto &id : rbIDs)
						{
							ids.push_back(id.get<unsigned int>());
						}
						rbIdsOfVerts.push_back(ids);
					}
				}

				// skinning Weights
				std::vector<std::vector<Real>> &skinningWeights(smd.m_SkinningWeights);
				if (sMesh.find("weightsOfVerts") != sMesh.end())
				{
					for (auto &vWeightsData : sMesh["weightsOfVerts"])
					{
						std::vector<Real> weights;
						for (auto &weight : vWeightsData)
						{
							weights.push_back(weight.get<Real>());
						}
						skinningWeights.push_back(weights);
					}
				}
			}

		}
	}
}

void StiffRodsSceneLoader::parseTreeModels(const nlohmann::json &j, const std::string &key, SceneData & sceneData)
{
	const nlohmann::json &child = j[key];
	{
		RodSceneData &rodSceneData = static_cast<RodSceneData&>(sceneData);
		rodSceneData.m_TreeModels.reserve(child.size());

		for (auto &tm : child)
		{
			rodSceneData.m_TreeModels.resize(rodSceneData.m_TreeModels.size() + 1);
			TreeModelData &tmd = rodSceneData.m_TreeModels.back();

			// id
			tmd.m_Id = 0;
			readValue(tm, "id", tmd.m_Id);

			// translation
			tmd.m_X.setZero();
			readVector(tm, "translation", tmd.m_X);

			// Orientation
			Vector3r axis;
			axis.setZero();
			Real angle = 0.0;
			if (readVector(tm, "rotationAxis", axis) &&
				readValue<Real>(tm, "rotationAngle", angle))
			{
				axis.normalize();
				tmd.m_Q = Quaternionr(AngleAxisr(angle, axis));
			}
			else
				tmd.m_Q = Quaternionr::Identity();

			// youngs modulus
			tmd.m_YoungsModulus = 0.;
			readValue(tm, "youngsModulus", tmd.m_YoungsModulus);

			// torsion modulus
			tmd.m_TorsionModulus = 0.;
			readValue(tm, "torsionModulus", tmd.m_TorsionModulus);

			// rigid body IDs
			if (tm.find("rbIds") != tm.end())
			{
				for (auto& id : tm["rbIds"])
				{
					tmd.m_SegmentIds.insert(id.get<unsigned int>());
				}
			}

			// joint IDs
			if (tm.find("jIds") != tm.end())
			{
				for (auto& id : tm["jIds"])
				{
					tmd.m_StretchBendingTwistingConstraintIds.insert(id.get<unsigned int>());
				}
			}

			// static Segment
			if (tm.find("staticSegments") != tm.end())
			{
				for (auto& id : tm["staticSegments"])
				{
					tmd.m_StaticSegments.insert(id.get<unsigned int>());
				}
			}

			// id
			tmd.m_SolverType = 12; // direct solver type == 12
			readValue(tm, "solverType", tmd.m_SolverType);
		}
	}
}

void StiffRodsSceneLoader::parseCompactTreeModels(const nlohmann::json &j, const std::string &key, SceneData & sceneData, const std::string &basePath)
{
	{
		const nlohmann::json &child = j[key];

		RodSceneData &rodSceneData = static_cast<RodSceneData&>(sceneData);
		rodSceneData.m_CompactTreeModels.reserve(child.size());

		for (auto &tm : child)
		{
			rodSceneData.m_CompactTreeModels.resize(rodSceneData.m_CompactTreeModels.size() + 1);
			CompactTreeModelData &ctmd = rodSceneData.m_CompactTreeModels.back();

			// id
			ctmd.m_Id = 0;
			readValue(tm, "id", ctmd.m_Id);

			// translation
			ctmd.m_X.setZero();
			readVector(tm, "translation", ctmd.m_X);

			// Orientation
			Vector3r axis;
			axis.setZero();
			Real angle = 0.0;
			if (readVector(tm, "rotationAxis", axis) &&
				readValue<Real>(tm, "rotationAngle", angle))
			{
				axis.normalize();
				ctmd.m_Q = Quaternionr(AngleAxisr(angle, axis));
			}
			else
				ctmd.m_Q = Quaternionr::Identity();

			// youngs modulus
			ctmd.m_YoungsModulus = 0.;
			readValue(tm, "youngsModulus", ctmd.m_YoungsModulus);

			// torsion modulus
			ctmd.m_TorsionModulus = 0.;
			readValue(tm, "torsionModulus", ctmd.m_TorsionModulus);
			
			// static Segment
			if (tm.find("staticSegments") != tm.end())
			{
				for (auto& id : tm["staticSegments"])
				{
					ctmd.m_StaticSegments.insert(id.get<unsigned int>());
				}
			}

			// solver type
			ctmd.m_SolverType = 12;
			readValue(tm, "solverType", ctmd.m_SolverType);

			// segment positions
			if (tm.find("segmentPositions") != tm.end())
			{
				for (auto &position : tm["segmentPositions"])
				{
					ctmd.m_SegmentPositions.push_back
						(readVectorValues<Real, 3>(position));
				}
			}

			// segment Orientations
			std::vector<Vector3r> tmpSegmentOrientationAxes;
			std::vector<Real> tmpSegmentOrientationAngles;

			if (tm.find("segmentOrientationAxes") != tm.end())
			{
				for (auto &axis : tm["segmentOrientationAxes"])
				{
					tmpSegmentOrientationAxes.push_back
						(readVectorValues<Real, 3>(axis));
				}
			}

			if (tm.find("segmentOrientationAngles") != tm.end())
			{
				for (auto &angle : tm["segmentOrientationAngles"])
				{
					tmpSegmentOrientationAngles.push_back
						(angle.get<Real>());
				}
			}

			for (size_t idx(0);
				idx < tmpSegmentOrientationAngles.size() && idx < tmpSegmentOrientationAxes.size();
				++idx)
			{
				Real angle(tmpSegmentOrientationAngles[idx]);
				Vector3r & axis(tmpSegmentOrientationAxes[idx]);
				axis.normalize();
				ctmd.m_SegmentOrientations.push_back(Quaternionr(AngleAxisr(angle, axis)));
			}

			// segment scales
			if (tm.find("segmentScales") != tm.end())
			{
				for (auto &scale : tm["segmentScales"])
				{
					ctmd.m_SegmentScales.push_back
						(readVectorValues<Real, 3>(scale));
				}
			}

			// segment densities
			if (tm.find("segmentDensities") != tm.end())
			{
				for (auto &density : tm["segmentDensities"])
				{
					ctmd.m_SegmentDensities.push_back
						(density.get<Real>());
				}
			}

			// segment model files
			if (tm.find("segmentModelFiles") != tm.end())
			{
				std::string geomFileName;
				for (auto &modelFile : tm["segmentModelFiles"])
				{
					geomFileName = modelFile.get<std::string>();
					if (FileSystem::isRelativePath(geomFileName))
					{
						geomFileName = basePath + "/" + geomFileName;
					}
					ctmd.m_SegmentModelFiles.push_back(geomFileName);					
				}
			}

			// segment friction coefficients
			if (tm.find("segmentFrictionCoefficients") != tm.end())
			{
				for (auto &fc : tm["segmentFrictionCoefficients"])
				{
					ctmd.m_SegmentFrictionCoefficients.push_back
						(fc.get<Real>());
				}
			}

			// segment restitution coefficients
			if (tm.find("segmentRestitutionCoefficients") != tm.end())
			{
				for (auto &rc : tm["segmentRestitutionCoefficients"])
				{
					ctmd.m_SegmentRestitutioCoefficients.push_back
						(rc.get<Real>());
				}
			}

			// segment collision category bits
			if (tm.find("segmentCollisionCategoryBits") != tm.end())
			{
				for (auto &ccb : tm["segmentCollisionCategoryBits"])
				{
					ctmd.m_SegmentCollisionCategoryBits.push_back
						(ccb.get<short>());
				}
			}

			// segment collision filter mask bits
			if (tm.find("segmentCollisionFilterMaskBits") != tm.end())
			{
				for (auto &cfmb : tm["segmentCollisionFilterMaskBits"])
				{
					ctmd.m_SegmentCollisionFilterMaskBits.push_back
						(cfmb.get<short>());
				}
			}

			// joint positions
			if (tm.find("jointPositions") != tm.end())
			{
				for (auto &jp : tm["jointPositions"])
				{
					ctmd.m_SBTConstraintPositions.push_back
						(readVectorValues<Real, 3>(jp));
				}
			}

			// joint bodies indices
			if (tm.find("jointBodiesIndices") != tm.end())
			{
				for (auto &jbi : tm["jointBodiesIndices"])
				{
					std::vector<unsigned int> bodyIndices;
					for (auto& item : jbi)
					{
						bodyIndices.push_back(item.get<unsigned int>());
					}
					ctmd.m_SBTConstraintBodiesIndices.push_back(bodyIndices);
				}
			}

			// joint stiffness modifier
			if (tm.find("jointStiffnessModifier") != tm.end())
			{
				for (auto &jsm : tm["jointStiffnessModifier"])
				{
					ctmd.m_SBTConstraintStiffnessModifier.push_back
						(readVectorValues<Real, 3>(jsm));
				}
			}
		}
	}
}

void StiffRodsSceneLoader::parseStretchBendingTwistingConstraints(const nlohmann::json &j, const std::string &key, SceneData & sceneData)
{
	{
		const nlohmann::json &child = j[key];
		RodSceneData &rodSceneData = static_cast<RodSceneData&>(sceneData);
		rodSceneData.m_StretchBendingTwistingConstraints.reserve(static_cast<size_t>(child.size()));

		for (auto &joint : child)
		{
			rodSceneData.m_StretchBendingTwistingConstraints.resize(rodSceneData.m_StretchBendingTwistingConstraints.size() + 1);
			StretchBendingTwistingConstraintData &cj = rodSceneData.m_StretchBendingTwistingConstraints.back();

			// id
			cj.m_Id = 0;
			readValue(joint, "id", cj.m_Id);

			// first rigid body Id
			cj.m_SegmentID[0] = 0;
			readValue(joint, "bodyID1", cj.m_SegmentID[0]);

			// second rigid body Id
			cj.m_SegmentID[1] = 0;
			readValue(joint, "bodyID2", cj.m_SegmentID[1]);

			// joint position
			cj.m_Position = Vector3r(0., 0., 0.);
			readVector(joint, "position", cj.m_Position);

			// youngs modulus
			cj.m_YoungsModulus = 0.;
			readValue(joint, "youngsModulus", cj.m_YoungsModulus);

			// torsion modulus
			cj.m_TorsionModulus = 0.;
			readValue(joint, "torsionModulus", cj.m_TorsionModulus);

			// stiffness modifier
			cj.m_StiffnessModifier = Vector3r(0., 0., 0.);
			readVector(joint, "stiffnessModifier", cj.m_StiffnessModifier);
		}
	}
}

void StiffRodsSceneLoader::parseRodAnimators(const nlohmann::json &j, const std::string &key, SceneData & sceneData)
{
	{
		const nlohmann::json &child = j[key];
		RodSceneData &rodSceneData = static_cast<RodSceneData&>(sceneData);
		rodSceneData.m_RodAnimators.reserve(static_cast<size_t>(child.size()));

		for (auto &anim : child)
		{
			rodSceneData.m_RodAnimators.resize(rodSceneData.m_RodAnimators.size() + 1);
			AnimatorData &ra = rodSceneData.m_RodAnimators.back();

			// id
			ra.m_id = 0;
			readValue(anim, "id", ra.m_id);

			// rod id
			ra.m_RodID = 0;
			readValue(anim, "rodID", ra.m_RodID);

			// segment id
			ra.m_SegmentID = 0;
			readValue(anim, "segmentID", ra.m_SegmentID);

			// starting time
			ra.m_StartingTime = 0;
			readValue(anim, "startingTime", ra.m_StartingTime);

			// ending time
			ra.m_EndingTime = 0;
			readValue(anim, "endingTime", ra.m_EndingTime);

			// type
			ra.m_Type = 0;
			readValue(anim, "type", ra.m_Type);

			// data
			if (anim.find("data") != anim.end())
			{
				for (auto &dataItem : anim["data"])
				{
					ra.m_Data.push_back(readVectorValues<Real, 3>(dataItem));
				}
			}

		}
	}
}

void PBD::StiffRodsSceneLoader::readStiffRodsScene(const std::string &fileName, SceneData &sceneData)
{
	try
	{
		std::ifstream input_file(fileName);
		if (!input_file.is_open())
		{
			std::cerr << "Cannot open file!\n";
			return;
		}
		m_json << input_file;

		std::string basePath = FileSystem::getFilePath(fileName);

		readValue(m_json, "Name", sceneData.m_sceneName);

		sceneData.m_camPosition = Vector3r(5.0, 10.0, 30.0);
		readVector(m_json, "cameraPosition", sceneData.m_camPosition);
		sceneData.m_camLookat = Vector3r(5.0, 0.0, 0.0);
		readVector(m_json, "cameraLookat", sceneData.m_camLookat);

		//////////////////////////////////////////////////////////////////////////
		// read general 
		//////////////////////////////////////////////////////////////////////////
		sceneData.m_timeStepSize = static_cast<Real>(0.005);
		sceneData.m_gravity = Vector3r(0, -static_cast<Real>(9.81), 0);
		if (m_json.find("Simulation") != m_json.end())
			readSimulation(m_json, "Simulation", sceneData);

		//////////////////////////////////////////////////////////////////////////
		// read rigid bodies
		//////////////////////////////////////////////////////////////////////////
		if (m_json.find("RigidBodies") != m_json.end())
			readRigidBodies(m_json, "RigidBodies", basePath, sceneData);

		//////////////////////////////////////////////////////////////////////////
		// read triangle models
		//////////////////////////////////////////////////////////////////////////
		if (m_json.find("TriangleModels") != m_json.end())
			readTriangleModels(m_json, "TriangleModels", basePath, sceneData);

		//////////////////////////////////////////////////////////////////////////
		// read tet models
		//////////////////////////////////////////////////////////////////////////
		if (m_json.find("TetModels") != m_json.end())
			readTetModels(m_json, "TetModels", basePath, sceneData);

		//////////////////////////////////////////////////////////////////////////
		// read ball joints
		//////////////////////////////////////////////////////////////////////////
		if (m_json.find("BallJoints") != m_json.end())
			readBallJoints(m_json, "BallJoints", sceneData);

		//////////////////////////////////////////////////////////////////////////
		// read ball-on-line joints
		//////////////////////////////////////////////////////////////////////////
		if (m_json.find("BallOnLineJoints") != m_json.end())
			readBallOnLineJoints(m_json, "BallOnLineJoints", sceneData);

		//////////////////////////////////////////////////////////////////////////
		// read hinge joints
		//////////////////////////////////////////////////////////////////////////
		if (m_json.find("HingeJoints") != m_json.end())
			readHingeJoints(m_json, "HingeJoints", sceneData);

		//////////////////////////////////////////////////////////////////////////
		// read universal joints
		//////////////////////////////////////////////////////////////////////////
		if (m_json.find("UniversalJoints") != m_json.end())
			readUniversalJoints(m_json, "UniversalJoints", sceneData);

		//////////////////////////////////////////////////////////////////////////
		// read slider joints
		//////////////////////////////////////////////////////////////////////////
		if (m_json.find("SliderJoints") != m_json.end())
			readSliderJoints(m_json, "SliderJoints", sceneData);

		//////////////////////////////////////////////////////////////////////////
		// read RigidBodyParticleBallJoints
		//////////////////////////////////////////////////////////////////////////
		if (m_json.find("RigidBodyParticleBallJoints") != m_json.end())
			readRigidBodyParticleBallJoints(m_json, "RigidBodyParticleBallJoints", sceneData);

		//////////////////////////////////////////////////////////////////////////
		// read RigidBodySprings
		//////////////////////////////////////////////////////////////////////////
		if (m_json.find("RigidBodySprings") != m_json.end())
			readRigidBodySprings(m_json, "RigidBodySprings", sceneData);

		//////////////////////////////////////////////////////////////////////////
		// read DistanceJoints
		//////////////////////////////////////////////////////////////////////////
		if (m_json.find("DistanceJoints") != m_json.end())
			readDistanceJoints(m_json, "DistanceJoints", sceneData);

		//////////////////////////////////////////////////////////////////////////
		// read TargetAngleMotorHingeJoint
		//////////////////////////////////////////////////////////////////////////
		if (m_json.find("TargetAngleMotorHingeJoints") != m_json.end())
			readTargetAngleMotorHingeJoints(m_json, "TargetAngleMotorHingeJoints", sceneData);

		//////////////////////////////////////////////////////////////////////////
		// read TargetVelocityMotorHingeJoint
		//////////////////////////////////////////////////////////////////////////
		if (m_json.find("TargetVelocityMotorHingeJoints") != m_json.end())
			readTargetVelocityMotorHingeJoints(m_json, "TargetVelocityMotorHingeJoints", sceneData);

		//////////////////////////////////////////////////////////////////////////
		// read TargetPositionMotorSliderJoint
		//////////////////////////////////////////////////////////////////////////
		if (m_json.find("TargetPositionMotorSliderJoints") != m_json.end())
			readTargetPositionMotorSliderJoints(m_json, "TargetPositionMotorSliderJoints", sceneData);

		//////////////////////////////////////////////////////////////////////////
		// read TargetVelocityMotorSliderJoint
		//////////////////////////////////////////////////////////////////////////
		if (m_json.find("TargetVelocityMotorSliderJoints") != m_json.end())
			readTargetVelocityMotorSliderJoints(m_json, "TargetVelocityMotorSliderJoints", sceneData);

		parseStiffRodsData(m_json, sceneData, basePath);
	}
	catch (std::exception& e)
	{
		LOG_ERR << e.what();
		exit(1);
	}
}
