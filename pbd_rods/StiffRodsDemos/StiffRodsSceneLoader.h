
#ifndef __STIFFRODSSCENELOADER_H__ 
#define __STIFFRODSSCENELOADER_H__

#include "Utils/SceneLoader.h"

#include <set>
#include <map>

namespace PBD
{
	class StiffRodsSceneLoader : public Utilities::SceneLoader
	{
	public:
		virtual ~StiffRodsSceneLoader() {}

		struct AnimatorData{
			unsigned int m_id;
			unsigned int m_RodID;
			unsigned int m_SegmentID;
			Real m_StartingTime;
			Real m_EndingTime;
			unsigned int m_Type;
			std::vector<Vector3r> m_Data;
		};

		struct StretchBendingTwistingConstraintData
		{
			unsigned int m_Id;
			unsigned int m_SegmentID[2];
			Real m_YoungsModulus;
			Real m_TorsionModulus;
			Vector3r m_Position;
			Vector3r m_StiffnessModifier;
		};

		struct TreeModelData{
			unsigned int m_Id;
			Vector3r m_X;
			Quaternionr m_Q;
			Real m_YoungsModulus;
			Real m_TorsionModulus;
			std::set<unsigned int> m_SegmentIds;
			std::set<unsigned int> m_StretchBendingTwistingConstraintIds;
			std::set<unsigned int> m_StaticSegments;
			unsigned int m_SolverType;
		};
		
		struct SkinningMeshData{
			unsigned int m_Id;
			std::string m_MeshFile;
			Vector3r m_X;
			Quaternionr m_Q;
			Vector3r m_Scale;
			std::vector<unsigned int> m_VertexIDs;
			std::vector<std::vector<unsigned int>> m_BodyIDs;
			std::vector<std::vector<Real>> m_SkinningWeights;
		};

		struct CompactTreeModelData{
			unsigned int m_Id;
			Vector3r m_X;
			Quaternionr m_Q;
			Real m_YoungsModulus;
			Real m_TorsionModulus;
			std::set<unsigned int> m_StaticSegments;
			unsigned int m_SolverType;
			std::vector<Vector3r> m_SegmentPositions;
			std::vector<Quaternionr> m_SegmentOrientations;
			std::vector<Vector3r> m_SegmentScales;
			std::vector<Real> m_SegmentDensities;
			std::vector<std::string> m_SegmentModelFiles;
			std::vector<Real> m_SegmentFrictionCoefficients;
			std::vector<Real> m_SegmentRestitutioCoefficients;
			std::vector<short> m_SegmentCollisionCategoryBits;
			std::vector<short> m_SegmentCollisionFilterMaskBits;
			std::vector<Vector3r> m_SBTConstraintPositions;
			std::vector<std::vector<unsigned int>> m_SBTConstraintBodiesIndices;
			std::vector<Vector3r> m_SBTConstraintStiffnessModifier;
		};

		struct RodSceneData : public SceneLoader::SceneData{
			std::vector<AnimatorData> m_RodAnimators;
			std::vector<StretchBendingTwistingConstraintData> m_StretchBendingTwistingConstraints;
			std::vector<TreeModelData> m_TreeModels;
			std::vector<SkinningMeshData> m_SkinningMeshes;
			std::vector<CompactTreeModelData> m_CompactTreeModels;
		};

		virtual void readStiffRodsScene(const std::string &fileName, SceneData &sceneData);
		
	protected:

		void parseStiffRodsData(const nlohmann::json &j, SceneData &sceneData, const std::string &basePath);
		void parseRodAnimators(const nlohmann::json &j, const std::string &key, SceneData & sceneData);
		void parseStretchBendingTwistingConstraints(const nlohmann::json &j, const std::string &key, SceneData & sceneData);
		void parseTreeModels(const nlohmann::json &j, const std::string &key, SceneData & sceneData);
		void parseCompactTreeModels(const nlohmann::json &j, const std::string &key, SceneData & sceneData, const std::string &basePath);
		void parseSkinningMeshes(const nlohmann::json &j, const std::string &key, SceneData & sceneData, const std::string &basePath);

		template <typename T, int size>
		static Eigen::Matrix<T, size, 1> readVectorValues(const nlohmann::json &j)
		{
			unsigned int index = 0u;
			Eigen::Matrix<T, size, 1> v(Eigen::Matrix<T, size, 1>::Zero());
			for (auto& item : j)
			{
				v[index++] = item.get<T>();
				if (index >= size)
					break;
			}

			return v;
		}
	};
}

#endif
