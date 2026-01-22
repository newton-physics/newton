#ifndef __NEIGHBORHOODSEARCHSPATIALHASHING_H__
#define __NEIGHBORHOODSEARCHSPATIALHASHING_H__

#include "Utils/Hashmap.h"
#include <vector>
#include "Common/Common.h"

typedef Eigen::Vector3i NeighborhoodSearchCellPos;

namespace Utilities
{
	template<>
	inline unsigned int hashFunction<NeighborhoodSearchCellPos*>(NeighborhoodSearchCellPos* const &key)
	{
		const int p1 = 73856093 * (*key)[0];
		const int p2 = 19349663 * (*key)[1];
		const int p3 = 83492791 * (*key)[2];
		return p1 + p2 + p3;
	}
}

namespace PBD
{
	class NeighborhoodSearchSpatialHashing
	{
	public: 
		NeighborhoodSearchSpatialHashing(const unsigned int numParticles = 0, const Real radius = 0.1, const unsigned int maxNeighbors = 60u, const unsigned int maxParticlesPerCell = 50u);
		~NeighborhoodSearchSpatialHashing();

		// Spatial hashing
		struct HashEntry
		{
			HashEntry() {};
			unsigned long timestamp;
			std::vector<unsigned int> particleIndices;
		};

		FORCE_INLINE static int floor(const Real v)
		{
			return (int)(v + 32768.f) - 32768;			// Shift to get positive values 
		}

		void cleanup();
		void neighborhoodSearch(Vector3r *x);
		void neighborhoodSearch(Vector3r *x, const unsigned int numBoundaryParticles, Vector3r *boundaryX);
		void update();
		unsigned int **getNeighbors() const;
		unsigned int *getNumNeighbors() const;
		const unsigned int getMaxNeighbors() const { return m_maxNeighbors;	}

		unsigned int getNumParticles() const;
		void setRadius(const Real radius);
		Real getRadius() const;

		FORCE_INLINE unsigned int n_neighbors(unsigned int i) const 
		{
			return m_numNeighbors[i];
		}
		FORCE_INLINE unsigned int neighbor(unsigned int i, unsigned int k) const 
		{
			return m_neighbors[i][k];
		}


	private: 
		unsigned int m_numParticles;
		unsigned int m_maxNeighbors;
		unsigned int m_maxParticlesPerCell;
		unsigned int **m_neighbors;
		unsigned int *m_numNeighbors;
		Real m_cellGridSize;
		Real m_radius2;
		unsigned int m_currentTimestamp;
		Utilities::Hashmap<NeighborhoodSearchCellPos*, HashEntry*> m_gridMap;
	};
}

#endif
