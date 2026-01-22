#include "NeighborhoodSearchSpatialHashing.h"

using namespace PBD;
using namespace Utilities;

NeighborhoodSearchSpatialHashing::NeighborhoodSearchSpatialHashing(const unsigned int numParticles, const Real radius, const unsigned int maxNeighbors, const unsigned int maxParticlesPerCell) :
	m_gridMap(numParticles*2)
{
	m_cellGridSize = radius;
	m_radius2 = radius*radius;
	m_numParticles = numParticles;
	m_maxParticlesPerCell = maxParticlesPerCell;
	m_maxNeighbors = maxNeighbors;

	m_numNeighbors = NULL;
	m_neighbors = NULL;

	if (numParticles != 0)
	{
		m_numNeighbors = new unsigned int[m_numParticles];
		m_neighbors = new unsigned int*[m_numParticles];
		for (unsigned int i = 0; i < m_numParticles; i++)
			m_neighbors[i] = new unsigned int[m_maxNeighbors];
	}

	m_currentTimestamp = 0;
}

NeighborhoodSearchSpatialHashing::~NeighborhoodSearchSpatialHashing()
{
	cleanup();
}

void NeighborhoodSearchSpatialHashing::cleanup()
{
	for (unsigned int i=0; i < m_numParticles; i++)
		delete [] m_neighbors[i];
	delete [] m_neighbors;
	delete [] m_numNeighbors;
	m_numParticles = 0;

	for (unsigned int i=0; i < m_gridMap.bucket_count(); i++)
	{
		Hashmap<NeighborhoodSearchCellPos*, NeighborhoodSearchSpatialHashing::HashEntry*>::KeyValueMap *kvMap = m_gridMap.getKeyValueMap(i);
		if (kvMap)
		{
			for (Hashmap<NeighborhoodSearchCellPos*, NeighborhoodSearchSpatialHashing::HashEntry*>::KeyValueMap::iterator iter=kvMap->begin(); iter != kvMap->end(); iter++)
			{
				NeighborhoodSearchSpatialHashing::HashEntry* entry = iter->second;
				delete entry;
				iter->second = NULL;
			}
		}
	}
}

unsigned int ** NeighborhoodSearchSpatialHashing::getNeighbors() const
{
	return m_neighbors;
}

unsigned int * NeighborhoodSearchSpatialHashing::getNumNeighbors() const
{
	return m_numNeighbors;
}

unsigned int NeighborhoodSearchSpatialHashing::getNumParticles() const
{
	return m_numParticles;
}

void NeighborhoodSearchSpatialHashing::setRadius(const Real radius)
{
	m_cellGridSize = radius;
	m_radius2 = radius*radius;
}

Real NeighborhoodSearchSpatialHashing::getRadius() const
{
	return sqrt(m_radius2);
}

void NeighborhoodSearchSpatialHashing::update()
{
	m_currentTimestamp++;
}


void NeighborhoodSearchSpatialHashing::neighborhoodSearch(Vector3r *x) 
{		
	const Real factor = static_cast<Real>(1.0)/m_cellGridSize;
	for (int i=0; i < (int) m_numParticles; i++)
	{
		const int cellPos1 = NeighborhoodSearchSpatialHashing::floor(x[i][0] * factor)+1;
		const int cellPos2 = NeighborhoodSearchSpatialHashing::floor(x[i][1] * factor)+1;
		const int cellPos3 = NeighborhoodSearchSpatialHashing::floor(x[i][2] * factor)+1;
		NeighborhoodSearchCellPos cellPos(cellPos1, cellPos2, cellPos3);
		HashEntry *&entry = m_gridMap[&cellPos];

		if (entry != NULL)
		{
			if (entry->timestamp != m_currentTimestamp)
			{
				entry->timestamp = m_currentTimestamp;
				entry->particleIndices.clear();
			}
		}
		else
		{
			HashEntry *newEntry = new HashEntry(); 	
			newEntry->particleIndices.reserve(m_maxParticlesPerCell);
			newEntry->timestamp = m_currentTimestamp; 
			entry = newEntry;
		}
		entry->particleIndices.push_back(i);
	}

	// loop over all 27 neighboring cells
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static) 
		for (int i=0; i < (int) m_numParticles; i++)
		{
			m_numNeighbors[i] = 0;
			const int cellPos1 = NeighborhoodSearchSpatialHashing::floor(x[i][0] * factor);
			const int cellPos2 = NeighborhoodSearchSpatialHashing::floor(x[i][1] * factor);
			const int cellPos3 = NeighborhoodSearchSpatialHashing::floor(x[i][2] * factor);
			for(unsigned char j=0; j < 3; j++)
			{				
				for(unsigned char k=0; k < 3; k++)
				{									
					for(unsigned char l=0; l < 3; l++)
					{
						NeighborhoodSearchCellPos cellPos(cellPos1+j, cellPos2+k, cellPos3+l);
						HashEntry * const *entry = m_gridMap.query(&cellPos);
					
						if ((entry != NULL) && (*entry != NULL) && ((*entry)->timestamp == m_currentTimestamp))
						{
							for (unsigned int m=0; m < (*entry)->particleIndices.size(); m++)
							{
								const unsigned int pi = (*entry)->particleIndices[m];
								if (pi != i)
								{
									const Real dist2 = (x[i]-x[pi]).squaredNorm();
									if (dist2 < m_radius2)
									{
										if (m_numNeighbors[i] < m_maxNeighbors)
											m_neighbors[i][m_numNeighbors[i]++] = pi;
//									else
// 											std::cout << "too many neighbors detected\n";
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

void NeighborhoodSearchSpatialHashing::neighborhoodSearch(Vector3r *x, const unsigned int numBoundaryParticles, Vector3r *boundaryX)
{		
	const Real factor = static_cast<Real>(1.0)/m_cellGridSize;
	for (int i=0; i < (int) m_numParticles; i++)
	{
		const int cellPos1 = NeighborhoodSearchSpatialHashing::floor(x[i][0] * factor)+1;
		const int cellPos2 = NeighborhoodSearchSpatialHashing::floor(x[i][1] * factor)+1;
		const int cellPos3 = NeighborhoodSearchSpatialHashing::floor(x[i][2] * factor)+1;
		NeighborhoodSearchCellPos cellPos(cellPos1, cellPos2, cellPos3);
		HashEntry *&entry = m_gridMap[&cellPos];

		if (entry != NULL)
		{
			if (entry->timestamp != m_currentTimestamp)
			{
				entry->timestamp = m_currentTimestamp;
				entry->particleIndices.clear();
			}
		}
		else
		{
			HashEntry *newEntry = new HashEntry(); 	
			newEntry->particleIndices.reserve(m_maxParticlesPerCell);
			newEntry->timestamp = m_currentTimestamp; 
			entry = newEntry;
		}
		entry->particleIndices.push_back(i);
	}

	for (int i = 0; i < (int)numBoundaryParticles; i++)
	{
		const int cellPos1 = NeighborhoodSearchSpatialHashing::floor(boundaryX[i][0] * factor) + 1;
		const int cellPos2 = NeighborhoodSearchSpatialHashing::floor(boundaryX[i][1] * factor) + 1;
		const int cellPos3 = NeighborhoodSearchSpatialHashing::floor(boundaryX[i][2] * factor) + 1;
		NeighborhoodSearchCellPos cellPos(cellPos1, cellPos2, cellPos3);
		HashEntry *&entry = m_gridMap[&cellPos];

		if (entry != NULL)
		{
			if (entry->timestamp != m_currentTimestamp)
			{
				entry->timestamp = m_currentTimestamp;
				entry->particleIndices.clear();
			}
		}
		else
		{
			HashEntry *newEntry = new HashEntry();
			newEntry->particleIndices.reserve(m_maxParticlesPerCell);
			newEntry->timestamp = m_currentTimestamp;
			entry = newEntry;
		}
		entry->particleIndices.push_back(m_numParticles + i);
	}

	// loop over all 27 neighboring cells
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i=0; i < (int) m_numParticles; i++)
		{
			m_numNeighbors[i] = 0;
			const int cellPos1 = NeighborhoodSearchSpatialHashing::floor(x[i][0] * factor);
			const int cellPos2 = NeighborhoodSearchSpatialHashing::floor(x[i][1] * factor);
			const int cellPos3 = NeighborhoodSearchSpatialHashing::floor(x[i][2] * factor);
			for(unsigned char j=0; j < 3; j++)
			{				
				for(unsigned char k=0; k < 3; k++)
				{									
					for(unsigned char l=0; l < 3; l++)
					{
						NeighborhoodSearchCellPos cellPos(cellPos1+j, cellPos2+k, cellPos3+l);
						HashEntry * const *entry = m_gridMap.query(&cellPos);
					
						if ((entry != NULL) && (*entry != NULL) && ((*entry)->timestamp == m_currentTimestamp))
						{
							for (unsigned int m=0; m < (*entry)->particleIndices.size(); m++)
							{
								const unsigned int pi = (*entry)->particleIndices[m];
								if (pi != i)
								{
									Real dist2;
									if (pi < m_numParticles)
										dist2 = (x[i]-x[pi]).squaredNorm();
									else
										dist2 = (x[i] - boundaryX[pi - m_numParticles]).squaredNorm();

									if (dist2 < m_radius2)
									{
										if (m_numNeighbors[i] < m_maxNeighbors)
											m_neighbors[i][m_numNeighbors[i]++] = pi;
// 										else
// 											std::cout << "too many neighbors detected\n";
									}
								}
							}
						}
					}
				}
			}
		}
	}
}