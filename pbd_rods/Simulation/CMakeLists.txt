add_library(Simulation
		AABB.h
		CollisionDetection.cpp
		CollisionDetection.h
		Constraints.cpp
		Constraints.h
		CubicSDFCollisionDetection.cpp
		CubicSDFCollisionDetection.h
		DistanceFieldCollisionDetection.cpp
		DistanceFieldCollisionDetection.h
		IDFactory.cpp
		IDFactory.h
		LineModel.cpp
		LineModel.h
		NeighborhoodSearchSpatialHashing.cpp
		NeighborhoodSearchSpatialHashing.h
		ParticleData.h
		RigidBody.h
		RigidBodyGeometry.cpp
		RigidBodyGeometry.h
		Simulation.cpp
		Simulation.h
		SimulationModel.cpp
		SimulationModel.h
		TetModel.cpp
		TetModel.h
		TimeManager.cpp
		TimeManager.h
		TimeStep.cpp
		TimeStep.h
		TimeStepController.cpp
		TimeStepController.h
		TriangleModel.cpp
		TriangleModel.h
		
		BoundingSphere.h
		BoundingSphereHierarchy.cpp
		BoundingSphereHierarchy.h
		kdTree.h
		kdTree.inl

		CMakeLists.txt
)

############################################################
# Discregrid
############################################################
target_include_directories(Simulation PUBLIC ${Discregrid_INCLUDE_DIR})
if (TARGET Ext_Discregrid)
	add_dependencies(Simulation Ext_Discregrid)
endif()


############################################################
# GenericParameters
############################################################
target_include_directories(Simulation PUBLIC ${GenericParameters_INCLUDE_DIR})
if(TARGET Ext_GenericParameters)
	add_dependencies(Simulation Ext_GenericParameters)
endif()

find_package( Eigen3 REQUIRED )
target_include_directories(Simulation PUBLIC ${EIGEN3_INCLUDE_DIR} )

target_link_libraries(Simulation PUBLIC PositionBasedDynamics)


install(TARGETS Simulation
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY .
DESTINATION include/Simulation
FILES_MATCHING PATTERN "*.h" PATTERN "*.inl")