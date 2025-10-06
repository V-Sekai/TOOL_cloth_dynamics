#pragma once

#include <string>

// Forward declaration to avoid circular dependency
namespace tool_cloth_dynamics {
class CapsuleRig;

/**
 * @brief Simple test suite for skeleton-capsule pipeline
 *
 * Verifies that all major components work together correctly.
 */
class SkeletonPipelineTest {
public:
	/**
	 * @brief Test skeleton loading from a simple OBJ structure
	 */
	static bool testSkeletonLoading();

	/**
	 * @brief Test capsule generation from skeleton
	 */
	static bool testCapsuleGeneration();

	/**
	 * @brief Test sparse octree functionality
	 */
	static bool testSparseOctree();

	/**
	 * @brief Test radius estimation using octree
	 */
	static bool testRadiusEstimation();

	/**
	 * @brief Test radius estimation on real mesh data
	 */
	static bool testRealMeshRadiusEstimation();

	/**
	 * @brief Test demo scene configuration
	 */
	static bool testDemoScenes();

	/**
	 * @brief Test capsule fitting with CAPSULE_FIT loss
	 */
	static bool testDemoCapsuleFitting();

	/**
	 * @brief Compute and report capsule fit loss metrics against avatar mesh
	 */
	static bool computeCapsuleFitLossReport(const CapsuleRig &rig, const std::string &avatar_mesh_path);

	/**
	 * @brief Demonstrate iterative optimization with loss improvement across iterations
	 */
	static bool testCapsuleFitOptimization(const CapsuleRig &rig, const std::string &avatar_mesh_path);

	/**
	 * @brief Compute loss for a single iteration (helper for optimization demo)
	 */
	static bool computeCapsuleFitLossSingleIteration(const CapsuleRig &rig, const std::string &avatar_mesh_path, int iteration);

	/**
	 * @brief Run all tests
	 */
	static bool runAllTests();
};

} // namespace tool_cloth_dynamics
