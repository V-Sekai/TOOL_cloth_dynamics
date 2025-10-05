#pragma once

namespace tool_cloth_dynamics {

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
	 * @brief Run all tests
	 */
	static bool runAllTests();
};

} // namespace tool_cloth_dynamics
