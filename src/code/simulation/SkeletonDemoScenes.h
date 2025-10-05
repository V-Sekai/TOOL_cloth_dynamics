#pragma once

#include "../engine/Constants.h"
#include "CapsuleGenerator.h"
#include "Simulation.h"
#include "SpringBoneExtender.h"
#include <string>

namespace tool_cloth_dynamics {

/**
 * @brief Demo scene configurations for skeleton-capsule pipeline
 *
 * Implements the demo scenes specified in the decision document,
 * showcasing various skeleton-to-capsule draping scenarios.
 */
namespace SkeletonDemoScenes {

/**
 * @brief Main demo: FoxGirl skeleton + jumpsuit draping
 *
 * Demonstrates the complete skeleton-capsule pipeline:
 * 1. Load FoxGirl skeleton from paired assets
 * 2. Generate tapered capsules
 * 3. Load jumpsuit garment
 * 4. Auto-pin garment to capsules
 * 5. Realistic cloth draping simulation
 */
struct FoxGirlJumpsuitDemo {
	static Simulation::SceneConfiguration create() {
		Simulation::SceneConfiguration config;

		// Demo identification
		config.name = "FoxGirl_Jumpsuit_Skeleton_Demo";

		// Load FoxGirl skeleton (paired assets)
		config.skeletonPath = "src/assets/meshes/avatars/FoxGirl/skeleton.obj";
		config.primitiveConfig = SKELETON_CAPSULES;

		// Load jumpsuit garment (paired assets)
		config.fabric.name = "garments/jumpsuit_dense/garment.obj";
		config.fabric.isModel = true;
		config.fabric.keepOriginalScalePoint = false;

		// Simulation parameters
		config.timeStep = 1.0 / 60.0; // 60 FPS
		config.stepNum = 300; // 5 seconds simulation

		// Physics settings for realistic draping
		config.fabric.k_stiff_stretching = 200.0;
		config.fabric.k_stiff_bending = 0.01;
		config.fabric.density = 0.15;

		// Enable gravity for cloth draping
		config.gravityEnabled = true;

		// Camera positioning for good view
		config.camPos = Vec3d(2.0, 1.5, 3.0);
		config.camFocusPos = Vec3d(0.0, 0.0, 0.0);
		config.camFocusPointType = CameraFocusPointType::CAMERA_POINT;

		// Disable wind for cleaner demo
		config.windConfig = WindConfig::NO_WIND;

		// Color scheme
		config.fabric.color = COLOR_IBM_BLUE60; // Nice blue for jumpsuit

		return config;
	}

	/**
	 * @brief Expected behavior timeline
	 */
	static std::string getExpectedBehavior() {
		return "Demo Timeline:\n"
			   "T=0s: Jumpsuit spawns above FoxGirl skeleton capsules\n"
			   "T=1s: Cloth falls, first contact with shoulder/torso capsules\n"
			   "T=2s: Cloth drapes over arms and torso\n"
			   "T=3s: Cloth settles around legs and lower body\n"
			   "T=5s: Cloth reaches final draped shape over skeleton\n"
			   "\n"
			   "Success Criteria:\n"
			   "- Cloth maintains garment-like shape\n"
			   "- No unrealistic stretching or tearing\n"
			   "- Smooth draping over all body parts\n"
			   "- Stable final configuration";
	}
};

/**
 * @brief Advanced demo: Multiple skeleton types
 *
 * Showcases different skeleton topologies:
 * - Humanoid (FoxGirl)
 * - Quadruped (if available)
 * - Custom skeleton structures
 */
struct MultiSkeletonDemo {
	static std::vector<Simulation::SceneConfiguration> create() {
		std::vector<Simulation::SceneConfiguration> demos;

		// Demo 1: Humanoid avatar draping
		Simulation::SceneConfiguration humanoid_demo;
		humanoid_demo.name = "Humanoid_Avatar_Demo";
		humanoid_demo.skeletonPath = "src/assets/meshes/avatars/FoxGirl/skeleton.obj";
		humanoid_demo.fabric.name = "avatars/FoxGirl/avatar.obj";
		humanoid_demo.primitiveConfig = SKELETON_CAPSULES;
		humanoid_demo.timeStep = 1.0 / 60.0;
		humanoid_demo.stepNum = 180; // 3 seconds
		humanoid_demo.fabric.color = COLOR_SKIN;
		demos.push_back(humanoid_demo);

		// Demo 2: Dense garment draping
		Simulation::SceneConfiguration garment_demo;
		garment_demo.name = "Dense_Garment_Demo";
		garment_demo.skeletonPath = "src/assets/meshes/garments/jumpsuit_dense/skeleton.obj";
		garment_demo.fabric.name = "garments/jumpsuit_dense/garment.obj";
		garment_demo.primitiveConfig = SKELETON_CAPSULES;
		garment_demo.timeStep = 1.0 / 60.0;
		garment_demo.stepNum = 240; // 4 seconds
		garment_demo.fabric.color = COLOR_IBM_ORANGE40;
		demos.push_back(garment_demo);

		return demos;
	}
};

/**
 * @brief Performance test: Large mesh handling
 *
 * Tests the sparse octree optimization with high-resolution meshes.
 */
struct PerformanceTestDemo {
	static Simulation::SceneConfiguration create() {
		Simulation::SceneConfiguration config;

		config.name = "Performance_Test_Large_Mesh";
		config.skeletonPath = "src/assets/meshes/avatars/FoxGirl/skeleton.obj";
		config.fabric.name = "garments/jumpsuit_dense/garment.obj"; // Use dense garment for testing
		config.primitiveConfig = SKELETON_CAPSULES;

		// Use advanced radius estimation (sparse octree)
		config.useAdvancedRadiusEstimation = true;

		// Performance-focused settings
		config.timeStep = 1.0 / 30.0; // 30 FPS for better performance
		config.stepNum = 150; // 5 seconds at 30 FPS

		// Reduce some quality for speed
		config.fabric.k_stiff_stretching = 150.0; // Slightly lower stiffness
		config.fabric.density = 0.12; // Lighter fabric

		return config;
	}
};

/**
 * @brief Spring bone demo: Hair and skirt physics
 *
 * Demonstrates VRM Spring Bone extension for secondary animation.
 */
struct SpringBoneDemo {
	static Simulation::SceneConfiguration create() {
		Simulation::SceneConfiguration config;

		config.name = "Spring_Bone_Hair_Skirt_Demo";
		config.skeletonPath = "src/assets/meshes/avatars/FoxGirl/skeleton.obj";
		config.fabric.name = "garments/LCL_Skirt_DressEvening_003/garment.obj";
		config.primitiveConfig = SKELETON_WITH_SPRING_BONES;

		// Enable spring bone extensions
		config.useSpringBones = true;
		config.springBoneSubdivisions = 4; // More segments for smoother motion
		config.springBoneTaperFactor = 0.85; // More aggressive tapering

		// Physics for dynamic motion
		config.timeStep = 1.0 / 60.0;
		config.stepNum = 600; // 10 seconds for full spring motion

		// Wind for spring bone motion
		config.windConfig = WindConfig::WIND_SIN;
		config.windEnabled = true;

		// Fabric settings for flowing garments
		config.fabric.k_stiff_stretching = 120.0; // More flexible
		config.fabric.k_stiff_bending = 0.005; // Lower bending stiffness
		config.fabric.density = 0.1; // Light fabric

		return config;
	}
};

/**
 * @brief Batch processing demo: Multiple assets
 *
 * Demonstrates processing multiple skeleton-mesh pairs in sequence.
 */
struct BatchProcessingDemo {
	struct AssetPair {
		std::string skeleton_path;
		std::string mesh_path;
		std::string demo_name;
		Vec3d fabric_color;
	};

	static std::vector<AssetPair> getAssetPairs() {
		return {
			{ "src/assets/meshes/avatars/FoxGirl/skeleton.obj",
					"src/assets/meshes/avatars/FoxGirl/avatar.obj",
					"FoxGirl_Avatar_Batch",
					COLOR_SKIN },
			{ "src/assets/meshes/garments/jumpsuit_dense/skeleton.obj",
					"src/assets/meshes/garments/jumpsuit_dense/garment.obj",
					"Jumpsuit_Dense_Batch",
					COLOR_IBM_BLUE60 },
			{ "src/assets/meshes/garments/Puffer_dense/skeleton.obj",
					"src/assets/meshes/garments/Puffer_dense/garment.obj",
					"Puffer_Dense_Batch",
					COLOR_IBM_ORANGE40 }
		};
	}

	static Simulation::SceneConfiguration createForAsset(const AssetPair &asset) {
		Simulation::SceneConfiguration config;

		config.name = asset.demo_name;
		config.skeletonPath = asset.skeleton_path;
		config.fabric.name = asset.mesh_path;
		config.fabric.color = asset.fabric_color;
		config.primitiveConfig = SKELETON_CAPSULES;

		// Standard batch processing settings
		config.timeStep = 1.0 / 60.0;
		config.stepNum = 180; // 3 seconds each
		config.fabric.isModel = true;

		return config;
	}
};

/**
 * @brief Create and run the main FoxGirl + Jumpsuit demo
 *
 * @return Configured simulation ready to run
 */
Simulation *createMainDemo();

/**
 * @brief Run all demo scenes in sequence
 *
 * Useful for automated testing and validation of the complete pipeline.
 *
 * @param output_directory Directory to save demo results
 * @return Success status
 */
bool runAllDemos(const std::string &output_directory = "demo_results/");

/**
 * @brief Validate demo scene assets
 *
 * Checks that all required skeleton.obj and mesh.obj files exist
 * and are properly paired before running demos.
 *
 * @return Validation report
 */
std::string validateDemoAssets();

} // namespace SkeletonDemoScenes

/**
 * @brief Additional scene configuration enums for skeleton demos
 */
enum SkeletonPrimitiveConfiguration {
	SKELETON_CAPSULES = 100, // Load capsules from skeleton.obj
	SKELETON_WITH_SPRING_BONES = 101 // Load skeleton + spring bone extensions
};

} // namespace tool_cloth_dynamics
