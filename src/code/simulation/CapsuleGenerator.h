#pragma once

#include "SkeletonLoader.h"
#include "TaperedCapsule.h"
#include <memory>
#include <string>
#include <vector>

// Forward declaration to avoid circular dependency
class Simulation;

namespace tool_cloth_dynamics {

// Forward declaration to avoid circular dependency
enum class GarmentType;

/**
 * @brief A complete rig of tapered capsules generated from a skeleton
 *
 * Manages the collection of capsules created from skeleton bones,
 * handles simulation integration, and provides asset pairing validation.
 */
class CapsuleRig {
public:
	/**
	 * @brief Generate capsule rig from skeleton with fixed radius
	 *
	 * @param skel Skeleton structure with bones
	 * @param radius Fixed radius for all capsules (MVP approach)
	 * @return Generated capsule rig
	 */
	static CapsuleRig generate(const Skeleton &skel, double radius = 0.1);

	/**
	 * @brief Generate capsule rig from paired assets directory with mesh-based radius estimation
	 *
	 * Enforces asset pairing constraints by loading skeleton and mesh
	 * from the same directory (e.g., "avatars/FoxGirl/").
	 * Computes capsule radii using mesh proximity queries with sparse octree.
	 *
	 * @param asset_directory Directory containing paired skeleton.obj + mesh.obj
	 * @param subdivisions_per_bone Number of capsules per bone for improved fitting (default: 3)
	 * @return Generated capsule rig with mesh-based radius estimation
	 * @throws std::runtime_error if assets are not properly paired
	 */
	static CapsuleRig generateFromPairedAssets(const std::string &asset_directory, int subdivisions_per_bone = 3);

	/**
	 * @brief Add all capsules to simulation as collision primitives
	 *
	 * @param simulation Target simulation to add capsules to
	 */
	void addToSimulation(::Simulation *simulation);

	/**
	 * @brief Pin garment vertices to nearby capsules using proximity
	 *
	 * Zero-configuration approach: automatically pins cloth vertices
	 * that are within pin_distance of any capsule surface.
	 *
	 * @param simulation Simulation containing cloth
	 * @param pin_distance Maximum distance for pinning (default: 0.05 = 5cm)
	 */
	void pinGarmentToCapsules(::Simulation *simulation, double pin_distance = 0.05);

	/**
	 * @brief Fit garment to capsule rig using bone-based inflation algorithm
	 *
	 * Simplified approach: uses four-phase inflation algorithm with bone-based
	 * anchor points for natural cloth draping.
	 *
	 * @param simulation Simulation containing cloth
	 */
	void fitGarmentToRig(::Simulation *simulation);

	/**
	 * @brief Validate that skeleton and mesh are properly paired
	 *
	 * Checks that both files exist in the same directory and are compatible.
	 *
	 * @param skeleton_path Path to skeleton.obj file
	 * @param mesh_path Path to mesh.obj file
	 * @return true if assets are properly paired
	 */
	static bool validateAssetPairing(
			const std::string &skeleton_path,
			const std::string &mesh_path);

	/**
	 * @brief Get the underlying skeleton structure
	 */
	const Skeleton &getSkeleton() const { return skeleton; }

	/**
	 * @brief Get all generated capsules
	 */
	const std::vector<std::unique_ptr<TaperedCapsule>> &getCapsules() const {
		return capsules;
	}

	/**
	 * @brief Get number of capsules in the rig
	 */
	size_t getCapsuleCount() const { return capsules.size(); }

	/**
	 * @brief Check if rig is valid (has capsules)
	 */
	bool isValid() const { return !capsules.empty(); }

	/**
	 * @brief Export all capsules in the rig to OBJ files
	 *
	 * Creates separate OBJ files for each capsule, named with capsule index.
	 *
	 * @param base_filename Base filename (will be suffixed with capsule index)
	 * @return true if all exports successful
	 */
	bool exportToOBJ(const std::string &base_filename) const;

	/**
	 * @brief Default constructor (public for Simulation member usage)
	 */
	CapsuleRig() = default;

	/**
	 * @brief Deleted copy constructor and assignment to prevent copying unique_ptr vectors
	 */
	CapsuleRig(const CapsuleRig &) = delete;
	CapsuleRig &operator=(const CapsuleRig &) = delete;

	/**
	 * @brief Move constructor and assignment
	 */
	CapsuleRig(CapsuleRig &&) = default;
	CapsuleRig &operator=(CapsuleRig &&) = default;

private:
	Skeleton skeleton;
	std::vector<std::unique_ptr<TaperedCapsule>> capsules;
	std::vector<std::vector<int>> bone_to_capsules_map; // 1:many mapping from bones to capsules
	std::string asset_source_directory; // Track pairing source
};

/**
 * @brief Utility class for generating capsule rigs from skeletons
 *
 * Implements the core logic for converting skeleton bones into
 * tapered capsules with various radius estimation strategies.
 */
class CapsuleGenerator {
public:
	/**
	 * @brief Generate capsules from skeleton bones with fixed radius
	 *
	 * MVP implementation: all capsules get the same radius.
	 *
	 * @param skeleton Input skeleton structure
	 * @param radius Fixed radius for all capsules
	 * @return Vector of generated tapered capsules
	 */
	static std::vector<std::unique_ptr<TaperedCapsule>> generateCapsules(
			const Skeleton &skeleton,
			double radius = 0.1);

	/**
	 * @brief Generate capsules with advanced radius estimation using octree
	 *
	 * Full implementation: estimates radii from mesh geometry using O(log N) octree queries.
	 *
	 * @param skeleton Input skeleton structure
	 * @param mesh_vertices Mesh vertex data for radius estimation
	 * @param use_tapered_radii If true, estimate different radii at bone ends for tapering
	 * @return Vector of generated tapered capsules with optimized radii
	 */
	static std::vector<std::unique_ptr<TaperedCapsule>> generateCapsulesWithAdvancedRadii(
			const Skeleton &skeleton,
			const MatXd &mesh_vertices,
			bool use_tapered_radii = true,
			int subdivisions_per_bone = 1);

	/**
	 * @brief Convert a single bone to a tapered capsule
	 *
	 * @param bone Input bone structure
	 * @param radius_top Radius at bone end (wider end)
	 * @param radius_bottom Radius at bone start (narrower end)
	 * @param color Capsule color for visualization
	 * @return Generated tapered capsule
	 */
	static std::unique_ptr<TaperedCapsule> boneToTaperedCapsule(
			const Bone &bone,
			double radius_top = 0.1,
			double radius_bottom = 0.1,
			Vec3d color = Vec3d(0.8, 0.7, 0.6) // Skin-like color
	);

	/**
	 * @brief Subdivide a bone into multiple aligned sub-bones for improved capsule resolution
	 *
	 * @param bone Input bone to subdivide
	 * @param num_subdivisions Number of sub-bones to create
	 * @return Vector of sub-bones along the original bone axis
	 */
	static std::vector<Bone> subdivideBone(const Bone &bone, int num_subdivisions);

	/**
	 * @brief Subdivide a bone into sub-bones with continuous radius assignment from pre-sampled profile
	 *
	 * Creates sub-bones with radii that smoothly transition along the full bone length,
	 * preventing discontinuities between adjacent capsules.
	 *
	 * @param bone Input bone to subdivide
	 * @param radius_profile Pre-sampled radius values along full bone length
	 * @param num_subdivisions Number of sub-bones to create
	 * @return Vector of (sub-bone, radii) pairs with continuous radius transitions
	 */
	static std::vector<std::pair<Bone, std::pair<double, double>>> subdivideBoneWithRadii(
			const Bone &bone,
			const std::vector<double> &radius_profile,
			int num_subdivisions);

	/**
	 * @brief Validate skeleton structure before processing
	 */
	static bool validateSkeleton(const Skeleton &skeleton);
};

} // namespace tool_cloth_dynamics
