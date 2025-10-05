#pragma once

#include "CapsuleGenerator.h"
#include "SkeletonLoader.h"
#include "TaperedCapsule.h"
#include <map>
#include <string>
#include <vector>

namespace tool_cloth_dynamics {

/**
 * @brief A single segment in a VRM Spring Bone chain
 *
 * Each segment is a tapered capsule connected to its parent via spring constraints.
 * Used for hair, skirts, tails, and other dynamic secondary animation.
 */
struct SpringBoneSegment {
	std::unique_ptr<TaperedCapsule> capsule;
	int parent_segment_id; // -1 for root segment
	double stiffness; // Spring constant to parent
	double damping; // Spring damping coefficient
	double gravity_power; // How much gravity affects this segment
	Vec3d gravity_dir; // Gravity direction (usually down)

	/**
	 * @brief Constructor
	 */
	SpringBoneSegment(
			std::unique_ptr<TaperedCapsule> cap,
			int parent_id = -1,
			double spring_stiffness = 100.0,
			double spring_damping = 0.5,
			double grav_power = 1.0,
			Vec3d grav_dir = Vec3d(0, -1, 0)) :
			capsule(std::move(cap)),
			parent_segment_id(parent_id),
			stiffness(spring_stiffness),
			damping(spring_damping),
			gravity_power(grav_power),
			gravity_dir(grav_dir.normalized()) {}
};

/**
 * @brief A chain of spring-connected bone segments
 *
 * Represents a single VRM Spring Bone chain (e.g., one strand of hair,
 * one part of a skirt). Multiple chains make up a complete spring bone system.
 */
class SpringBoneChain {
public:
	std::vector<SpringBoneSegment> segments;
	std::string name; // "hair_front", "skirt_back", etc.
	int root_bone_id; // Which skeleton bone this chain attaches to

	/**
	 * @brief Subdivide a bone into spring segments
	 *
	 * @param bone Bone to subdivide
	 * @param num_subdivisions Number of segments to create
	 * @param taper_factor Radius reduction per segment (0.9 = 10% smaller each segment)
	 * @param spring_stiffness Spring constant between segments
	 * @param spring_damping Damping coefficient
	 * @return Spring bone chain
	 */
	static SpringBoneChain subdivide(
			const Bone &bone,
			int num_subdivisions = 3,
			double taper_factor = 0.9,
			double spring_stiffness = 100.0,
			double spring_damping = 0.5);

	/**
	 * @brief Add all segments to simulation
	 */
	void addToSimulation(Simulation *simulation);

	/**
	 * @brief Get total number of segments in chain
	 */
	size_t getSegmentCount() const { return segments.size(); }

	/**
	 * @brief Check if chain is valid
	 */
	bool isValid() const { return !segments.empty(); }

private:
	SpringBoneChain() = default;
};

/**
 * @brief VRM Spring Bone configuration parameters
 *
 * Matches VRM specification for spring bone settings.
 * See: https://vrm.dev/en/univrm/components/univrm_secondary/
 */
struct VRMSpringBoneParams {
	std::string bone_name; // "hair_front", "skirt_back", etc.
	int subdivisions = 3; // Number of segments
	double stiffness = 100.0; // Spring constant
	double damping = 0.5; // Spring damping
	double gravity_power = 1.0; // How much gravity affects
	Vec3d gravity_dir = Vec3d(0, -1, 0); // Gravity direction
	double taper_factor = 0.9; // Radius reduction per segment
	double radius_scale = 1.0; // Scale factor for capsule radii
};

/**
 * @brief Complete VRM Spring Bone configuration
 */
struct VRMSpringBoneConfig {
	std::map<std::string, VRMSpringBoneParams> bone_configs;

	/**
	 * @brief Load from VRM-style JSON configuration
	 */
	static VRMSpringBoneConfig loadFromJSON(const std::string &filepath);

	/**
	 * @brief Save to JSON file
	 */
	void saveToJSON(const std::string &filepath) const;

	/**
	 * @brief Create default configuration for common bone types
	 */
	static VRMSpringBoneConfig createDefault();
};

/**
 * @brief Utility class for extending skeletons with VRM Spring Bones
 *
 * Takes a base skeleton and adds subdivided spring bone chains for
 * secondary animation of hair, clothing, accessories, etc.
 */
class SpringBoneExtender {
public:
	/**
	 * @brief Configuration for spring bone extension
	 */
	struct Config {
		int subdivisions_per_bone = 3;
		double taper_factor = 0.9;
		double spring_stiffness = 100.0;
		double spring_damping = 0.5;
		double gravity_power = 1.0;
		Vec3d gravity_dir = Vec3d(0, -1, 0);
	};

	/**
	 * @brief Extend a capsule rig with spring bones
	 *
	 * @param base_rig Base skeleton rig
	 * @param config Spring bone configuration
	 * @return Extended rig with spring bones
	 */
	static CapsuleRig extend(const CapsuleRig &base_rig, const Config &config);

	/**
	 * @brief Extend with VRM Spring Bone configuration
	 *
	 * @param base_rig Base skeleton rig
	 * @param vrm_config VRM-style configuration
	 * @return Extended rig with VRM spring bones
	 */
	static CapsuleRig extendWithVRMSpringBones(
			const CapsuleRig &base_rig,
			const VRMSpringBoneConfig &vrm_config);

	/**
	 * @brief Create spring bone chains for specific bone types
	 *
	 * @param skeleton Input skeleton
	 * @param bone_types Types to create chains for ("hair", "skirt", "tail")
	 * @param config Spring bone configuration
	 * @return Vector of spring bone chains
	 */
	static std::vector<SpringBoneChain> createChainsForBoneTypes(
			const Skeleton &skeleton,
			const std::vector<std::string> &bone_types,
			const Config &config);

	/**
	 * @brief Auto-detect potential spring bone locations
	 *
	 * Analyzes skeleton structure to suggest which bones would benefit
	 * from spring bone subdivision (long bones, end effectors, etc.).
	 *
	 * @param skeleton Input skeleton
	 * @return Vector of bone indices suitable for spring bones
	 */
	static std::vector<int> detectSpringBoneCandidates(const Skeleton &skeleton);

private:
	/**
	 * @brief Create a spring bone chain from a single bone
	 */
	static SpringBoneChain createChainFromBone(
			const Bone &bone,
			const Config &config,
			const std::string &chain_name = "");

	/**
	 * @brief Validate spring bone configuration
	 */
	static bool validateConfig(const Config &config);
};

/**
 * @brief Demo configurations for common spring bone setups
 */
namespace SpringBonePresets {
/**
 * @brief Hair spring bone configuration
 */
VRMSpringBoneConfig createHairConfig();

/**
 * @brief Skirt/dress spring bone configuration
 */
VRMSpringBoneConfig createSkirtConfig();

/**
 * @brief Tail spring bone configuration (for characters with tails)
 */
VRMSpringBoneConfig createTailConfig();

/**
 * @brief Complete character configuration (hair + clothing)
 */
VRMSpringBoneConfig createFullCharacterConfig();
} //namespace SpringBonePresets

} // namespace tool_cloth_dynamics
