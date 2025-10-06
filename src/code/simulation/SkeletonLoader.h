#pragma once

#include "../engine/UtilityFunctions.h"
#include <map>
#include <string>
#include <vector>

namespace tool_cloth_dynamics {

/**
 * @brief Single bone transformation (position, rotation, scale)
 */
struct BoneTransform {
	Vec3d translation = Vec3d::Zero();
	Quat rotation = Quat::Identity();
	Vec3d scale = Vec3d::Ones();

	BoneTransform() = default;
	BoneTransform(const Vec3d &t, const Quat &r, const Vec3d &s) :
			translation(t), rotation(r), scale(s) {}
};

/**
 * @brief Complete pose of a skeleton at a single point in time
 *
 * Maps bone names to their transformations for animation and retargeting.
 */
struct SkeletonPose {
	std::map<std::string, BoneTransform> bone_transforms;

	/**
	 * @brief Check if pose contains a specific bone
	 */
	bool hasBone(const std::string &bone_name) const {
		return bone_transforms.find(bone_name) != bone_transforms.end();
	}

	/**
	 * @brief Get transformation for a bone (returns identity if not found)
	 */
	BoneTransform getBoneTransform(const std::string &bone_name) const {
		auto it = bone_transforms.find(bone_name);
		return (it != bone_transforms.end()) ? it->second : BoneTransform();
	}
};

/**
 * @brief Represents a single bone in a skeleton
 *
 * Each bone connects two joints and can generate a tapered capsule
 * following the KHR_implicit_shapes specification.
 */
struct Bone {
	Vec3d start; // Joint position (e.g., shoulder)
	Vec3d end; // Joint position (e.g., elbow)
	int parent_id; // -1 for root, index into skeleton bones
	std::string name; // Optional: "upper_arm_L", "spine", etc.

	/**
	 * @brief Get bone axis direction (normalized)
	 */
	Vec3d getAxis() const {
		Vec3d axis = end - start;
		double len = axis.norm();
		return (len > 1e-10) ? axis / len : Vec3d(0, 1, 0);
	}

	/**
	 * @brief Get bone length (distance between joints)
	 */
	double getLength() const {
		return (end - start).norm();
	}

	/**
	 * @brief Get bone center point
	 */
	Vec3d getCenter() const {
		return (start + end) * 0.5;
	}
};

/**
 * @brief Complete skeleton structure with bones and joints
 *
 * Loaded from skeleton.obj files with vertices (joints) and line segments (bones).
 * Compatible with cloth-fit retargeting constraints.
 */
struct Skeleton {
	std::vector<Bone> bones;
	std::vector<Vec3d> joints; // All unique joint positions

	/**
	 * @brief Build skeleton from OBJ file with vertices and line segments
	 *
	 * @param filepath Path to skeleton.obj file
	 * @return Loaded skeleton structure
	 * @throws std::runtime_error if file cannot be loaded or parsed
	 */
	static Skeleton fromOBJ(const std::string &filepath);

	/**
	 * @brief Get total number of bones
	 */
	size_t getBoneCount() const { return bones.size(); }

	/**
	 * @brief Get total number of joints
	 */
	size_t getJointCount() const { return joints.size(); }

	/**
	 * @brief Validate skeleton structure
	 *
	 * @return true if skeleton has valid bones and joints
	 */
	bool isValid() const {
		return !bones.empty() && !joints.empty();
	}
};

/**
 * @brief Utility class for loading skeleton data from OBJ files
 *
 * Handles the parsing of skeleton.obj files containing:
 * - v x y z (vertices = joint positions)
 * - l v1 v2 (line segments = bones connecting joints)
 *
 * Compatible with cloth-fit's edge matching requirements for retargeting.
 */
class SkeletonLoader {
public:
	/**
	 * @brief Load skeleton from OBJ file
	 *
	 * @param filepath Path to skeleton.obj file
	 * @return Loaded skeleton structure
	 * @throws std::runtime_error if file cannot be loaded or format is invalid
	 */
	static Skeleton loadFromOBJ(const std::string &filepath);

	/**
	 * @brief Parse just the bone line segments from OBJ file
	 *
	 * @param filepath Path to skeleton.obj file
	 * @return Vector of bone endpoints (start, end) pairs
	 */
	static std::vector<std::pair<Vec3d, Vec3d>> loadBones(const std::string &filepath);

	/**
	 * @brief Validate that two skeletons have identical edge structure
	 *
	 * This implements cloth-fit's are_same_edges() constraint for retargeting.
	 * Both skeletons must have identical line segments (same connectivity).
	 *
	 * @param skeleton_a First skeleton
	 * @param skeleton_b Second skeleton
	 * @return true if edge structures match exactly
	 */
	static bool validateEdgeCompatibility(const Skeleton &skeleton_a, const Skeleton &skeleton_b);

private:
	/**
	 * @brief Parse vertices (joint positions) from OBJ file
	 */
	static std::vector<Vec3d> parseVertices(const std::string &filepath);

	/**
	 * @brief Parse line segments (bones) from OBJ file
	 */
	static std::vector<std::pair<int, int>> parseLines(const std::string &filepath);

	/**
	 * @brief Check if file exists and is readable
	 */
	static bool validateFile(const std::string &filepath);
};

} // namespace tool_cloth_dynamics
