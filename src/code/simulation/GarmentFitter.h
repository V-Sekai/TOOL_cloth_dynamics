#pragma once

#include "../engine/Macros.h" // For Vec3d
#include <Eigen/Dense> // For Eigen::MatrixXd
#include <string>
#include <vector>

// Forward declaration to avoid circular dependency
class Simulation;

namespace tool_cloth_dynamics {

// Forward declarations to avoid circular dependencies
class CapsuleRig;
class TaperedCapsule;

/**
 * @brief Bone-based anchor point for garment attachment
 */
struct AnchorPoint {
	std::string bone_name; // Name of the skeleton bone
	Vec3d bone_position; // Position of the bone center
	int bone_index; // Index of the bone in skeleton
	double stiffness; // Attachment strength (0-1)
	double max_distance; // Maximum attachment distance

	// Constructor for proper initialization
	AnchorPoint(std::string name, Vec3d pos, int index, double stiff, double dist) :
			bone_name(std::move(name)), bone_position(pos), bone_index(index), stiffness(stiff), max_distance(dist) {}
};

/**
 * @brief Four-phase garment inflation algorithm for natural cloth draping
 *
 * Simplified approach: uses bone-based anchor points directly from skeleton
 * instead of complex garment classification and mesh analysis.
 */
class GarmentFitter {
public:
	/**
	 * @brief Fit garment to capsule rig using bone-based inflation algorithm
	 *
	 * @param sim Simulation containing cloth mesh
	 * @param rig Capsule rig for collision and anchoring
	 */
	static void fitGarmentToRig(
			Simulation *sim,
			const CapsuleRig &rig);

private:
	/**
	 * @brief Generate bone-based anchor points directly from skeleton bones
	 */
	static std::vector<AnchorPoint> generateBoneAnchors(const CapsuleRig &rig);

	/**
	 * @brief Apply bone-based anchor points
	 */
	static void applyBoneAnchors(
			Simulation *sim,
			const CapsuleRig &rig,
			const std::vector<AnchorPoint> &anchors);

	/**
	 * @brief Phase 1: Collapse garment toward body centroid
	 */
	static void collapseToCentroid(
			Simulation *sim,
			const CapsuleRig &rig);

	/**
	 * @brief Phase 2: Expand with repulsive forces and collision guidance
	 */
	static void expandWithCollisions(
			Simulation *sim,
			const CapsuleRig &rig);

	/**
	 * @brief Phase 4: Refine drape under physics
	 */
	static void refineDrape(
			Simulation *sim,
			const CapsuleRig &rig);

	/**
	 * @brief Compute body centroid from capsule rig
	 */
	static Vec3d computeBodyCentroid(const CapsuleRig &rig);

	/**
	 * @brief Apply repulsive force from capsule surface
	 */
	static Vec3d computeRepulsiveForce(
			const Vec3d &cloth_vertex,
			const TaperedCapsule *capsule,
			double strength);

	/**
	 * @brief Find capsule by semantic name
	 */
	static const TaperedCapsule *findCapsuleByName(
			const CapsuleRig &rig,
			const std::string &name);
};

} // namespace tool_cloth_dynamics
