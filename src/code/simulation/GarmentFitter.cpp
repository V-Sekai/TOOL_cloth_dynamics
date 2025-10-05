#include "GarmentFitter.h"
#include "AttachmentSpring.h"
#include "CapsuleGenerator.h"
#include "Simulation.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace tool_cloth_dynamics {

void GarmentFitter::fitGarmentToRig(
		Simulation *sim,
		const CapsuleRig &rig) {
	if (!sim) {
		std::cerr << "ERROR: Null simulation provided to GarmentFitter" << std::endl;
		return;
	}

	std::cout << "Starting four-phase garment inflation algorithm..." << std::endl;

	// Phase 1: Initialize at center
	collapseToCentroid(sim, rig);

	// Phase 2: Expand with physics
	expandWithCollisions(sim, rig);

	// Phase 3: Apply bone-based anchors
	auto anchors = generateBoneAnchors(rig);
	applyBoneAnchors(sim, rig, anchors);

	// Phase 4: Refine under gravity
	refineDrape(sim, rig);

	std::cout << "Garment inflation completed with " << anchors.size() << " bone anchors" << std::endl;
}

// Phase 1: Collapse garment toward body centroid
void GarmentFitter::collapseToCentroid(
		Simulation *sim,
		const CapsuleRig &rig) {
	Vec3d centroid = computeBodyCentroid(rig);
	const double collapse_factor = 0.3; // Collapse 30% toward centroid

	std::cout << "Phase 1: Collapsing garment toward centroid " << centroid.transpose() << std::endl;

	// Move each cloth vertex toward centroid
	for (size_t i = 0; i < sim->particles.size(); ++i) {
		Vec3d &pos = sim->particles[i].pos;
		Vec3d to_centroid = centroid - pos;
		pos += to_centroid * collapse_factor;
	}

	// Reset velocities to prevent instability
	for (auto &particle : sim->particles) {
		particle.velocity = Vec3d::Zero();
	}
}

// Phase 2: Expand with repulsive forces and collision guidance
void GarmentFitter::expandWithCollisions(
		Simulation *sim,
		const CapsuleRig &rig) {
	const int expansion_steps = 50;
	const double repulsion_strength = 2.0;

	std::cout << "Phase 2: Expanding garment with collision guidance (" << expansion_steps << " steps)" << std::endl;

	// Store original simulation settings
	bool original_gravity = Simulation::gravityEnabled;
	bool original_wind = Simulation::windEnabled;
	bool original_contact = Simulation::contactEnabled;

	// Enable collision detection but disable gravity and wind for controlled expansion
	sim->setWindAncCollision(false, true, false);

	// Add repulsive forces as external forces
	for (int step = 0; step < expansion_steps; ++step) {
		// Apply repulsive forces from capsules as external forces
		for (size_t i = 0; i < sim->particles.size(); ++i) {
			Vec3d total_force = Vec3d::Zero();

			for (const auto &capsule : rig.getCapsules()) {
				Vec3d repulsive_force = computeRepulsiveForce(
						sim->particles[i].pos, capsule.get(), repulsion_strength);
				total_force += repulsive_force;
			}

			// Add repulsive force to particle's external force field
			sim->external_force_field.segment(i * 3, 3) += total_force;
		}

		// Step the simulation using DiffCloth's physics
		sim->step();

		// Clear external forces for next step
		sim->external_force_field.setZero();
	}

	// Restore original simulation settings
	sim->setWindAncCollision(original_wind, original_contact, false);
	Simulation::gravityEnabled = original_gravity;
}

// Phase 3: Apply bone-based anchor points
void GarmentFitter::applyBoneAnchors(
		Simulation *sim,
		const CapsuleRig &rig,
		const std::vector<AnchorPoint> &anchors) {
	std::cout << "Phase 3: Applying " << anchors.size() << " bone-based anchor points" << std::endl;

	for (size_t anchor_idx = 0; anchor_idx < anchors.size(); ++anchor_idx) {
		const auto &anchor = anchors[anchor_idx];

		// Find the corresponding capsule for this bone
		if (anchor.bone_index >= rig.getCapsules().size()) {
			std::cout << "Warning: Bone index " << anchor.bone_index << " out of range" << std::endl;
			continue;
		}

		const auto &target_capsule = rig.getCapsules()[anchor.bone_index];

		// Find closest cloth vertex to bone position within max_distance
		int best_vertex = -1;
		double best_distance = anchor.max_distance;

		for (size_t i = 0; i < sim->particles.size(); ++i) {
			double dist = (sim->particles[i].pos - anchor.bone_position).norm();
			if (dist < best_distance) {
				best_distance = dist;
				best_vertex = static_cast<int>(i);
			}
		}

		if (best_vertex >= 0) {
			// Create attachment spring
			AttachmentSpring attachment(
					best_vertex,
					&sim->particles,
					static_cast<int>(sim->sysMat[0].fixedPoints.size()),
					&sim->sysMat[0].fixedPoints);

			// Set stiffness
			AttachmentSpring::k_stiff = anchor.stiffness;

			// Add fixed point at bone position
			FixedPoint fixed_point;
			fixed_point.pos = anchor.bone_position;
			sim->sysMat[0].fixedPoints.push_back(fixed_point);

			sim->sysMat[0].attachments.push_back(attachment);

			std::cout << "  Anchored vertex " << best_vertex << " to bone '" << anchor.bone_name
					  << "' (stiffness: " << anchor.stiffness << ")" << std::endl;
		}
	}
}

// Phase 4: Refine drape under physics
void GarmentFitter::refineDrape(
		Simulation *sim,
		const CapsuleRig &rig) {
	const int refinement_steps = 100;

	std::cout << "Phase 4: Refining drape under physics (" << refinement_steps << " steps)" << std::endl;

	// Store original simulation settings
	bool original_gravity = Simulation::gravityEnabled;
	bool original_wind = Simulation::windEnabled;
	bool original_contact = Simulation::contactEnabled;

	// Enable gravity and collision for natural drape settling
	sim->setWindAncCollision(false, true, true);
	Simulation::gravityEnabled = true;

	// Run physics simulation with gravity to settle cloth
	for (int step = 0; step < refinement_steps; ++step) {
		sim->step();
	}

	// Restore original simulation settings
	sim->setWindAncCollision(original_wind, original_contact, false);
	Simulation::gravityEnabled = original_gravity;
}

Vec3d GarmentFitter::computeBodyCentroid(const CapsuleRig &rig) {
	Vec3d centroid = Vec3d::Zero();
	size_t count = 0;

	for (const auto &capsule : rig.getCapsules()) {
		centroid += capsule->center;
		count++;
	}

	return count > 0 ? centroid / count : Vec3d(0, 0, 0);
}

Vec3d GarmentFitter::computeRepulsiveForce(
		const Vec3d &cloth_vertex,
		const TaperedCapsule *capsule,
		double strength) {
	Vec3d closest_point, normal;
	double dist = capsule->closestPointOnSurface(cloth_vertex, closest_point, normal);

	if (dist < 0.001) {
		// Inside capsule - strong repulsive force
		return normal * strength * 10.0;
	} else if (dist < 0.1) {
		// Near capsule - moderate repulsive force
		double force_magnitude = strength / (dist * dist + 0.01);
		return normal * force_magnitude;
	}

	return Vec3d::Zero();
}

const TaperedCapsule *GarmentFitter::findCapsuleByName(
		const CapsuleRig &rig,
		const std::string &name) {
	// Simplified: just return first capsule for now
	// In full implementation, would map names to specific capsules
	if (!rig.getCapsules().empty()) {
		return rig.getCapsules()[0].get();
	}
	return nullptr;
}

/**
 * @brief Generate bone-based anchor points directly from skeleton bones
 */
std::vector<AnchorPoint> GarmentFitter::generateBoneAnchors(const CapsuleRig &rig) {
	std::vector<AnchorPoint> anchors;
	const auto &skeleton = rig.getSkeleton();
	const auto &bones = skeleton.bones;

	for (size_t i = 0; i < bones.size(); ++i) {
		const auto &bone = bones[i];
		// Use bone center as anchor position
		Vec3d bone_center = (bone.start + bone.end) * 0.5;

		// Create anchor point for this bone
		anchors.emplace_back(bone.name, bone_center, static_cast<int>(i), 0.8, 0.15);
	}

	return anchors;
}

} // namespace tool_cloth_dynamics
