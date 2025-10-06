#include "SkeletonPipelineTest.h"
#include "../engine/MeshFileHandler.h"
#include "../optimization/OptimizationTaskConfigurations.h"
#include "../supports/Logging.h"
#include "CapsuleGenerator.h"
#include "SkeletonDemoScenes.h"
#include "SkeletonLoader.h"
#include "SparseOctree.h"

// Include here to avoid circular dependency in header
#include "CapsuleGenerator.h"
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>

namespace tool_cloth_dynamics {

bool SkeletonPipelineTest::computeCapsuleFitLossReport(const CapsuleRig &rig, const std::string &avatar_mesh_path) {
	try {
		// Load avatar mesh for loss computation
		std::vector<Vec3d> avatar_vertices;
		std::vector<Vec3i> avatar_faces;
		MeshFileHandler::loadOBJFile(avatar_mesh_path.c_str(), avatar_vertices, avatar_faces);

		if (avatar_vertices.empty()) {
			return false; // Avatar mesh not found
		}

		// Convert to Eigen matrix
		MatrixXd avatar_V(3, avatar_vertices.size());
		for (size_t i = 0; i < avatar_vertices.size(); ++i) {
			avatar_V(0, i) = avatar_vertices[i].x();
			avatar_V(1, i) = avatar_vertices[i].y();
			avatar_V(2, i) = avatar_vertices[i].z();
		}

		// Build octree for efficient distance queries
		SparseOctree octree;
		octree.build(avatar_V);

		// Compute surface distance loss (L_surface)
		double surface_loss = 0.0;
		int surface_samples = 0;
		const int avatar_sample_rate = std::max(1, static_cast<int>(avatar_vertices.size() / 1000)); // Sample ~1000 points

		for (size_t i = 0; i < avatar_vertices.size(); i += avatar_sample_rate) {
			const Vec3d &vertex = avatar_vertices[i];

			// Find closest capsule surface
			double min_distance = std::numeric_limits<double>::max();

			for (const auto &capsule_ptr : rig.getCapsules()) {
				const TaperedCapsule *capsule = capsule_ptr.get();
				if (!capsule)
					continue;

				// Get distance from vertex to capsule surface
				Vec3d closest_point;
				Vec3d surface_normal;
				double distance = capsule->closestPointOnSurface(vertex, closest_point, surface_normal);
				min_distance = std::min(min_distance, distance);
			}

			// Accumulate surface loss (penalize penetration and excessive distance)
			if (min_distance > 0.1) { // Allow small gaps (< 10cm)
				surface_loss += min_distance * min_distance;
			}
			surface_samples++;
		}

		if (surface_samples > 0) {
			surface_loss /= surface_samples; // Normalize by sample count
		}

		// Compute volume coverage loss (L_volume) - simplified sampling
		double coverage_loss = 0.0;
		int coverage_samples = 0;

		// Sample at regular intervals along all capsule surfaces
		for (const auto &capsule_ptr : rig.getCapsules()) {
			const TaperedCapsule *capsule = capsule_ptr.get();
			if (!capsule)
				continue;

			// Sample points on capsule surface
			const int radial_samples = 6;
			const int height_samples = 5;

			for (int h = 0; h < height_samples; ++h) {
				for (int r = 0; r < radial_samples; ++r) {
					// Generate surface point on capsule
					double height_fraction = static_cast<double>(h) / (height_samples - 1);
					double angle = 2.0 * M_PI * r / radial_samples;

					// Interpolate radius along height (handle tapering)
					double local_radius = capsule->radius_bottom +
							height_fraction * (capsule->radius_top - capsule->radius_bottom);

					// Create surface point
					Vec3d local_point;
					local_point.x() = local_radius * std::cos(angle);
					local_point.y() = -capsule->mid_height + 2 * capsule->mid_height * height_fraction;
					local_point.z() = local_radius * std::sin(angle);

					// Transform to world coordinates
					Mat3x3d rotation_matrix = capsule->buildRotationMatrix(Vec3d(0, 1, 0), capsule->axis);
					Vec3d world_point = rotation_matrix * local_point + capsule->center;

					// Check distance to avatar mesh
					double distance_to_mesh = octree.nearestDistance(world_point);

					// Penalize points too far from mesh (uncovered volume)
					if (distance_to_mesh > 0.15) { // Allow 15cm margin
						coverage_loss += 1.0; // Binary coverage loss
					}
					coverage_samples++;
				}
			}
		}

		if (coverage_samples > 0) {
			coverage_loss /= coverage_samples; // Normalize to [0,1] range
		}

		// Compute regularization loss (capsule size constraints)
		double regularization_loss = 0.0;
		for (const auto &capsule_ptr : rig.getCapsules()) {
			const TaperedCapsule *capsule = capsule_ptr.get();
			if (!capsule)
				continue;

			// Penalize capsules that are too small or too large
			double avg_radius = (capsule->radius_top + capsule->radius_bottom) * 0.5;

			if (avg_radius < 0.02)
				regularization_loss += (0.02 - avg_radius) * 100; // Min 2cm
			if (avg_radius > 1.0)
				regularization_loss += (avg_radius - 1.0) * 10; // Max 1m

			// Penalize extreme tapering
			double radius_ratio = std::max(capsule->radius_top, capsule->radius_bottom) /
					std::max(capsule->radius_bottom, capsule->radius_top);
			if (radius_ratio > 5.0)
				regularization_loss += (radius_ratio - 5.0) * 0.1;
		}

		// Total CAPSULE_FIT loss
		double total_loss = surface_loss + coverage_loss + regularization_loss;

		// Report loss breakdown
		std::cout << "--- CAPSULE_FIT Loss Analysis ---" << std::endl;
		std::cout << "  Surface Distance Loss (L_surface): " << surface_loss << std::endl;
		std::cout << "  Volume Coverage Loss (L_volume): " << coverage_loss << std::endl;
		std::cout << "  Regularization Loss (L_regular): " << regularization_loss << std::endl;
		std::cout << "  Total CAPSULE_FIT Loss: " << total_loss << std::endl;

		// Quality assessment
		if (total_loss < 0.1) {
			std::cout << "🎯 Excellent fit - low visual distortion, high fidelity" << std::endl;
		} else if (total_loss < 0.5) {
			std::cout << "✅ Good fit - acceptable balance of visual vs simulation quality" << std::endl;
		} else if (total_loss < 2.0) {
			std::cout << "⚠️  Poor fit - noticeable visual artifacts, marginal simulation quality" << std::endl;
		} else {
			std::cout << "❌ Bad fit - significant visual loss, poor simulation accuracy" << std::endl;
		}

		return true;

	} catch (const std::exception &e) {
		std::cout << "⚠️  Loss computation failed: " << e.what() << std::endl;
		return false;
	}
}

} // namespace tool_cloth_dynamics
