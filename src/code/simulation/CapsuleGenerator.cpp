#include "CapsuleGenerator.h"
#include "AttachmentSpring.h"
#include "Simulation.h"
#include "SparseOctree.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>

// Include here to avoid circular dependency in header
#include "GarmentFitter.h"

namespace tool_cloth_dynamics {

CapsuleRig CapsuleRig::generate(const Skeleton &skel, double radius) {
	if (!CapsuleGenerator::validateSkeleton(skel)) {
		throw std::runtime_error("Invalid skeleton provided to CapsuleRig::generate");
	}

	CapsuleRig rig;
	rig.skeleton = skel;
	rig.capsules = std::move(CapsuleGenerator::generateCapsules(skel, radius));

	// Build bone-to-capsule mapping
	rig.bone_to_capsule_map.resize(skel.getBoneCount());
	for (size_t i = 0; i < skel.getBoneCount(); ++i) {
		rig.bone_to_capsule_map[i] = static_cast<int>(i);
	}

	std::cout << "Generated CapsuleRig with " << rig.getCapsuleCount() << " capsules" << std::endl;

	return rig;
}

CapsuleRig CapsuleRig::generateFromPairedAssets(const std::string &asset_directory) {
	// Construct expected file paths
	std::string skeleton_path = asset_directory + "/skeleton.obj";
	std::string mesh_path = asset_directory + "/avatar.obj";

	// Try different mesh file names
	if (!std::filesystem::exists(mesh_path)) {
		mesh_path = asset_directory + "/garment.obj";
	}
	if (!std::filesystem::exists(mesh_path)) {
		mesh_path = asset_directory + "/mesh.obj";
	}

	// Validate asset pairing
	if (!validateAssetPairing(skeleton_path, mesh_path)) {
		throw std::runtime_error("Assets not properly paired in directory: " + asset_directory);
	}

	// Load skeleton
	Skeleton skeleton = SkeletonLoader::loadFromOBJ(skeleton_path);

	// Load mesh for radius estimation
	std::vector<Vec3d> mesh_vertices;
	std::vector<Vec3i> mesh_faces;
	MeshFileHandler::loadOBJFile(mesh_path.c_str(), mesh_vertices, mesh_faces);

	// Convert mesh to Eigen matrix for octree
	MatXd mesh_V(3, mesh_vertices.size());
	for (size_t i = 0; i < mesh_vertices.size(); ++i) {
		mesh_V(0, i) = mesh_vertices[i].x();
		mesh_V(1, i) = mesh_vertices[i].y();
		mesh_V(2, i) = mesh_vertices[i].z();
	}

	// Generate rig with mesh-based radius estimation
	CapsuleRig rig;
	rig.skeleton = skeleton;
	rig.capsules = std::move(CapsuleGenerator::generateCapsulesWithAdvancedRadii(skeleton, mesh_V, true));

	// Build bone-to-capsule mapping
	rig.bone_to_capsule_map.resize(skeleton.getBoneCount());
	for (size_t i = 0; i < skeleton.getBoneCount(); ++i) {
		rig.bone_to_capsule_map[i] = static_cast<int>(i);
	}

	rig.asset_source_directory = asset_directory;

	std::cout << "Generated CapsuleRig from paired assets with mesh-based radius estimation: " << asset_directory << std::endl;

	return rig;
}

void CapsuleRig::addToSimulation(Simulation *simulation) {
	if (!simulation) {
		std::cerr << "Error: Null simulation provided to CapsuleRig::addToSimulation" << std::endl;
		return;
	}

	for (auto &capsule : capsules) {
		// Add to primitives list for collision detection
		simulation->primitives.push_back(capsule.get());

		// Add to rendering list for visualization
		simulation->allPrimitivesToRender.push_back(capsule.get());
	}

	std::cout << "Added " << capsules.size() << " capsules to simulation" << std::endl;
}

void CapsuleRig::pinGarmentToCapsules(Simulation *simulation, double pin_distance) {
	if (!simulation) {
		std::cerr << "Error: Null simulation provided to CapsuleRig::pinGarmentToCapsules" << std::endl;
		return;
	}

	int pins_created = 0;

	// Check each cloth vertex against all capsules
	for (int i = 0; i < simulation->particles.size(); ++i) {
		Vec3d cloth_vertex = simulation->particles[i].pos;
		bool pinned = false;

		for (const auto &capsule : capsules) {
			Vec3d closest_point;
			Vec3d surface_normal;
			double distance = capsule->closestPointOnSurface(cloth_vertex, closest_point, surface_normal);

			if (distance < pin_distance) {
				// Create attachment spring using existing constructor
				AttachmentSpring attachment(
						i, // p1_idx: vertex index
						&simulation->particles, // pArr: particle array
						static_cast<int>(simulation->sysMat[0].fixedPoints.size()), // pfixed_idx: new fixed point index
						&simulation->sysMat[0].fixedPoints // fixed points array
				);

				// Add fixed point at current particle position
				FixedPoint fixed_point;
				fixed_point.pos = cloth_vertex;
				simulation->sysMat[0].fixedPoints.push_back(fixed_point);

				simulation->sysMat[0].attachments.push_back(attachment);
				pins_created++;
				pinned = true;
				break; // Pin to first close capsule only
			}
		}
	}

	std::cout << "Created " << pins_created << " garment pins to capsules" << std::endl;
}

void CapsuleRig::fitGarmentToRig(Simulation *simulation) {
	if (!simulation) {
		std::cerr << "Error: Null simulation provided to CapsuleRig::fitGarmentToRig" << std::endl;
		return;
	}

	// Use the simplified four-phase garment inflation algorithm with bone-based anchors
	GarmentFitter::fitGarmentToRig(simulation, *this);
}

bool CapsuleRig::validateAssetPairing(const std::string &skeleton_path, const std::string &mesh_path) {
	// Check if both files exist
	if (!std::filesystem::exists(skeleton_path)) {
		std::cout << "Warning: Skeleton file not found: " << skeleton_path << std::endl;
		return false;
	}

	if (!std::filesystem::exists(mesh_path)) {
		std::cout << "Warning: Mesh file not found: " << mesh_path << std::endl;
		return false;
	}

	// Check if they're in the same directory (pairing constraint)
	std::filesystem::path skeleton_dir = std::filesystem::path(skeleton_path).parent_path();
	std::filesystem::path mesh_dir = std::filesystem::path(mesh_path).parent_path();

	if (skeleton_dir != mesh_dir) {
		std::cout << "Warning: Assets not in same directory - skeleton: " << skeleton_dir.string() << ", mesh: " << mesh_dir.string() << std::endl;
		return false;
	}

	// Basic file size validation (skeleton files should be much smaller than mesh files)
	try {
		auto skeleton_size = std::filesystem::file_size(skeleton_path);
		auto mesh_size = std::filesystem::file_size(mesh_path);

		if (skeleton_size > mesh_size) {
			std::cout << "Warning: Suspicious file sizes - skeleton larger than mesh" << std::endl;
			return false;
		}
	} catch (const std::filesystem::filesystem_error &e) {
		std::cout << "Warning: Could not check file sizes: " << e.what() << std::endl;
		// Continue anyway - size check is not critical
	}

	return true;
}

std::vector<std::unique_ptr<TaperedCapsule>> CapsuleGenerator::generateCapsules(
		const Skeleton &skeleton,
		double radius) {
	if (!validateSkeleton(skeleton)) {
		throw std::runtime_error("Invalid skeleton provided to CapsuleGenerator::generateCapsules");
	}

	std::vector<std::unique_ptr<TaperedCapsule>> capsules;
	capsules.reserve(skeleton.getBoneCount());

	for (const Bone &bone : skeleton.bones) {
		auto capsule = boneToTaperedCapsule(bone, radius, radius);
		capsules.push_back(std::move(capsule));
	}

	std::cout << "Generated " << capsules.size() << " capsules from skeleton (fixed radius: "
			  << radius << ")" << std::endl;

	return capsules;
}

std::vector<std::unique_ptr<TaperedCapsule>> CapsuleGenerator::generateCapsulesWithAdvancedRadii(
		const Skeleton &skeleton,
		const MatXd &mesh_vertices,
		bool use_tapered_radii) {
	if (!validateSkeleton(skeleton)) {
		throw std::runtime_error("Invalid skeleton provided to CapsuleGenerator::generateCapsulesWithAdvancedRadii");
	}

	if (mesh_vertices.cols() == 0) {
		std::cout << "Warning: Empty mesh provided, falling back to fixed radius" << std::endl;
		return generateCapsules(skeleton, 0.1);
	}

	// Build octree for O(log N) radius estimation
	SparseOctree octree;
	octree.build(mesh_vertices);

	std::vector<std::unique_ptr<TaperedCapsule>> capsules;
	capsules.reserve(skeleton.getBoneCount());

	for (const Bone &bone : skeleton.bones) {
		if (use_tapered_radii) {
			// Estimate different radii at both ends for tapered effect
			auto [radius_start, radius_end] = RadiusEstimator::estimateTaper(bone, octree);

			// Apply safety factors to prevent too-thin capsules
			radius_start = std::max(radius_start * 0.8, 0.02); // Min 2cm
			radius_end = std::max(radius_end * 0.8, 0.02);

			auto capsule = boneToTaperedCapsule(bone, radius_end, radius_start, COLOR_SKIN);
			capsules.push_back(std::move(capsule));
		} else {
			// Use single radius estimated along bone center
			double radius = RadiusEstimator::estimateRadius(bone, octree);
			radius = std::max(radius * 0.8, 0.02); // Safety factor and minimum

			auto capsule = boneToTaperedCapsule(bone, radius, radius, COLOR_SKIN);
			capsules.push_back(std::move(capsule));
		}
	}

	std::cout << "Generated " << capsules.size() << " capsules with advanced radius estimation"
			  << (use_tapered_radii ? " (tapered)" : " (uniform)") << std::endl;

	return capsules;
}

std::unique_ptr<TaperedCapsule> CapsuleGenerator::boneToTaperedCapsule(
		const Bone &bone,
		double radius_top,
		double radius_bottom,
		Vec3d color) {
	Vec3d center = bone.getCenter();
	Vec3d axis = bone.getAxis();
	double height = bone.getLength();

	// Create tapered capsule
	auto capsule = std::make_unique<TaperedCapsule>(
			center,
			radius_top,
			radius_bottom,
			height,
			axis,
			color);

	return capsule;
}

bool CapsuleGenerator::validateSkeleton(const Skeleton &skeleton) {
	if (!skeleton.isValid()) {
		std::cerr << "Error: Skeleton validation failed: empty bones or joints" << std::endl;
		return false;
	}

	// Check for degenerate bones (zero length)
	for (const Bone &bone : skeleton.bones) {
		if (bone.getLength() < 1e-10) {
			std::cout << "Warning: Found degenerate bone with zero length" << std::endl;
			// Don't fail completely, just warn
		}
	}

	return true;
}

bool CapsuleRig::exportToOBJ(const std::string &filename) const {
	if (!isValid()) {
		std::cerr << "Error: Invalid CapsuleRig - no capsules to export" << std::endl;
		return false;
	}

	// Collect all vertices and faces from all capsules
	std::vector<Vec3d> all_vertices;
	std::vector<Vec3i> all_faces;
	size_t vertex_offset = 0;

	for (const auto &capsule : capsules) {
		if (!capsule)
			continue;

		if (!capsule->mesh_generated) {
			// Generate mesh if not already done
			const_cast<TaperedCapsule *>(capsule.get())->generateMesh();
		}

		if (capsule->vertices.empty() || capsule->faces.empty()) {
			std::cerr << "Warning: Capsule has no mesh data, skipping" << std::endl;
			continue;
		}

		// Add vertices
		all_vertices.insert(all_vertices.end(), capsule->vertices.begin(), capsule->vertices.end());

		// Add faces with adjusted indices
		for (const Vec3i &face : capsule->faces) {
			Vec3i adjusted_face(
					face[0] + vertex_offset,
					face[1] + vertex_offset,
					face[2] + vertex_offset);
			all_faces.push_back(adjusted_face);
		}

		vertex_offset += capsule->vertices.size();
	}

	if (all_vertices.empty() || all_faces.empty()) {
		std::cerr << "Error: No valid mesh data found in any capsule" << std::endl;
		return false;
	}

	// Export combined mesh to OBJ file
	MeshFileHandler::writeOBJFile(filename.c_str(), all_vertices, all_faces);

	std::cout << "Exported " << capsules.size() << " capsules to " << filename
			  << " (" << all_vertices.size() << " vertices, " << all_faces.size() << " faces)" << std::endl;

	return true;
}

} // namespace tool_cloth_dynamics
