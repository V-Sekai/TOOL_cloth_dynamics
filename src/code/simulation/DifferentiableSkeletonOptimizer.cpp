#include "DifferentiableSkeletonOptimizer.h"
#include "CapsuleGenerator.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace tool_cloth_dynamics {

// L-BFGS objective function for skeleton optimization
class SkeletonOptimizationObjective {
private:
	const Eigen::MatrixXd &target_mesh_vertices_;
	const Eigen::MatrixXi &target_mesh_faces_;
	const Skeleton &template_skeleton_;
	mutable Skeleton current_skeleton_;

public:
	SkeletonOptimizationObjective(
			const Eigen::MatrixXd &mesh_vertices,
			const Eigen::MatrixXi &mesh_faces,
			const Skeleton &template_skeleton) :
			target_mesh_vertices_(mesh_vertices), target_mesh_faces_(mesh_faces), template_skeleton_(template_skeleton), current_skeleton_(template_skeleton) {}

	// L-BFGS objective function: f(x) where x is flattened joint positions
	double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
		// Reconstruct skeleton from optimization variables
		updateSkeletonFromVariables(x);

		// Compute total loss
		double coverage_loss = DifferentiableSkeletonOptimizer::computeCoverageLoss(
				current_skeleton_, target_mesh_vertices_, target_mesh_faces_);
		double plausibility_loss = DifferentiableSkeletonOptimizer::computePlausibilityLoss(
				current_skeleton_);

		double total_loss = coverage_loss + plausibility_loss;

		// Compute gradients (simplified finite differences for now)
		computeGradients(x, grad);

		return total_loss;
	}

private:
	void updateSkeletonFromVariables(const Eigen::VectorXd &x) {
		current_skeleton_ = template_skeleton_;
		size_t num_joints = current_skeleton_.joints.size();

		// Update joint positions from optimization variables
		for (size_t i = 0; i < num_joints && i * 3 + 2 < x.size(); ++i) {
			current_skeleton_.joints[i] = Vec3d(x[i * 3], x[i * 3 + 1], x[i * 3 + 2]);
		}

		// Update bone positions to match new joint positions
		for (auto &bone : current_skeleton_.bones) {
			// Find corresponding joint indices and update
			for (size_t j = 0; j < current_skeleton_.joints.size(); ++j) {
				if ((bone.start - current_skeleton_.joints[j]).norm() < 1e-6) {
					bone.start = current_skeleton_.joints[j];
				}
				if ((bone.end - current_skeleton_.joints[j]).norm() < 1e-6) {
					bone.end = current_skeleton_.joints[j];
				}
			}
		}
	}

	void computeGradients(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
		const double eps = 1e-6;
		grad.resize(x.size());
		grad.setZero();

		// Finite differences for gradients
		Eigen::VectorXd x_plus = x;
		Eigen::VectorXd dummy_grad(x.size()); // Temporary gradient vector
		for (int i = 0; i < x.size(); ++i) {
			x_plus[i] += eps;
			double f_plus = (*this)(x_plus, dummy_grad); // Don't need grad here

			x_plus[i] -= 2 * eps;
			double f_minus = (*this)(x_plus, dummy_grad);

			grad[i] = (f_plus - f_minus) / (2 * eps);
			x_plus[i] = x[i]; // Reset
		}
	}
};

Skeleton DifferentiableSkeletonOptimizer::optimizeFromMesh(
		const Eigen::MatrixXd &mesh_vertices,
		const Eigen::MatrixXi &mesh_faces,
		const Skeleton &template_skeleton,
		int max_iterations) {
	std::cout << "Optimizing skeleton from mesh with " << mesh_vertices.cols()
			  << " vertices and " << mesh_faces.cols() << " faces using L-BFGS" << std::endl;

	// Set up L-BFGS parameters
	LBFGSpp::LBFGSParam<double> param;
	param.epsilon = 1e-6; // Convergence tolerance
	param.max_iterations = max_iterations;
	param.m = 10; // L-BFGS memory

	// Create objective function
	SkeletonOptimizationObjective objective(mesh_vertices, mesh_faces, template_skeleton);

	// Initialize optimization variables from template skeleton
	size_t num_joints = template_skeleton.joints.size();
	Eigen::VectorXd x(3 * num_joints);
	for (size_t i = 0; i < num_joints; ++i) {
		x[i * 3] = template_skeleton.joints[i][0];
		x[i * 3 + 1] = template_skeleton.joints[i][1];
		x[i * 3 + 2] = template_skeleton.joints[i][2];
	}

	// Create L-BFGS solver
	LBFGSpp::LBFGSSolver<double> solver(param);

	// Optimize
	double fx;
	int niter = solver.minimize(objective, x, fx);

	std::cout << "L-BFGS optimization completed in " << niter
			  << " iterations, final loss: " << fx << std::endl;

	// Extract optimized skeleton
	Skeleton result = template_skeleton;
	for (size_t i = 0; i < num_joints; ++i) {
		result.joints[i] = Vec3d(x[i * 3], x[i * 3 + 1], x[i * 3 + 2]);
	}

	// Update bone positions
	for (auto &bone : result.bones) {
		for (size_t j = 0; j < result.joints.size(); ++j) {
			if ((bone.start - result.joints[j]).norm() < 1e-6) {
				bone.start = result.joints[j];
			}
			if ((bone.end - result.joints[j]).norm() < 1e-6) {
				bone.end = result.joints[j];
			}
		}
	}

	return result;
}

Skeleton DifferentiableSkeletonOptimizer::retargetSkeleton(
		const Skeleton &source_skeleton,
		const Skeleton &target_template,
		int max_iterations) {
	// Convert source skeleton to mesh representation
	auto [mesh_vertices, mesh_faces] = skeletonToMesh(source_skeleton);

	std::cout << "Retargeting skeleton: " << source_skeleton.getBoneCount()
			  << " source bones → " << target_template.getBoneCount()
			  << " target bones" << std::endl;

	// Use mesh-to-skeleton optimization with source-as-mesh
	return optimizeFromMesh(mesh_vertices, mesh_faces, target_template, max_iterations);
}

CapsuleRig DifferentiableSkeletonOptimizer::refineParameters(
		const Skeleton &skeleton,
		const Eigen::MatrixXd &target_mesh_vertices,
		const Eigen::MatrixXi &target_mesh_faces) {
	// Build sparse octree for efficient nearest neighbor queries
	SparseOctree octree;
	octree.build(target_mesh_vertices);

	// Generate initial capsule rig
	CapsuleRig rig = CapsuleRig::generate(skeleton, 0.1);

	// Refine capsule parameters using mesh proximity
	for (size_t i = 0; i < rig.getCapsuleCount(); ++i) {
		auto capsule = rig.getCapsules()[i].get();
		if (!capsule)
			continue;

		// Sample points along capsule axis
		Vec3d center = capsule->center;
		Vec3d axis = capsule->axis;
		double height = capsule->mid_height;

		std::vector<double> radii;
		int num_samples = 10;

		for (int j = 0; j < num_samples; ++j) {
			double t = j / (num_samples - 1.0);
			Vec3d sample_point = center + axis * (t - 0.5) * height;

			// Find nearest mesh distance
			double dist = octree.nearestDistance(sample_point);
			radii.push_back(dist);
		}

		// Use median radius for robustness
		std::sort(radii.begin(), radii.end());
		double median_radius = radii[radii.size() / 2];

		// Update capsule with refined radius
		capsule->radius_top = median_radius;
		capsule->radius_bottom = median_radius;
	}

	return rig;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXi> DifferentiableSkeletonOptimizer::skeletonToMesh(
		const Skeleton &skeleton) {
	const auto &joints = skeleton.joints;
	const auto &bones = skeleton.bones;

	size_t num_joints = joints.size();
	size_t num_bones = bones.size();

	// Create vertices from joint positions
	Eigen::MatrixXd vertices(3, num_joints);
	for (size_t i = 0; i < num_joints; ++i) {
		vertices.col(i) = joints[i];
	}

	// Create faces connecting bones to form a mesh
	// Each bone becomes a triangular prism connecting start/end joints
	std::vector<Eigen::Vector3i> faces;

	// For each bone, create faces connecting it to neighboring bones
	for (size_t i = 0; i < num_bones; ++i) {
		const auto &bone = bones[i];

		// Find joint indices
		auto start_it = std::find(joints.begin(), joints.end(), bone.start);
		auto end_it = std::find(joints.begin(), joints.end(), bone.end);

		if (start_it == joints.end() || end_it == joints.end()) {
			continue; // Skip invalid bones
		}

		size_t start_idx = start_it - joints.begin();
		size_t end_idx = end_it - joints.begin();

		// Create triangular faces along the bone
		// This creates a simple mesh representation that preserves bone structure

		// Add faces for bone connectivity (simplified mesh generation)
		// In a full implementation, this would create a more sophisticated mesh
		// that better represents the skeletal structure for optimization

		// For now, create minimal connectivity
		if (start_idx < end_idx) {
			// Create a triangular face using bone endpoints and a midpoint
			Vec3d midpoint = (bone.start + bone.end) * 0.5;

			// Add midpoint as additional vertex if needed
			vertices.conservativeResize(3, vertices.cols() + 1);
			vertices.col(vertices.cols() - 1) = midpoint;

			size_t mid_idx = vertices.cols() - 1;

			// Create triangular faces
			faces.push_back(Eigen::Vector3i(start_idx, end_idx, mid_idx));
		}
	}

	// Convert faces vector to matrix
	Eigen::MatrixXi faces_matrix(3, faces.size());
	for (size_t i = 0; i < faces.size(); ++i) {
		faces_matrix.col(i) = faces[i];
	}

	return { vertices, faces_matrix };
}

double DifferentiableSkeletonOptimizer::computeCoverageLoss(
		const Skeleton &skeleton,
		const Eigen::MatrixXd &mesh_vertices,
		const Eigen::MatrixXi &mesh_faces) {
	// Placeholder: compute distance from mesh vertices to nearest skeleton bones
	double total_loss = 0.0;
	size_t sample_count = std::min(size_t(100), static_cast<size_t>(mesh_vertices.cols()));

	for (size_t i = 0; i < sample_count; ++i) {
		Vec3d mesh_point = mesh_vertices.col(i);

		// Find minimum distance to any bone
		double min_distance = std::numeric_limits<double>::max();

		for (const auto &bone : skeleton.bones) {
			// Compute distance from point to bone segment
			Vec3d bone_axis = bone.getAxis();
			Vec3d to_point = mesh_point - bone.start;
			double projection = to_point.dot(bone_axis);
			double bone_length = bone.getLength();

			// Clamp projection to bone segment
			projection = std::max(0.0, std::min(bone_length, projection));

			// Compute closest point on bone
			Vec3d closest_point = bone.start + projection * bone_axis;
			double distance = (mesh_point - closest_point).norm();

			min_distance = std::min(min_distance, distance);
		}

		total_loss += min_distance * min_distance; // Squared distance
	}

	return total_loss / sample_count;
}

double DifferentiableSkeletonOptimizer::computePlausibilityLoss(const Skeleton &skeleton) {
	double loss = 0.0;

	// Basic plausibility checks
	for (const auto &bone : skeleton.bones) {
		double length = bone.getLength();

		// Penalize very short or very long bones
		if (length < 0.01) {
			loss += 1000.0; // Very short bones are bad
		} else if (length > 2.0) {
			loss += (length - 2.0) * 10.0; // Penalize excessively long bones
		}

		// Check for bones with zero length
		if (length < 1e-6) {
			loss += 10000.0;
		}
	}

	return loss;
}

double DifferentiableSkeletonOptimizer::computeRetargetingLoss(
		const Skeleton &current_skeleton,
		const Skeleton &source_skeleton,
		const Skeleton &target_template) {
	double loss = 0.0;

	// Preserve proportions from target template
	if (current_skeleton.bones.size() == target_template.bones.size()) {
		for (size_t i = 0; i < current_skeleton.bones.size(); ++i) {
			double current_length = current_skeleton.bones[i].getLength();
			double template_length = target_template.bones[i].getLength();

			if (template_length > 0) {
				double ratio_diff = std::abs(current_length / template_length - 1.0);
				loss += ratio_diff * ratio_diff * 100.0; // Strong penalty for proportion changes
			}
		}
	}

	return loss;
}

Eigen::MatrixXd DifferentiableSkeletonOptimizer::computeGradients(
		double loss,
		const Skeleton &skeleton) {
	// Placeholder: return zero gradients
	// In a real implementation, this would compute analytical or numerical gradients
	Eigen::MatrixXd gradients(3, skeleton.joints.size());
	gradients.setZero();
	return gradients;
}

void DifferentiableSkeletonOptimizer::updateSkeleton(
		Skeleton &skeleton,
		const Eigen::MatrixXd &gradients,
		double learning_rate) {
	// Update joint positions based on gradients
	for (size_t i = 0; i < skeleton.joints.size(); ++i) {
		if (i < gradients.cols()) {
			Vec3d gradient = gradients.col(i);
			skeleton.joints[i] -= learning_rate * gradient;
		}
	}

	// Update bone positions to match new joint positions
	for (auto &bone : skeleton.bones) {
		// Find corresponding joint indices and update
		auto start_it = std::find(skeleton.joints.begin(), skeleton.joints.end(), bone.start);
		auto end_it = std::find(skeleton.joints.begin(), skeleton.joints.end(), bone.end);

		if (start_it != skeleton.joints.end()) {
			bone.start = *start_it;
		}
		if (end_it != skeleton.joints.end()) {
			bone.end = *end_it;
		}
	}
}

} // namespace tool_cloth_dynamics
