#pragma once

#include "CapsuleGenerator.h"
#include "SkeletonLoader.h"
#include "SparseOctree.h"
#include <LBFGS.h>
#include <LBFGSpp/Param.h>
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace tool_cloth_dynamics {

/**
 * @brief Differentiable skeleton optimization for mesh-to-skeleton fitting
 *
 * Uses L-BFGS optimization to fit skeletons to mesh geometries,
 * enabling automatic skeleton generation and retargeting.
 */
class DifferentiableSkeletonOptimizer {
public:
	/**
	 * @brief Fit skeleton to mesh using template initialization
	 *
	 * @param mesh_vertices Mesh vertex positions (3 x n_vertices)
	 * @param mesh_faces Mesh face indices (3 x n_faces)
	 * @param template_skeleton Initial skeleton template
	 * @param max_iterations Maximum optimization iterations
	 * @return Optimized skeleton fitted to mesh
	 */
	static Skeleton optimizeFromMesh(
			const Eigen::MatrixXd &mesh_vertices,
			const Eigen::MatrixXi &mesh_faces,
			const Skeleton &template_skeleton,
			int max_iterations = 100);

	/**
	 * @brief Retarget skeleton by treating source as mesh
	 *
	 * Converts source skeleton to mesh representation and fits target template
	 * using the same optimization pipeline as mesh-to-skeleton fitting.
	 *
	 * @param source_skeleton Source skeleton (treated as mesh)
	 * @param target_template Target skeleton template
	 * @param max_iterations Maximum optimization iterations
	 * @return Retargeted skeleton with source topology adapted to target structure
	 */
	static Skeleton retargetSkeleton(
			const Skeleton &source_skeleton,
			const Skeleton &target_template,
			int max_iterations = 100);

	/**
	 * @brief Refine capsule parameters for existing skeleton
	 *
	 * @param skeleton Input skeleton
	 * @param target_mesh_vertices Target mesh vertices
	 * @param target_mesh_faces Target mesh faces
	 * @return Refined capsule rig
	 */
	static CapsuleRig refineParameters(
			const Skeleton &skeleton,
			const Eigen::MatrixXd &target_mesh_vertices,
			const Eigen::MatrixXi &target_mesh_faces);

public:
	/**
	 * @brief Convert skeleton to mesh representation for optimization
	 *
	 * Creates a mesh from skeleton joints with connectivity that preserves
	 * bone structure for optimization.
	 *
	 * @param skeleton Source skeleton
	 * @return Pair of vertices and faces representing the skeleton as mesh
	 */
	static std::pair<Eigen::MatrixXd, Eigen::MatrixXi> skeletonToMesh(
			const Skeleton &skeleton);

	/**
	 * @brief Compute coverage loss between skeleton and mesh
	 *
	 * Measures how well the skeleton covers the mesh volume.
	 *
	 * @param skeleton Current skeleton
	 * @param mesh_vertices Target mesh vertices
	 * @param mesh_faces Target mesh faces
	 * @return Coverage loss value
	 */
	static double computeCoverageLoss(
			const Skeleton &skeleton,
			const Eigen::MatrixXd &mesh_vertices,
			const Eigen::MatrixXi &mesh_faces);

	/**
	 * @brief Compute skeleton plausibility constraints
	 *
	 * Ensures anatomical correctness (joint angles, bone lengths, etc.)
	 *
	 * @param skeleton Current skeleton
	 * @return Plausibility loss value
	 */
	static double computePlausibilityLoss(const Skeleton &skeleton);

	/**
	 * @brief Compute retargeting constraints for skeleton-to-skeleton
	 *
	 * Preserves proportions and topology during retargeting.
	 *
	 * @param current_skeleton Current optimized skeleton
	 * @param source_skeleton Original source skeleton (as mesh reference)
	 * @param target_template Target template skeleton
	 * @return Retargeting constraint loss
	 */
	static double computeRetargetingLoss(
			const Skeleton &current_skeleton,
			const Skeleton &source_skeleton,
			const Skeleton &target_template);

	/**
	 * @brief Compute gradients for optimization
	 *
	 * @param loss Current loss value
	 * @param skeleton Current skeleton
	 * @return Gradient matrix for joint positions
	 */
	static Eigen::MatrixXd computeGradients(
			double loss,
			const Skeleton &skeleton);

	/**
	 * @brief Update skeleton joints using gradients
	 *
	 * @param skeleton Skeleton to update (modified in-place)
	 * @param gradients Gradient matrix
	 * @param learning_rate Optimization step size
	 */
	static void updateSkeleton(
			Skeleton &skeleton,
			const Eigen::MatrixXd &gradients,
			double learning_rate);
};

} // namespace tool_cloth_dynamics
