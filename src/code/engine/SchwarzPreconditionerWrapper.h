/*
 * Eigen-compatible wrapper for the Schwarz preconditioner (backward BiCGSTAB).
 * Builds mesh topology from cloth mesh and converts BlockCsrForm to SE float format.
 * See docs/PRECONDITIONER_INVESTIGATION.md and the integration plan.
 */
#ifndef OMEGAENGINE_SCHWARZPRECONDITIONERWRAPPER_H
#define OMEGAENGINE_SCHWARZPRECONDITIONERWRAPPER_H

#include "BlockCsrForm.h"
#include "Macros.h"
#include <vector>

namespace OmegaEngine {

/**
 * Mesh topology in the format expected by the Schwarz preconditioner:
 * edges (Int4: v0, v1, opp0, opp1), faces (Int4: v0, v1, v2, 0), vertex neighbours (CSR).
 * Built once from cloth mesh; passed to the preconditioner at PreparePreconditioner.
 */
struct SchwarzPreconditionerTopology {
	int numVerts = 0;
	int numEdges = 0;
	int numFaces = 0;
	std::vector<float> positions;   /* 3 * numVerts, contiguous float for SeVec3fSimd* */
	std::vector<int> edges;         /* 4 * numEdges: v0,v1,opp0,opp1 per edge */
	std::vector<int> faces;         /* 4 * numFaces: v0,v1,v2,0 per face */
	std::vector<int> neighbourStarts;
	std::vector<int> neighbourIndices;

	/** Build from triangle mesh (p0_idx, p1_idx, p2_idx per triangle). */
	void build(int nVerts, const std::vector<int>& triP0,
		const std::vector<int>& triP1, const std::vector<int>& triP2);
	void clear();
};

/**
 * Eigen-compatible preconditioner for BiCGSTAB: compute(SpMat), solve(rhs).
 * Uses BlockCsrForm extraction and SE Schwarz preconditioner internally.
 */
class SchwarzPreconditionerWrapper {
public:
	SchwarzPreconditionerWrapper();
	~SchwarzPreconditionerWrapper();

	/** Set mesh topology (call when mesh is available, e.g. before first backward solve). */
	void setTopology(const SchwarzPreconditionerTopology& topo);

	/** Set current particle positions (3*n doubles). Call before factorize so compute() can use them. */
	void setPositions(const double* positions, int n);

	/** Eigen preconditioner API: prepare from system matrix. Uses positions from last setPositions() if not passed. */
	void compute(const SpMat& A, const double* positions = nullptr);

	/** Eigen preconditioner API: return preconditioned residual (z = M^{-1} * rhs). */
	VecXd solve(const VecXd& rhs) const;

	/** Eigen preconditioner API: return last computation status. */
	Eigen::ComputationInfo info() const { return Eigen::Success; }

private:
	struct Impl;
	mutable Impl* impl_ = nullptr;
};

} // namespace OmegaEngine

#endif
