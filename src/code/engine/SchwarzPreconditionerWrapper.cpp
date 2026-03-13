/*
 * Schwarz preconditioner wrapper: topology build, BlockCsrForm → float, SE preconditioner.
 */
#include "SchwarzPreconditionerWrapper.h"
#include "CollisionStencilAdapter.h"
#include <algorithm>
#include <set>
#include <cstring>

#include "SeSchwarzPreconditioner.h"
#include "SeCsr.h"
#include "SeVector.h"
#include "SeVectorSimd.h"
#include "SeMatrix.h"

SE_USING_NAMESPACE

namespace OmegaEngine {

// --- Topology -----------------------------------------------------------------

void SchwarzPreconditionerTopology::build(int nVerts,
	const std::vector<int>& triP0, const std::vector<int>& triP1, const std::vector<int>& triP2)
{
	clear();
	numVerts = nVerts;
	numFaces = static_cast<int>(triP0.size());

	/* Faces: 4 ints each (v0, v1, v2, 0) */
	faces.resize(4 * static_cast<size_t>(numFaces));
	for (size_t i = 0; i < triP0.size(); i++) {
		faces[4 * i + 0] = triP0[i];
		faces[4 * i + 1] = triP1[i];
		faces[4 * i + 2] = triP2[i];
		faces[4 * i + 3] = 0;
	}

	/* Unique edges (v0, v1) with v0 < v1; store as (v0, v1, 0, 0) for Int4 */
	std::set<std::pair<int, int>> edgeSet;
	for (size_t i = 0; i < triP0.size(); i++) {
		int a = triP0[i], b = triP1[i], c = triP2[i];
		edgeSet.insert({ std::min(a, b), std::max(a, b) });
		edgeSet.insert({ std::min(b, c), std::max(b, c) });
		edgeSet.insert({ std::min(a, c), std::max(a, c) });
	}
	numEdges = static_cast<int>(edgeSet.size());
	edges.resize(4 * static_cast<size_t>(numEdges));
	size_t idx = 0;
	for (const auto& e : edgeSet) {
		edges[4 * idx + 0] = e.first;
		edges[4 * idx + 1] = e.second;
		edges[4 * idx + 2] = 0;
		edges[4 * idx + 3] = 0;
		idx++;
	}

	/* Vertex neighbours from edges */
	std::vector<std::vector<int>> vertNeighbours(static_cast<size_t>(numVerts));
	for (const auto& e : edgeSet) {
		vertNeighbours[static_cast<size_t>(e.first)].push_back(e.second);
		vertNeighbours[static_cast<size_t>(e.second)].push_back(e.first);
	}
	neighbourStarts.resize(static_cast<size_t>(numVerts) + 1);
	neighbourStarts[0] = 0;
	for (int v = 0; v < numVerts; v++) {
		neighbourStarts[static_cast<size_t>(v + 1)] =
			neighbourStarts[static_cast<size_t>(v)] + static_cast<int>(vertNeighbours[static_cast<size_t>(v)].size());
	}
	neighbourIndices.resize(static_cast<size_t>(neighbourStarts[numVerts]));
	for (int v = 0; v < numVerts; v++) {
		std::memcpy(neighbourIndices.data() + neighbourStarts[v],
			vertNeighbours[static_cast<size_t>(v)].data(),
			vertNeighbours[static_cast<size_t>(v)].size() * sizeof(int));
	}

	/* Positions buffer: 4 floats per vertex for SeVec3fSimd alignment */
	positions.resize(4 * static_cast<size_t>(numVerts), 0.f);
}

void SchwarzPreconditionerTopology::clear() {
	numVerts = numEdges = numFaces = 0;
	positions.clear();
	edges.clear();
	faces.clear();
	neighbourStarts.clear();
	neighbourIndices.clear();
}

// --- Wrapper impl -------------------------------------------------------------

struct SchwarzPreconditionerWrapper::Impl {
	SE::SeSchwarzPreconditioner prec;
	SchwarzPreconditionerTopology topo;
	BlockCsrForm blockForm;
	std::vector<SE::SeMatrix3f> diagFloat;
	std::vector<SE::SeMatrix3f> offDiagFloat;
	std::vector<int> csrRanges;
	std::vector<unsigned int> efCounts, eeCounts, vfCounts;
	std::vector<SE::SeVec3fSimd> residualBuf;
	std::vector<SE::SeVec3fSimd> zBuf;
	SE::SeCsr<int> neighboursCsr;
	int lastN = -1;
	bool topologySet = false;

	void ensureCountArrays(int nV, int nE) {
		if (efCounts.size() < static_cast<size_t>(nE + 1)) {
			efCounts.resize(static_cast<size_t>(nE + 1), 0u);
			eeCounts.resize(static_cast<size_t>(nE + 1), 0u);
		}
		if (vfCounts.size() < static_cast<size_t>(nV + 1)) {
			vfCounts.resize(static_cast<size_t>(nV + 1), 0u);
		}
		efCounts[static_cast<size_t>(nE)] = 0;
		eeCounts[static_cast<size_t>(nE)] = 0;
		vfCounts[static_cast<size_t>(nV)] = 0;
	}
};

SchwarzPreconditionerWrapper::SchwarzPreconditionerWrapper() {
	impl_ = new Impl();
}

SchwarzPreconditionerWrapper::~SchwarzPreconditionerWrapper() {
	delete impl_;
	impl_ = nullptr;
}

void SchwarzPreconditionerWrapper::setTopology(const SchwarzPreconditionerTopology& topo) {
	impl_->topo = topo;
	impl_->topologySet = true;
	/* Fill SE CSR for neighbours via InitIdxs */
	std::vector<std::vector<int>> vertNeighbours(static_cast<size_t>(topo.numVerts));
	for (int v = 0; v < topo.numVerts; v++) {
		int start = topo.neighbourStarts[static_cast<size_t>(v)];
		int end = topo.neighbourStarts[static_cast<size_t>(v) + 1];
		vertNeighbours[static_cast<size_t>(v)].assign(
			topo.neighbourIndices.begin() + start,
			topo.neighbourIndices.begin() + end);
	}
	impl_->neighboursCsr.InitIdxs(vertNeighbours);
}

void SchwarzPreconditionerWrapper::setPositions(const double* positions, int n) {
	Impl& i = *impl_;
	if (!positions || n <= 0) return;
	i.topo.positions.resize(4 * static_cast<size_t>(n), 0.f);
	for (int v = 0; v < n; v++) {
		i.topo.positions[4 * static_cast<size_t>(v) + 0] = static_cast<float>(positions[3 * v + 0]);
		i.topo.positions[4 * static_cast<size_t>(v) + 1] = static_cast<float>(positions[3 * v + 1]);
		i.topo.positions[4 * static_cast<size_t>(v) + 2] = static_cast<float>(positions[3 * v + 2]);
	}
}

void SchwarzPreconditionerWrapper::compute(const SpMat& A, const double* positions) {
	Impl& i = *impl_;
	const int n = static_cast<int>(A.rows() / 3);
	if (n <= 0 || A.rows() != 3 * n || A.cols() != 3 * n)
		return;
	if (!i.topologySet || i.topo.numVerts != n) {
		/* No topology or size mismatch: cannot use Schwarz; leave preconditioner idle (solve will return rhs). */
		return;
	}

	extractBlockCsrFromSpMat(A, i.blockForm);
	if (i.blockForm.numVerts != static_cast<size_t>(n))
		return;

	/* Copy positions into topo.positions (4 floats per vert for SeVec3fSimd); use argument or last setPositions() */
	if (positions) {
		i.topo.positions.resize(4 * static_cast<size_t>(n), 0.f);
		for (int v = 0; v < n; v++) {
			i.topo.positions[4 * static_cast<size_t>(v) + 0] = static_cast<float>(positions[3 * v + 0]);
			i.topo.positions[4 * static_cast<size_t>(v) + 1] = static_cast<float>(positions[3 * v + 1]);
			i.topo.positions[4 * static_cast<size_t>(v) + 2] = static_cast<float>(positions[3 * v + 2]);
		}
	} else if (i.topo.positions.size() < 4 * static_cast<size_t>(n)) {
		i.topo.positions.resize(4 * static_cast<size_t>(n), 0.f);
	}
	i.prec.m_positions = reinterpret_cast<const SE::SeVec3fSimd*>(i.topo.positions.data());
	i.prec.m_edges = reinterpret_cast<const SE::Int4*>(i.topo.edges.data());
	i.prec.m_faces = reinterpret_cast<const SE::Int4*>(i.topo.faces.data());
	i.prec.m_neighbours = &i.neighboursCsr;

	/* Allocate if dimensions changed */
	if (i.lastN != n) {
		i.prec.AllocatePrecoditioner(n, i.topo.numEdges, i.topo.numFaces);
		i.lastN = n;
	}

	/* Block CSR → float */
	i.diagFloat.resize(static_cast<size_t>(n));
	for (int v = 0; v < n; v++) {
		for (int r = 0; r < 3; r++)
			for (int c = 0; c < 3; c++)
				i.diagFloat[static_cast<size_t>(v)](r, c) = static_cast<float>(i.blockForm.diagonal[static_cast<size_t>(v)](r, c));
	}
	i.offDiagFloat.resize(i.blockForm.csrBlocks.size());
	for (size_t k = 0; k < i.blockForm.csrBlocks.size(); k++) {
		for (int r = 0; r < 3; r++)
			for (int c = 0; c < 3; c++)
				i.offDiagFloat[k](r, c) = static_cast<float>(i.blockForm.csrBlocks[k](r, c));
	}
	i.csrRanges = i.blockForm.csrRanges;

	i.ensureCountArrays(n, i.topo.numEdges);

	i.prec.PreparePreconditioner(
		i.diagFloat.data(), i.offDiagFloat.data(), i.csrRanges.data(),
		nullptr, nullptr, nullptr,
		i.efCounts.data(), i.eeCounts.data(), i.vfCounts.data());
}

VecXd SchwarzPreconditionerWrapper::solve(const VecXd& rhs) const {
	Impl& i = *impl_;
	const int n = static_cast<int>(rhs.size() / 3);
	if (n <= 0 || rhs.size() != 3 * static_cast<Eigen::Index>(n)) {
		return rhs;
	}
	if (!i.topologySet || i.lastN != n) {
		return rhs;  /* No preconditioner prepared */
	}

	i.residualBuf.resize(static_cast<size_t>(n));
	i.zBuf.resize(static_cast<size_t>(n));
	for (int v = 0; v < n; v++) {
		i.residualBuf[static_cast<size_t>(v)] = SE::SeVec3fSimd(
			static_cast<float>(rhs(3 * v + 0)),
			static_cast<float>(rhs(3 * v + 1)),
			static_cast<float>(rhs(3 * v + 2)));
	}
	i.prec.Preconditioning(i.zBuf.data(), i.residualBuf.data(), n);
	VecXd out = VecXd::Zero(3 * n);
	for (int v = 0; v < n; v++) {
		out(3 * v + 0) = static_cast<double>(i.zBuf[static_cast<size_t>(v)][0]);
		out(3 * v + 1) = static_cast<double>(i.zBuf[static_cast<size_t>(v)][1]);
		out(3 * v + 2) = static_cast<double>(i.zBuf[static_cast<size_t>(v)][2]);
	}
	return out;
}

} // namespace OmegaEngine
