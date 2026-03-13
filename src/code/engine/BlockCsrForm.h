/*
 * Block CSR form for the backward-pass system matrix (3x3 blocks per vertex).
 * Allows feeding external preconditioners (e.g. Schwarz) that expect
 * diagonal + CSR off-diagonals. See docs/PRECONDITIONER_INVESTIGATION.md.
 */
#ifndef OMEGAENGINE_BLOCKCSRFORM_H
#define OMEGAENGINE_BLOCKCSRFORM_H

#include "Macros.h"
#include <utility>
#include <vector>

namespace OmegaEngine {

/** 3x3 block (one per vertex for diagonal, or per neighbor for off-diagonals). */
using Mat3d = Eigen::Matrix<double, 3, 3>;

/**
 * Block-CSR representation of a 3n x 3n matrix (n = num vertices).
 * - diagonal[i] = 3x3 block for (i,i).
 * - For row block i, off-diagonal blocks are in csrBlocks[csrRanges[i] .. csrRanges[i+1]-1],
 *   with column vertex indices in csrColIndex[csrRanges[i] .. csrRanges[i+1]-1].
 */
struct BlockCsrForm {
	int numVerts = 0;
	std::vector<Mat3d> diagonal;
	std::vector<int> csrRanges;       // length numVerts+1
	std::vector<int> csrColIndex;     // column vertex index per off-diagonal block
	std::vector<Mat3d> csrBlocks;     // off-diagonal 3x3 blocks in row order

	void clear() {
		numVerts = 0;
		diagonal.clear();
		csrRanges.clear();
		csrColIndex.clear();
		csrBlocks.clear();
	}

	/** Total number of off-diagonal blocks. */
	size_t numOffDiagonalBlocks() const {
		return csrBlocks.size();
	}
};

/**
 * Extract block-CSR form from an Eigen sparse matrix A (must be 3n x 3n, n = number of vertices).
 * Assumes row/column indices are grouped by vertex: block (i,j) = A(3*i:3*i+3, 3*j:3*j+3).
 */
inline void extractBlockCsrFromSpMat(const SpMat& A, BlockCsrForm& out) {
	out.clear();
	const int n = static_cast<int>(A.rows() / 3);
	if (A.rows() != 3 * n || A.cols() != 3 * n)
		return;
	out.numVerts = n;
	out.diagonal.resize(static_cast<size_t>(n));
	out.csrRanges.resize(static_cast<size_t>(n + 1));
	for (int i = 0; i < n; i++)
		out.diagonal[static_cast<size_t>(i)].setZero();
	// Column-major: iterate by column, accumulate into row-block i, col-block j.
	std::vector<std::vector<std::pair<int, Mat3d>>> rowBlocks(static_cast<size_t>(n));
	for (size_t idx = 0; idx < rowBlocks.size(); idx++)
		rowBlocks[idx].clear();
	for (int col = 0; col < A.cols(); col++) {
		int j = col / 3;
		int cj = col % 3;
		for (typename SpMat::InnerIterator it(A, col); it; ++it) {
			int row = static_cast<int>(it.row());
			int i = row / 3;
			int ri = row % 3;
			double v = it.value();
			if (i == j) {
				out.diagonal[static_cast<size_t>(i)](ri, cj) = v;
			} else {
				std::vector<std::pair<int, Mat3d>>& list = rowBlocks[static_cast<size_t>(i)];
				size_t k = 0;
				for (; k < list.size(); k++)
					if (list[k].first == j) break;
				if (k == list.size())
					list.push_back({ j, Mat3d::Zero() });
				list[k].second(ri, cj) = v;
			}
		}
	}
	out.csrRanges[0] = 0;
	for (int i = 0; i < n; i++) {
		const auto& list = rowBlocks[static_cast<size_t>(i)];
		out.csrRanges[static_cast<size_t>(i + 1)] = static_cast<int>(out.csrRanges[static_cast<size_t>(i)] + static_cast<int>(list.size()));
		for (const auto& p : list) {
			out.csrColIndex.push_back(p.first);
			out.csrBlocks.push_back(p.second);
		}
	}
}

} // namespace OmegaEngine

#endif
