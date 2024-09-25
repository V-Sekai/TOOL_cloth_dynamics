/*
 * MIT License
 *
 * Copyright (c) 2024-present K. S. Ernest (Fire) Lee
 * Copyright (c) 2022-2024 Yifei Li
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

//
// Created by Yifei Li on 3/17/21.
// Email: liyifei@csail.mit.edu
//

#ifndef OMEGAENGINE_TRIANGLEBENDING_H
#define OMEGAENGINE_TRIANGLEBENDING_H

#include "Constraint.h"
#include "Particle.h"

class TriangleBending : public Constraint {
public:
	int p0_idx, p1_idx, p2_idx, p3_idx;
	std::vector<int> idx_arr;
	std::vector<Particle> &pArr;
	Mat3x12d de_dxnew;
	static double k_stiff;
	Mat3x12d grad, A_w;
	double constrainWeightSqrt;
	Vec4d weightVert;
	Vec12d dfi_dk_buffer;
	Mat12x12d hessianBuffer;
	double n, A0, A1;

	TriangleBending(int p0_idx, int p1_idx, int p2_idx, int p3_idx,
			std::vector<Particle> &pArr);

	Vec12d bendingForce(const VecXd &x_new);

	Vec12d dfi_dk(const VecXd &x_new);

	// Compute d[v * (dproj/dx)]/dx.
	// dproj/dx: 3 x 12 matrix.
	// v: 3d (row) vector.
	Mat12x12d projectSecondOrderJacobian(const VecXd &x_new, const Vec3d &v);

	void setConstraintWeight() override {
		constrainWeightSqrt = std::sqrt(TriangleBending::k_stiff * 3.0 / (A0 + A1));
	};

	void addConstraint(std::vector<Triplet> &tri, int &c_idx,
			bool withWeight) override;

	VecXd project(const VecXd &x_vec) const override;

	void projectBackward(const VecXd &x_vec, TripleVector &triplets) override;
	void projectBackwardPrecompute(const VecXd &x_vec) override;
	double evaluateEnergy(const VecXd &x_new) override;

	Mat3x12d backwardGradient(const VecXd &x_vec);
};

#endif // OMEGAENGINE_TRIANGLEBENDING_H
