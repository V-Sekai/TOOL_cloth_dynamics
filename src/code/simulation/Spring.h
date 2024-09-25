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
// Created by Yifei Li on 9/24/20.
//

#ifndef OMEGAENGINE_SPRING_H
#define OMEGAENGINE_SPRING_H

#include "../engine/Macros.h"
#include "Constraint.h"
#include "Eigen/Sparse"
#include "Particle.h"

class Spring : public Constraint {
public:
	double sqrtConstraintWeight;
	int p1_idx;
	int p2_idx;
	std::vector<Particle> &pArr;
	static double k_stiff;
	double k_s; // spring stiffness
	double l0; // rest length
	Mat3x3d dp_dx1, dp_dx2;
	Spring(int p1_idx, int p2_idx, std::vector<Particle> &pArr, double ks,
			double L) :
			Constraint(Constraint::ConstraintType::CONSTRAINT_SPRING_STRETCH, 3),
			p1_idx(p1_idx),
			p2_idx(p2_idx),
			pArr(pArr),
			k_s(ks),
			l0(L),
			sqrtConstraintWeight(std::sqrt(Spring::k_stiff)) {}

	void setConstraintWeight() override {
		sqrtConstraintWeight = std::sqrt(Spring::k_stiff);
	};

	Particle *p1() const {
		assert(pArr[p1_idx].idx == p1_idx);
		return &pArr[p1_idx];
	}

	Particle *p2() const {
		assert(pArr[p2_idx].idx == p2_idx);
		return &pArr[p2_idx];
	}

	Vec6d dforce_dk(const VecXd &x_vec) const;

	Vec3d p1_vec3(const VecXd &x) const { return x.segment(p1()->idx * 3, 3); }

	Vec3d p2_vec3(const VecXd &x) const { return x.segment(p2()->idx * 3, 3); }

	Vec6d getForce(const VecXd &x_vec, double k) const {
		Vec3d dX = (p2_vec3(x_vec) - p1_vec3(x_vec));

		double dL = dX.norm() - l0;
		Vec3d dir = dX.normalized();
		Vec3d f_s = k * dir * dL;
		//  Vec3d f_d = k_d * dX.dot(p2v - p1v) * dir;
		//  Vec3d f = f_s + f_d;
		Vec3d f = f_s;
		Vec6d ret;
		ret.segment(0, 3) = f;
		ret.segment(3, 3) = -f;
		return ret;
	}

	Vec6d getForce(const VecXd &x_vec) const;

	Mat3x3d getStretchingHessian(const VecXd &x_vec) const;

	Eigen::VectorXd project(const VecXd &x_vec) const override;

	void projectBackward(const VecXd &x_vec, TripleVector &triplets) override;
	void projectBackwardPrecompute(const VecXd &x_vec) override;
	double evaluateEnergy(const VecXd &x_new) override;

	void addConstraint(std::vector<Triplet> &tri, int &c_idx,
			bool withWeight) override;
};

#endif // OMEGAENGINE_SPRING_H
