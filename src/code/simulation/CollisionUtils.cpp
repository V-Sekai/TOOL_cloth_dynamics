#include "CollisionUtils.h"

#include <limits>

Eigen::Vector3d tool_cloth_dynamics::collision::closestPointTriangle(const Eigen::Vector3d &p, const Eigen::Vector3d &a,
		const Eigen::Vector3d &b, const Eigen::Vector3d &c, Eigen::Vector3d &bary, ClosestPointOnTriangleType &pointType) {
	Eigen::Vector3d ab = b - a;
	Eigen::Vector3d ac = c - a;
	Eigen::Vector3d ap = p - a;

	double d1 = ab.dot(ap);
	double d2 = ac.dot(ap);
	if (d1 <= 0.0 && d2 <= 0.0) {
		pointType = ClosestPointOnTriangleType::AtA;
		bary = Eigen::Vector3d(1.0, 0.0, 0.0);
		return a;
	}

	Eigen::Vector3d bp = p - b;
	double d3 = ab.dot(bp);
	double d4 = ac.dot(bp);
	if (d3 >= 0.0 && d4 <= d3) {
		pointType = ClosestPointOnTriangleType::AtB;
		bary = Eigen::Vector3d(0.0, 1.0, 0.0);
		return b;
	}

	Eigen::Vector3d cp = p - c;
	double d5 = ab.dot(cp);
	double d6 = ac.dot(cp);
	if (d6 >= 0.0 && d5 <= d6) {
		pointType = ClosestPointOnTriangleType::AtC;
		bary = Eigen::Vector3d(0.0, 0.0, 1.0);
		return c;
	}

	double vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
		double v = d1 / (d1 - d3);
		pointType = ClosestPointOnTriangleType::AtAB;
		bary = Eigen::Vector3d(1 - v, v, 0.0);
		return a + v * ab;
	}

	double vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
		double v = d2 / (d2 - d6);
		pointType = ClosestPointOnTriangleType::AtAC;
		bary = Eigen::Vector3d(1 - v, 0.0, v);
		return a + v * ac;
	}

	double va = d3 * d6 - d5 * d4;
	if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
		double v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		pointType = ClosestPointOnTriangleType::AtBC;
		bary = Eigen::Vector3d(0.0, 1 - v, v);
		return b + v * (c - b);
	}

	double denom = 1.0 / (va + vb + vc);
	double v = vb * denom;
	double w = vc * denom;
	pointType = ClosestPointOnTriangleType::AtInterior;
	bary = Eigen::Vector3d(1 - v - w, v, w);
	return a + v * ab + w * ac;
}

double tool_cloth_dynamics::collision::getClosestDistanceBetweenSegments(const Eigen::Vector3d &p0, const Eigen::Vector3d &p1,
		const Eigen::Vector3d &q0, const Eigen::Vector3d &q1) {
	Eigen::Vector3d p = p1 - p0;
	Eigen::Vector3d q = q1 - q0;
	Eigen::Vector3d r = p0 - q0;

	double a = p.dot(p);
	double b = p.dot(q);
	double c = q.dot(q);
	double d = p.dot(r);
	double e = q.dot(r);

	double s = 0.0;
	double t = 0.0;

	double det = a * c - b * b;
	if (det > 1e-5) {
		double bte = b * e;
		double ctd = c * d;

		if (bte <= ctd) {
			if (e <= 0.0) {
				s = (d >= a ? 1.0 : (d > 0.0 ? -d / a : 0.0));
				t = 0.0;
			} else if (e < c) {
				s = 0.0;
				t = e / c;
			} else {
				s = (b - d >= a ? 1.0 : (b - d > 0.0 ? (b - d) / a : 0.0));
				t = 1.0;
			}
		} else {
			s = bte - ctd;
			if (s >= det) {
				if (b + e <= 0.0) {
					s = (d <= 0.0 ? 0.0 : (d < a ? -d / a : 1.0));
					t = 0.0;
				} else if (b + e < c) {
					s = 1.0;
					t = (b + e) / c;
				} else {
					s = (b - d <= 0.0 ? 0.0 : (b - d < a ? (b - d) / a : 1.0));
					t = 1.0;
				}
			} else {
				double ate = a * e;
				double btd = b * d;

				if (ate <= btd) {
					s = (d <= 0.0 ? 0.0 : (d >= a ? 1.0 : -d / a));
					t = 0.0;
				} else {
					t = ate - btd;
					if (t >= det) {
						s = (b - d <= 0.0 ? 0.0 : (b - d < a ? (b - d) / a : 1.0));
						t = 1.0;
					} else {
						s /= det;
						t /= det;
					}
				}
			}
		}
	} else {
		if (e <= 0.0) {
			s = (d <= 0.0 ? 0.0 : (d >= a ? 1.0 : -d / a));
			t = 0.0;
		} else if (e >= c) {
			s = (b - d <= 0.0 ? 0.0 : (b - d >= a ? 1.0 : (b - d) / a));
			t = 1.0;
		} else {
			s = 0.0;
			t = e / c;
		}
	}

	Eigen::Vector3d ps = (1.0 - s) * p0 + s * p1;
	Eigen::Vector3d qt = (1.0 - t) * q0 + t * q1;
	Eigen::Vector3d st = qt - ps;
	return st.norm();
}

std::vector<tool_cloth_dynamics::collision::Contact> tool_cloth_dynamics::collision::detectPointTriCollisions(const Eigen::MatrixXd &cloth_V, const Eigen::MatrixXi &cloth_F,
		const Eigen::MatrixXd &obstacle_V, const Eigen::MatrixXi &obstacle_F, double max_dist) {
	std::vector<tool_cloth_dynamics::collision::Contact> contacts;
	for (int i = 0; i < obstacle_F.cols(); ++i) {
		Eigen::Vector3d a = obstacle_V.col(obstacle_F(0, i));
		Eigen::Vector3d b = obstacle_V.col(obstacle_F(1, i));
		Eigen::Vector3d c = obstacle_V.col(obstacle_F(2, i));

		for (int j = 0; j < cloth_V.cols(); ++j) {
			Eigen::Vector3d p = cloth_V.col(j);
			Eigen::Vector3d bary;
			ClosestPointOnTriangleType type;
			Eigen::Vector3d closest = closestPointTriangle(p, a, b, c, bary, type);
			double dist = (p - closest).norm();
			if (dist < max_dist) {
				// Penetration if signed dist <0, but for proto, use depth = -dist if inside.
				// For simple, assume basic distance, sign via normal dot.
				Eigen::Vector3d normal = (p - closest).normalized();
				double depth = -dist; // Assume negative for penetration.
				contacts.push_back({ closest, normal, depth, j, bary });
			}
		}
	}
	return contacts;
}

// Note: For edge-edge, similar loop over cloth edges vs obstacle edges, using getClosestDistanceBetweenSegments if < threshold, compute midpoints/normal.
