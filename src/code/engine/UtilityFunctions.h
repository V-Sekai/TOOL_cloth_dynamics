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
// Created by Yifei Li on 4/27/21.
// Email: liyifei@csail.mit.edu
//

#ifndef OMEGAENGINE_UTILITYFUNCTIONS_H
#define OMEGAENGINE_UTILITYFUNCTIONS_H

#include "Macros.h"
#include <sys/stat.h>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>

static void insertIntoTriplets33(TripleVector &triplets,
		Eigen::Matrix<double, 3, 3> src, int startRow,
		int startCol) {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (std::abs(src(i, j)) > 1e-10) {
				triplets.emplace_back(startRow + i, startCol + j, src(i, j));
			}
		}
	}
}

template <int rows, int cols>
static void insertIntoTriplets(TripleVector &triplets,
		Eigen::Matrix<double, rows, cols> &src,
		int startRow, int startCol) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (std::abs(src(i, j)) > 1e-10) {
				triplets.emplace_back(startRow + i, startCol + j, src(i, j));
			}
		}
	}
}

template <int rows, int cols>
static void insertIntoTriplets(TripleVector &triplets,
		Eigen::Matrix<double, rows, cols> &src,
		int blockRows, int blockCols, int srcStartRows,
		int srcStartCols, int dstStartRow,
		int dstStartCol) {
	for (int i = 0; i < blockRows; i++) {
		for (int j = 0; j < blockCols; j++) {
			double val = src(srcStartRows + i, srcStartCols + j);
			if (std::abs(val) > 1e-12) {
				triplets.emplace_back(dstStartRow + i, dstStartCol + j, val);
			}
		}
	}
}

template <int m, int n, int p, int q>
static Eigen::Matrix<double, m * p, n * q>
kronecker(const Eigen::Matrix<double, m, n> &A,
		const Eigen::Matrix<double, p, q> &B) {
	Eigen::Matrix<double, m * p, n * q> C;
	C.setZero();
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			for (int k = 0; k < p; k++)
				for (int l = 0; l < q; l++)
					C(i * p + k, j * q + l) = A(i, j) * B(k, l);

	return C;
}

static void checkFolderExistsAndCreate(std::string path) {
	std::filesystem::create_directories(path);
}

typedef Eigen::Transform<double, 3, Eigen::Affine> Rotation;
static Rotation axisToRotation(Vec3d finalDir, Vec3d initialDir) {
	Rotation rot;
	rot.setIdentity();
	finalDir.normalize();
	initialDir.normalize();
	if ((finalDir - initialDir).norm() > 1e-5) {
		Vec3d perp = initialDir.cross(finalDir);
		double angle = std::acos(finalDir.dot(initialDir));
		rot = Eigen::AngleAxis(angle, perp.normalized());
	}
	return rot;
}

static std::vector<std::string>
listFiles(std::string path, std::string extension) // extension = ".txt"
{
	std::vector<std::string> filePaths;
	for (const auto &entry : std::filesystem::directory_iterator(path)) {
		if (entry.path().extension() == extension) {
			filePaths.push_back(entry.path().string());
		}
	}
	return filePaths;
}

static std::vector<std::string> listDirectory(std::string path) {
	std::vector<std::string> filePaths;
	for (const std::filesystem::directory_entry &entry :
			std::filesystem::directory_iterator(path)) {
		filePaths.emplace_back(entry.path().string());
	}
	return filePaths;
}

static std::string currentTimestampToStr() {
	time_t rawtime;
	struct tm *timeinfo;
	char buffer[80];

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", timeinfo);
	std::string str(buffer);

	return str;
}

static glm::vec3 toGlm(Vec3d v) { return glm::vec3(v[0], v[1], v[2]); }

static Vec3d toEigen(glm::vec3 v) { return Vec3d(v[0], v[1], v[2]); }

static std::string vec3d2str(Vec3d v, int precision = 2) {
	std::ostringstream streamObj3;
	streamObj3 << std::fixed;
	streamObj3 << std::setprecision(precision);
	streamObj3 << "(" << v[0] << "," << v[1] << "," << v[2] << ")";

	return streamObj3.str();
}

static std::string vec2str(VecXd v, int precision = 2) {
	std::ostringstream streamObj3;
	streamObj3 << std::fixed;
	streamObj3 << std::setprecision(precision);
	streamObj3 << "(" << v[0];

	for (int i = 1; i < v.rows(); i++)
		streamObj3 << "," << v[i];

	streamObj3 << ")";

	return streamObj3.str();
}

template <int m>
static std::string vec2str(Eigen::Matrix<double, m, 1> v, int precision = 2) {
	std::ostringstream streamObj3;
	streamObj3 << std::fixed;
	streamObj3 << std::setprecision(precision);
	streamObj3 << "(" << v[0];

	for (int i = 1; i < m; i++)
		streamObj3 << "," << v[i];

	streamObj3 << ")";

	return streamObj3.str();
}

static std::string vecXd2str(VecXd v, int precision = 2) {
	std::ostringstream streamObj3;
	streamObj3 << std::fixed;
	streamObj3 << std::setprecision(precision);
	streamObj3 << "(" << v[0];

	for (int i = 1; i < v.rows(); i++)
		streamObj3 << "," << v[i];

	streamObj3 << ")";

	return streamObj3.str();
}

static std::string d2str(double v, int precision = 2) {
	std::ostringstream streamObj3;
	streamObj3 << std::fixed;
	streamObj3 << std::setprecision(precision);
	streamObj3 << v;
	return streamObj3.str();
}

static std::string d2scistr(double v, int precision = 2) {
	std::ostringstream streamObj3;
	streamObj3 << std::scientific;
	streamObj3 << std::setprecision(precision);
	streamObj3 << v;
	return streamObj3.str();
}

static std::string int2str(int v) { return std::to_string(v); }

static std::string lld2str(long long v, int precision = 2) {
	std::ostringstream streamObj3;
	streamObj3 << std::fixed;
	streamObj3 << std::setprecision(precision);
	streamObj3 << v;
	return streamObj3.str();
}

static void writeStringToFile(std::string fileName, std::string content) {
	std::ofstream myfile;
	myfile.open(fileName.c_str(), std::ofstream::app);
	myfile << content;
	myfile.close();
}

#endif // OMEGAENGINE_UTILITYFUNCTIONS_H
