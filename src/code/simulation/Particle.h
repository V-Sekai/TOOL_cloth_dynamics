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

#ifndef OMEGAENGINE_PARTICLE_H
#define OMEGAENGINE_PARTICLE_H

#include "../engine/Macros.h"

class Particle {
public:
	double mass;
	int idx;
	Vec3d pos_rest; // position where strain will be 0
	Vec3d pos_init; // starting position
	Vec3d pos;
	double radii;
	double area; // the distributed area onto this particle

	Vec3d velocity_init;
	Vec3d normal;
	Vec2i planeCoord; // int coordiant in 2D cloth plane1
	Vec3d velocity;

	Particle(double mass, const Vec3d &rest_pos, const Vec3d &pos_initial,
			const Vec3d &velocity, const Vec2i &planeCoord, int idx) :
			mass(mass), pos_rest(rest_pos), pos_init(pos_initial), pos(pos_initial), velocity(velocity), velocity_init(velocity), planeCoord(planeCoord), normal(0, 0, 0), idx(idx){};

	void addNormal(const Vec3d &n) { normal += n; }

	void clearNormal() { normal = Vec3d(0.0, 0.0, 0.0); }

	void printState() const {
		std::printf("Paricle %d x=(%.2f,%.2f,%.2f) v=(%.9f,%.9f,%.9f)\n", idx,
				pos[0], pos[1], pos[2], velocity[0], velocity[1], velocity[2]);
	}
};

#endif // OMEGAENGINE_PARTICLE_H
