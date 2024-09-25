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
// Created by Yifei Li on 3/3/21.
// Email: liyifei@csail.mit.edu
//

#ifndef OMEGAENGINE_SIMULATIONCONSTANTS_H
#define OMEGAENGINE_SIMULATIONCONSTANTS_H

#include "../simulation/Simulation.h"

class OptimizationTaskConfigurations {
public:
	static Simulation::FabricConfiguration normalFabric6lowres,
			slopeFabricRestOnPlane, conitnuousNormalTestFabric, tshirt1000,
			agenthat579, sock482, dressv7khandsUpDrape, sphereFabric, normalFabric6;

	static Simulation::SceneConfiguration simpleScene, rotatingSphereScene,
			windScene, tshirtScene, hatScene, sockScene, dressScene,
			continousNormalScene, slopeSimplifiedScene;

	static Simulation::TaskConfiguration demoSphere, demoTshirt, demoWInd,
			demoHat, demoSock, demoDress, demoWindSim2Real, demoSlope;
	static std::vector<Simulation::SceneConfiguration> sceneConfigArrays;

	static std::map<int, Simulation::TaskConfiguration> demoNumToConfigMap;
};

#endif // OMEGAENGINE_SIMULATIONCONSTANTS_H
