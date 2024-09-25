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
// Created by Yifei Li on 3/15/21.
// Email: liyifei@csail.mit.edu
//

#ifndef OMEGAENGINE_BACKWARDTASKSOLVER_H
#define OMEGAENGINE_BACKWARDTASKSOLVER_H

#include "../engine/Macros.h"
#include "../engine/UtilityFunctions.h"
#include "../simulation/Simulation.h"
#include "OptimizationTaskConfigurations.h"
#include "OptimizeHelper.h"
#include <time.h>

class BackwardTaskSolver {
public:
	static void
	optimizeLBFGS(Simulation *system, OptimizeHelper &helper, int FORWARD_STEPS,
			int demoNum, bool isRandom, int srandSeed,
			const std::function<void(const std::string &)> &setTextBoxCB);

	static void
	setWindSim2realInitialParams(Simulation::ParamInfo &paramGroundtruth,
			Simulation::BackwardTaskInformation &taskInfo,
			Simulation *system);

	static void setDemoSceneConfigAndConvergence(
			Simulation *system, int demoNum,
			Simulation::BackwardTaskInformation &taskInfo);

	static void
	resetSplineConfigsForControlTasks(int demoNum, Simulation *system,
			Simulation::ParamInfo &paramGroundtruth);

	static void setLossFunctionInformationAndType(LossType &lossType,
			Simulation::LossInfo &lossInfo,
			Simulation *system,
			int demoNum);

	static void
	setInitialConditions(int demoNum, Simulation *system,
			Simulation::ParamInfo &paramGroundtruth,
			Simulation::BackwardTaskInformation &taskInfo);

	static void
	solveDemo(Simulation *system,
			const std::function<void(const std::string &)> &setTextBoxCB,
			int demoNum, bool isRandom, int srandSeed);

	static OptimizeHelper getOptimizeHelper(Simulation *system, int demoNum);

	static OptimizeHelper *getOptimizeHelperPointer(Simulation *system,
			int demoNum);
};

#endif // OMEGAENGINE_BACKWARDTASKSOLVER_H
