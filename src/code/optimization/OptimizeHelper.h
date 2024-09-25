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
// Created by Yifei Li on 5/15/21.
// Email: liyifei@csail.mit.edu
//

#ifndef OMEGAENGINE_OPTIMIZEHELPER_H
#define OMEGAENGINE_OPTIMIZEHELPER_H

#include "../engine/Macros.h"
#include "../simulation/Simulation.h"
#include "OptimizationTaskConfigurations.h"

class OptimizeHelper {
public:
	struct Offsets {
		int offset_dL_dfwind;
		int offset_dL_dfext;
		int offset_dL_density;
		int offset_dL_dspline;
		int offset_calcdL_calccontrolPoints;
		int offset_dL_dfwind_timestep;
		int offset_dL_dx0;
		int offset_dL_k[Constraint::CONSTRAINT_NUM];
		int offset_dL_dconstantForceField;
		std::vector<int> offset_dL_dmu = {};
	};

	Simulation::LossInfo lossInfo;
	Simulation::BackwardTaskInformation taskInfo;
	Simulation *system;
	Simulation::TaskSolveStatistics statistics;
	std::string experimentName;

	VecXd paramLowerBound, paramUpperBound;
	std::vector<std::string> paramName;
	std::vector<bool> paramLogScaleTransformOn;
	bool stiffnessLogScale = false;
	bool densityLogScale = false;
	bool windScale = false;

	VecXd param_mean, param_scale;
	int FORWARD_STEPS;
	LossType lossType;
	Simulation::ParamInfo param_guess, param_actual;
	Demos demoNum;
	int totalParamNumber;
	int totalSplineParamNumber, iter = 0;
	bool hasGroundtruthParam = false;

	std::pair<std::vector<Simulation::ForwardInformation>,
			std::vector<Simulation::BackwardInformation>>
			lastBacakwardOptRecord;
	std::pair<Simulation::ParamInfo, double> lastGuess;

	Offsets offset = {};

	OptimizeHelper(Demos demoNum, Simulation *system,
			Simulation::LossInfo &lossInfo,
			Simulation::BackwardTaskInformation &taskInfo,
			LossType lossType, int FORWARD_STEPS,
			Simulation::ParamInfo paramActual);

	OptimizeHelper(OptimizeHelper &other) {
		lossInfo = other.lossInfo;
		taskInfo = other.taskInfo;
		system = other.system;
		paramLowerBound = other.paramLowerBound;
		paramUpperBound = other.paramUpperBound;
		paramName = other.paramName;
		param_mean = other.param_mean;
		param_scale = other.param_scale;
		FORWARD_STEPS = other.FORWARD_STEPS;
		lossType = other.lossType;
		param_guess = other.param_guess;
		demoNum = other.demoNum;
		totalParamNumber = other.totalParamNumber;
		totalSplineParamNumber = other.totalSplineParamNumber;
		iter = other.iter;
		offset = other.offset;
		hasGroundtruthParam = other.hasGroundtruthParam;
		lastBacakwardOptRecord = other.lastBacakwardOptRecord;
		lastGuess = other.lastGuess;
	}

	VecXd getLowerBound();

	VecXd getUpperBound();

	std::pair<VecXd, int> clampParameter(VecXd &in);

	Simulation::ParamInfo vecXdToParamInfo(const VecXd &x);

	std::pair<bool, VecXd> parameterFromRandSeed(int randSeed);

	VecXd paramInfoToVecXd(Simulation::ParamInfo &param);
	VecXd gradientInfoToVecXd(Simulation::BackwardInformation &backwardInfo);
	VecXd getActualParam();
	bool paramIsWithinBound(VecXd x);

	double runSimulationAndGetLoss(const VecXd &x);
	Simulation::BackwardInformation
	runSimulationAndGetLossAndGradient(const VecXd &x);
	std::vector<Simulation::BackwardInformation>
	runSimulationAndGetLossAndGradients(const VecXd &x);
	VecXd getRandomParam(int randSeed = 0);

	void setParameterBounds(Simulation::BackwardTaskInformation &taskInfo);

	void saveLastIter();

	double operator()(const VecXd &x, VecXd &grad);
};

#endif // OMEGAENGINE_OPTIMIZEHELPER_H
