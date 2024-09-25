
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

#include "engine/UtilityFunctions.h"
#include "optimization/BackwardTaskSolver.h"
#include "optimization/OptimizationTaskConfigurations.h"
#include "simulation/Simulation.h"
#include "supports/Logging.h"
#include <mutex>
#include <queue>
using namespace glm;

char *getCmdOption(char **begin, char **end, const std::string &option) {
	char **itr = std::find(begin, end, option);
	if (itr != end && ++itr != end) {
		return *itr;
	}
	return 0;
}

static std::string getEnvVar(const std::string &name) {
	if (const char *value = std::getenv(name.c_str())) {
		return std::string(value);
	} else {
		return "";
	}
}

int main(int argc, char *argv[]) {
	int n_threads = std::thread::hardware_concurrency();
	std::string NUM_THREADS_ENV_VAR = getEnvVar("OMP_NUM_THREADS");
	if (NUM_THREADS_ENV_VAR.empty()) {
		Logging::logWarning(
				"OMP_NUM_THREADS not specified in your environment variable. Defaulting to number of physical cores: " + std::to_string(n_threads) + "\n");
	} else {
		n_threads = std::stoi(NUM_THREADS_ENV_VAR);
		Logging::logColor("OMP_NUM_THREADS=" + std::to_string(n_threads) + "\n",
				Logging::GREEN);
	}
	bool parallelizeEigen = true;
	if (OPENMP_ENABLED) {
		Eigen::initParallel();
		omp_set_num_threads(n_threads);
		if (parallelizeEigen) {
			Eigen::setNbThreads(n_threads);
		} else {
			Eigen::setNbThreads(1);
		}
		int n = Eigen::nbThreads();
		std::printf("Eigen threads: %d\n", n);
#pragma omp parallel default(none)
		{
#pragma omp single
			printf("OpenMP num_threads = %d\n", omp_get_num_threads());
		}
	}

	enum Modes {
		BACKWARD_TASK, /* 1 */
		BACKWARD_TASK_DEMO, /* 2 */
	};
	std::vector<std::string> validDemos = { "tshirt", "sock", "hat", "sphere",
		"dress" };
	char *demoNameStr = getCmdOption(argv, argv + argc, "-demo");
	char *randSeedStr = getCmdOption(argv, argv + argc, "-seed");
	char *expStr = getCmdOption(argv, argv + argc, "-exp");
	if (!demoNameStr || argc == 1) {
		std::string message = "WARNING: No command line argument provided.\n"
							  "Usage: Please specify -demo [tshirt, sock, hat, sphere, dress] followed by -seed [number]\n"
							  "For example: -demo tshirt -seed 12345\n"
							  "\nDetails of each demo:\n"
							  "- T-shirt: System Identification - Optimize wind model and cloth material parameters to match target trajectory. 4278 Dof, h=1/90s, 250 Timesteps, 6 Design Parameters\n"
							  "- Sock: Trajectory Optimization - Optimize manipulator end effector trajectories to put on the sock. 3165 Dof, h=1/160s, 400 Timesteps, 36 Design Parameters\n"
							  "- Hat: Trajectory Optimization - Optimize manipulator end effector trajectories to move the hat onto the head. 1737 Dof, h=1/100s, 400 Timesteps, 18 Design Parameters\n"
							  "- Sphere: System Identification - Optimize the frictional coefficient between the sphere and the cloth to match target trajectory. 1875 Dof, h=1/180s, 350 Timesteps, 1 Design Parameters\n"
							  "- Dress: Inverse Design - Optimize dress material parameters so that the spinning angle of the dress is 50 degrees. 10902 Dof, h=1/120s, 125 Timesteps, 2 Design Parameters";
		Logging::logFatal(message);
		Logging::logFatal("Exiting program...");
		exit(0);
	} else {
		std::string demoName = std::string(demoNameStr);
		Demos demo;
		if (demoName == "tshirt") {
			demo = Demos::DEMO_WIND_TSHIRT;
		} else if (demoName == "sock") {
			demo = Demos::DEMO_WEAR_SOCK;
		} else if (demoName == "hat") {
			demo = Demos::DEMO_WEAR_HAT;
		} else if (demoName == "sphere") {
			demo = Demos::DEMO_SPHERE_ROTATE;
		} else if (demoName == "dress") {
			demo = Demos::DEMO_DRESS_TWIRL;
		}
		if (!randSeedStr) {
			Logging::logFatal("Please specify " + std::string("-seed") + "\n");
			exit(0);
		}

		Simulation::SceneConfiguration initSceneProfile =
				OptimizationTaskConfigurations::hatScene;
		Simulation *clothSystem =
				Simulation::createSystem(initSceneProfile, Vec3d(0, 0, 0), true);
		BackwardTaskSolver::solveDemo(
				clothSystem, [&](const std::string &v) {}, demo, true, std::atoi(randSeedStr));
		delete clothSystem;
		std::printf("Exiting program...\n");
	}

	return 0;
}
