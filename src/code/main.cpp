
#include "engine/UtilityFunctions.h"
#include "optimization/BackwardTaskSolver.h"
#include "optimization/OptimizationTaskConfigurations.h"
#include "simulation/Simulation.h"
#include "supports/Logging.h"
#include <mutex>
#include <queue>
using namespace glm;

void runBackwardTask(int demoIdx, bool isRandom, int srandSeed) {
	Simulation::SceneConfiguration initSceneProfile =
			OptimizationTaskConfigurations::hatScene;
	Simulation *clothSystem =
			Simulation::createSystem(initSceneProfile, Vec3d(0, 0, 0), true);

	BackwardTaskSolver::solveDemo(
			clothSystem, [&](const std::string &v) {}, demoIdx, isRandom, srandSeed);

	delete clothSystem;
}

char *getCmdOption(char **begin, char **end, const std::string &option) {
	char **itr = std::find(begin, end, option);
	if (itr != end && ++itr != end) {
		return *itr;
	}
	return 0;
}

void checkCmdOptionExistsAndValid(std::string option, char *input,
		std::vector<std::string> options) {
	if (!input) {
		Logging::logFatal("Please specify " + std::string(option) + "\n");
		exit(0);
	}

	if (options.empty())
		return;
	for (std::string candidate : options) {
		if (candidate == std::string(input))
			return;
	}

	Logging::logFatal("Invalid option for  " + std::string(option) + "\n");
	exit(0);
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
	checkCmdOptionExistsAndValid("-demo", demoNameStr, validDemos);

	if (argc == 1) {
		Logging::logFatal(
				"WARNING: No command line argument.\n Please specify "
				"-demo [tshirt, sock, hat] -seed [number]\n");
		Logging::logFatal("Exiting program...\n");
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
		checkCmdOptionExistsAndValid("-seed", randSeedStr, {});
		runBackwardTask(demo, true, std::atoi(randSeedStr));
		std::printf("Exiting program...\n");
	}

	return 0;
}
