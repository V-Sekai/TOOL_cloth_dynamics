# Standalone build for the MetalCGSolver wrapper + its correctness test.
# This Makefile is deliberately separate from DiffCloth's CMake build
# (in build_diffcloth/) and from tests/slang_validate/Makefile — it
# proves the C++ ↔ Obj-C++ ↔ Metal wrapper works in isolation before
# we wire it into Simulation.cpp.
#
# Prerequisite:
#   make -C ../../../tests/slang_validate test
# (so that spmv.metallib, saxpby.metallib, dot_reduce.metallib exist).

REPO_ROOT  := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../..)
METALLIB_DIR := $(REPO_ROOT)/tests/slang_validate/build
BUILD      := $(CURDIR)/build

CXX        ?= clang++
CXXFLAGS   ?= -std=c++17 -O2 -Wall -Wextra
INCLUDES   := -I.

EIGEN_INC := $(REPO_ROOT)/external/eigen
LBFGSPP_INC := $(REPO_ROOT)/external/LBFGSpp/include

.PHONY: test test-avbd test-inverse-design clean

test: $(BUILD)/test_metal_cg_solver
	@echo "=== Smoke test of the wrapper (no Eigen, no DiffCloth)."
	$(BUILD)/test_metal_cg_solver $(METALLIB_DIR)

test-avbd: $(BUILD)/test_avbd_solver
	@echo "=== Smoke test of AvbdSolver — all 6 AVBD PSOs build."
	$(BUILD)/test_avbd_solver $(METALLIB_DIR)

test-inverse-design: $(BUILD)/test_avbd_inverse_design
	@echo "=== Inverse-design MVP (CHI-14 Brick A): LBFGSpp fits k via AVBD adjoint."
	$(BUILD)/test_avbd_inverse_design $(METALLIB_DIR)

$(BUILD)/test_metal_cg_solver: test_metal_cg_solver.cpp \
                                $(BUILD)/MetalCGSolver.o
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	  -framework Metal -framework Foundation \
	  -o $@ $^

$(BUILD)/test_avbd_solver: test_avbd_solver.cpp $(BUILD)/AvbdSolver.o
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	  -framework Metal -framework Foundation \
	  -o $@ $^

$(BUILD)/test_avbd_inverse_design: test_avbd_inverse_design.cpp $(BUILD)/AvbdSolver.o
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I$(EIGEN_INC) -I$(LBFGSPP_INC) \
	  -framework Metal -framework Foundation \
	  -o $@ $^

$(BUILD)/MetalCGSolver.o: MetalCGSolver.mm MetalCGSolver.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -fobjc-arc -c -o $@ MetalCGSolver.mm

$(BUILD)/AvbdSolver.o: AvbdSolver.mm AvbdSolver.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -fobjc-arc -c -o $@ AvbdSolver.mm

clean:
	rm -rf $(BUILD)
