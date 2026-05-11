# Real-input validators for the Cloth.SlangCodegen.* kernels.
#
# Each kernel is converted by `slangc -target cpp` to a CPU function
# `main_0(varyingInput, params, globals)`, then compiled together
# with a hand-written test harness that:
#   1. populates the kernel's input buffers with known values,
#   2. invokes the kernel as a single dispatch,
#   3. asserts the output matches an independent CPU reference.
#
# `make test` runs every harness; non-zero exit on any failure.
#
# Tests are listed as `<test_name>:<kernel_name>` pairs so multiple
# tests can share the same kernel emit if needed.
#
# Prerequisite: a slangc toolchain at <repo>/bin/.slang/, installed by
#   misc/install-slang.sh
# The bin/slangc wrapper is a shim that exec's bin/.slang/bin/slangc.

REPO_ROOT  := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
SLANGC     := $(REPO_ROOT)/bin/slangc
SLANG_DIR  := $(REPO_ROOT)/bin/.slang
EMITTED    := $(CURDIR)/build
LEAN_DIR   := $(REPO_ROOT)/lean

CXX        ?= clang++
# -Wno-unused-parameter silences clang's complaint about
# `entryPointParams_0` in slangc -target cpp output (the parameter is
# part of the Slang ABI and unused by every compute kernel we emit).
CXXFLAGS   ?= -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter
INCLUDES   := -I$(SLANG_DIR)/include -I$(EMITTED)

# Per-test-to-kernel mapping. test_name → kernel_name.
TESTS      := spring_project:spring_project triangle_bending:triangle_bending \
              attachment_project:attachment_project \
              triangle_project:triangle_project \
              saxpby:saxpby \
              spmv:spmv \
              dot_reduce_serial:dot_reduce_serial \
              assemble_b:assemble_b

TEST_NAMES := $(foreach t,$(TESTS),$(word 1,$(subst :, ,$(t))))
KERNEL_FOR  = $(word 2,$(subst :, ,$(filter $(1):%,$(TESTS))))

KERNELS    := $(sort $(foreach t,$(TESTS),$(word 2,$(subst :, ,$(t)))))
SHADERS    := $(addprefix $(EMITTED)/,$(addsuffix _emit.cpp,$(KERNELS)))
BINS       := $(addprefix $(EMITTED)/,$(addsuffix _test,$(TEST_NAMES))) \
              $(EMITTED)/cg_demo_test \
              $(EMITTED)/pd_step_demo_test \
              $(EMITTED)/pd_multistep_demo_test \
              $(EMITTED)/pd_gravity_demo_test

.PHONY: test clean emit-shaders

test: $(BINS)
	@fails=0; \
	for b in $(BINS); do \
	  echo "=== $$b"; $$b || fails=$$(( fails + 1 )); \
	done; \
	if [ $$fails -ne 0 ]; then echo "FAILED: $$fails"; exit 1; fi; \
	echo "all kernels OK"

# Make sure the .slang files exist before we ask slangc for cpp output.
emit-shaders:
	@cd $(LEAN_DIR) && lake exe emit_shaders /tmp/emitted_shaders >/dev/null

# slang → cpp source.
$(EMITTED)/%_emit.cpp: emit-shaders
	@mkdir -p $(EMITTED)
	$(SLANGC) -target cpp -stage compute -entry main \
	  -o $@ /tmp/emitted_shaders/$*.slang

# Per-test build rule. Each test_name maps to a kernel_name via the
# TESTS list; the test source #include's its kernel's emit.
.SECONDEXPANSION:
$(EMITTED)/%_test: %_test.cpp $$(EMITTED)/$$(call KERNEL_FOR,%)_emit.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

# Multi-kernel composition test. cg_demo dispatches both spmv and saxpby
# from a single TU (see header comment in cg_demo_test.cpp for the
# namespace + SLANG_PRELUDE_EXTERN_C-undef pattern that lets two
# slangc-emitted kernels coexist).
$(EMITTED)/cg_demo_test: cg_demo_test.cpp $(EMITTED)/spmv_emit.cpp \
                         $(EMITTED)/saxpby_emit.cpp \
                         $(EMITTED)/dot_reduce_serial_emit.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

# End-to-end PD step composition. Co-includes six slangc-emit.cpp
# files in one TU via the namespace + EXTERN_C-undef trick.
$(EMITTED)/pd_step_demo_test: pd_step_demo_test.cpp \
                              $(EMITTED)/spring_project_emit.cpp \
                              $(EMITTED)/attachment_project_emit.cpp \
                              $(EMITTED)/assemble_b_emit.cpp \
                              $(EMITTED)/spmv_emit.cpp \
                              $(EMITTED)/saxpby_emit.cpp \
                              $(EMITTED)/dot_reduce_serial_emit.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

# Multi-step time integration. Same six kernels as pd_step_demo, wrapped
# in an outer time loop with velocity tracking and damping.
$(EMITTED)/pd_multistep_demo_test: pd_multistep_demo_test.cpp \
                                   $(EMITTED)/spring_project_emit.cpp \
                                   $(EMITTED)/attachment_project_emit.cpp \
                                   $(EMITTED)/assemble_b_emit.cpp \
                                   $(EMITTED)/spmv_emit.cpp \
                                   $(EMITTED)/saxpby_emit.cpp \
                                   $(EMITTED)/dot_reduce_serial_emit.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

# Hanging-mass demo. Same six kernels as pd_multistep_demo, with
# gravity added to the predictor.
$(EMITTED)/pd_gravity_demo_test: pd_gravity_demo_test.cpp \
                                 $(EMITTED)/spring_project_emit.cpp \
                                 $(EMITTED)/attachment_project_emit.cpp \
                                 $(EMITTED)/assemble_b_emit.cpp \
                                 $(EMITTED)/spmv_emit.cpp \
                                 $(EMITTED)/saxpby_emit.cpp \
                                 $(EMITTED)/dot_reduce_serial_emit.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -rf $(EMITTED)
