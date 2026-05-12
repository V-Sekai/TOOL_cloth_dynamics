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

# Metal kernels we benchmark on real GPU (Tier-3 perf). The full slang →
# air → metallib pipeline runs once per kernel listed here.
METAL_KERNELS := spring_project saxpby spmv dot_reduce_serial dot_reduce assemble_b cg_alpha cg_beta saxpby_indirect spmv_df32 saxpby_indirect_df32 spring_force vbd_init vbd_solve_apply vbd_gather_spring vbd_gather_attachment vbd_gather_triangle vbd_gather_bending attachment_force_al attachment_dual_update triangle_membrane_force_al triangle_membrane_dual_update triangle_bending_force_al triangle_bending_dual_update self_collision_scan vbd_solve_apply_backward vbd_init_backward vbd_gather_triangle_backward vbd_gather_attachment_backward vbd_gather_bending_backward vbd_gather_spring_backward attachment_force_al_backward spring_force_backward triangle_bending_force_al_backward triangle_membrane_force_al_backward
METALLIBS     := $(addprefix $(EMITTED)/,$(addsuffix .metallib,$(METAL_KERNELS)))

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
              assemble_b:assemble_b \
              cg_alpha:cg_alpha \
              cg_beta:cg_beta \
              saxpby_indirect:saxpby_indirect \
              spmv_df32:spmv_df32 \
              saxpby_indirect_df32:saxpby_indirect_df32 \
              spring_force:spring_force \
              attachment_force:attachment_force \
              triangle_membrane_force:triangle_membrane_force \
              triangle_bending_force:triangle_bending_force \
              vbd_init:vbd_init \
              vbd_solve_apply:vbd_solve_apply \
              vbd_gather_spring:vbd_gather_spring \
              vbd_gather_attachment:vbd_gather_attachment \
              vbd_gather_triangle:vbd_gather_triangle \
              vbd_gather_bending:vbd_gather_bending \
              attachment_dual_update:attachment_dual_update \
              attachment_force_al:attachment_force_al \
              triangle_membrane_dual_update:triangle_membrane_dual_update \
              triangle_membrane_force_al:triangle_membrane_force_al \
              triangle_bending_dual_update:triangle_bending_dual_update \
              triangle_bending_force_al:triangle_bending_force_al \
              spring_force_backward:spring_force_backward \
              triangle_bending_force_al_backward:triangle_bending_force_al_backward \
              triangle_membrane_force_al_backward:triangle_membrane_force_al_backward

TEST_NAMES := $(foreach t,$(TESTS),$(word 1,$(subst :, ,$(t))))
KERNEL_FOR  = $(word 2,$(subst :, ,$(filter $(1):%,$(TESTS))))

KERNELS    := $(sort $(foreach t,$(TESTS),$(word 2,$(subst :, ,$(t)))))
SHADERS    := $(addprefix $(EMITTED)/,$(addsuffix _emit.cpp,$(KERNELS)))
BINS       := $(addprefix $(EMITTED)/,$(addsuffix _test,$(TEST_NAMES))) \
              $(EMITTED)/cg_demo_test \
              $(EMITTED)/pd_step_demo_test \
              $(EMITTED)/pd_multistep_demo_test \
              $(EMITTED)/pd_gravity_demo_test \
              $(EMITTED)/perf_baseline_test \
              $(EMITTED)/metal_perf_test

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

# spring_force_backward FD validator — co-includes the forward and
# backward emits via the namespace + EXTERN_C-undef trick. Tests that
# the analytic adjoint matches central-difference gradients on the
# forward spring-force kernel within 2% relative tolerance.
$(EMITTED)/spring_force_backward_test: spring_force_backward_test.cpp \
                                       $(EMITTED)/spring_force_emit.cpp \
                                       $(EMITTED)/spring_force_backward_emit.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

# triangle_bending_force_al_backward FD validator — same co-include pattern.
$(EMITTED)/triangle_bending_force_al_backward_test: triangle_bending_force_al_backward_test.cpp \
                                                    $(EMITTED)/triangle_bending_force_al_emit.cpp \
                                                    $(EMITTED)/triangle_bending_force_al_backward_emit.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

# triangle_membrane_force_al_backward FD validator — co-include forward + backward.
# Membrane backward is the frozen-R approximation; FD tolerance is generous
# (5% on positions, since the Sifakis-cookie cancellation is only exact at
# converged states — see the kernel header for the math).
$(EMITTED)/triangle_membrane_force_al_backward_test: triangle_membrane_force_al_backward_test.cpp \
                                                     $(EMITTED)/triangle_membrane_force_al_emit.cpp \
                                                     $(EMITTED)/triangle_membrane_force_al_backward_emit.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

# Perf baseline. Times each of the five hottest kernels on a
# realistic-N input through slangc-cpp's sequential simulator. Read
# the harness's header comment for what this measures (regression
# floor, relative cost ranking, scaling sanity) vs what it does NOT
# measure (GPU perf — Tier 3 work).
$(EMITTED)/perf_baseline_test: perf_baseline_test.cpp \
                               $(EMITTED)/spring_project_emit.cpp \
                               $(EMITTED)/saxpby_emit.cpp \
                               $(EMITTED)/spmv_emit.cpp \
                               $(EMITTED)/dot_reduce_serial_emit.cpp \
                               $(EMITTED)/assemble_b_emit.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

# ---- Tier-3 Metal perf ------------------------------------------------
# slang -> .metal source -> .air -> .metallib, then an Objective-C++
# dispatcher loads each .metallib and times kernel invocations on the
# real GPU. The pipeline mirrors curvenet's Layer 2 (slangc-metal) but
# adds runtime dispatch + timing.
#
# Each rule below is target-specific:
#   build/%.metal     comes from slangc -target metal on /tmp/emitted_shaders/%.slang
#   build/%.air       comes from xcrun metal -c on build/%.metal
#   build/%.metallib  comes from xcrun metallib on build/%.air
#
# direct_delta_mush hits an upstream slangc Metal-target codegen bug
# ("Unknown addressspace encountered"), but none of the five kernels
# we benchmark here use float4x4 + multi-buffer indexing, so we're
# clear.

$(EMITTED)/%.metal: emit-shaders
	@mkdir -p $(EMITTED)
	$(SLANGC) -target metal -stage compute -entry main \
	  -o $@ /tmp/emitted_shaders/$*.slang

$(EMITTED)/%.air: $(EMITTED)/%.metal
	xcrun -sdk macosx metal -c $< -o $@

$(EMITTED)/%.metallib: $(EMITTED)/%.air
	xcrun -sdk macosx metallib $< -o $@

# Metal dispatcher needs all five .metallib files alongside the .mm
# source. -fobjc-arc enables Objective-C ARC for the imported headers.
$(EMITTED)/metal_perf_test: metal_perf_test.mm $(METALLIBS)
	$(CXX) -std=c++17 -O2 -fobjc-arc -Wall -Wextra \
	  -framework Metal -framework Foundation \
	  -o $@ $<

clean:
	rm -rf $(EMITTED)
