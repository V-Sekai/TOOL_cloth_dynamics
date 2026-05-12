// Smoke test for AvbdSolver — verifies all six AVBD kernel metallibs
// load and build PSOs successfully. This is the PR-E stub validation;
// actual GPU dispatch isn't tested yet (that's follow-up PRs).
//
// Usage:
//   ./test_avbd_solver <metallib_dir>
//
// Expects <metallib_dir> to contain:
//   vbd_init.metallib
//   vbd_gather_spring.metallib
//   vbd_gather_attachment.metallib
//   vbd_gather_triangle.metallib
//   vbd_gather_bending.metallib
//   vbd_solve_apply.metallib

#include "AvbdSolver.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <metallib_dir>\n", argv[0]);
        return 2;
    }

    cloth::AvbdSolver solver(argv[1]);
    if (!solver.ok()) {
        std::fprintf(stderr, "test_avbd_solver: construction failed\n");
        return 1;
    }

    // Stub: step() is a no-op that should return 0 when ok.
    const int rc = solver.step();
    if (rc != 0) {
        std::fprintf(stderr, "test_avbd_solver: step() returned %d, expected 0\n", rc);
        return 1;
    }

    std::printf("test_avbd_solver: OK (all 6 PSOs built, step stub returned 0)\n");
    return 0;
}
