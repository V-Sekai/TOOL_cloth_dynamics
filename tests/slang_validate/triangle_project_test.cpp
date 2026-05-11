// Real-input validator for Cloth.SlangCodegen.TriangleProject — the
// DiffCloth PD in-plane stretch local step (4th and final constraint
// kernel).
//
// Reference: src/code/simulation/Triangle.cpp:project +
// projectToManifold. With inv_deltaUV = identity, F is the raw current
// edges; the kernel does Gram-Schmidt to get an orthonormal frame,
// then closes the frame's polar projection in closed form (no SVD).
//
// Test fixtures (2 triangles sharing the same 4 vertices):
//
//   triangle 0 (sheared, det(F_2D) = 1 = rest area):
//     p0 = (0,0,0), p1 = (1,0,0), p2 = (0.5, 1, 0)
//     F_2D = [1, 0.5; 0, 1]     (after Gram-Schmidt)
//     x = a+d = 2, y = -b = -0.5, rn = sqrt(4.25) = sqrt(17)/2
//     R = (1/rn)·[2, 0.5; -0.5, 2]
//     newF.col(0) = (1,0,0)·(2/rn) + (0,1,0)·(-0.5/rn)
//                 = (4/sqrt(17), -1/sqrt(17), 0)
//     newF.col(1) = (1,0,0)·(0.5/rn) + (0,1,0)·(2/rn)
//                 = (1/sqrt(17), 4/sqrt(17), 0)
//
//   triangle 1 (rest pose at identity):
//     p0 = (0,0,0), p1 = (1,0,0), p2 = (0, 1, 0)
//     F_2D = [1 0; 0 1] → R = I → newF.col(0) = (1,0,0), col(1) = (0,1,0)
//
// Output buffer layout: 2 float3 per triangle, [2c+0] = newF.col(0),
// [2c+1] = newF.col(1). Padded slots (c ≥ 2): sqrtWeight = 0 so each
// output pair is (0,0,0).

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "triangle_project_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_TRI_REAL = 2;
    constexpr uint32_t GROUP_SIZE = 64;

    // Three vertices shared by both triangles (idx referenced; v3 unused).
    std::vector<Vector<float, 3>> positions = {
        Vector<float, 3>(0.0f, 0.0f, 0.0f),
        Vector<float, 3>(1.0f, 0.0f, 0.0f),
        Vector<float, 3>(0.5f, 1.0f, 0.0f),
        Vector<float, 3>(0.0f, 1.0f, 0.0f),
    };

    // Output: 2 float3 per triangle, padded to 2*GROUP_SIZE.
    std::vector<Vector<float, 3>> projected(2 * GROUP_SIZE,
                                            Vector<float, 3>(0.0f));

    // Triangle 0: (v0, v1, v2)  — sheared.
    // Triangle 1: (v0, v1, v3)  — rest pose at identity.
    std::vector<uint32_t> idx(3 * GROUP_SIZE, 0u);
    idx[0] = 0u; idx[1] = 1u; idx[2] = 2u;
    idx[3] = 0u; idx[4] = 1u; idx[5] = 3u;

    std::vector<float> sqrtWeight(GROUP_SIZE, 0.0f);
    sqrtWeight[0] = 1.0f;
    sqrtWeight[1] = 1.0f;

    GlobalParams_0 gp{};
    gp.positions_0.data   = positions.data();    gp.positions_0.count   = positions.size();
    gp.projected_0.data   = projected.data();    gp.projected_0.count   = projected.size();
    gp.idx_0.data         = idx.data();          gp.idx_0.count         = idx.size();
    gp.sqrtWeight_0.data  = sqrtWeight.data();   gp.sqrtWeight_0.count  = sqrtWeight.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    // Reference: mirror the kernel arithmetic in fp32 on the host so we
    // bit-match even on irrational expected values like 1/sqrt(17).
    auto expectedFor = [&](uint32_t c, float out[2][3]) {
        const auto& p0 = positions[idx[3*c + 0]];
        const auto& p1 = positions[idx[3*c + 1]];
        const auto& p2 = positions[idx[3*c + 2]];
        float f0x = p1.x - p0.x, f0y = p1.y - p0.y, f0z = p1.z - p0.z;
        float f1x = p2.x - p0.x, f1y = p2.y - p0.y, f1z = p2.z - p0.z;
        float a   = std::sqrt(f0x*f0x + f0y*f0y + f0z*f0z);
        float e1x = f0x / a, e1y = f0y / a, e1z = f0z / a;
        float b   = e1x*f1x + e1y*f1y + e1z*f1z;
        float tx  = f1x - b*e1x, ty = f1y - b*e1y, tz = f1z - b*e1z;
        float d   = std::sqrt(tx*tx + ty*ty + tz*tz);
        float e2x = tx / d, e2y = ty / d, e2z = tz / d;
        float xv  = a + d;
        float yv  = 0.0f - b;
        float rn  = std::sqrt(xv*xv + yv*yv);
        float r00 = xv / rn, r01 = b / rn;
        float r10 = yv / rn, r11 = xv / rn;
        float n0x = e1x*r00 + e2x*r10;
        float n0y = e1y*r00 + e2y*r10;
        float n0z = e1z*r00 + e2z*r10;
        float n1x = e1x*r01 + e2x*r11;
        float n1y = e1y*r01 + e2y*r11;
        float n1z = e1z*r01 + e2z*r11;
        const float sw = sqrtWeight[c];
        out[0][0] = sw * n0x; out[0][1] = sw * n0y; out[0][2] = sw * n0z;
        out[1][0] = sw * n1x; out[1][1] = sw * n1y; out[1][2] = sw * n1z;
    };

    int   fails        = 0;
    float max_abs_diff = 0.0f;
    for (uint32_t c = 0; c < N_TRI_REAL; ++c) {
        float ref[2][3];
        expectedFor(c, ref);
        for (int col = 0; col < 2; ++col) {
            for (int k = 0; k < 3; ++k) {
                const float got = projected[2*c + col][k];
                const float r   = ref[col][k];
                const float dd  = std::fabs(got - r);
                if (dd > max_abs_diff) max_abs_diff = dd;
                if (dd > 1e-6f) {
                    if (fails < 5) {
                        std::fprintf(stderr,
                            "triangle_project mismatch at c=%u, col=%d, k=%d: got %g, expected %g (diff %g)\n",
                            c, col, k, got, r, dd);
                    }
                    ++fails;
                }
            }
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "triangle_project: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("triangle_project: %u/%u triangles OK (max_abs_diff=%g, %.1fms)\n",
                    N_TRI_REAL, N_TRI_REAL, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "triangle_project: %d FAIL out of %u\n",
                 fails, N_TRI_REAL * 6u);
    return 1;
}
