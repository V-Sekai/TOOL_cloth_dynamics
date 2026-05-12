// AvbdBackwardShim implementation — CHI-14 Brick C.
//
// Pure aggregation: no GPU access beyond what AvbdSolver's public
// accessors expose. Lives in the slang_solver/ tree so it can be
// linked without DiffCloth.

#include "AvbdBackwardShim.h"

#include <cstdint>
#include <vector>

namespace cloth {

namespace {

// Constraint type indices — must mirror DiffCloth's
// `Constraint::ConstraintType` enum (Constraint.h). Kept as named
// constants here to avoid taking a dependency on DiffCloth's headers.
constexpr int IDX_SPRING_STRETCH    = 0;
constexpr int IDX_ATTACHMENT        = 1;
constexpr int IDX_TRIANGLE          = 2;
constexpr int IDX_TRIANGLE_BENDING  = 3;

double sumD(const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += double(x);
    return s;
}

}  // anonymous namespace

int avbdBackwardShim(AvbdSolver& solver,
                     uint32_t nVerts,
                     uint32_t nAttach,
                     uint32_t nTri,
                     uint32_t nBend,
                     const double* dL_dx_in,
                     const double* vertex_area,
                     AvbdParamGradients& out) {
    // Downcast dL_dx to float for the GPU adjoint dispatch. fp32 is
    // the working precision throughout the AVBD pipeline; the
    // accumulated double-precision aggregates below recover most of
    // the lost precision through summation.
    std::vector<float> v_loss(3 * nVerts);
    for (uint32_t i = 0; i < 3 * nVerts; ++i) {
        v_loss[i] = static_cast<float>(dL_dx_in[i]);
    }
    if (solver.stepBackward(v_loss.data()) != 0) return -1;

    // Reset (caller-supplied accumulator semantics are not used here —
    // each shim call yields a fresh per-step gradient, consistent with
    // DiffCloth's PD `stepBackward` which produces one BackwardInfo
    // per step and lets the multi-step driver accumulate).
    for (int i = 0; i < 4; ++i) out.dL_dk_pertype[i] = 0.0;
    out.dL_ddensity = 0.0;

    // Per-vertex position cotangent.
    std::vector<float> dPos;
    solver.readPositionsGrad(dPos);
    out.dL_dx.assign(3 * nVerts, 0.0);
    for (uint32_t i = 0; i < 3 * nVerts; ++i) {
        out.dL_dx[i] = double(dPos[i]);
    }

    // Per-vertex predicted cotangent (CHI-113). Empty if init-backward
    // hasn't been dispatched — caller falls back to zero dL_dwind /
    // dL_dfext in that case.
    std::vector<float> dPred;
    solver.readPredictedGrad(dPred);
    if (!dPred.empty()) {
        out.dL_dpredicted.assign(3 * nVerts, 0.0);
        for (uint32_t i = 0; i < 3 * nVerts && i < dPred.size(); ++i) {
            out.dL_dpredicted[i] = double(dPred[i]);
        }
    } else {
        out.dL_dpredicted.clear();
    }

    // Density (per-vertex mass cotangent × area).
    std::vector<float> dMass;
    solver.readMassGrad(dMass);
    if (vertex_area != nullptr) {
        double acc = 0.0;
        for (uint32_t v = 0; v < nVerts; ++v) {
            acc += double(dMass[v]) * vertex_area[v];
        }
        out.dL_ddensity = acc;
    }

    // Spring stiffness sum (DiffCloth assumes uniform k per type).
    if (nAttach != 0 || nTri != 0 || nBend != 0 || true /* always check spring */) {
        std::vector<float> dRestLen, dSpringStiff;
        solver.readSpringGrad(dRestLen, dSpringStiff);
        out.dL_dk_pertype[IDX_SPRING_STRETCH] = sumD(dSpringStiff);
    }

    // Attachment: per-attachment fixedPos, lambda, summed stiffness.
    if (nAttach > 0) {
        std::vector<float> dFixed, dAtK, dLam;
        solver.readAttachGrad(dFixed, dAtK, dLam);
        out.dL_dxfixed.assign(3 * nAttach, 0.0);
        for (uint32_t i = 0; i < 3 * nAttach; ++i) {
            out.dL_dxfixed[i] = double(dFixed[i]);
        }
        out.dL_dlambda_attach.assign(3 * nAttach, 0.0);
        for (uint32_t i = 0; i < 3 * nAttach; ++i) {
            out.dL_dlambda_attach[i] = double(dLam[i]);
        }
        out.dL_dk_pertype[IDX_ATTACHMENT] = sumD(dAtK);
    } else {
        out.dL_dxfixed.clear();
        out.dL_dlambda_attach.clear();
    }

    // Triangle: stiffness sum + per-tri lambda0/lambda1.
    if (nTri > 0) {
        std::vector<float> dK, dL0, dL1;
        solver.readTriGrad(dK, dL0, dL1);
        out.dL_dk_pertype[IDX_TRIANGLE] = sumD(dK);
        out.dL_dlambda0_tri.assign(3 * nTri, 0.0);
        out.dL_dlambda1_tri.assign(3 * nTri, 0.0);
        for (uint32_t i = 0; i < 3 * nTri; ++i) {
            out.dL_dlambda0_tri[i] = double(dL0[i]);
            out.dL_dlambda1_tri[i] = double(dL1[i]);
        }
    } else {
        out.dL_dlambda0_tri.clear();
        out.dL_dlambda1_tri.clear();
    }

    // Bending: stiffness sum + per-bend lambda. nTarget cotangent is
    // not exposed in DiffCloth's BackwardInformation (rest-pose
    // geometry is bind-time); intentionally dropped here.
    if (nBend > 0) {
        std::vector<float> dN, dK, dLam;
        solver.readBendGrad(dN, dK, dLam);
        out.dL_dk_pertype[IDX_TRIANGLE_BENDING] = sumD(dK);
        out.dL_dlambda_bend.assign(3 * nBend, 0.0);
        for (uint32_t i = 0; i < 3 * nBend; ++i) {
            out.dL_dlambda_bend[i] = double(dLam[i]);
        }
    } else {
        out.dL_dlambda_bend.clear();
    }

    return 0;
}

}  // namespace cloth
