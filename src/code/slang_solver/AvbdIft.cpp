#include "AvbdIft.h"
#include "AvbdSolver.h"

#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>

namespace cloth {

struct AvbdIft::Impl {
    AvbdSolver& solver;
    uint32_t nV = 0;
    bool topologyReady = false;

    // Static topology.
    std::vector<std::vector<uint32_t>> nbr;          // N[u]: u and every vertex sharing a clique
    std::vector<int> colorOf;                        // distance-2 coloring
    int numColors = 0;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> pairsOfColor;  // (u, w), w has the color

    // Cached factorization.
    using SpMat = Eigen::SparseMatrix<double>;
    Eigen::SimplicialLDLT<SpMat> ldlt;
    bool haveFactor = false;
    int refactors = 0, solveCount = 0;

    explicit Impl(AvbdSolver& s) : solver(s) {}

    void buildTopology() {
        std::vector<uint32_t> flat; std::vector<uint8_t> sizes;
        solver.getConstraintCliques(flat, sizes);
        uint32_t maxV = 0;
        for (uint32_t v : flat) maxV = std::max(maxV, v);
        nV = maxV + 1;
        std::vector<std::set<uint32_t>> adj(nV);
        size_t at = 0;
        for (uint8_t sz : sizes) {
            for (int a = 0; a < sz; ++a)
                for (int b = 0; b < sz; ++b) adj[flat[at + a]].insert(flat[at + b]);
            at += sz;
        }
        nbr.assign(nV, {});
        for (uint32_t u = 0; u < nV; ++u) { adj[u].insert(u); nbr[u].assign(adj[u].begin(), adj[u].end()); }
        // Distance-2 greedy coloring: vertices within two hops get different colors.
        colorOf.assign(nV, -1);
        std::vector<char> used;
        for (uint32_t v = 0; v < nV; ++v) {
            used.assign(nV < 256 ? 256 : size_t(numColors) + 2, 0);
            if (used.size() < size_t(numColors) + 2) used.resize(size_t(numColors) + 2, 0);
            for (uint32_t u : nbr[v])
                for (uint32_t w : nbr[u])
                    if (colorOf[w] >= 0 && size_t(colorOf[w]) < used.size()) used[colorOf[w]] = 1;
            int c = 0;
            while (size_t(c) < used.size() && used[c]) ++c;
            colorOf[v] = c;
            numColors = std::max(numColors, c + 1);
        }
        pairsOfColor.assign(numColors, {});
        for (uint32_t w = 0; w < nV; ++w)
            for (uint32_t u : nbr[w]) pairsOfColor[colorOf[w]].emplace_back(u, w);
        topologyReady = true;
    }
};

AvbdIft::AvbdIft(AvbdSolver& solver) : impl_(new Impl(solver)) {}
AvbdIft::~AvbdIft() = default;
int AvbdIft::refactorizations() const { return impl_->refactors; }
int AvbdIft::solves() const { return impl_->solveCount; }

int AvbdIft::solve(const float* x, const float* y, const float* fixedPos, const float* c,
                   int maxIter, float relTol, std::vector<float>& z, float* relRes) {
    Impl& I = *impl_;
    AvbdSolver& S = I.solver;
    if (!S.ok()) return -1;
    if (!I.topologyReady) I.buildTopology();
    const uint32_t nV = I.nV;
    const size_t n = 3 * size_t(nV);
    z.assign(n, 0.0f);
    if (relRes) *relRes = 0.0f;
    ++I.solveCount;

    double cNorm = 0.0;
    for (size_t i = 0; i < n; ++i) cNorm += double(c[i]) * double(c[i]);
    cNorm = std::sqrt(cNorm);
    if (cNorm == 0.0) return 0;

    S.updateState(x, y);
    if (fixedPos) S.updateAttachmentFixedPos(fixedPos);

    double lo[3] = { 1e30, 1e30, 1e30 }, hi[3] = { -1e30, -1e30, -1e30 };
    for (uint32_t v = 0; v < nV; ++v)
        for (int k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], double(x[3 * v + k]));
            hi[k] = std::max(hi[k], double(x[3 * v + k]));
        }
    const double extent = std::sqrt((hi[0]-lo[0])*(hi[0]-lo[0]) + (hi[1]-lo[1])*(hi[1]-lo[1]) +
                                    (hi[2]-lo[2])*(hi[2]-lo[2])) + 1e-12;

    std::vector<float> xp(n), xm(n), gp, gm;
    auto hvp = [&](const std::vector<double>& v, std::vector<double>& out) -> bool {
        double vmax = 0.0;
        for (double e : v) vmax = std::max(vmax, std::fabs(e));
        if (vmax == 0.0) { std::fill(out.begin(), out.end(), 0.0); return true; }
        const double eps = 1e-3 * extent / vmax;
        for (size_t i = 0; i < n; ++i) {
            xp[i] = float(double(x[i]) + eps * v[i]);
            xm[i] = float(double(x[i]) - eps * v[i]);
        }
        if (S.computeGradient(xp.data(), gp, nullptr) != 0) return false;
        if (S.computeGradient(xm.data(), gm, nullptr) != 0) return false;
        for (size_t i = 0; i < n; ++i) out[i] = (double(gp[i]) - double(gm[i])) / (2.0 * eps);
        return true;
    };

    auto refactor = [&]() -> bool {
        using Trip = Eigen::Triplet<double>;
        std::vector<Trip> trips;
        std::vector<double> e(n), col(n);
        for (int cidx = 0; cidx < I.numColors; ++cidx) {
            const auto& pairs = I.pairsOfColor[cidx];
            if (pairs.empty()) continue;
            for (int a = 0; a < 3; ++a) {
                std::fill(e.begin(), e.end(), 0.0);
                for (const auto& uw : pairs) if (uw.first == uw.second) e[3 * size_t(uw.second) + a] = 1.0;
                if (!hvp(e, col)) return false;
                for (const auto& uw : pairs)
                    for (int r = 0; r < 3; ++r)
                        trips.emplace_back(int(3 * uw.first + r), int(3 * uw.second + a), col[3 * size_t(uw.first) + r]);
            }
        }
        const int dim = int(n);
        Impl::SpMat H(dim, dim);
        H.setFromTriplets(trips.begin(), trips.end());
        Impl::SpMat Hs = (H + Impl::SpMat(H.transpose())) * 0.5;
        // Factor, shifting the diagonal until the matrix is positive definite.
        double maxDiag = 0.0;
        for (int i = 0; i < int(n); ++i) maxDiag = std::max(maxDiag, std::fabs(Hs.coeff(i, i)));
        double shift = 0.0;
        for (int attempt = 0; attempt < 8; ++attempt) {
            Impl::SpMat A = Hs;
            if (shift > 0.0)
                for (int i = 0; i < int(n); ++i) A.coeffRef(i, i) += shift;
            I.ldlt.compute(A);
            bool good = I.ldlt.info() == Eigen::Success;
            if (good) {
                const auto d = I.ldlt.vectorD();
                for (int i = 0; i < d.size(); ++i) if (!(d[i] > 0.0)) { good = false; break; }
            }
            if (good) { I.haveFactor = true; ++I.refactors; return true; }
            shift = (shift == 0.0) ? 1e-6 * maxDiag : shift * 10.0;
        }
        return false;
    };

    auto pcg = [&](std::vector<double>& zd, double& res) -> int {
        std::vector<double> r(n), s(n), p(n), Hp(n);
        std::fill(zd.begin(), zd.end(), 0.0);
        for (size_t i = 0; i < n; ++i) r[i] = c[i];
        auto prec = [&](const std::vector<double>& rr, std::vector<double>& out) {
            Eigen::VectorXd b = Eigen::Map<const Eigen::VectorXd>(rr.data(), long(n));
            Eigen::VectorXd sol = I.ldlt.solve(b);
            Eigen::Map<Eigen::VectorXd> outMap(out.data(), long(n));
            outMap = sol;
        };
        prec(r, s);
        p = s;
        double rs = 0.0;
        for (size_t i = 0; i < n; ++i) rs += r[i] * s[i];
        res = cNorm;
        int it = 0;
        for (; it < maxIter; ++it) {
            if (!hvp(p, Hp)) return -1;
            double pHp = 0.0;
            for (size_t i = 0; i < n; ++i) pHp += p[i] * Hp[i];
            if (!(pHp > 0.0)) break;
            const double alpha = rs / pHp;
            for (size_t i = 0; i < n; ++i) { zd[i] += alpha * p[i]; r[i] -= alpha * Hp[i]; }
            res = 0.0;
            for (size_t i = 0; i < n; ++i) res += r[i] * r[i];
            res = std::sqrt(res);
            if (res <= double(relTol) * cNorm) { ++it; break; }
            prec(r, s);
            double rsNew = 0.0;
            for (size_t i = 0; i < n; ++i) rsNew += r[i] * s[i];
            const double beta = rsNew / rs;
            rs = rsNew;
            for (size_t i = 0; i < n; ++i) p[i] = s[i] + beta * p[i];
        }
        return it;
    };

    std::vector<double> zd(n);
    double res = cNorm;
    int its = -1;
    if (I.haveFactor) its = pcg(zd, res);
    if (its < 0 || res > double(relTol) * cNorm) {
        // Cached factor is stale for this step: rebuild at the current state.
        if (!refactor()) return -2;
        its = pcg(zd, res);
        if (its < 0) return -1;
    }
    for (size_t i = 0; i < n; ++i) z[i] = float(zd[i]);
    if (relRes) *relRes = float(res / cNorm);
    S.updateState(x, y);   // leave the solver at the converged positions
    return its;
}

}  // namespace cloth
