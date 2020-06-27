// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2020 Monir Hadji <hadji.szs@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#include "optimal_transport_interpolation.h"

namespace utils
{
    // TODO: express it without memory usage? bench
    const auto gen_gaussian = [](int sigma, int siz) {
        const int lo = -1 * std::ceil(siz / 2);
        const int hi = std::ceil(siz / 2);

        VectorXd ret = VectorXd::Constant(hi - lo + 1, 1.);

        double sumv = 0.;
        // pragma simd
        for (int i = 0, v = lo; v <= hi; ++i, ++v) {
            ret(i) = v;
            // std::clog << ret(i) << " ";
            ret(i) = std::exp(-(ret(i)*ret(i) / 2.*(sigma*sigma)));
            sumv += ret(i);
        }

        // std::reduce or std::accumulate
        for (int i = 0; i < ret.size(); ++i)
            ret(i) /= sumv;
        // std::clog << std::endl;
        return ret;
    };

    template<typename T>
    struct less {
        bool operator()(const T& lhs, const T& rhs) const
        {
            return lhs < rhs;
        };
    };

    template<typename T, class Compare>
    const T& clamp(T&& v, T&& lo, T&& hi, Compare comp) {
        return assert(!comp(hi, lo)),
            comp(v, lo) ? lo : comp(hi, v) ? hi : v;
    }

    template<typename T>
    const T& clamp(T&& v, T&& lo, T&& hi) {
        return clamp(v, lo, hi, less<T>());
    }
}

inline
void clamp_n(VectorXd& q, double zero)
{
    for (int i = 0; i < int(q.size()); ++i)
    {
        if (q[i] > zero || q[i] < -zero) continue;
        if (q[i] >= 0.0) q[i] = zero;
        else             q[i] = -zero;
    }
}

struct {
    double upperEntropy = 0.8;
    double tolerance = 1e-7;
    double diffIters = 10;
    double maxIters = 1000;
    double epsilon = 1e-50;
    double gamma = 0.0;
} mOpt;

IGL_INLINE bool otbarycenter(
    const std::vector<Eigen::MatrixXd>& shapes,
    const Eigen::RowVector3i& res, // grid resolution of the input scalar fields
    const VectorXd& area,
    const VectorXd& alpha,
    const int num_iter,
    const int nb_iter,
    VectorXd& barycenter)
{
    auto p = shapes;
    auto& mArea = area;
    auto& q = barycenter;

    const int gridsize = 40;

    const double mu = res(0) / (gridsize - 20);
    // const int nbvoxel = res(0)*res(1)*res(2);
    VectorXd H = utils::gen_gaussian(mu, mu*(gridsize - 10.));

    const auto get = [&res] (VectorXd& p, int xi, int yi, int zi)
        -> double&
        { return p[xi + res(0)*(yi + res(1)*zi)]; };

    const auto imfilter = [&] (VectorXd I, const VectorXd& h, int dim)
        {
            const int w = res(1);
            VectorXd vres = I;
            // sort of 1d convolution
            const auto idx = [&](int i) {
                int lo = 0, hi = w - 1;
                return utils::clamp(i, lo, hi);
            };

            int DIM = h.size() / 2;
            for (int zi = 0; zi < res(2); zi++) {
                for (int yi = 0; yi < res(1); yi++) {
                    for (int xi = 0; xi < res(0); xi++) {
                        double res = 0.;
                        for (int p = 0; p < h.size(); ++p) {
                            if (dim == 1)
                                res += h(p) * get(I, xi, yi, idx(zi + p - DIM));
                            else if (dim == 2)
                                res += h(p) * get(I, xi, idx(yi + p - DIM), zi);
                            else
                                res += h(p) * get(I, idx(xi + p - DIM), yi, zi);
                        }
                        get(vres, xi, yi, zi) = res;
                    }
                }
            }
            return vres;
        };

    const auto Kv = [&] (VectorXd& p) {
        // apply a gaussian filter in each dimension
        // p is the N*N*N voxelization in one column (i.e size == [N*N*N, 1])
        //VectorXd ret = p;
        p = p.array() * area.array();
        return imfilter(imfilter(imfilter(p, H, 1), H, 2), H, 3);
    };

    for (int k = 0; k < p.size(); ++k) {
        const double sumv = (p[k].array() * mArea.array()).sum();
        p[k] = p[k].array() / sumv;
    }

    // sanity check
    assert(p.size() > 0);
    assert((int)p.size() == alpha.size());
    //for (int i = 0; i < alpha.size(); ++i)
    //    std::cout << alpha(i) << " ; ";
    //std::cout << "\n<< alpha" << std::endl;

    // auto firstres = p[0]._gridsf._res;
    // for (int i = 1; i < p.size(); ++i) {
    //     assert(firstres == p[i]._gridsf._res && "grid res need to be all equals");
    //     //std::cout << p[i]._gridsf._res << std::endl;
    // }

    assert(mArea.size() > 0);
    for (unsigned i = 0; i < p.size(); ++i)
    {
        assert(!(p[i].hasNaN()) && "NaN values in the input");
        //std::clog << p[i].size() << " ==? " << mArea.size() << std::endl;
        assert(p[i].size() == mArea.size());
        //std::clog << (p[i].array() * mArea.array()).sum() << " ==? " << 1 << std::endl;
        //std::clog << std::abs((p[i].array() * mArea.array()).sum() - 1.0) << std::endl;
        assert(std::abs((p[i].array() * mArea.array()).sum() - 1.0) < 3.2e-5);
    }

    q = p[0];

    // init
    const unsigned k = p.size();
    const unsigned n = mArea.size();
    const double opteps = mOpt.epsilon;//1e-50;
    //const double opteps = std::numeric_limits<double>::lowest();

    //alpha /= alpha.sum();
    for (int i = 0; i < int(k); ++i)
        p[i].array() += opteps;

    q = VectorXd::Constant(n, 1.);
    std::vector<VectorXd> v(k, VectorXd::Constant(n, 1.));
    std::vector<VectorXd> w(k, VectorXd::Constant(n, 1.));

    int iter = 0;
    for (; iter < nb_iter; ++iter)
    {
        // update w
// #pragma omp parallel for
        for (int i = 0; i < int(k); ++i)
        {
            VectorXd tmp = Kv(v[i]);
            clamp_n(tmp, opteps);
            w[i] = p[i].array() / tmp.array();
            assert(!w[i].hasNaN());
        }

        //std::clog << "compute d" << std::endl;
        // compute auxiliar d
        std::vector<VectorXd> d(k);
        for (int i = 0; i < int(k); ++i)
        {
            d[i] = v[i].array() * Kv(w[i]).array();
            clamp_n(d[i], opteps);
            assert(!d[i].hasNaN());
        }

        //std::clog << "compute q" << std::endl;
        // update barycenter q
        VectorXd prev = q;
        q = VectorXd::Zero(n);
        assert(!alpha.hasNaN());
        for (int i = 0; i < int(k); ++i)
        {
            //std::cout << "i: " << i << "/" << k << std::endl;
            //std::cout << "d: " << d[i] << std::endl;
            q = q.array() + alpha[i] * d[i].array().log();
            assert(!q.hasNaN());
        }

        q = q.array().exp();
        assert(!q.hasNaN());

        // // Optional
        // const double useSharpening = false;
        // if (useSharpening && iter > 0)
        //     sharpen(q);

        //std::clog << "update v" << std::endl;
        // update v
        for (int i = 0; i < int(k); ++i)
        {
            v[i] = v[i].array() * (q.array() / d[i].array());
        }

        // stop criterion
        const VectorXd diff = prev - q;//(prev - q)*mArea;
        VectorXd tmp = area.array() * diff.array();
        const double res =  diff.dot(tmp);
        // if (verbose > 0)
        //   std::cout << green << "Iter " << white << iter
        //             << ": " << res << std::endl;
        //const double tol = 1e-7;
        assert(!q.hasNaN());
        if (iter > 1 && res < mOpt.tolerance) break;
    }

    return true;
}
