#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>

#include <algorithm>

#include "MeshUtils.h"

#define NBSHAPE 2

using Eigen::VectorXd;

// in order to bypass --std=c++1z
// with C++17 just replace utils by std
namespace utils
{
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

inline
double dot_product(const VectorXd& area, const VectorXd& x, const VectorXd& y)
{
    VectorXd tmp = area.array() * y.array();
    return x.dot(tmp);
}

template<int N> // Adapt N to state the dimension (2d or 3d..)
struct GridConv {
    int _width = N;
    int _height = N;
    int _depth = N;
    int _niter = 100;

    VectorXd _area;
    VectorXd _H;

    GridConv() = default;
    GridConv(const VectorXd h, Eigen::RowVector3i& res)
        : _width(res(1)), _height(res(2)), _depth(res(0)), _H(h) {
        //std::cout << "GRIDCONV : " << res << std::endl;
    }

    double& get(VectorXd& p, int xi, int yi, int zi)
    {
        // array3D[i][j][k]
        // int depth = n, width = n;
        return p[xi + _depth*(yi + _width*zi)];
    };

    double dotProduct(const VectorXd& areaW, const VectorXd& x, const VectorXd& y) const {
        VectorXd tmp = areaW.array() * y.array();
        return x.dot(tmp);
    }

    VectorXd imfilter(VectorXd I, const VectorXd& h, int dim) {
        int w = _width;
        VectorXd vres = I;
        // sort of 1d convolution

        const auto idx = [&](int i) {
            int lo = 0, hi = w - 1;
            return utils::clamp(i, lo, hi);
        };

        int DIM = h.size() / 2;
        //std::cout << "DIM: " << DIM
        //          << " h: " << h.size() << std::endl;

        for (int zi = 0; zi < _height; zi++) {
            for (int yi = 0; yi < _width; yi++) {
                for (int xi = 0; xi < _depth; xi++) {
                    double res = 0.;
                    // std::clog << "[";
                    for (int p = 0; p < h.size(); ++p) {
                        //std::clog<<h(p)<< "*"<<get(test, k, zi, idx(i+p-DIM), ntest)<<"+";
                        //res += h(p) * get(test, k, zi, idx(i+p-DIM), ntest);
                        //res += h(p) * get(test, k, idx(zi+p-DIM), i, ntest);
                        if (dim == 1)
                            res += h(p) * get(I, xi, yi, idx(zi + p - DIM));
                        else if (dim == 2)
                            res += h(p) * get(I, xi, idx(yi + p - DIM), zi);
                        else
                            res += h(p) * get(I, idx(xi + p - DIM), yi, zi);
                    }
                    get(vres, xi, yi, zi) = res;
                    // std::clog << res << " ";
                }
                // std::clog << std::endl;
            }
            // std::clog << "\n next layer " << std::endl;
        }
        return vres;
    }

    VectorXd Kv(VectorXd& p, VectorXd& H) {
        // apply a gaussian filter in each dimension
        // p is the N*N*N voxelization in one column (i.e size == [N*N*N, 1])
        //VectorXd ret = p;
        p = p.array() * _area.array();
        return imfilter(imfilter(imfilter(p, H, 1), H, 2), H, 3);
    };

    VectorXd Kv(VectorXd& p) {
        return Kv(p, _H);
    };
#define SQ(X) ((X)*(X))
    VectorXd convoWassersteinBarycenter(std::vector<ScalarField> p,
                                        VectorXd& mArea,
                                        VectorXd& alpha,
                                        int num_iter, int nb_iter) {
        //double ratio = double(num_iter) / double(nb_iter);
        //double rc0 = 0.15*std::min(_width, std::min(_height, _depth));
        //double cy0 = _width / 2.;//+ ratio*rc0;
        //double cz0 = _height / 2.; //- ratio*rc0;
        //double cx0 = _depth / 2.;

        //for (int zi = 0; zi < _height; zi++) {
        //    for (int yi = 0; yi < _width; yi++) {
        //        for (int xi = 0; xi < _depth; xi++) {
        //            double dist = std::sqrt(SQ(xi-cx0) + SQ(yi-cy0) + SQ(zi-cz0));
        //            double cval = (rc0 - dist);
        //            get(mArea, xi, yi, zi) = cval < 0.5? 0.5: cval;
        //            //2. + 0.03*std::sin(0.01*num_iter+xi + yi + zi);
        //        }
        //    }
        //}

        //mArea /= mArea.maxCoeff();

        for (int k = 0; k < p.size(); ++k) {
            const double sumv = (p[k].values().array() * mArea.array()).sum();
            p[k]._values = p[k]._values.array() / sumv;
        }
        auto& areaWeights = mArea;
        this->_area = mArea;

        // sanity check
        assert(p.size() > 0);
        assert((int)p.size() == alpha.size());
        //for (int i = 0; i < alpha.size(); ++i)
        //    std::cout << alpha(i) << " ; ";
        //std::cout << "\n<< alpha" << std::endl;

        auto firstres = p[0]._gridsf._res;
        for (int i = 1; i < p.size(); ++i) {
            assert(firstres == p[i]._gridsf._res && "grid res need to be all equals");
            //std::cout << p[i]._gridsf._res << std::endl;
        }

        assert(mArea.size() > 0);
        for (unsigned i = 0; i < p.size(); ++i)
        {
            assert(!(p[i].values().hasNaN()) && "NaN values in the input");
            //std::clog << p[i].values().size() << " ==? " << mArea.size() << std::endl;
            assert(p[i].values().size() == mArea.size());
            //std::clog << (p[i].values().array() * mArea.array()).sum() << " ==? " << 1 << std::endl;
            //std::clog << std::abs((p[i].values().array() * mArea.array()).sum() - 1.0) << std::endl;
            assert(std::abs((p[i].values().array() * mArea.array()).sum() - 1.0) < 3.2e-5);
        }

        VectorXd q = p[0].values();

        // init
        const unsigned k = p.size();
        const unsigned n = mArea.size();
        const double opteps = mOpt.epsilon;//1e-50;
        //const double opteps = std::numeric_limits<double>::lowest();

        //alpha /= alpha.sum();
        for (int i = 0; i < int(k); ++i)
            p[i].values().array() += opteps;

        q = VectorXd::Constant(n, 1.);
        std::vector<VectorXd> v(k, VectorXd::Constant(n, 1.));
        std::vector<VectorXd> w(k, VectorXd::Constant(n, 1.));

        int iter = 0;
        for (; iter < _niter; ++iter)
        {
            // update w
#pragma omp parallel for
            for (int i = 0; i < int(k); ++i)
            {
                VectorXd tmp = Kv(v[i]);
                clamp_n(tmp, opteps);
                w[i] = p[i].values().array() / tmp.array(); // FIXME p[i] or q ?
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

            // Optional
            const double useSharpening = false;
            if (useSharpening && iter > 0)
                sharpen(q);

            //std::clog << "update v" << std::endl;
            // update v
            for (int i = 0; i < int(k); ++i)
            {
                v[i] = v[i].array() * (q.array() / d[i].array());
            }

            // stop criterion
            const VectorXd diff = prev - q;//(prev - q)*mArea;
            const double res = dotProduct(mArea, diff, diff);
            // if (verbose > 0)
            //   std::cout << green << "Iter " << white << iter
            //             << ": " << res << std::endl;
            //const double tol = 1e-7;
            assert(!q.hasNaN());
            if (iter > 1 && res < mOpt.tolerance) break;
        }

        return q;
    }

    struct {
        double upperEntropy = 0.8;
        double tolerance = 1e-7;
        double diffIters = 10;
        double maxIters = 1000;
        double epsilon = 1e-50;
        double gamma = 0.0;
    } mOpt;

    void sharpen(VectorXd& q) const
    {
        const double tol = mOpt.tolerance;

        // q has low entropy
        const double maginalEntropy = computeMarginalEntropy(q, 1.0);
        const double val = maginalEntropy - mOpt.upperEntropy;
        std::cout << "\n\n \t\t[maginalEntropy] " << maginalEntropy << std::endl;
        if (std::abs(val) < tol) return;

        // binary search for 'attempts' times
        double alpha0 = 0.8;
        double alpha1 = 1.2;
        unsigned attempt = 100;
        const double beta = binarySearch(q, alpha0, alpha1, attempt, tol);
        std::cout << "\n\n \t\t[beta] " << beta << " [attempt] " << attempt << std::endl;
        if (attempt == 0) return; // binary search failed
        q = q.array().pow(beta);  // finally, sharpening
    }

    double binarySearch(const VectorXd& q, double x0, double x1, unsigned& attempt, double tol) const
    {
        if (attempt == 0) return 1.0; // abort
        attempt--;

        double f0 = computeMarginalEntropy(q, x0) - mOpt.upperEntropy;
        if (std::abs(f0) < tol) return x0; // x0 is root

        double f1 = computeMarginalEntropy(q, x1) - mOpt.upperEntropy;
        if (std::abs(f1) < tol) return x1; // x1 is root

        if (std::abs(f1 - f0) < tol) { attempt = 0; return 1.0; } // abort

        double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        return binarySearch(q, x1, x2, attempt, tol);
    }

    double computeMarginalEntropy(const VectorXd& q, double beta) const
    {
        // - \int_M q(x)^beta*log(q(x)^beta) dx
        VectorXd l = beta * q.array().log();
        VectorXd e = l.array().exp();
        return -dotProduct(_area, e, l);
    }
};

struct WeightProvider {
    std::vector<VectorXd> mat_poids; // one VectorXd per step [step][shape]
    int nbstep   = 0;
    int nbshapes = 0;

    WeightProvider() = default;
    VectorXd& get_step(int i) {
        return mat_poids[i];
    }

    void assign_2danim(std::vector<double>& vecw)
    {
        assert(vecw.size() == nbstep && mat_poids.size() == nbstep
               && "ill-formed call to set the weight provider");

        // get weights values for the animation
        std::vector<double> wval(nbstep);
        for (int istep = 0; istep < nbstep; ++istep) {
            double tot = 0.;
            for (int k = 0; k < NBSHAPE; ++k)
                tot += mat_poids[istep](k);

            wval[istep] = 1. - tot;
        }

        // rescale values
        //
        const double valmin = *std::min_element(vecw.cbegin(), vecw.cend());
        const double valmax = *std::max_element(vecw.cbegin(), vecw.cend());

        const utils::interval inwin { valmin, valmax };
        const utils::interval shapewin { NBSHAPE, float(nbshapes - 1) };

        for(auto& dv: vecw)
            dv = utils::mapval(inwin, shapewin, dv);

        // y values are related to the choosen frame (keyshapes)
        // middle of the interpolation (2d animation transition)
        for (int istep = 0; istep < nbstep; ++istep)
        {
            const float poids = wval[istep];
            const int selected_shape_fl = std::floor(vecw[istep]);
            const int selected_shape_ce = std::ceil(vecw[istep]);
            //utils::clamp(vecw[istep], double(shapewin.l), double(shapewin.u));
            mat_poids[istep](selected_shape_fl) = poids;
            mat_poids[istep](selected_shape_ce) = poids;
        }
    }

    // if idxshape==-1 : assign to 2d animation columns
    void assign_shape(std::vector<double>& vecw, int idxshape)
    {
        assert(vecw.size() == nbstep && mat_poids.size() == nbstep
               && "ill-formed call to set the weight provider");

        for (int istep = 0; istep < nbstep; ++istep)
            mat_poids[istep](idxshape) = vecw[istep];
    }

    void set_dimension(int nstep, int nshape)
    {
        this->nbstep = nstep;
        this->nbshapes = nshape;
        mat_poids.resize(nbstep, VectorXd::Constant(nbshapes, 0.));
    }

    void normalize()
    {
        for (int istep = 0; istep < nbstep; ++istep) {
            double ctot = 0.;
            for (int ishape = 0; ishape < nbshapes; ++ishape)
                ctot += mat_poids[istep](ishape);
            std::clog << "step#" << istep << "\t" << std::flush;
            assert(ctot > 0. && "can't be zero");
            const double invtot = 1. / ctot;
            for (int ishape = 0; ishape < nbshapes; ++ishape)
                mat_poids[istep](ishape) *= invtot;

            for (int ishape = 0; ishape < nbshapes; ++ishape) {
                std::cout << "[" << mat_poids[istep](ishape) << "] ";
            }
            std::cout << std::endl;
        }
    }

    void print()
    {
        for (int istep = 0; istep < nbstep; ++istep) {
            std::cout << " [" << istep << "] ";
            for (int ishape = 0; ishape < nbshapes; ++ishape) {
                std::cout << " " << mat_poids[istep](ishape) << "  ;  ";
            }
            std::cout << std::endl;
        }
    }

    //void compute_bybernstein() {
    //VectorXd poids = VectorXd::Constant(sf.size(), 1.);
    //const double bu = 1.0 - ct;
    //const double bv = ct;
    // need to generated weights automatically by the number of input shapes

    // can't use metaprog, nbr of shapes can be dynamic in maya,
    // or maybe just generate them with a max nbshapes
    // but the computation time for bernstein might be peanuts compared to ot
    //poids[0] = bernstein<NBSHAPE - 1, 0>(ct);//1.-bu-bv;
    //poids[1] = bernstein<NBSHAPE - 1, 1>(ct);//bu;
    //poids[2] = bernstein<NBSHAPE - 1, 2>(ct);//bv;
    //}
};

struct OTMeshInterface {
    std::vector<IGLMesh> inmeshes;
    std::vector<ScalarField> sf;
    std::vector<MCWrapper> out3DSequence;
    WeightProvider weightProvider;
    int nbSample = 0;
    int gridsize = 40; // in order to have a fast 1st preview

    OTMeshInterface() = default;
    OTMeshInterface(std::vector<IGLMesh>&& in)
        : inmeshes(std::move(in))
    {
        preprocess();
    }

    void reset()
    {
        inmeshes.clear();
        sf.clear();
        out3DSequence.clear();
        weightProvider.mat_poids.clear();
    }

    void preprocess()
    {
        // preprocess the input meshes to have a bbox of similar size
        // find the biggest bbox
        double cmax = 0.;
        for (auto& m : inmeshes) {
            const RowVector3d diagvec = m._v.colwise().maxCoeff() - m._v.colwise().minCoeff();
            double currval = diagvec.norm();
            if (cmax < currval)
                cmax = currval;
        }
        // scale
        for (auto& m : inmeshes) {
            const RowVector3d diagvec = m._v.colwise().maxCoeff() - m._v.colwise().minCoeff();
            double scale_value = cmax / diagvec.norm();
            m._v *= scale_value;
        }
    }

    void toscalarfield()
    {
        const int s = gridsize;
        //sf.clear();
        sf.reserve(inmeshes.size());
        for (auto& imesh : inmeshes) {
            sf.emplace_back(imesh, s);
        }
        //sf.emplace_back(inmeshes[0], s);
        //sf.emplace_back(inmeshes[1], s);

        //// find the largest grid
        int maxnb = 0, imax = 0;
        for (int i = 0; i < sf.size(); ++i) {
            auto cval = sf[0]._gridsf.nbelmt();
            if (maxnb < cval) {
                maxnb = cval;
                imax = i;
            }
        }

        // force the same grid size for each input (the largest one for everyone)
        Eigen::RowVector3i maxres = sf[imax]._gridsf._res;
        int maxelmt = std::max(maxres(2), std::max(maxres(0), maxres(1)));
        maxres(0) = maxres(1) = maxres(2) = maxelmt;

        for (int i = 0; i < sf.size(); ++i) {
            sf[i]._gridsf._res = maxres;
            sf[i]._gridsf.compute_grid();

            // small hack to avoid overlap with bbox
            const RowVector3d diagvec = inmeshes[i]._v.colwise().maxCoeff() - inmeshes[i]._v.colwise().minCoeff();

            //inmeshes[i]._v *= 1.3; // downscale
                                    //inmeshes[i]._v.array() += 0.01; // move it a bit away from the origin

            sf[i].compute_from_mesh(inmeshes[i]);
            //sf[i]._gridsf._vmin = inmeshes[i]._v.colwise().minCoeff();
            //sf[i]._gridsf._vmax = inmeshes[i]._v.colwise().maxCoeff();
        }

        //for (auto& currsf : sf) {
        //    currsf._gridsf._res = maxres;
        //    currsf._gridsf.compute();
        //    currsf.compute_from_mesh();
        //}

        // sanity check
        auto& firstsf = sf[0];
        for (auto& currsf : sf) {
            assert(firstsf._gridsf == currsf._gridsf && "different grid");
        }
    }

    void update_weight_dimension() {
        const int nbshapes = sf.size();
        const int nbstep = nbshapes + nbSample;
        weightProvider.set_dimension(nbstep, nbshapes);
    }

    void compute_step_keyframe()
    {
        const int initsf = 0;
        const int nbshapes = sf.size();
        const int nbstep = nbshapes + nbSample;

        const float ratio_in_key = 0.9f; // 0<- close to input ; 1<- close to keyshapes
                                         //weightProvider.compute_byratio(ratio_in_key);
                                         //weightProvider.compute_byqueue();

        out3DSequence.resize(nbstep);
        auto rv = sf[0]._gridsf._res;
        const double mu = rv(0) / (0.33*double(gridsize));
        const int nbvoxel = sf[0]._gridsf.nbelmt();
        VectorXd H = utils::gen_gaussian(mu, mu*(0.166*double(gridsize)));
        //std::clog << H.size() << "\n" << H << std::endl;

        //VectorXd areaW = VectorXd::Constant(nbvoxel, 1.0);
        //for (int i = 0; i < areaW.size(); i++) {
        //    areaW(i) = sin(std::sqrt(1.+double(i)));
        //}
        GridConv<-1> grid = { H, sf[0]._gridsf._res };

        ScalarField& init0 = sf[initsf];
        // not the same for each input
        const auto pmin0 = init0._gridsf._vmin;
        const auto pmax0 = init0._gridsf._vmax;
        std::vector<std::pair<RowVector3d, RowVector3d>> allvec;
        allvec.reserve(sf.size());

        for (int i = 0; i < sf.size(); ++i) {
            allvec.emplace_back(std::make_pair(sf[i]._gridsf._vmin - pmin0,
                                sf[i]._gridsf._vmax - pmax0));
        }

#pragma omp parallel for
        for (int i = 0; i < nbstep; i++) {
            auto& poids = weightProvider.get_step(i);

            ScalarField cpy_inter_sf = init0;
            //assert(poids.size() == allvec.size() && "should be the same size (==number of input shapes)");
            for (int j = 0; j < poids.size(); j++) {
                cpy_inter_sf._gridsf._vmin += poids[j] * allvec[j].first;
                cpy_inter_sf._gridsf._vmax += poids[j] * allvec[j].second;
            }

            cpy_inter_sf._gridsf.compute_grid();
            //std::cout << "\n alpha: " << poids[0] << ";" << poids[1] << std::endl;
            //std::cout << ">> start core computation" << std::endl;

            const int nbvoxel = sf[0]._gridsf.nbelmt();
            VectorXd areaW = VectorXd::Constant(nbvoxel, 1.0);

            SFWeightVec inter_val = grid.convoWassersteinBarycenter(sf, areaW, poids, i, nbstep);
            //std::cout << "<< end core computation" << std::endl;

            //assert(cpy_inter_sf._values.size() == inter_val.size()
            //       && "dimension problem, hint: grid size");
            //assert(!inter_val.hasNaN() && "barycenter computation problem");
            //assert(inter_val.sum() != 0 && "empty barycenter ?!");
            //vec2file("vecval/intersf.dat", inter_sf._values);

            //////
            const bool infer_isovalue = false;
            const bool normalize_q = false;
            auto& _values = cpy_inter_sf._values;
            auto& _isovalue = cpy_inter_sf._isovalue;

            _values = inter_val;
            if (normalize_q) {
                //_values /= _values.maxCoeff();
                _values /= _values.sum();
                //utils::rescale(_values, {0., 1.});
            }

            const double sumval = std::abs(1. - _values.sum());

            //assert( < grid.mOpt.tolerance
            //       && "barycenter values need to be normalized");

            if (infer_isovalue) {
                //_isovalue = 0.5*(_values.maxCoeff() - _values.minCoeff());
                _isovalue = _values(_values.size() / 2);
            }
            else {
                _isovalue = 0;
                for (int i = 0; i < poids.size(); i++) {
                    //const auto& sfv = sf[i].values();
                    //const double cisov = utils::mapval({ sfv.minCoeff(), sfv.maxCoeff() },
                    //                                   { 0., 1. },
                    //                                   sf[i]._isovalue);
                    const double cisov = sf[i]._isovalue;
                    _isovalue += poids[i] * cisov;
                }
            }

            out3DSequence[i]._insf = cpy_inter_sf;
            out3DSequence[i].compute();
#pragma omp critical
            {
                std::clog << "\t[step " << i << "] done" << std::endl;
            }
        }

        // move all to the origin
        //for (auto& mc : out3DSequence) {
        //    auto& cmesh = mc._out;

        //    const auto vmin = cmesh._v.colwise().minCoeff();
        //    const auto vmax = cmesh._v.colwise().maxCoeff();
        //    const auto vvec = vmax - vmin;

        //    // place the mesh at the origin
        //    cmesh._v.array() = cmesh._v.rowwise() - (vmin + 0.5*vvec);
        //}
    }

    void compute_step(int maxiter, int initsf)
    {
        out3DSequence.resize(maxiter);
        auto rv = sf[0]._gridsf._res;
        const double mu = rv(0) / (gridsize - 20);
        const int nbvoxel = sf[0]._gridsf.nbelmt();
        VectorXd H = utils::gen_gaussian(mu, mu*(gridsize - 10.));
        //std::clog << H.size() << "\n" << H << std::endl;

        //VectorXd areaW = VectorXd::Constant(nbvoxel, 1.0);
        //for (int i = 0; i < areaW.size(); i++) {
        //    areaW(i) = sin(std::sqrt(1.+double(i)));
        //}

        GridConv<-1> grid = { H, sf[0]._gridsf._res };

        ScalarField& init0 = sf[initsf];
        // not the same for each input
        const auto pmin0 = init0._gridsf._vmin;
        const auto pmax0 = init0._gridsf._vmax;
        std::vector<std::pair<RowVector3d, RowVector3d>> allvec;
        allvec.reserve(sf.size());

        for (int i = 0; i < sf.size(); ++i) {
            allvec.emplace_back(std::make_pair(sf[i]._gridsf._vmin - pmin0,
                                sf[i]._gridsf._vmax - pmax0));
        }
#pragma omp parallel for
        for (int i = 0; i < maxiter; i++) {
            const double ct = double(i) / double(maxiter);
            VectorXd poids = VectorXd::Constant(sf.size(), 1.);
            //const double bu = 1.0 - ct;
            //const double bv = ct;
            // need to generated weights automatically by the number of input shapes

            // can't use metaprog, nbr of shapes can be dynamic in maya, or maybe just generate them with a max nbshapes
            // but the computation time for bernstein might be peanuts compared to ot
            poids[0] = bernstein<NBSHAPE - 1, 0>(ct);//1.-bu-bv;
            poids[1] = bernstein<NBSHAPE - 1, 1>(ct);//bu;
            //poids[2] = bernstein<NBSHAPE - 1, 2>(ct);//bv;

            ScalarField cpy_inter_sf = init0;

            assert(poids.size() == allvec.size() && "should be the same size (==number of input shapes)");
            for (int i = 0; i < poids.size(); i++) {
                cpy_inter_sf._gridsf._vmin += poids[i] * allvec[i].first;
                cpy_inter_sf._gridsf._vmax += poids[i] * allvec[i].second;
            }
            cpy_inter_sf._gridsf.compute_grid();
            //std::cout << "\n alpha: " << poids[0] << ";" << poids[1] << std::endl;
            //std::cout << ">> start core computation" << std::endl;

            const int nbvoxel = sf[0]._gridsf.nbelmt();
            VectorXd areaW = VectorXd::Constant(nbvoxel, 1.0);

            SFWeightVec inter_val = grid.convoWassersteinBarycenter(sf, areaW, poids, i, maxiter);
            //std::cout << "<< end core computation" << std::endl;

            //assert(cpy_inter_sf._values.size() == inter_val.size()
            //       && "dimension problem, hint: grid size");
            //assert(!inter_val.hasNaN() && "barycenter computation problem");
            //assert(inter_val.sum() != 0 && "empty barycenter ?!");
            //vec2file("vecval/intersf.dat", inter_sf._values);

            //////
            const bool infer_isovalue = false;
            const bool normalize_q = false;
            auto& _values = cpy_inter_sf._values;
            auto& _isovalue = cpy_inter_sf._isovalue;

            _values = inter_val;
            if (normalize_q) {
                //_values /= _values.maxCoeff();
                _values /= _values.sum();
                //utils::rescale(_values, {0., 1.});
            }

            const double sumval = std::abs(1. - _values.sum());
            std::cout << "\n\t sumval= " << sumval << std::endl;

            //assert( < grid.mOpt.tolerance
            //       && "barycenter values need to be normalized");

            if (infer_isovalue) {
                //_isovalue = 0.5*(_values.maxCoeff() - _values.minCoeff());
                _isovalue = _values(_values.size()/2);
            }
            else {
                _isovalue = 0;
                for (int i = 0; i < poids.size(); i++) {
                    //const auto& sfv = sf[i].values();
                    //const double cisov = utils::mapval({ sfv.minCoeff(), sfv.maxCoeff() },
                    //                                   { 0., 1. },
                    //                                   sf[i]._isovalue);
                    const double cisov = sf[i]._isovalue;
                    _isovalue += poids[i] * cisov;
                }
            }

            out3DSequence[i]._insf = cpy_inter_sf;
            out3DSequence[i].compute();

            std::clog << "\t\t\t[step " << i << "] done" << std::endl;
        }
    }
};
