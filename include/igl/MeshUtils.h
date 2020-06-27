#pragma once

#include <type_traits>

#include <igl/pathinfo.h>
#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <igl/readMESH.h>
#include <igl/faces_first.h>
#include <igl/readTGF.h>
#include <igl/launch_medit.h>
#include <igl/boundary_conditions.h>
#include <igl/writeOBJ.h>
#include <igl/writeOFF.h>
#include <igl/writeDMAT.h>
#include <igl/writeMESH.h>
#include <igl/normalize_row_sums.h>
#include <igl/bbw.h>
#include <igl/cotmatrix.h>
#include <igl/copyleft/marching_cubes.h>
#include <igl/signed_distance.h>

#include <fstream>


// simple couple struct to have the code a bit clearer
template<typename MESHTYPE>
struct MeshCouple {
    MESHTYPE src;  // mesh of the _initial_ state for the interpolation
    MESHTYPE dest; // mesh of the _final_ state for the interpolation
};

using namespace Eigen;

using VertexVec = Eigen::MatrixXd;
using FaceVec   = Eigen::MatrixXi;
using SFWeightVec = Eigen::VectorXd;

// Template functions to estimate the binominal coefficient
template<uint8_t n, uint8_t k>
struct binomial {
    static constexpr int value = (binomial<n - 1, k - 1>::value + binomial<n - 1, k>::value);
};

template<>
struct binomial<0, 0> {
    static constexpr int value = 1;
};

template<uint8_t n>
struct binomial<n, 0> {
    static constexpr int value = 1;
};

template<uint8_t n>
struct binomial<n, n> {
    static constexpr int value = 1;
};

// Template bernstein polynomial
template <uint8_t n, uint8_t k>
float bernstein(const float val) {
    constexpr float binom_coeff = binomial<n, k>::value;
    const float a = std::pow(val, k);
    const float b = std::pow(1 - val, n - k);
    return binom_coeff * a * b;
}


namespace utils
{
    const auto vec2file = [](const std::string& f, const VectorXd& v) {
        std::ofstream file(f);
        if (file.is_open())
        {
            file << v;
        }
    };

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

    const auto isequal = [](const VectorXd& lhs, const VectorXd& rhs) {
        bool ret = true;
        assert(lhs.size() == rhs.size() && "not the same size");

        int i = 0;
        const double f = 1.;
        //const double epsilon = 0.000000001; // OK for Kv comparison
        const double epsilon = 0.001; // OK for the gaussian comparison
        for (; ret && i < lhs.size(); ++i) {
            std::clog << "#" << i << " "
                << f*lhs(i) << " ==? " << f*rhs(i) << std::endl;
            ret = std::abs(f*lhs(i) - f*rhs(i)) < epsilon;
        }

        if (!ret)
            std::cerr << "Error in the iteration #" << i << std::endl;

        return ret;
    };

    const auto readcsv = [](const std::string& fn) {
        std::ifstream input(fn);
        std::clog << "\n\n" << fn << "\n\n" << std::endl;

        const int nblines = std::count(std::istreambuf_iterator<char>(input),
                                       std::istreambuf_iterator<char>(), '\n');
        std::cout << "NBLINES: " << nblines << std::endl;
        input.seekg(0);
        int i = 0;
        VectorXd ret = VectorXd::Constant(nblines, 1.);

        for (std::string line; getline(input, line); ++i)
        {
            ret(i) = std::stod(line);
        }

        return ret;
    };

    struct interval {
        double l, u;
        friend std::ostream& operator<<(std::ostream& o, const interval& i) {
            return o << " [ " << i.l << " ; " << i.u << " ] ";
        }
    };

    const auto rescale = [](VectorXd& v, interval id) {
        const double minv = v.minCoeff();
        const double maxv = v.maxCoeff();
        const double invlen = 1. / (maxv - minv);

        // FIXME: have a look regarding the perf of this, might be too expensive
        // do not know that much about eigen lib
        v.array() = (v.array() - minv) * invlen * (id.u - id.l);
        v.array() = v.array() + id.l;
    };

    // `ia` = source interval
    // `ib` = destination
    // `t` the value in `ia` that we want in `ib`
    const auto mapval = [](interval ia, interval ib, double t) {
        const double a = ia.l, b = ia.u;
        const double c = ib.l, d = ib.u;
        return c + (t - a)*((d - c) / (b - a));
    };
}

struct IGLMesh {
    std::string _fn;
    VertexVec _v;
    FaceVec   _f;

    IGLMesh() = default;
    IGLMesh(const IGLMesh&) = default;
    IGLMesh(const std::string& fn)
        : _fn(fn)
    {
        load_mesh_from_file(fn);
    }

    bool load_mesh_from_file(const std::string& mesh_filename)
    {
        using namespace std;
        using namespace igl;
        auto& V = _v;
        auto& F = _f;
        std::string dirname, basename, extension, filename;
        pathinfo(mesh_filename, dirname, basename, extension, filename);
        transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        bool success = false;
        if (extension == "obj")
        {
            success = readOBJ(mesh_filename, V, F);
        }
        else if (extension == "off")
        {
            success = readOFF(mesh_filename, V, F);
        }
        else
        {
            cerr << "Error: Unknown shape file format extension: ." << extension << endl;
            return false;
        }

        std::cout << "load done from " << mesh_filename << std::endl;
        return success;
    }

    // Read a surface mesh from a {.obj|.off|.mesh} files
    // Inputs:
    //   mesh_filename  path to {.obj|.off|.mesh} file
    // Outputs:
    //   V  #V by 3 list of mesh vertex positions
    //   F  #F by 3 list of triangle indices
    // Returns true only if successfuly able to read file


    bool export_mesh() {
        std::string dirname, basename, extension, filename;
        igl::pathinfo(_fn, dirname, basename, extension, filename);
        std::string outfile = dirname + "/out/out_" + filename + "." + extension;
        if (extension == "obj")
            igl::writeOBJ(outfile, _v, _f);
        else if (extension == "off")
            igl::writeOFF(outfile, _v, _f);

        std::cout << "export done to " << outfile << std::endl;
        return true;
    }
};

struct GridScalarField {
    Eigen::RowVector3i _res;
    Eigen::MatrixXd    _grid;
    RowVector3d _vmax;
    RowVector3d _vmin;

    GridScalarField() = default;
    GridScalarField(const GridScalarField&) = default;
    GridScalarField& operator=(const GridScalarField&) = default;

    GridScalarField(RowVector3d& vmin, RowVector3d& vmax, int s)
        : _vmin(vmin), _vmax(vmax) {
        compute_res(s);
    }

    GridScalarField(const Eigen::MatrixXd& V, int s)
    {
        _vmin = V.colwise().minCoeff();
        _vmax = V.colwise().maxCoeff();
        compute_res(s);
        //std::cout << "\nres: [ " << _res(0) << " ; " << _res(1) << " ; " << _res(2) << " ]" << std::endl;
    }

    void compute_res(int s) {
        _res = (s*((_vmax - _vmin) / (_vmax - _vmin).maxCoeff())).cast<int>();
    }

    friend bool operator==(const GridScalarField& a, const GridScalarField& b) {
        return b.nbelmt() == a.nbelmt()
            && b._res(0) == a._res(0)
            && b._res(1) == a._res(1)
            && b._res(2) == a._res(2)
            && b._grid.size() == a._grid.size();
    }

    int nbelmt() const {
        return _res(0)*_res(1)*_res(2);
    }

    void compute_grid() {
        double depth  = _res(0);
        double width  = _res(1);
        double height = _res(2);

        assert(nbelmt() != 0 && "empty grid");
        _grid = MatrixXd(nbelmt(), 3);
        //std::cout << "\nres: [ " << _res(0) << " ; " << _res(1) << " ; " << _res(2) << " ]" << std::endl;
        for (int zi = 0; zi < height; zi++)
        {
            const auto lerp = [&](const int di, const int d)->double
            { return _vmin(d) + (double)di / (double)(_res(d) - 1)*(_vmax(d) - _vmin(d)); };
            const double z = lerp(zi, 2);
            for (int yi = 0; yi < width; yi++)
            {
                const double y = lerp(yi, 1);
                for (int xi = 0; xi < depth; xi++)
                {
                    const double x = lerp(xi, 0);
                    _grid.row(xi + depth*(yi + width*zi)) = RowVector3d(x, y, z);
                }
            }
        }
    }
};

//#define DEFAULT_METHOD igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER

struct ScalarField {
    double epsilon = 0.0001;
    // method for sf computation (e.g. winding numbers)
    igl::SignedDistanceType _method = igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER;
    GridScalarField         _gridsf;
    SFWeightVec             _values; // all values of the scalar field in one column #(N*N*N)
    double                  _isovalue = 0.;

    ScalarField() = default;
    ScalarField(const ScalarField& sf) = default;
    ScalarField& operator=(const ScalarField&) = default;

    ScalarField(const IGLMesh& mesh, int s,
                const igl::SignedDistanceType method = igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER)
        : _method(method), _gridsf(mesh._v, s) // FIXME: define correctly `s`
    { }

    double& get(int xi, int yi, int zi)
    {
        return _values[xi + _gridsf._res(0)*(yi + _gridsf._res(1)*zi)];
    };

    SFWeightVec& values() { return _values; }

    void post_process() {
        // map the current interval to 0-1 and normalization
        utils::interval initial_range{ _values.minCoeff(), _values.maxCoeff() };
        utils::rescale(_values, {0., 1.});
        _values /= _values.sum();
        // compute the zero in the new interval
        _isovalue = utils::mapval(initial_range, { 0., _values.maxCoeff() }, 0.);
        assert(std::abs(1. - _values.sum()) < 0.01
               && "error during the scalar field post process");
    }

    // the scalar field is computed from this input mesh
    void compute_from_mesh(const IGLMesh& _inmesh, bool winding=true) {
        const igl::SignedDistanceType method = winding ? igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER :
            igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL;
        VectorXi I_; MatrixXd C_, N_; // FIXME: investigate if we need to use those
        igl::signed_distance(_gridsf._grid, _inmesh._v, _inmesh._f,
                             method, _values, I_, C_, N_);

        // log
        auto& outq = _values;
        //print();

        // sanity check
        assert(!outq.hasNaN()  && "signed distance computation might be ill-formed");
        assert(outq.sum() != 0 && "empty scalar field?!");
        post_process();
    }

    void print() {
        auto& outq = _values;
        std::cout << "min val: " << outq.minCoeff()
            << "\tmax val: " << outq.maxCoeff()
            << "\nmean: " << outq.mean()
            << "\tsum: " << outq.sum() << "\tisovalue: " << _isovalue
            << std::endl;
    }
};

struct MCWrapper { // marching cube wrapper
    ScalarField  _insf; // out mesh is computed from this scalar field
    IGLMesh      _out; // out mesh
    bool         _binit    = false; // whether you want a binary output (warning: lot of artifact but could be interesting for artist)
    MCWrapper() = default;
    MCWrapper(const ScalarField& sf)
        : _insf(sf)
    {
        const auto& sfval = _insf._values;
        assert((std::is_same<VectorXd, std::decay<decltype(sfval)>::type>::value)
                && "types coherency is ill-formed");
    }

    double compute_isovalue(bool assign=false) {
        const auto& sfval = _insf._values;
        double newval = (sfval.maxCoeff() - sfval.minCoeff()) / 2.;
        if (assign) _insf._isovalue = newval;
        return newval;
    }

    void compute(const double isov = -1.) {
        const auto& sfval = _insf._values;
        const auto& res = _insf._gridsf._res;
        const auto& grid = _insf._gridsf._grid;

        const double used_isovalue = isov != -1. ? isov : _insf._isovalue;

        VectorXd bin_sfval = sfval;
        if (_binit)
            std::for_each(bin_sfval.data(), bin_sfval.data() + bin_sfval.size(),
                          [u=used_isovalue](double& b) { b = (b > u ? 1 : (b < u ? -1 : u)); });

        //std::cout << ">> MARCHING CUBE" << std::endl;
        //_insf.print();
        //std::cout << "\tused isovalue : " << used_isovalue
        //    << "\tused resogrid : " << res
        //    << "\n>> end" << std::endl;
        assert(sfval.size() != 0 && "empty scalar field?");
        assert(_insf._gridsf.nbelmt() != 0 && "empty grid?");
        assert(!sfval.hasNaN() && "input scalar field is ill-formed");
        assert(sfval.minCoeff() < used_isovalue && used_isovalue < sfval.maxCoeff()
               && "input isovalue is degenerated (outside the scalar field range)");

        igl::copyleft::marching_cubes(_binit ? bin_sfval : sfval,
                                      grid, res(0), res(1), res(2), used_isovalue,
                                      _out._v, _out._f);

        assert(_out._v.size() != 0 && _out._f.size() != 0
               && "empty");
    }

    //const FaceVec&     get_in_f() const { return _insf._inmesh._f; }
    //const VertexVec&   get_in_v() const { return _insf._inmesh._v; }
    const FaceVec&     get_out_f() const { return _out._f; }
    const VertexVec&   get_out_v() const { return _out._v; }
    const SFWeightVec& get_in_sfvalues() const { return _insf._values; }
    const RowVector3d& vmin() const { return _insf._gridsf._vmin; }
    const RowVector3d& vmax() const { return _insf._gridsf._vmax; }
};
