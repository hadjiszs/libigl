#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <array>
#include <type_traits>
#include <thread>
#include <igl/opengl/glfw/Viewer.h>

#include <igl/ConvWassersteinCore.h>

#include "tutorial_shared_path.h"

using namespace utils;
using namespace Eigen;

int main(int argc, char** argv)
{
    const bool use_sharp = false;

    // Edges of the bounding box
    Eigen::MatrixXi E_box(12, 2);
    E_box <<
        0, 1,
        1, 2,
        2, 3,
        3, 0,
        4, 5,
        5, 6,
        6, 7,
        7, 4,
        0, 4,
        1, 5,
        2, 6,
        7, 3;
    // Find the bounding box
    const auto getvbox_w = [&](auto vmin, auto vmax) {
        auto& m = vmin;
        auto& M = vmax;

        // Corners of the bounding box
        Eigen::MatrixXd V_box(8, 3);
        V_box <<
            m(0), m(1), m(2),
            M(0), m(1), m(2),
            M(0), M(1), m(2),
            m(0), M(1), m(2),
            m(0), m(1), M(2),
            M(0), m(1), M(2),
            M(0), M(1), M(2),
            m(0), M(1), M(2);
        return V_box;
    };

    const auto getvbox = [&](const Eigen::VectorXd& V) {
        return getvbox_w(V.colwise().minCoeff(), V.colwise().maxCoeff());
    };

    const auto plotbbox = [&](igl::opengl::glfw::Viewer& viewer, const MCWrapper& mc,
                              const Eigen::RowVector3d& color) {
        auto V_box = getvbox_w(mc.vmin(), mc.vmax());
        viewer.data().add_points(V_box, color);

        // Plot the edges of the bounding box
        for (unsigned i = 0; i < E_box.rows(); ++i)
            viewer.data().add_edges
            (
            V_box.row(E_box(i, 0)),
            V_box.row(E_box(i, 1)),
            color
            );
    };

    //
    // PREPROCESS MESH
    //
    std::cout << "Computing scalar field ..." << std::endl;
    std::vector<IGLMesh> inmeshes; inmeshes.reserve(NBSHAPE);

    //inmeshes.emplace_back("meshes/hand1.off");
    inmeshes.emplace_back(TUTORIAL_SHARED_PATH "/torus.off");
    inmeshes.emplace_back(TUTORIAL_SHARED_PATH "/bull.off");
    //inmeshes.emplace_back("meshes/sphere_102.off");

    // I prefer to construct the Eigen representation of input meshes vector outside
    // in order to keep the OTMeshInterface agnostic from where the input is coming from
    OTMeshInterface otcore{std::move(inmeshes)};
    otcore.toscalarfield();

    // PRE-COMPUTE EACH STEP (LUT for displacement interpolation)
    const int maxiter = 5;
    //intervec.resize(maxiter);
    otcore.compute_step(maxiter, 0);
    auto& intervec = otcore.out3DSequence;

    //
    // VIEW
    //
    igl::opengl::glfw::Viewer viewer;
    //MCWrapper& currmc = inter_mc;
    //MCWrapper& currmc = mc[1];
    const double step   = 0.05;
    const double stepiv = 0.01;

    // for bbox of view of input shapes only
    //std::vector<MCWrapper> mc; mc.reserve(NBSHAPE);
    //for (auto& scalarfield : sf) {

    //    mc.emplace_back(scalarfield);
    //    mc.back().compute();
    //}

    const auto updatebbox = [&](MCWrapper& current) {
        //for (auto& minit : mc)
        //    plotbbox(viewer, minit,);

        auto color = Eigen::RowVector3d(0., 1., 0.2);
        for(auto& m: otcore.inmeshes)
        {
            const RowVector3d vmin = m._v.colwise().minCoeff();
            const RowVector3d vmax = m._v.colwise().maxCoeff();
            auto V_box = getvbox_w(vmin, vmax);
            viewer.data().add_points(V_box, color);

            // Plot the edges of the bounding box
            for (unsigned i = 0; i < E_box.rows(); ++i)
                viewer.data().add_edges
                (
                V_box.row(E_box(i, 0)),
                V_box.row(E_box(i, 1)),
                color
                );
        }

        plotbbox(viewer, current, Eigen::RowVector3d(0, 0.2, 1));
    };

    const int curr = 0;
    const auto& select_mesh = [](igl::opengl::glfw::Viewer & viewer,
                                 const VertexVec& v, const FaceVec& f) {
        viewer.data().clear();
        std::cout << "V: " << v.size() << "\t F: " << f.size() << std::endl;
        viewer.data().set_mesh(v, f);
    };

    bool reverseit = false;
    // const double isovstep = 0.001;
    int it = 0;
    //const double isov = 0.3;
    //const bool autoiso = true;

    select_mesh(viewer, intervec[it].get_out_v(), intervec[it].get_out_f());
    updatebbox(intervec[it]);

    viewer.callback_key_down =
        [&](igl::opengl::glfw::Viewer & viewer, unsigned char key, int mod)->bool
    {
        switch (key)
        {
        default:
            return false;
        //case '1':
        //    //select_mesh(viewer, mc[0].get_out_v(), mc[0].get_out_f());
        //    alpha[0] += step;
        //    alpha[1] = 1. - alpha[0];
        //    if (alpha[0] < 0. || alpha[0] > 1.
        //        || alpha[1] < 0. || alpha[1] > 1.) {
        //        alpha[0] = 0.;
        //        alpha[1] = 1.;
        //    }
        //    computewasser();
        //    inter_mc.compute(0.3);
        //    goto updateoutmesh;
        //case '2':
        //    alpha[1] += step;
        //    alpha[0] = 1. - alpha[1];
        //    if (alpha[0] < 0. || alpha[0] > 1.
        //        || alpha[1] < 0. || alpha[1] > 1.) {
        //        alpha[0] = 0.;
        //        alpha[1] = 1.;
        //    }
        //    computewasser();
        //    inter_mc.compute(0.3);
        //    goto updateoutmesh;
        //case '3':
        //{
        //    const double iv = currmc._isovalue;
        //    std::cout << "increase isovalue from " << iv << " to " << iv + stepiv << std::endl;
        //    currmc.compute(iv + stepiv);
        //}
        //    goto updateoutmesh;
        //case '4':
        //{
        //    const double iv = currmc._isovalue;
        //    std::cout << "decrease isovalue from " << iv << " to " << iv - stepiv << std::endl;
        //    currmc.compute(iv - stepiv);
        //}
        //    goto updateoutmesh;
        //case '5':
        //    //select_mesh(viewer, mc[0].get_out_v(), mc[0].get_out_f());
        //    isov -= isovstep;
        //    viewer.data().clear();
        //    intervec[it].compute(autoiso? -1.: isov);
        //    viewer.data().set_mesh(intervec[it].get_out_v(), intervec[it].get_out_f());
        //    updatebbox(intervec[it]);
        //    break;
        //case '6':
        //    //select_mesh(viewer, mc[1].get_out_v(), mc[1].get_out_f());
        //    isov += isovstep;
        //    viewer.data().clear();
        //    intervec[it].compute(autoiso ? -1. : isov);
        //    viewer.data().set_mesh(intervec[it].get_out_v(), intervec[it].get_out_f());
        //    updatebbox(intervec[it]);
        //    break;
        case '7':
            if (it == 0) reverseit = false;
            else if (it == intervec.size()) reverseit = true;
            if (reverseit) it--;

            viewer.data().clear();
            //intervec[it].compute(autoiso ? -1. : isov);
            viewer.data().set_mesh(intervec[it].get_out_v(), intervec[it].get_out_f());
            updatebbox(intervec[it]);
            if (!reverseit) it++;
            break;
        //case '8':
        //    select_mesh(viewer, mc[0].get_out_v(), mc[0].get_out_f());
        //    updatebbox(intervec[it]);
        //    break;
        //case '9':
        //    select_mesh(viewer, mc[1].get_out_v(), mc[1].get_out_f());
        //    updatebbox(intervec[it]);
        //    break;
        updateoutmesh:
            //select_mesh(viewer, currmc.get_out_v(), currmc.get_out_f());
            //updatebbox(currmc);
            break;
        }

        viewer.data().set_face_based(true);
        return true;
    };

    viewer.callback_post_draw = [&] (igl::opengl::glfw::Viewer & viewer) -> bool {
        if (it == 0) reverseit = false;
        else if (it == intervec.size()) reverseit = true;
        if (reverseit) it--;

        viewer.data().clear();
        //intervec[it].compute(autoiso ? -1. : isov);
        viewer.data().set_mesh(intervec[it].get_out_v(), intervec[it].get_out_f());
        updatebbox(intervec[it]);
        if (!reverseit) it++;

        return true;
    };

    viewer.core_list[viewer.core_index(0)].animation_max_fps = 15;
    viewer.core_list[viewer.core_index(0)].is_animating = true;
    viewer.launch();

    return 0;
}
