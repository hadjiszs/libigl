#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <array>
#include <type_traits>
#include <thread>
#include <igl/opengl/glfw/Viewer.h>

#include <igl/optimal_transport_interpolation.h>

#include "tutorial_shared_path.h"

using namespace Eigen;

int main(int argc, char** argv)
{
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
    otcore.compute_step(maxiter, 0);
    auto& intervec = otcore.out3DSequence;

    //
    // VIEW
    //
    igl::opengl::glfw::Viewer viewer;
    const double step   = 0.05;
    const double stepiv = 0.01;

    const int curr = 0;
    const auto& select_mesh = [](igl::opengl::glfw::Viewer & viewer,
                                 const VertexVec& v, const FaceVec& f) {
        viewer.data().clear();
        std::cout << "V: " << v.size() << "\t F: " << f.size() << std::endl;
        viewer.data().set_mesh(v, f);
    };

    bool reverseit = false;
    int it = 0;

    select_mesh(viewer, intervec[it].get_out_v(), intervec[it].get_out_f());
    updatebbox(intervec[it]);

    viewer.callback_key_down =
        [&](igl::opengl::glfw::Viewer & viewer, unsigned char key, int mod)->bool
    {
        switch (key)
        {
        default:
            return false;
        case '7':
            if (it == 0) reverseit = false;
            else if (it == intervec.size()) reverseit = true;
            if (reverseit) it--;

            viewer.data().clear();
            //intervec[it].compute(autoiso ? -1. : isov);
            viewer.data().set_mesh(intervec[it].get_out_v(), intervec[it].get_out_f());
            // updatebbox(intervec[it]);
            if (!reverseit) it++;
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
        viewer.data().set_mesh(intervec[it].get_out_v(), intervec[it].get_out_f());
        if (!reverseit) it++;

        return true;
    };

    viewer.core_list[viewer.core_index(0)].animation_max_fps = 15;
    viewer.core_list[viewer.core_index(0)].is_animating = true;
    viewer.launch();

    return 0;
}
