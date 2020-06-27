// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2020 Monir Hadji <hadji.szs@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef IGL_OPTIMAL_TRANSPORT_INTERPOLATION_H
#define IGL_OPTIMAL_TRANSPORT_INTERPOLATION_H
#include "igl_inline.h"
#include <Eigen/Core>
#include <vector>

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace igl
{
    IGL_INLINE bool otbarycenter(
        const std::vector<Eigen::MatrixXd>& shapes,
        const Eigen::RowVector3i& res, // grid resolution of the input scalar fields
        const VectorXd& area,
        const VectorXd& alpha,
        const int num_iter,
        const int nb_iter,
        VectorXd& barycenter);
}

#ifndef IGL_STATIC_LIBRARY
#  include "optimal_transport_interpolation.cpp"
#endif

#endif // IGL_OPTIMAL_TRANSPORT_INTERPOLATION_H
