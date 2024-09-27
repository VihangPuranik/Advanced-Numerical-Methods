#pragma once

#include "types.hpp"
#include <Eigen/Dense>

//----------------GradientBegin----------------
/// Compute the gradients of q_bar.
/** Note: this routine does not care if q are the conserved
 *  or primitive variables.
 *
 * @param [out] dqdx  approximation of dq/dx. Has shape (n_cells, 4).
 * @param [out] dqdy  approximation of dq/dy. Has shape (n_cells, 4).
 * @param       q_bar the cell-averages of `q`. Has shape (n_cells, 4).
 * @param       mesh
 */
void compute_gradients(Eigen::MatrixXd &dqdx,
                       Eigen::MatrixXd &dqdy,
                       const Eigen::MatrixXd &q_bar,
                       const Mesh &mesh) {



    dqdx.setZero();
    dqdy.setZero();

    int n_cells = mesh.getNumberOfTriangles();

    for (int i = 0; i < n_cells; ++i) {
        // Let's first deal with the case where we can't compute the gradient
        // via Gauss.
        if (!mesh.isValidNeighbour(i, 0) || !mesh.isValidNeighbour(i, 1)
            || !mesh.isValidNeighbour(i, 2)) {
            continue;
        }


        double area = mesh.getTriangleArea(i);


        // w.l.o.g. there are enough neighbours.
        for (int k = 0; k < 3; ++k) {
            auto n = mesh.getUnitNormal(i, k);
            double length = mesh.getEdgeLength(i, k);

            int j = mesh.getNeighbour(i, k);

            Eigen::Vector2d xi = mesh.getCellCenter(i);
            Eigen::Vector2d xj = mesh.getCellCenter(j);
            Eigen::Vector2d x_ij = mesh.getEdgeCenter(i, k);

            double al = (x_ij - xi).norm() / (xi - xj).norm();

            //// ANCSE_COMMENT Compute the gradient of all 4 components of q_bar
            //// ANCSE_CUT_START_TEMPLATE
            EulerState q_ij = (1.0 - al) * q_bar.row(i) + al * q_bar.row(j);

            dqdx.row(i) += q_ij * n[0] * length / area;
            dqdy.row(i) += q_ij * n[1] * length / area;

            //// ANCSE_END_TEMPLATE
        }
    }


}
//----------------GradientEnd----------------
