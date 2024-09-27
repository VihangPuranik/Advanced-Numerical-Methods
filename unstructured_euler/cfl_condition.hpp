#pragma once

#include <fmt/format.h>

#include "euler.hpp"
#include "mesh.hpp"

class CFLCondition {
  public:
    explicit CFLCondition(const Mesh &mesh) : dx(mesh.getMinimumInradius()) {}


//----------------CFLBegin----------------
    double operator()(const Eigen::MatrixXd &U) const {

        //// ANCSE_COMMENT compute the cfl condition here
        //// ANCSE_COMMENT you can use `assert_valid_timestep` to check if
        //// ANCSE_COMMENT the computed value is valid.

        double max_ev = 0.0;

        int n_cells = U.rows();

        //// ANCSE_CUT_START_TEMPLATE
#pragma omp parallel for
        for (int i = 0; i < n_cells; ++i) {
            max_ev = std::max(max_ev, euler::maxEigenValue(U.row(i)));
        }

        double dt_cfl = cfl_number * dx / max_ev;
        assert_valid_timestep(dt_cfl);

        return dt_cfl;
        //// ANCSE_RETURN_VALUE 0.001;
        //// ANCSE_END_TEMPLATE
    }
//----------------CFLEnd----------------

    void assert_valid_timestep(double dt_cfl) const {
        if (dt_cfl <= 0.0 || !std::isfinite(dt_cfl)) {
            throw std::runtime_error(
                fmt::format("Non-positive timestep: dt = {:.3e}", dt_cfl));
        }
    }

  private:
    double dx;
    double cfl_number = 0.45;
};