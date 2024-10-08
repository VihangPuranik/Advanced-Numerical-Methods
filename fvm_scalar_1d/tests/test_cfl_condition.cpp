#include <Eigen/Dense>
#include <ancse/cfl_condition.hpp>
#include <gtest/gtest.h>

//// ANCSE_CUT_START_TEMPLATE
// This will check the CFL condition for Burgers equation.
void check_cfl_condition(const CFLCondition &cfl_condition,
                         const Grid &grid,
                         double cfl_number) {
    int n_ghost = grid.n_ghost;
    int n_cells = grid.n_cells;

    Eigen::VectorXd u(n_cells);
    for (int i = 0; i < n_cells; ++i) {
        u[i] = -i * i;
    }
    double max_abs_u = (n_cells - n_ghost - 1) * (n_cells - n_ghost - 1);

    for (int i = 0; i < n_ghost; ++i) {
        u[i] = 2.0 * max_abs_u;
        u[n_cells - n_ghost + i] = 2.0 * max_abs_u;
    }

    double dt_cfl_approx = cfl_condition(u);
    double dt_cfl_exact = cfl_number * grid.dx / max_abs_u;

    ASSERT_DOUBLE_EQ(dt_cfl_approx, dt_cfl_exact);
}

TEST(CFLCondition, Example) {
    auto n_ghost = 2;
    auto n_cells = 10 + 2 * n_ghost;
    auto grid = Grid({0.9, 1.0}, n_cells, n_ghost);
    auto model = Model();
    double cfl_number = 0.5;

    auto cfl_condition = StandardCFLCondition(grid, model, cfl_number);
    check_cfl_condition(cfl_condition, grid, cfl_number);
}
//// ANCSE_END_TEMPLATE
