#include <Eigen/Dense>
#include <ancse/cfl_condition.hpp>
#include <gtest/gtest.h>

//// ANCSE_CUT_START_TEMPLATE
// This will check the CFL condition for Euler equation.
void check_cfl_condition_euler(const CFLCondition &cfl_condition,
                         const Grid &grid,
                         double cfl_number) {
    int n_ghost = grid.n_ghost;
    int n_cells = grid.n_cells;
    int n_vars  = 3;

    Eigen::MatrixXd u(n_vars, n_cells);
    // set u to initial conditions for Sod shock tube
    // left state
    for (int i = 0; i < n_cells/2; ++i) {
        u.col(i) << 1, 0, 1.5;
    }
    // right state
    for (int i = n_cells/2; i < n_cells; ++i) {
        u.col(i) << 0.125, 0, 0.15;
    }
    // ghost cells states modified
    // to yield max eigenvalue higher than the interior cells.
    for (int i = 0; i < n_ghost; ++i) {
        u.col(i) << 1, 1, 1.5;
        u.col(n_cells - n_ghost + i) << 0.125, 1, 0.15;
    }

    double max_eigval = 1.2909944487358056;

    double dt_cfl_approx = cfl_condition(u);
    double dt_cfl_exact = cfl_number * grid.dx / max_eigval;
    ASSERT_DOUBLE_EQ(dt_cfl_approx, dt_cfl_exact);
}

TEST(CFLCondition, Euler) {
    auto n_ghost = 2;
    auto n_cells = 6 + 2 * n_ghost;
    auto grid = Grid({0.9, 1.0}, n_cells, n_ghost);
    std::shared_ptr<Model> model = std::make_shared<Euler>();
    double cfl_number = 0.5;

    auto cfl_condition = StandardCFLCondition<FVM>(grid, model, cfl_number);
    check_cfl_condition_euler(cfl_condition, grid, cfl_number);
}
//// ANCSE_END_TEMPLATE
