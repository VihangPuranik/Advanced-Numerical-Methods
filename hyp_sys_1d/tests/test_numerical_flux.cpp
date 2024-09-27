#include <gtest/gtest.h>

#include <ancse/numerical_flux.hpp>

template <class NumericalFlux>
void check_consistency_euler(const NumericalFlux &nf) {

    auto model = Euler();
    int n_vars = model.get_nvars();

    Eigen::MatrixXd u(n_vars, 3);
    u.col(0) << 1, 0, 1.5;
    u.col(1) << 0.125, 0.25, 0.40;
    u.col(2) << 0.11432, -0.11432, 0.26432;

    double TOL = 1E-10;
    for (int k=0; k<u.cols(); k++) {
        ASSERT_LE( (model.flux(u.col(k)) - nf(u.col(k), u.col(k))).norm(), TOL);
    }
}

TEST(TestCentralFlux, consistency) {
    std::shared_ptr<Model> model = std::make_shared<Euler>();
    auto central_flux = CentralFlux(model);

    check_consistency_euler(central_flux);
}

//// ANCSE_CUT_START_TEMPLATE
TEST(TestRusanov, consistency) {
    std::shared_ptr<Model> model = std::make_shared<Euler>();
    auto rusanov = Rusanov(model);

    check_consistency_euler(rusanov);
}
//// ANCSE_END_TEMPLATE

//// ANCSE_CUT_START_TEMPLATE
TEST(TestLaxFriedrichs, consistency) {
    std::shared_ptr<Model> model = std::make_shared<Euler>();
    auto grid = Grid({0.0, 1.0}, 14, 2);
    auto sim_time = std::make_shared<SimulationTime>(0.0, 0.1, 0.2, 0);
    auto lf = LaxFriedrichs(grid, model, sim_time);

    check_consistency_euler(lf);
}
//// ANCSE_END_TEMPLATE

//// ANCSE_CUT_START_TEMPLATE
TEST(TestRoeEuler, consistency) {
    std::shared_ptr<Model> model = std::make_shared<Euler>();
    auto roe = Roe(model);

    check_consistency_euler(roe);
}
//// ANCSE_END_TEMPLATE

//// ANCSE_CUT_START_TEMPLATE
TEST(TestHLL, consistency) {
    std::shared_ptr<Model> model = std::make_shared<Euler>();
    auto hll = HLL(model);

    check_consistency_euler(hll);
}
//// ANCSE_END_TEMPLATE

//// ANCSE_CUT_START_TEMPLATE
TEST(TestHLLCEuler, consistency) {
    auto model = std::make_shared<Euler>();
    auto hllc = HLLCEuler(model);

    check_consistency_euler(hllc);
}
//// ANCSE_END_TEMPLATE

//// ANCSE_CUT_START_TEMPLATE
template <class NumericalFlux>
void check_consistency_shallow_water(const NumericalFlux &nf) {

    auto model = ShallowWater();
    int n_vars = model.get_nvars();

    Eigen::MatrixXd u(n_vars, 3);
    u.col(0) << 1, 0;
    u.col(1) << 0.125, 0.25;
    u.col(2) << 0.11432, -0.11432;

    double TOL = 1E-10;
    for (int k=0; k<u.cols(); k++) {
        ASSERT_LE( (model.flux(u.col(k)) - nf(u.col(k), u.col(k))).norm(), TOL);
    }
}

TEST(TestRoeShallowWater, consistency) {
    std::shared_ptr<Model> model = std::make_shared<ShallowWater>();
    auto roe = Roe(model);

    check_consistency_shallow_water(roe);
}
//// ANCSE_END_TEMPLATE
