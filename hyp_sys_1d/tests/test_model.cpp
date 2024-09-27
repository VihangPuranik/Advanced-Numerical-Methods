#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <iostream>
#include <ancse/model.hpp>

//// ANCSE_CUT_START_TEMPLATE
TEST(TestEuler, primitives)
{
    auto model = Euler();
    int n_vars = model.get_nvars();
    ASSERT_EQ(n_vars, 3);

    double true_rho = 0.125;
    double true_v   = -2;
    double true_E   = 0.40;
    double true_p   = 0.1;
    double true_c   = 1.1547005383792517;
    double true_H   = 4;

    Eigen::VectorXd u_cons(n_vars);
    u_cons << true_rho, true_rho*true_v, true_E;

    double rho, v, p;
    std::tie (rho, v, p) = model.primitive(u_cons);
    double E = model.energy(rho, v, p);
    double c = model.sound_speed(rho, p);
    double H = model.enthalpy(rho, E, p);

    ASSERT_DOUBLE_EQ(rho, true_rho);
    ASSERT_DOUBLE_EQ(v  , true_v  );
    ASSERT_DOUBLE_EQ(E  , true_E  );
    ASSERT_DOUBLE_EQ(p  , true_p  );
    ASSERT_DOUBLE_EQ(c  , true_c  );
    ASSERT_DOUBLE_EQ(H  , true_H  );
}


TEST(TestEuler, flux)
{
    auto model = Euler();
    int n_vars = model.get_nvars();
    ASSERT_EQ(n_vars, 3);

    double rho = 0.125;
    double v   = -2;
    double E   = 0.40;

    Eigen::VectorXd true_flux(n_vars);
    true_flux << -0.25, 0.6, -1;

    double true_maxeigval = 3.1547005383792517;

    Eigen::MatrixXd true_eigvecs(n_vars, n_vars);
    true_eigvecs <<  1                 ,  1,  1,
                    -3.1547005383792517, -2, -0.8452994616207483,
                     6.309401076758503 ,  2,  1.6905989232414966;

    Eigen::VectorXd u_cons(n_vars);
    u_cons << rho, rho*v, E;

    Eigen::VectorXd flux = model.flux(u_cons);
    double maxeigval = model.max_eigenvalue(u_cons);
    Eigen::MatrixXd eigvecs = model.eigenvectors(u_cons);

    double TOL = 1E-10;
    ASSERT_LE((true_flux - flux).norm(), TOL);
    ASSERT_DOUBLE_EQ(maxeigval, true_maxeigval);
    ASSERT_LE((true_eigvecs - eigvecs).norm(), TOL);
}


TEST(TestShallowWater, primitives)
{
    auto model = ShallowWater();
    int n_vars = model.get_nvars();
    ASSERT_EQ(n_vars, 2);

    double true_h = 4.0;
    double true_v = 1;
    double true_c = 2.0;

    Eigen::VectorXd u_cons(n_vars);
    u_cons << true_h, true_h*true_v;

    double h, v;
    std::tie (h, v) = model.primitive(u_cons);
    double c = model.sound_speed(h);

    ASSERT_DOUBLE_EQ(h, true_h);
    ASSERT_DOUBLE_EQ(v, true_v);
    ASSERT_DOUBLE_EQ(c, true_c);
}


TEST(TestShallowWater, flux)
{
    auto model = ShallowWater();
    int n_vars = model.get_nvars();
    ASSERT_EQ(n_vars, 2);

    double h = 4;
    double v = 1.5;

    Eigen::VectorXd true_flux(n_vars);
    true_flux << 6.0, 17.0;

    double true_maxeigval = 3.5;

    Eigen::MatrixXd true_eigvecs(n_vars, n_vars);
    true_eigvecs <<  1   ,  1  ,
                    -0.5 , +3.5;

    Eigen::VectorXd u_cons(n_vars);
    u_cons << h, h*v;

    Eigen::VectorXd flux = model.flux(u_cons);
    double maxeigval = model.max_eigenvalue(u_cons);
    Eigen::MatrixXd eigvecs = model.eigenvectors(u_cons);

    double TOL = 1E-10;
    ASSERT_LE((true_flux - flux).norm(), TOL);
    ASSERT_DOUBLE_EQ(maxeigval, true_maxeigval);
    ASSERT_LE((true_eigvecs - eigvecs).norm(), TOL);
}
//// ANCSE_END_TEMPLATE
