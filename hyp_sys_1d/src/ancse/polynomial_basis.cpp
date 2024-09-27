#include "../../include/ancse/polynomial_basis.hpp"

//----------------PolyBasisBegin----------------
/// Computes the Legendre polynomial basis
/// at a given reference point xi \in [0,1]
Eigen::VectorXd PolynomialBasis :: operator() (double xi) const
{
    Eigen::VectorXd basis(p+1);


    //// ANCSE_CUT_START_TEMPLATE
    basis(0) = 1;
    if (p > 0) {
        double z = 2*xi - 1;
        basis(1) = sqrt(3)*z;
        if (p > 1) {
            basis(2) = 0.5*sqrt(5)*(3*z*z-1);
        }
    }

    return (basis*scaling_factor);
    //// ANCSE_END_TEMPLATE
    //// ANCSE_RETURN_VALUE Eigen::VectorXd::Zero(p+1);
}

/// Computes the derivative of Legendre polynomial basis
/// at a given reference point xi \in [0,1]
Eigen::VectorXd PolynomialBasis :: deriv (double xi) const
{
    Eigen::VectorXd basis_deriv(p+1);

    //// ANCSE_CUT_START_TEMPLATE
    basis_deriv(0) = 0;
    if (p > 0) {
        double z = 2*xi - 1;
        basis_deriv(1) = 2*sqrt(3);
        if (p > 1) {
            basis_deriv(2) = 6*sqrt(5)*z;
        }
    }

    return (basis_deriv*scaling_factor);
    //// ANCSE_END_TEMPLATE
    //// ANCSE_RETURN_VALUE Eigen::VectorXd::Zero(p+1);
}
//----------------PolyBasisEnd----------------
