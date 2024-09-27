#include <ancse/dg_handler.hpp>

//----------------DGHandlerBegin1----------------
/// build solution from DG coefficients and the basis
/// pre-evaluated at a certain point
Eigen::VectorXd DGHandler
:: build_sol(const Eigen::VectorXd& u,
             const Eigen::VectorXd& basis) const
{
    Eigen::VectorXd uSol = Eigen::VectorXd::Zero(n_vars);

    //// ANCSE_CUT_START_TEMPLATE
    for (int i = 0; i < n_vars; i++) {
        for (int q = 0; q < n_coeff; q++) {
            uSol(i) += u(q + i*n_coeff)*basis(q);
        }
    }
    //// ANCSE_END_TEMPLATE

    return uSol;
}

/// build solution from DG coefficients at a given reference point
Eigen::VectorXd DGHandler 
:: build_sol(const Eigen::VectorXd& u,
             double xi) const
{
    Eigen::VectorXd uSol(n_vars);

    //// ANCSE_CUT_START_TEMPLATE
    auto basis = poly_basis(xi);
    uSol = build_sol(u, basis);
    //// ANCSE_END_TEMPLATE

    return uSol;
}

/// build cell average
Eigen::MatrixXd DGHandler
:: build_cell_avg (const Eigen::MatrixXd& u) const
{
    auto n_cells = u.cols();
    Eigen::MatrixXd u0 (n_vars, n_cells);

    //// ANCSE_CUT_START_TEMPLATE
    double basis0 = (poly_basis(1.0))(0);
    for (int j = 0; j < n_cells; j++) {
        for (int i = 0; i < n_vars; i++) {
            u0(i,j) = u(i*n_coeff, j)*basis0;
        }
    }
    //// ANCSE_END_TEMPLATE

    return u0;
}
//----------------DGHandlerEnd1----------------

//----------------DGHandlerBegin2----------------
/// build split solution uSol_m = u0 + um, uSol_p = u0 - up
/// from DG coefficients
std::tuple <Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
DGHandler :: build_split_sol(const Eigen::MatrixXd& u) const
{
    auto n_cells = u.cols();
    if (n_coeff > 3) {
        throw std::runtime_error(
            "Limiter not implemented for higher than 3rd order");
    }
    
    auto u0 = build_cell_avg(u);
    Eigen::MatrixXd um = Eigen::MatrixXd::Zero (n_vars, n_cells);
    Eigen::MatrixXd up = Eigen::MatrixXd::Zero (n_vars, n_cells);

    //// ANCSE_CUT_START_TEMPLATE
    if (n_coeff > 1) {
        auto basis_m = poly_basis(1.0);
        auto basis_p = poly_basis(0.0);

        for (int j = 0; j < n_cells; j++) {
            for (int i = 0; i < n_vars; i++) {
                for (int q = 1; q < n_coeff; q++) {
                    um(i,j) += u(q + i*n_coeff, j)*basis_m(q);
                    up(i,j) -= u(q + i*n_coeff, j)*basis_p(q);
                }
            }
        }
    }
    //// ANCSE_END_TEMPLATE

    return {std::move(u0), std::move(um), std::move(up)};
}

/// build DG coefficients from uSol_m = u0 + um, uSol_p = u0 - up
void DGHandler :: compute_limit_coeffs (Eigen::MatrixXd &u,
                                        Eigen::MatrixXd &um,
                                        Eigen::MatrixXd &up) const
{
    if (n_coeff == 1) {
        return;
    }
    else if (n_coeff > 3) {
        throw std::runtime_error(
            "Limiter not implemented for higher than 3rd order");
    }
    
    //// ANCSE_CUT_START_TEMPLATE
    auto n_cells = u.cols();
    auto basis = poly_basis(1.0);

    if (n_coeff == 2) {
        for (int j = 0; j < n_cells; j++) {
            for (int i = 0; i < n_vars; i++) {
                u(1 + i*n_coeff, j) = um(i,j)/basis(1);
            }
        }
    } else if (n_coeff == 3) {
        for (int j = 0; j < n_cells; j++) {
            for (int i = 0; i < n_vars; i++) {
                u(1 + i*n_coeff, j) = 0.5*(um(i,j) + up(i,j))/basis(1);
                u(2 + i*n_coeff, j) = 0.5*(um(i,j) - up(i,j))/basis(2);
            }
        }
    }
    //// ANCSE_END_TEMPLATE
}
//----------------DGHandlerEnd2----------------

