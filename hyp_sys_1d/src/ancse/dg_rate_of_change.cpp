#include <ancse/dg_rate_of_change.hpp>

#include <Eigen/Dense>
#include <ancse/config.hpp>
#include <ancse/polynomial_basis.hpp>
#include <ancse/dg_handler.hpp>
#include <ancse/numerical_flux.hpp>
#include <fmt/format.h>

//----------------DGRateOfChangeBegin----------------
/// DG numerical flux term
template <class NumericalFlux>
void DGRateOfChange<NumericalFlux>
:: eval_numerical_flux (Eigen::MatrixXd &dudt,
                        const Eigen::MatrixXd &u0) const
{

    auto n_cells = grid.n_cells;
    auto n_ghost = grid.n_ghost;
    auto n_vars = model->get_nvars();
    int n_coeff = 1 + poly_basis.get_degree();

    Eigen::VectorXd fL = Eigen::VectorXd::Zero(n_vars);
    Eigen::VectorXd fR = Eigen::VectorXd::Zero(n_vars);
    Eigen::VectorXd uL(n_vars), uR(n_vars);

    //// ANCSE_CUT_START_TEMPLATE
    //// ANCSE_COMMENT implement the loop for DG numerical flux term.

    /// evaluate basis to the left and right of a cell interface
    auto basisL = poly_basis(1.0);
    auto basisR = poly_basis(0.0);

    for (int j = n_ghost-1; j < n_cells - n_ghost; ++j)
    {
        uL = dg_handler.build_sol(u0.col(j), basisL);
        uR = dg_handler.build_sol(u0.col(j+1), basisR);

        fL = fR;
        fR = numerical_flux(uL, uR);

        for (int i = 0; i < n_vars; i++) {
            for (int q = 0; q < n_coeff; q++) {
                dudt(q + i*n_coeff, j)
                        += fL(i)*basisR(q) - fR(i)*basisL(q);
            }
        }
    }
    //// ANCSE_END_TEMPLATE
}

/// DG volume integral term
template <class NumericalFlux>
void DGRateOfChange<NumericalFlux>
:: eval_volume_integral(Eigen::MatrixXd &dudt,
                        const Eigen::MatrixXd &u0) const
{
    //// ANCSE_CUT_START_TEMPLATE
    //// ANCSE_COMMENT implement the loop for DG volume integral.
    int n_coeff = 1 + poly_basis.get_degree();
    if (n_coeff == 1) {
        return;
    }

    auto n_cells = grid.n_cells;
    auto n_ghost = grid.n_ghost;
    auto n_vars = model->get_nvars();

    Eigen::VectorXd flux = Eigen::VectorXd::Zero(n_vars);
    Eigen::VectorXd uSol(n_vars);

    int n_quad = quad_points.size();
    Eigen::MatrixXd basis(n_coeff, n_quad);
    Eigen::MatrixXd basis_deriv(n_coeff, n_quad);

    /// eval basis and its derivate for all quadrature points
    for (int k = 0; k < quad_points.size(); k++) {
        basis.col(k) = poly_basis(quad_points(k));
        basis_deriv.col(k) = poly_basis.deriv(quad_points(k));
    }

    for (int j = n_ghost-1; j < n_cells - n_ghost; ++j)
    {
        for (int k = 0; k < n_quad; k++)
        {
            uSol = dg_handler.build_sol(u0.col(j), basis.col(k));
            flux = model->flux(uSol);

            auto basis_deriv_ = basis_deriv.col(k);
            for (int i = 0; i < n_vars; i++) {
                for (int q = 0; q < n_coeff; q++) {
                    dudt(q + i*n_coeff, j)
                            += quad_weights(k)*flux(i)*basis_deriv_(q);
                }
            }
        }
    }
    //// ANCSE_END_TEMPLATE
}
//----------------DGRateOfChangeEnd----------------

#define REGISTER_NUMERICAL_FLUX(token, FluxType, flux)          \
    if (config["flux"] == (token)) {                            \
        return std::make_shared< DGRateOfChange<FluxType> >(    \
            grid, model, flux, poly_basis, dg_handler);                     \
    }


std::shared_ptr<RateOfChange> make_dg_rate_of_change(
    const nlohmann::json &config,
    const Grid &grid,
    const std::shared_ptr<Model> &model,
    const PolynomialBasis &poly_basis,
    const DGHandler &dg_handler,
    const std::shared_ptr<SimulationTime> &simulation_time)
{

    //// ANCSE_CUT_START_TEMPLATE
    //// ANCSE_COMMENT Register the other numerical fluxes.

    REGISTER_NUMERICAL_FLUX("lax_friedrichs",
                            LaxFriedrichs,
                            LaxFriedrichs(grid, model, simulation_time))

    REGISTER_NUMERICAL_FLUX("rusanov", Rusanov, Rusanov(model))

    REGISTER_NUMERICAL_FLUX("roe", Roe, Roe(model))

    REGISTER_NUMERICAL_FLUX("hll", HLL, HLL(model))

    if (model->get_name().compare("euler") == 0) {
        auto model_euler = std::dynamic_pointer_cast<Euler>(model);

        REGISTER_NUMERICAL_FLUX("hllc", HLLCEuler, HLLCEuler(model_euler))
    }

    //// ANCSE_END_TEMPLATE

    throw std::runtime_error(
        fmt::format("Unknown numerical flux. {}",
                    std::string(config["flux"])));
}

#undef REGISTER_NUMERICAL_FLUX
