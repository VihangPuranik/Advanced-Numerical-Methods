#ifndef HYPSYS1D_NUMERICAL_FLUX_HPP
#define HYPSYS1D_NUMERICAL_FLUX_HPP

#include <memory>
#include <iostream>

#include <ancse/grid.hpp>
#include <ancse/model.hpp>
#include <ancse/simulation_time.hpp>

/// Central flux.
/** This flux works does not depend on the model.
 * It is also unconditionally a bad choice.
 */
class CentralFlux {
  public:
    // Note: the interface for creating fluxes will give you access
    //       to the following:
    //         - model
    //         - grid
    //         - shared_ptr to simulation_time
    //       Therefore, try to only use a subset of those three in your
    //       constructors.
    explicit CentralFlux(const std::shared_ptr<Model> &model)
        : model(model) {}

    /// Compute the numerical flux given the left and right trace.
    Eigen::VectorXd operator()(const Eigen::VectorXd &uL,
                               const Eigen::VectorXd &uR) const
    {
        auto fL = model->flux(uL);
        auto fR = model->flux(uR);

        return 0.5 * (fL + fR);
    }

  private:
    std::shared_ptr<Model> model;
};


/// Lax-Friedrichs numerical flux.
/** This flux works for any model. */
//----------------FluxLFBegin----------------
class LaxFriedrichs {
  public:
    // Note: This version is a bit tricky. A numerical flux should be
    //       a function of the two trace values at the interface,
    //       i.e. what we call `uL`, `uR`.
    //       However, it requires 'dt' and 'dx'. Therefore,
    //       these need to be made available to the flux.
    //       This is one of the reasons why `SimulationTime`.
    LaxFriedrichs(const Grid &grid,
                  const std::shared_ptr<Model> &model,
                  std::shared_ptr<SimulationTime> simulation_time)
        : simulation_time(std::move(simulation_time)),
          grid(grid),
          model(model) {}

    Eigen::VectorXd operator()(const Eigen::VectorXd &uL,
                               const Eigen::VectorXd &uR) const {
        double dx = grid.dx;
        double dt = simulation_time->dt;
        //// ANCSE_CUT_START_TEMPLATE

        auto fL = model->flux(uL);
        auto fR = model->flux(uR);

        return 0.5 * ((fL + fR) - dx / dt * (uR - uL));
        //// ANCSE_END_TEMPLATE
        //// ANCSE_RETURN_VALUE Eigen::VectorXd();
    }

  private:
    std::shared_ptr<SimulationTime> simulation_time;
    Grid grid;
    std::shared_ptr<Model> model;
};
//----------------FluxLFEnd----------------


/// Rusanov's flux (or local Lax-Friedrichs).
/** This flux works for any model. */
//----------------FluxRusanovBegin----------------
class Rusanov {
  public:
    explicit Rusanov(const std::shared_ptr<Model> &model)
        : model(model) {}

    Eigen::VectorXd operator()(const Eigen::VectorXd &uL,
                               const Eigen::VectorXd &uR) const
    {
        //// ANCSE_CUT_START_TEMPLATE
        double a = std::max(model->max_eigenvalue(uL),
                            model->max_eigenvalue(uR));

        auto fL = model->flux(uL);
        auto fR = model->flux(uR);

        return 0.5 * ((fL + fR) - a * (uR - uL));
        //// ANCSE_END_TEMPLATE
        //// ANCSE_RETURN_VALUE Eigen::VectorXd();
    }

  private:
    std::shared_ptr<Model> model;
};
//----------------FluxRusanovEnd----------------

/// Roe flux.
/** This requires knowledge about the model.
 *  It is also well-known for generating unphysical weak solutions.
 */
//----------------FluxRoeBegin----------------
class Roe{
  public:
    explicit Roe(const std::shared_ptr<Model> &model)
        : model(model) {}

    Eigen::VectorXd operator()(const Eigen::VectorXd &uL,
                               const Eigen::VectorXd &uR) const
    {
        //// ANCSE_CUT_START_TEMPLATE
        auto fL = model->flux(uL);
        auto fR = model->flux(uR);

        auto uM = model->roe_avg(uL, uR);
        auto eigvalsM = model->eigenvalues(uM);
        auto eigvecsM = model->eigenvectors(uM);

        return 0.5*((fL + fR) - eigvecsM*(eigvalsM.cwiseAbs().asDiagonal())
                                *eigvecsM.fullPivLu().solve(uR-uL));

        //// ANCSE_END_TEMPLATE
        //// ANCSE_RETURN_VALUE Eigen::VectorXd();
    }

  private:
    std::shared_ptr<Model> model;
};
//----------------FluxRoeEnd----------------



/// HLL flux.
/** This requires knowledge about the model. */
//----------------FluxHLLBegin---------------- 
class HLL {
  public:
    explicit HLL(const std::shared_ptr<Model> &model) : model(model) {}

    Eigen::VectorXd operator()(const Eigen::VectorXd &uL,
                               const Eigen::VectorXd &uR) const
    {
        //// ANCSE_CUT_START_TEMPLATE
        auto fL = model->flux(uL);
        double sL = model->eigenvalues(uL).minCoeff();

        auto fR = model->flux(uR);
        double sR = model->eigenvalues(uR).maxCoeff();

        // compute Roe average
        auto uM = model->roe_avg(uL, uR);
        auto eigvalsM = model->eigenvalues(uM);
        // Einfeldt-batten speeds
        sL = std::min(sL, eigvalsM.minCoeff());
        sR = std::max(sR, eigvalsM.maxCoeff());

        // numerical flux
        Eigen::VectorXd f = Eigen::VectorXd::Zero(uM.size());
        if (sL > 0) {
            f = fL;
        } else if (sL <=0 && sR >= 0) { // fs
            f = ((sR*fL - sL*fR)+ sL*sR*(uR - uL))/(sR - sL);
        } else if (sR < 0) {
            f = fR;
        }

        return f;
        //// ANCSE_END_TEMPLATE
        //// ANCSE_RETURN_VALUE Eigen::VectorXd();
    }

  private:
    std::shared_ptr<Model> model;
};
//----------------FluxHLLEnd---------------- 

/// HLLC flux
/** This requires knowledge about the model.
 *  This version is for the Euler equation.
 */
//----------------FluxHLLCEulerBegin----------------  
class HLLCEuler {
  public:
    explicit HLLCEuler(const std::shared_ptr<Euler> &model)
        : model(model) {
        n_vars = model->get_nvars();
    }

    Eigen::VectorXd operator()(const Eigen::VectorXd &uL,
                               const Eigen::VectorXd &uR) const
    {
        //// ANCSE_CUT_START_TEMPLATE
        auto fL = model->flux(uL);
        auto fR = model->flux(uR);

        // left state
        auto [rhoL, vL, pL] = model->primitive(uL);
        double cL  = model->sound_speed(rhoL, pL);
        Eigen::VectorXd eigvals_uL(n_vars);
        eigvals_uL << vL - cL, vL, vL + cL;
        double sL = eigvals_uL.minCoeff();

        // right state
        auto [rhoR, vR, pR] = model->primitive(uR);
        double cR  = model->sound_speed(rhoR, pR);
        Eigen::VectorXd eigvals_uR(n_vars);
        eigvals_uR << vR - cR, vR, vR + cR;
        double sR = eigvals_uR.maxCoeff();

        // Roe pressure
        double sM = (pR - pL + (rhoL*vL)*(sL - vL) - (rhoR*vR)*(sR - vR))
                          /(rhoL*(sL - vL) - rhoR*(sR - vR));
        double pM = 0.5*(pR + pL + rhoL*(sL - vL)*(sM - vL)
                       + rhoR*(sM - vR)*(sR - vR));

        // numerical flux
        Eigen::VectorXd f = Eigen::VectorXd::Zero(fL.size());
        if (sL >= 0) {
            f = fL;
        }
        else if(sL <=0 && sM > 0) //fLs
        {
            Eigen::VectorXd u_prim(n_vars);
            double rhoLs = rhoL*(vL - sL)/(sM - sL);
            u_prim << rhoLs, sM, pM;
            auto uLs = model->prim_to_cons(u_prim);
            f = fL + sL*(uLs - uL);
        }
        else if(sM <=0 && sR >= 0) //fRs
        {
            Eigen::VectorXd u_prim(n_vars);
            double rhoRs = rhoR*(vR - sR)/(sM - sR);
            u_prim << rhoRs, sM, pM;
            auto uRs = model->prim_to_cons(u_prim);
            f = fR + sR*(uRs - uR);
        }
        else if (sR <= 0) {
            f = fR;
        }

        return f;
        //// ANCSE_END_TEMPLATE
        //// ANCSE_RETURN_VALUE Eigen::VectorXd();
    }

  private:
    std::shared_ptr<Euler> model;
    int n_vars;
};
//----------------FluxHLLCEulerEnd----------------  



#endif // HYPSYS1D_NUMERICAL_FLUX_HPP
