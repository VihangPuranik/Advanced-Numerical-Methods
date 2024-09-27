#include <ancse/model.hpp>

#include <iostream>
#include <fmt/format.h>


///------------------///
/// Euler equations  ///
///------------------///

//// ANCSE_COMMENT For task 1g you can use the functions implemented in model.hpp
//----------------ModelEulerBegin----------------
Eigen::VectorXd Euler::flux(const Eigen::VectorXd &u) const
{
    //// ANCSE_CUT_START_TEMPLATE
    double rho, v, p;
    std::tie (rho, v, p) = primitive(u);
    double E = energy(rho, v, p);

    Eigen::VectorXd f(n_vars);
    f << u(1), rho*v*v + p, (E+p)*v;

    return f;
    //// ANCSE_END_TEMPLATE
    //// ANCSE_RETURN_VALUE Eigen::VectorXd::Zero(n_vars);
}

Eigen::VectorXd Euler::eigenvalues(const Eigen::VectorXd &u) const
{
    //// ANCSE_CUT_START_TEMPLATE
    double rho, v, p;
    std::tie (rho, v, p) = primitive(u);
    double c = sound_speed(rho, p);

    Eigen::VectorXd eigvals(n_vars);
    eigvals << v - c, v, v + c;

    return eigvals;
    //// ANCSE_END_TEMPLATE
    //// ANCSE_RETURN_VALUE Eigen::VectorXd::Zero(n_vars);
}

Eigen::MatrixXd Euler::eigenvectors(const Eigen::VectorXd &u) const
{
    //// ANCSE_CUT_START_TEMPLATE
    double rho, v, p;
    std::tie (rho, v, p) = primitive(u);
    double E = energy(rho, v, p);
    double H = enthalpy(rho, E, p);

    return eigenvectors(v, H);
    //// ANCSE_END_TEMPLATE
    //// ANCSE_RETURN_VALUE Eigen::MatrixXd::Zero(n_vars, n_vars);
}

double Euler::max_eigenvalue(const Eigen::VectorXd &u) const
{
    //// ANCSE_CUT_START_TEMPLATE
    return (eigenvalues(u).cwiseAbs()).maxCoeff();
    //// ANCSE_END_TEMPLATE
    //// ANCSE_RETURN_VALUE 0;
}


Eigen::VectorXd Euler::cons_to_prim(const Eigen::VectorXd &u_cons) const
{
    //// ANCSE_CUT_START_TEMPLATE
    double rho, v, p;
    std::tie (rho, v, p) = primitive(u_cons);

    Eigen::VectorXd u_prim(n_vars);
    u_prim << rho, v, p;

    return u_prim;
    //// ANCSE_END_TEMPLATE
    //// ANCSE_RETURN_VALUE Eigen::VectorXd::Zero(n_vars);
}

Eigen::VectorXd Euler::prim_to_cons(const Eigen::VectorXd &u_prim) const
{
    //// ANCSE_CUT_START_TEMPLATE
    Eigen::VectorXd u_cons(n_vars);
    u_cons(0) = u_prim(0);
    u_cons(1) = u_prim(0)*u_prim(1);
    u_cons(2) = u_prim(2)/(gamma-1) + 0.5*u_prim(0)*u_prim(1)*u_prim(1);

    return u_cons;
    //// ANCSE_END_TEMPLATE
    //// ANCSE_RETURN_VALUE Eigen::VectorXd::Zero(n_vars);
}

Eigen::VectorXd Euler::roe_avg(const Eigen::VectorXd &uL,
                               const Eigen::VectorXd &uR) const
{
    // left state
    double rhoL, vL, pL;
    std::tie (rhoL, vL, pL) = this->primitive(uL);
    double EL = this->energy(rhoL, vL, pL);
    double HL = this->enthalpy(rhoL, EL, pL);

    // right state
    double rhoR, vR, pR;
    std::tie (rhoR, vR, pR) = this->primitive(uR);
    double ER = this->energy(rhoR, vR, pR);
    double HR = this->enthalpy(rhoR, ER, pR);

    // Roe state
    double rhoS = 0.5*(rhoL + rhoR);
    double tmp = (sqrt(rhoL) + sqrt(rhoR));
    double vS  = (sqrt(rhoL)*vL + sqrt(rhoR)*vR)/tmp;
    double HS  = (sqrt(rhoL)*HL + sqrt(rhoR)*HR)/tmp;

    Eigen::VectorXd uS(n_vars);
    uS(0) = rhoS;
    uS(1) = uS(0)*vS;
    uS(2) = rhoS*HS/gamma + 0.5*(gamma-1)/gamma*rhoS*vS*vS;

    return uS;
}
//----------------ModelEulerEnd----------------

//// ANCSE_CUT_START_TEMPLATE
///--------------------------///
/// Shallow-water equations  ///
///--------------------------///
Eigen::VectorXd ShallowWater::flux(const Eigen::VectorXd &u) const
{
    double h, v;
    std::tie (h, v) = primitive(u);

    Eigen::VectorXd f(2);
    f << u(1), 0.5*g*h*h + h*v*v;

    return f;
}

Eigen::VectorXd ShallowWater::eigenvalues(const Eigen::VectorXd &u) const
{
    double h, v;
    std::tie (h, v) = primitive(u);
    double c = sound_speed(h);

    return eigenvalues(v, c);
}

Eigen::MatrixXd ShallowWater::eigenvectors(const Eigen::VectorXd &u) const
{
    double h, v;
    std::tie (h, v) = primitive(u);

    return eigenvectors(h, v);
}

double ShallowWater::max_eigenvalue(const Eigen::VectorXd &u) const
{
    return (eigenvalues(u).cwiseAbs()).maxCoeff();
}

Eigen::VectorXd ShallowWater::cons_to_prim(const Eigen::VectorXd &u_cons) const
{
    Eigen::VectorXd u_prim(n_vars);
    u_prim << u_cons(0), u_cons(1)/u_cons(0);
    return u_prim;
}

Eigen::VectorXd ShallowWater::prim_to_cons(const Eigen::VectorXd &u_prim) const
{
    Eigen::VectorXd u_cons(n_vars);
    u_cons(0) = u_prim(0);
    u_cons(1) = u_prim(0)*u_prim(1);
    return u_cons;
}

Eigen::VectorXd ShallowWater::roe_avg(const Eigen::VectorXd &uL,
                                      const Eigen::VectorXd &uR) const
{
    // left state
    auto [hL, vL] = primitive(uL);

    // right state
    auto [hR, vR] = primitive(uR);

    // Roe state
    double hS = 0.5*(hL + hR);
    double tmp = (sqrt(hL) + sqrt(hR));
    double vS  = (sqrt(hL)*vL + sqrt(hR)*vR)/tmp;

    Eigen::VectorXd uS(n_vars);
    uS(0) = hS;
    uS(1) = uS(0)*vS;

    return uS;
}
//// ANCSE_END_TEMPLATE

#define REGISTER_MODEL(token, ModelType)      \
    if (config["model"] == (token)) {         \
        return std::make_shared<ModelType>(); \
    }

std::shared_ptr<Model> make_model (const nlohmann::json &config)
{
    REGISTER_MODEL("burgers", Burgers)
    //// ANCSE_COMMENT implement and register your models here
    //// ANCSE_CUT_START_TEMPLATE
    REGISTER_MODEL("euler", Euler)

    REGISTER_MODEL("shallow_water", ShallowWater)
    //// ANCSE_END_TEMPLATE

    throw std::runtime_error(
        fmt::format("Unknown model. {}", std::string(config["flux"])));
}
