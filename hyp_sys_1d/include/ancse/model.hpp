#ifndef HYPSYS1D_MODEL_HPP
#define HYPSYS1D_MODEL_HPP

#include <cmath>
#include <memory>

#include <Eigen/Dense>
#include <ancse/config.hpp>

/// Interface for implementing different models,
/// eg. Euler equations, Shallow-water equations
///
/// Add more functions to this interface if needed.
class Model {
  public:
    virtual ~Model() = default;

    virtual Eigen::VectorXd flux(const Eigen::VectorXd &u) const = 0;
    virtual Eigen::VectorXd eigenvalues(const Eigen::VectorXd &u) const = 0;
    virtual Eigen::MatrixXd eigenvectors(const Eigen::VectorXd &u) const = 0;
    virtual double max_eigenvalue(const Eigen::VectorXd &u) const = 0;

    virtual Eigen::VectorXd cons_to_prim(const Eigen::VectorXd &u) const = 0;
    virtual Eigen::VectorXd prim_to_cons(const Eigen::VectorXd &u) const = 0;

    virtual Eigen::VectorXd roe_avg(const Eigen::VectorXd &uL,
                                    const Eigen::VectorXd &uR) const = 0;

    virtual int get_nvars() const = 0;
    virtual std::string get_name() const = 0;
};

class Burgers : public Model {
  public:

    Eigen::VectorXd flux(const Eigen::VectorXd &u) const override
    {
        Eigen::VectorXd f(n_vars);
        f(0) = 0.5*u(0)*u(0);

        return f;
    }

    Eigen::VectorXd eigenvalues(const Eigen::VectorXd &u) const override
    {
        Eigen::VectorXd eigvals(n_vars);
        eigvals(0) = u(0);

        return eigvals;
    }

    Eigen::MatrixXd eigenvectors(const Eigen::VectorXd &) const override
    {
        Eigen::MatrixXd eigvecs(n_vars, n_vars);
        eigvecs (0,0) = 1;

        return eigvecs;
    }

    double max_eigenvalue(const Eigen::VectorXd &u) const override {
        return (eigenvalues(u).cwiseAbs()).maxCoeff();
    }

    Eigen::VectorXd cons_to_prim(const Eigen::VectorXd &u) const override {
        return u;
    }

    Eigen::VectorXd prim_to_cons(const Eigen::VectorXd &u) const override {
        return u;
    }

    Eigen::VectorXd roe_avg(const Eigen::VectorXd &,
                            const Eigen::VectorXd &) const override
    {return Eigen::VectorXd::Zero(n_vars);}

    int get_nvars() const override
    {
        return n_vars;
    }

    std::string get_name() const override
    {
        return "burgers";
    }

  private:
    int n_vars = 1;
};

/// Euler equations
class Euler : public Model {
  public:
    
    Eigen::VectorXd flux(const Eigen::VectorXd &u) const override;
    
    Eigen::VectorXd eigenvalues(const Eigen::VectorXd &u) const override;

    Eigen::MatrixXd eigenvectors(const Eigen::VectorXd &u) const override;
    
    double max_eigenvalue(const Eigen::VectorXd &u) const override;

    Eigen::VectorXd cons_to_prim(const Eigen::VectorXd &u_cons) const override;

    Eigen::VectorXd prim_to_cons(const Eigen::VectorXd &u_prim) const override;

    Eigen::VectorXd roe_avg(const Eigen::VectorXd &,
                            const Eigen::VectorXd &) const override;

    ///  ANCSE_COMMENT Add more functions if needed.
    //----------------ModelEulerBegin----------------
    inline std::tuple<double, double, double>
    primitive(const Eigen::VectorXd &u_cons) const
    {

        ///  ANCSE_COMMENT Convert conservative to primitive;
        ///  ANCSE_COMMENT double rho=0.;
        ///  ANCSE_COMMENT double v=0.;
        ///  ANCSE_COMMENT double p=0.;
        ///  ANCSE_COMMENT return std::make_tuple (rho, v, p);


        //// ANCSE_CUT_START_TEMPLATE
        double rho = u_cons(0);
        double v   = u_cons(1)/u_cons(0);
        double E   = u_cons(2);
        double p = pressure(rho, v, E);
        return std::make_tuple (rho, v, p);
        //// ANCSE_END_TEMPLATE
    }

    inline double pressure(double rho, double v, double E) const
    {
        return (gamma-1)*(E - 0.5*rho*v*v);
    }

    inline double sound_speed(double rho, double p) const
    {
        return sqrt(gamma*p/rho);
    }

    inline double enthalpy(double rho, double E, double p) const
    {
        return (E+p)/rho;
    }

    inline double energy(double rho, double v, double p) const
    {
        return (p/(gamma-1) + 0.5*rho*v*v);
    }

    inline Eigen::VectorXd eigenvalues(double v, double c) const
    {

        Eigen::VectorXd eigvals(n_vars);
        ///  ANCSE_COMMENT Compute eigenvalues
        //// ANCSE_CUT_START_TEMPLATE
        eigvals << v - c, v, v + c;
        //// ANCSE_END_TEMPLATE
        return eigvals;
    }

    inline Eigen::MatrixXd eigenvectors(double v, double H) const
    {
        Eigen::MatrixXd eigvecs(n_vars, n_vars);
        ///  ANCSE_COMMENT Compute eigenvectors
        //// ANCSE_CUT_START_TEMPLATE
        double c = sqrt((gamma-1)*(H - 0.5*v*v));

        eigvecs << 1       , 1       , 1      ,
                   v - c   , v       , v + c  ,
                   H - v*c , 0.5*v*v , H + v*c;
        //// ANCSE_END_TEMPLATE

        return eigvecs;
    }
    //----------------ModelEulerEnd----------------


    void set_gamma(double gamma_)
    {
        gamma = gamma_;
    }

    double get_gamma() const
    {
        return gamma;
    }

    int get_nvars() const override
    {
        return n_vars;
    }

    std::string get_name() const override
    {
        return "euler";
    }

  private:
    int n_vars = 3;
    double gamma = 5./3.;
};

//// ANCSE_CUT_START_TEMPLATE
/// Shallow-water equations
class ShallowWater : public Model {
  public:

    Eigen::VectorXd flux(const Eigen::VectorXd &u) const override;

    Eigen::VectorXd eigenvalues(const Eigen::VectorXd &u) const override;

    Eigen::MatrixXd eigenvectors(const Eigen::VectorXd &u) const override;

    double max_eigenvalue(const Eigen::VectorXd &u) const override;

    Eigen::VectorXd cons_to_prim(const Eigen::VectorXd &u) const override;

    Eigen::VectorXd prim_to_cons(const Eigen::VectorXd &u) const override;

    Eigen::VectorXd roe_avg(const Eigen::VectorXd &,
                            const Eigen::VectorXd &) const override;

    inline std::pair<double, double>
    primitive(const Eigen::VectorXd &u_cons) const
    {
        double h = u_cons(0);
        double v   = u_cons(1)/u_cons(0);
        return std::make_pair (h, v);
    }

    inline double sound_speed(double h) const
    {
        return sqrt(g*h);
    }

    inline Eigen::VectorXd eigenvalues(double v, double c) const
    {
        Eigen::VectorXd eigvals(n_vars);
        eigvals << v - c, v + c;
        return eigvals;
    }

    inline Eigen::MatrixXd eigenvectors(double h, double v) const
    {
        Eigen::MatrixXd eigvecs(n_vars, n_vars);
        double c = sound_speed(h);

        eigvecs << 1       , 1    ,
                   v - c   , v + c;

        return eigvecs;
    }

    int get_nvars() const override
    {
        return n_vars;
    }

    std::string get_name() const override
    {
        return "shallow_water";
    }

  private:
    int n_vars = 2;
    double g = 1; // gravity constant
};
//// ANCSE_END_TEMPLATE

std::shared_ptr<Model> make_model (const nlohmann::json &config);

#endif // HYPSYS1D_MODEL_HPP
