#include <ancse/fvm_rate_of_change.hpp>

#include <Eigen/Dense>
#include <ancse/numerical_flux.hpp>
#include <ancse/reconstruction.hpp>
#include <fmt/format.h>

#define REGISTER_NUMERICAL_FLUX(token, FluxType, flux)                         \
    if (config["flux"] == (token)) {                                           \
        return std::make_shared<FVMRateOfChange<FluxType, Reconstruction>>(    \
            grid, model, flux, reconstruction);                                       \
    }

template <class Reconstruction>
std::shared_ptr<RateOfChange>
deduce_numerical_flux(const nlohmann::json &config,
                      const Grid &grid,
                      const std::shared_ptr<Model> &model,
                      const std::shared_ptr<SimulationTime> &simulation_time,
                      const Reconstruction &reconstruction)
{
    REGISTER_NUMERICAL_FLUX("central_flux", CentralFlux, CentralFlux(model))

    //// ANCSE_CUT_START_TEMPLATE
    //// ANCSE_COMMENT Register the other numerical fluxes.
    REGISTER_NUMERICAL_FLUX("lax_friedrichs",
                            LaxFriedrichs,
                            LaxFriedrichs(grid, model, simulation_time))

    REGISTER_NUMERICAL_FLUX("rusanov", Rusanov, Rusanov(model))

    REGISTER_NUMERICAL_FLUX("roe", Roe, Roe(model))

    REGISTER_NUMERICAL_FLUX("hll", HLL, HLL(model))

    //// ANCSE_END_TEMPLATE

    if (model->get_name().compare("euler") == 0) {
        auto model_euler = std::dynamic_pointer_cast<Euler>(model);

        REGISTER_NUMERICAL_FLUX("hllc", HLLCEuler, HLLCEuler(model_euler))
    }
    //// ANCSE_CUT_START_TEMPLATE
    else if (model->get_name().compare("shallow_water") == 0) {
        auto model_sw = std::dynamic_pointer_cast<ShallowWater>(model);
    }
    //// ANCSE_END_TEMPLATE


    throw std::runtime_error(
        fmt::format("Unknown numerical flux. {}", std::string(config["flux"])));
}
#undef REGISTER_NUMERICAL_FLUX

#define REGISTER_RECONSTRUCTION(token, reconstruction)                         \
    if (config["reconstruction"] == token) {                                   \
        return deduce_numerical_flux(                                          \
            config, grid, model, simulation_time, reconstruction);                     \
    }

std::shared_ptr<RateOfChange> make_fvm_rate_of_change(
    const nlohmann::json &config,
    const Grid &grid,
    const std::shared_ptr<Model> &model,
    const std::shared_ptr<SimulationTime> &simulation_time)
{
    REGISTER_RECONSTRUCTION("o1", PWConstantReconstruction{})

    //// ANCSE_CUT_START_TEMPLATE
    //// ANCSE_COMMENT Register piecewise linear reconstructions.
    if (config["reconstruction_variable"] == "conserved") {
        REGISTER_RECONSTRUCTION("minmod", (PWLinearReconstruction<MinMod,Conserved>(MinMod{})) )
        REGISTER_RECONSTRUCTION("superbee", (PWLinearReconstruction<SuperBee,Conserved>(SuperBee{})) )
        REGISTER_RECONSTRUCTION("mc", (PWLinearReconstruction<MonotonizedCentral,Conserved>(MonotonizedCentral{})) )
    }
    else if (config["reconstruction_variable"] == "primitive") {
        REGISTER_RECONSTRUCTION("minmod", (PWLinearReconstruction<MinMod,Primitive>(model, MinMod{})) )
        REGISTER_RECONSTRUCTION("superbee", (PWLinearReconstruction<SuperBee,Primitive>(model, SuperBee{})) )
        REGISTER_RECONSTRUCTION("mc", (PWLinearReconstruction<MonotonizedCentral,Primitive>(model, MonotonizedCentral{})) )
    }
    //// ANCSE_END_TEMPLATE

    throw std::runtime_error(fmt::format(
        "Unknown reconstruction. [{}]", std::string(config["reconstruction"])));
}

#undef REGISTER_RECONSTRUCTION
