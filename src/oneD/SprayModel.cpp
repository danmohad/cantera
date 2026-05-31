//! @file SprayModel.cpp

// This file is part of Cantera. See License.txt in the top-level directory or
// at https://cantera.org/license.txt for license and copyright information.

#include "cantera/oneD/SprayModel.h"
#include "cantera/base/ctexceptions.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/transport/Transport.h"

#include <algorithm>
#include <cmath>

using namespace std;

namespace Cantera
{

void MonodisperseSprayModel::setLiquidProperties(
    const LiquidFuelProperties& properties)
{
    if (properties.density <= 0.0) {
        throw CanteraError("MonodisperseSprayModel::setLiquidProperties",
            "Liquid density must be positive.");
    }
    if (properties.cp <= 0.0) {
        throw CanteraError("MonodisperseSprayModel::setLiquidProperties",
            "Liquid specific heat must be positive.");
    }
    if (properties.latentHeat <= 0.0) {
        throw CanteraError("MonodisperseSprayModel::setLiquidProperties",
            "Latent heat must be positive.");
    }
    if (properties.boilingTemperature <= 0.0) {
        throw CanteraError("MonodisperseSprayModel::setLiquidProperties",
            "Boiling temperature must be positive.");
    }
    if (properties.saturationPressure <= 0.0) {
        throw CanteraError("MonodisperseSprayModel::setLiquidProperties",
            "Reference saturation pressure must be positive.");
    }
    m_liquid = properties;
}

double MonodisperseSprayModel::diameter(double dropletMass) const
{
    if (dropletMass <= 0.0) {
        return 0.0;
    }
    if (m_liquid.density <= 0.0) {
        throw CanteraError("MonodisperseSprayModel::diameter",
            "Liquid properties have not been specified.");
    }
    return pow(6.0 * dropletMass / (Pi * m_liquid.density), 1.0/3.0);
}

double MonodisperseSprayModel::dropletMass(double diameter) const
{
    if (diameter <= 0.0) {
        throw CanteraError("MonodisperseSprayModel::dropletMass",
            "Droplet diameter must be positive.");
    }
    if (m_liquid.density <= 0.0) {
        throw CanteraError("MonodisperseSprayModel::dropletMass",
            "Liquid properties have not been specified.");
    }
    return Pi / 6.0 * m_liquid.density * pow(diameter, 3);
}

double MonodisperseSprayModel::surfaceFuelMassFraction(
    ThermoPhase& thermo, size_t fuelIndex, double dropletTemperature) const
{
    double wFuel = thermo.molecularWeight(fuelIndex);
    double rv = GasConstant / wFuel;
    double exponent = -m_liquid.latentHeat / rv *
                      (1.0 / dropletTemperature - 1.0 / m_liquid.boilingTemperature);
    double pSat = m_liquid.saturationPressure * exp(exponent);
    pSat = std::clamp(pSat, 0.0, 0.999 * thermo.pressure());
    double xFuelSurface = pSat / thermo.pressure();

    double xCarrier = 0.0;
    double carrierWeight = 0.0;
    for (size_t k = 0; k < thermo.nSpecies(); k++) {
        if (k == fuelIndex) {
            continue;
        }
        double xk = thermo.moleFraction(k);
        xCarrier += xk;
        carrierWeight += xk * thermo.molecularWeight(k);
    }
    if (xCarrier > Tiny) {
        carrierWeight /= xCarrier;
    } else {
        carrierWeight = thermo.meanMolecularWeight();
    }

    double denom = xFuelSurface * wFuel + (1.0 - xFuelSurface) * carrierWeight;
    return xFuelSurface * wFuel / std::max(denom, Tiny);
}

double MonodisperseSprayModel::filmCorrection(double transferNumber)
{
    if (fabs(transferNumber) < 1e-8) {
        return 1.0;
    }
    double limited = std::max(transferNumber, -0.95);
    return pow(1.0 + limited, 0.7) * log1p(limited) / limited;
}

SprayModelResult MonodisperseSprayModel::eval(
    ThermoPhase& thermo, Transport& transport, size_t fuelIndex, double dropletMass,
    double dropletTemperature, double dropletVelocity, double gasVelocity,
    double dropletSpreadRate, double gasSpreadRate) const
{
    SprayModelResult result;
    if (dropletMass <= 0.0) {
        return result;
    }

    double diameter_ = diameter(dropletMass);
    double radius = 0.5 * diameter_;
    double density = thermo.density();
    double viscosity = transport.viscosity();
    double conductivity = transport.thermalConductivity();
    double cpGas = thermo.cp_mass();

    vector<double> diff(thermo.nSpecies());
    transport.getMixDiffCoeffs(span<double>(diff.data(), diff.size()));
    double fuelDiff = std::max(diff[fuelIndex], Tiny);

    double axialSlip = gasVelocity - dropletVelocity;
    double radialSlipScale = radius * (gasSpreadRate - dropletSpreadRate);
    double slip = sqrt(axialSlip * axialSlip + radialSlipScale * radialSlipScale);
    result.reynoldsNumber = density * slip * diameter_ / std::max(viscosity, Tiny);

    double prandtl = cpGas * viscosity / std::max(conductivity, Tiny);
    double schmidt = viscosity / std::max(density * fuelDiff, Tiny);
    double nu0 = 2.0 + 0.552 * sqrt(result.reynoldsNumber) * pow(prandtl, 1.0/3.0);
    double sh0 = 2.0 + 0.552 * sqrt(result.reynoldsNumber) * pow(schmidt, 1.0/3.0);

    double yFuelSurface = surfaceFuelMassFraction(thermo, fuelIndex, dropletTemperature);
    double yFuelGas = thermo.massFraction(fuelIndex);
    result.massTransferNumber = std::max(
        (yFuelSurface - yFuelGas) / std::max(1.0 - yFuelSurface, Tiny), 0.0);

    result.sherwoodNumber = 2.0 + (sh0 - 2.0) / filmCorrection(result.massTransferNumber);
    if (result.massTransferNumber > 0.0) {
        result.evaporationRate = Pi * diameter_ * density * fuelDiff
            * result.sherwoodNumber * log1p(result.massTransferNumber);
    }

    vector<double> cpbar(thermo.nSpecies());
    thermo.getPartialMolarCp(span<double>(cpbar.data(), cpbar.size()));
    double cpFuel = cpbar[fuelIndex] / thermo.molecularWeight(fuelIndex);
    result.heatTransferNumber = cpFuel * (thermo.temperature() - dropletTemperature)
        / std::max(m_liquid.latentHeat, Tiny);

    result.nusseltNumber = 2.0 + (nu0 - 2.0) / filmCorrection(result.heatTransferNumber);
    double heatCorrection = 1.0;
    if (fabs(result.heatTransferNumber) > 1e-8) {
        heatCorrection = log1p(std::max(result.heatTransferNumber, -0.95))
            / result.heatTransferNumber;
    }
    result.heatTransferRate = 2.0 * Pi * radius * conductivity
        * result.nusseltNumber * (thermo.temperature() - dropletTemperature)
        * heatCorrection;

    double dragCoefficient = 3.0 * Pi * viscosity * diameter_;
    result.dragAxial = dragCoefficient * axialSlip;
    result.dragSpreadRate = dragCoefficient * (gasSpreadRate - dropletSpreadRate);

    return result;
}

}
