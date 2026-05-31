//! @file SprayModel.h

// This file is part of Cantera. See License.txt in the top-level directory or
// at https://cantera.org/license.txt for license and copyright information.

#ifndef CT_SPRAYMODEL_H
#define CT_SPRAYMODEL_H

#include "cantera/base/ct_defs.h"

namespace Cantera
{

class ThermoPhase;
class Transport;

//! Liquid fuel properties used by monodisperse droplet evaporation models.
struct LiquidFuelProperties
{
    double density = Undef; //!< Liquid density [kg/m^3]
    double cp = Undef; //!< Liquid specific heat [J/kg/K]
    double latentHeat = Undef; //!< Latent heat of vaporization [J/kg]
    double boilingTemperature = Undef; //!< Normal boiling temperature [K]
    double saturationPressure = OneAtm; //!< Saturation pressure at boiling point [Pa]
};

//! Result of a single-droplet spray closure evaluation.
struct SprayModelResult
{
    double evaporationRate = 0.0; //!< Single-droplet evaporation rate [kg/s]
    double heatTransferRate = 0.0; //!< Heat transfer from gas to droplet [W]
    double dragAxial = 0.0; //!< Axial drag force on droplet [N]
    double dragSpreadRate = 0.0; //!< Radial drag force per radius on droplet [N/m]
    double reynoldsNumber = 0.0; //!< Droplet Reynolds number
    double nusseltNumber = 2.0; //!< Droplet Nusselt number
    double sherwoodNumber = 2.0; //!< Droplet Sherwood number
    double massTransferNumber = 0.0; //!< Spalding mass transfer number
    double heatTransferNumber = 0.0; //!< Spalding heat transfer number
};

//! Abramzon-Sirignano style film model for single-component droplets.
class MonodisperseSprayModel
{
public:
    //! Set liquid properties for the evaporating species.
    void setLiquidProperties(const LiquidFuelProperties& properties);

    //! Liquid properties used by this model.
    const LiquidFuelProperties& liquidProperties() const {
        return m_liquid;
    }

    //! Return a droplet diameter from a droplet mass [m].
    double diameter(double dropletMass) const;

    //! Return a droplet mass from a droplet diameter [kg].
    double dropletMass(double diameter) const;

    //! Evaluate evaporation, heat transfer, and drag for one droplet.
    SprayModelResult eval(ThermoPhase& thermo, Transport& transport, size_t fuelIndex,
                          double dropletMass, double dropletTemperature,
                          double dropletVelocity, double gasVelocity,
                          double dropletSpreadRate, double gasSpreadRate) const;

private:
    double surfaceFuelMassFraction(ThermoPhase& thermo, size_t fuelIndex,
                                   double dropletTemperature) const;
    static double filmCorrection(double transferNumber);

    LiquidFuelProperties m_liquid;
};

}

#endif
