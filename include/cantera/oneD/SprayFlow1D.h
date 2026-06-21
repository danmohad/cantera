//! @file SprayFlow1D.h

// This file is part of Cantera. See License.txt in the top-level directory or
// at https://cantera.org/license.txt for license and copyright information.

#ifndef CT_SPRAYFLOW1D_H
#define CT_SPRAYFLOW1D_H

#include "Flow1D.h"
#include "SprayModel.h"

namespace Cantera
{

//! One-dimensional gas flow with a monodisperse evaporating liquid spray.
class SprayFlow1D : public Flow1D
{
public:
    //! Create a spray flow domain.
    SprayFlow1D(shared_ptr<Solution> phase, const string& id="", size_t points=1);

    string domainType() const override;

    //! Set the evaporating gas-phase fuel species.
    void setSprayFuel(const string& species);

    //! Gas-phase fuel species used by the spray model.
    string sprayFuel() const;

    //! Set liquid properties used by the droplet model.
    void setLiquidProperties(double density, double cp, double latentHeat,
                             double boilingTemperature,
                             double saturationPressure=OneAtm);

    //! Set inlet droplet diameter [m].
    void setDropletDiameter(double diameter);

    //! Set the minimum droplet diameter before droplets are treated as dry [m].
    void setMinimumDropletDiameter(double diameter);

    //! Minimum droplet diameter before droplets are treated as dry [m].
    double minimumDropletDiameter() const {
        return m_minDropletDiameter;
    }

    //! Set inlet liquid mass density [kg/m^3].
    void setLiquidMassDensity(double density);

    //! Set inlet liquid mass flux [kg/m^2/s].
    void setLiquidMassFlux(double massFlux);

    //! Set inlet liquid temperature [K].
    void setLiquidTemperature(double temperature);

    //! Set inlet droplet velocity [m/s].
    void setDropletVelocity(double velocity);

    //! Set inlet droplet spread rate [1/s].
    void setDropletSpreadRate(double spreadRate);

    //! Select the spray inlet side: `0` for left, `1` for right.
    void setSprayInlet(int side);

    //! Return spray inlet side: `0` for left, `1` for right.
    int sprayInlet() const {
        return m_inletSide;
    }

    //! Set whether droplets are constrained to no slip in free-flow flames.
    void setFreeFlowNoSlip(bool noSlip) {
        m_freeFlowNoSlip = noSlip;
        needJacUpdate();
    }

    //! Return whether free-flow droplets are constrained to no slip.
    bool freeFlowNoSlip() const {
        return m_freeFlowNoSlip;
    }

    //! Set whether droplet axial reversal raises an error.
    void setDropletReversalCheck(bool check) {
        m_checkDropletReversal = check;
        setSprayBounds();
        needJacUpdate();
    }

    //! Return whether droplet axial reversal raises an error.
    bool dropletReversalCheck() const {
        return m_checkDropletReversal;
    }

    //! Enable or disable all spray source terms in the gas-phase equations.
    void setGasPhaseSpraySourcesEnabled(bool enabled) {
        m_enableGasPhaseMassSource = enabled;
        m_enableGasPhaseSpeciesSource = enabled;
        m_enableGasPhaseEnergySource = enabled;
        m_enableGasPhaseMomentumSource = enabled;
        needJacUpdate();
    }

    //! Return whether all spray source terms are enabled in gas equations.
    bool gasPhaseSpraySourcesEnabled() const {
        return m_enableGasPhaseMassSource && m_enableGasPhaseSpeciesSource
            && m_enableGasPhaseEnergySource && m_enableGasPhaseMomentumSource;
    }

    //! Enable or disable the spray continuity source in the gas phase.
    void setGasPhaseSprayMassSourceEnabled(bool enabled) {
        m_enableGasPhaseMassSource = enabled;
        needJacUpdate();
    }

    //! Return whether the spray continuity source is enabled in the gas phase.
    bool gasPhaseSprayMassSourceEnabled() const {
        return m_enableGasPhaseMassSource;
    }

    //! Enable or disable spray species source terms in the gas phase.
    void setGasPhaseSpraySpeciesSourceEnabled(bool enabled) {
        m_enableGasPhaseSpeciesSource = enabled;
        needJacUpdate();
    }

    //! Return whether spray species source terms are enabled in the gas phase.
    bool gasPhaseSpraySpeciesSourceEnabled() const {
        return m_enableGasPhaseSpeciesSource;
    }

    //! Enable or disable the spray energy source in the gas phase.
    void setGasPhaseSprayEnergySourceEnabled(bool enabled) {
        m_enableGasPhaseEnergySource = enabled;
        needJacUpdate();
    }

    //! Return whether the spray energy source is enabled in the gas phase.
    bool gasPhaseSprayEnergySourceEnabled() const {
        return m_enableGasPhaseEnergySource;
    }

    //! Enable or disable the spray momentum source in the gas phase.
    void setGasPhaseSprayMomentumSourceEnabled(bool enabled) {
        m_enableGasPhaseMomentumSource = enabled;
        needJacUpdate();
    }

    //! Return whether the spray momentum source is enabled in the gas phase.
    bool gasPhaseSprayMomentumSourceEnabled() const {
        return m_enableGasPhaseMomentumSource;
    }

    //! Enable or disable all source terms in the droplet equations.
    void setDropletSourcesEnabled(bool enabled) {
        m_enableDropletEvaporation = enabled;
        m_enableDropletHeatTransfer = enabled;
        m_enableDropletAxialDrag = enabled;
        m_enableDropletSpreadDrag = enabled;
        setSprayBounds();
        needJacUpdate();
    }

    //! Return whether all droplet-equation source terms are enabled.
    bool dropletSourcesEnabled() const {
        return m_enableDropletEvaporation && m_enableDropletHeatTransfer
            && m_enableDropletAxialDrag && m_enableDropletSpreadDrag;
    }

    //! Enable or disable evaporation terms in liquid density and droplet mass.
    void setDropletEvaporationEnabled(bool enabled) {
        m_enableDropletEvaporation = enabled;
        needJacUpdate();
    }

    //! Return whether evaporation terms are enabled in droplet equations.
    bool dropletEvaporationEnabled() const {
        return m_enableDropletEvaporation;
    }

    //! Enable or disable heat transfer in the droplet temperature equation.
    void setDropletHeatTransferEnabled(bool enabled) {
        m_enableDropletHeatTransfer = enabled;
        needJacUpdate();
    }

    //! Return whether heat transfer is enabled in droplet equations.
    bool dropletHeatTransferEnabled() const {
        return m_enableDropletHeatTransfer;
    }

    //! Enable or disable drag in droplet velocity and spread-rate equations.
    void setDropletDragEnabled(bool enabled) {
        m_enableDropletAxialDrag = enabled;
        m_enableDropletSpreadDrag = enabled;
        setSprayBounds();
        needJacUpdate();
    }

    //! Return whether drag is enabled in droplet equations.
    bool dropletDragEnabled() const {
        return m_enableDropletAxialDrag && m_enableDropletSpreadDrag;
    }

    //! Enable or disable axial drag in the droplet velocity equation.
    void setDropletAxialDragEnabled(bool enabled) {
        m_enableDropletAxialDrag = enabled;
        setSprayBounds();
        needJacUpdate();
    }

    //! Return whether axial drag is enabled in the droplet velocity equation.
    bool dropletAxialDragEnabled() const {
        return m_enableDropletAxialDrag;
    }

    //! Set multiplier for axial drag in the droplet velocity equation.
    void setDropletAxialDragMultiplier(double multiplier) {
        if (multiplier < 0.0) {
            throw CanteraError("SprayFlow1D::setDropletAxialDragMultiplier",
                "The axial drag multiplier must be non-negative.");
        }
        m_dropletAxialDragMultiplier = multiplier;
        needJacUpdate();
    }

    //! Multiplier for axial drag in the droplet velocity equation.
    double dropletAxialDragMultiplier() const {
        return m_dropletAxialDragMultiplier;
    }

    //! Enable or disable radial drag in the droplet spread-rate equation.
    void setDropletSpreadDragEnabled(bool enabled) {
        m_enableDropletSpreadDrag = enabled;
        setSprayBounds();
        needJacUpdate();
    }

    //! Return whether radial drag is enabled in the droplet spread-rate equation.
    bool dropletSpreadDragEnabled() const {
        return m_enableDropletSpreadDrag;
    }

    //! Set multiplier for radial drag in the droplet spread-rate equation.
    void setDropletSpreadDragMultiplier(double multiplier) {
        if (multiplier < 0.0) {
            throw CanteraError("SprayFlow1D::setDropletSpreadDragMultiplier",
                "The spread-rate drag multiplier must be non-negative.");
        }
        m_dropletSpreadDragMultiplier = multiplier;
        needJacUpdate();
    }

    //! Multiplier for radial drag in the droplet spread-rate equation.
    double dropletSpreadDragMultiplier() const {
        return m_dropletSpreadDragMultiplier;
    }

    //! Volumetric evaporation source term [kg/m^3/s].
    double sprayEvaporationRate(size_t j) const {
        return m_sprayMassSource[j];
    }

    //! Heat transferred from the gas to the liquid [W/m^3].
    double sprayHeatTransferRate(size_t j) const {
        return m_sprayHeatTransfer[j];
    }

    //! Gas energy source term induced by the spray [W/m^3].
    double sprayGasEnergySource(size_t j) const {
        return m_sprayEnergySource[j];
    }

    //! Gas radial momentum source term induced by the spray [N/m^3].
    double sprayGasMomentumSource(size_t j) const {
        return m_sprayMomentumSource[j];
    }

    //! Droplet diameter at a grid point [m].
    double dropletDiameter(size_t j) const {
        return m_sprayModel.diameter(dropletMass(j));
    }

    //! Droplet Reynolds number.
    double dropletReynoldsNumber(size_t j) const {
        return m_reynoldsNumber[j];
    }

    //! Droplet Nusselt number.
    double dropletNusseltNumber(size_t j) const {
        return m_nusseltNumber[j];
    }

    //! Droplet Sherwood number.
    double dropletSherwoodNumber(size_t j) const {
        return m_sherwoodNumber[j];
    }

    void resize(size_t components, size_t points) override;
    void resetBadValues(span<double> xg) override;
    void _getInitialSoln(span<double> x) override;

    string componentName(size_t n) const override;
    size_t componentIndex(const string& name, bool checkAlias=true) const override;
    bool hasComponent(const string& name, bool checkAlias=true) const override;
    bool componentActive(size_t n) const override;

protected:
    AnyMap getMeta() const override;
    void setMeta(const AnyMap& state) override;
    void updateProperties(size_t jg, span<const double> x,
                          size_t jmin, size_t jmax) override;
    void evalAdditionalEquations(span<const double> x, span<double> rsd,
                                 span<int> diag, double rdt, size_t jmin,
                                 size_t jmax) override;
    double continuitySource(span<const double> x, size_t j) const override;
    double momentumSource(span<const double> x, size_t j) const override;
    double energySource(span<const double> x, size_t j) const override;
    double speciesSource(span<const double> x, size_t k, size_t j) const override;

    size_t liquidMassDensityIndex() const {
        return c_offset_Y + m_nsp;
    }

    size_t dropletMassIndex() const {
        return liquidMassDensityIndex() + 1;
    }

    size_t dropletVelocityIndex() const {
        return liquidMassDensityIndex() + 2;
    }

    size_t dropletSpreadRateIndex() const {
        return liquidMassDensityIndex() + 3;
    }

    size_t dropletTemperatureIndex() const {
        return liquidMassDensityIndex() + 4;
    }

    double liquidMassDensity(span<const double> x, size_t j) const {
        return x[index(liquidMassDensityIndex(), j)];
    }

    double dropletMass(span<const double> x, size_t j) const {
        return x[index(dropletMassIndex(), j)];
    }

    double dropletVelocity(span<const double> x, size_t j) const {
        return x[index(dropletVelocityIndex(), j)];
    }

    double dropletSpreadRate(span<const double> x, size_t j) const {
        return x[index(dropletSpreadRateIndex(), j)];
    }

    double dropletTemperature(span<const double> x, size_t j) const {
        return x[index(dropletTemperatureIndex(), j)];
    }

    double dropletMass(size_t j) const;
    double dropletDerivative(span<const double> x, size_t component, size_t j) const;
    double dropletVelocityGradientTerm(span<const double> x, size_t j) const;
    double liquidMassFlux(span<const double> x, size_t j) const;
    double minimumDropletMass() const;
    bool dropletLoadingIsEmpty(span<const double> x, size_t j) const;
    bool dropletIsDry(span<const double> x, size_t j) const;
    void checkDropletReversal(span<const double> x, size_t j) const;
    void updateSpraySources(span<const double> x, size_t jmin, size_t jmax);
    void setSprayBounds();
    void checkSprayReady() const;
    double inletDropletVelocity(span<const double> x, size_t j) const;

    static constexpr size_t nposFuel = npos;

    MonodisperseSprayModel m_sprayModel;
    size_t m_fuelIndex = nposFuel;
    double m_inletDropletDiameter = Undef;
    double m_inletDropletMass = Undef;
    double m_inletLiquidMassDensity = Undef;
    double m_inletLiquidMassFlux = Undef;
    double m_inletLiquidTemperature = Undef;
    double m_inletDropletVelocity = Undef;
    double m_inletDropletSpreadRate = 0.0;
    double m_minDropletDiameter = 1e-7;
    double m_reversalTolerance = 1e-12;
    int m_inletSide = 0;
    bool m_freeFlowNoSlip = true;
    bool m_checkDropletReversal = true;
    bool m_enableGasPhaseMassSource = true;
    bool m_enableGasPhaseSpeciesSource = true;
    bool m_enableGasPhaseEnergySource = true;
    bool m_enableGasPhaseMomentumSource = true;
    bool m_enableDropletEvaporation = true;
    bool m_enableDropletHeatTransfer = true;
    bool m_enableDropletAxialDrag = true;
    bool m_enableDropletSpreadDrag = true;
    double m_dropletAxialDragMultiplier = 1.0;
    double m_dropletSpreadDragMultiplier = 1.0;

    vector<double> m_sprayMassSource;
    vector<double> m_sprayHeatTransfer;
    vector<double> m_sprayEnergySource;
    vector<double> m_sprayMomentumSource;
    vector<double> m_dropletAxialDrag;
    vector<double> m_dropletSpreadDrag;
    vector<double> m_reynoldsNumber;
    vector<double> m_nusseltNumber;
    vector<double> m_sherwoodNumber;
};

}

#endif
