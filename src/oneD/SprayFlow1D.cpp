//! @file SprayFlow1D.cpp

// This file is part of Cantera. See License.txt in the top-level directory or
// at https://cantera.org/license.txt for license and copyright information.

#include "cantera/oneD/SprayFlow1D.h"
#include "cantera/oneD/refine.h"
#include "cantera/base/SolutionArray.h"
#include "cantera/transport/Transport.h"

#include <algorithm>

using namespace std;

namespace Cantera
{

namespace
{

constexpr double MinDropletTemperature = 1.0;

string normalizeComponentName(const string& name)
{
    if (name == "liquid_mass_density") {
        return "liquid-mass-density";
    } else if (name == "droplet_mass") {
        return "droplet-mass";
    } else if (name == "droplet_velocity") {
        return "droplet-velocity";
    } else if (name == "droplet_spread_rate") {
        return "droplet-spread-rate";
    } else if (name == "droplet_temperature") {
        return "droplet-temperature";
    }
    return name;
}

}

SprayFlow1D::SprayFlow1D(shared_ptr<Solution> phase, const string& id, size_t points)
    : Flow1D(phase, id, points)
{
    m_nv = c_offset_Y + m_nsp + 5;
    vector<double> grid(m_z.begin(), m_z.end());
    setupGrid(grid);
    setSprayBounds();
}

string SprayFlow1D::domainType() const
{
    if (m_isFree) {
        return "spray-free-flow";
    }
    if (m_usesLambda) {
        return "spray-axisymmetric-flow";
    }
    return "spray-unstrained-flow";
}

void SprayFlow1D::setSprayFuel(const string& species)
{
    size_t k = m_thermo->speciesIndex(species);
    if (k == npos) {
        throw CanteraError("SprayFlow1D::setSprayFuel",
            "Species '{}' is not present in the gas phase.", species);
    }
    m_fuelIndex = k;
    needJacUpdate();
}

string SprayFlow1D::sprayFuel() const
{
    if (m_fuelIndex == npos) {
        return "";
    }
    return m_thermo->speciesName(m_fuelIndex);
}

void SprayFlow1D::setLiquidProperties(double density, double cp, double latentHeat,
                                      double boilingTemperature,
                                      double saturationPressure)
{
    LiquidFuelProperties properties;
    properties.density = density;
    properties.cp = cp;
    properties.latentHeat = latentHeat;
    properties.boilingTemperature = boilingTemperature;
    properties.saturationPressure = saturationPressure;
    m_sprayModel.setLiquidProperties(properties);
    if (m_inletDropletDiameter != Undef) {
        m_inletDropletMass = m_sprayModel.dropletMass(m_inletDropletDiameter);
    }
    setSprayBounds();
    needJacUpdate();
}

void SprayFlow1D::setDropletDiameter(double diameter)
{
    if (diameter <= 0.0) {
        throw CanteraError("SprayFlow1D::setDropletDiameter",
            "Droplet diameter must be positive.");
    }
    if (diameter <= m_minDropletDiameter) {
        throw CanteraError("SprayFlow1D::setDropletDiameter",
            "Droplet diameter must be greater than the minimum droplet diameter.");
    }
    m_inletDropletDiameter = diameter;
    m_inletDropletMass = m_sprayModel.dropletMass(diameter);
    setSprayBounds();
    needJacUpdate();
}

void SprayFlow1D::setMinimumDropletDiameter(double diameter)
{
    if (diameter <= 0.0) {
        throw CanteraError("SprayFlow1D::setMinimumDropletDiameter",
            "Minimum droplet diameter must be positive.");
    }
    if (m_inletDropletDiameter != Undef && diameter >= m_inletDropletDiameter) {
        throw CanteraError("SprayFlow1D::setMinimumDropletDiameter",
            "Minimum droplet diameter must be smaller than the inlet droplet diameter.");
    }
    m_minDropletDiameter = diameter;
    setSprayBounds();
    needJacUpdate();
}

void SprayFlow1D::setLiquidMassDensity(double density)
{
    if (density < 0.0) {
        throw CanteraError("SprayFlow1D::setLiquidMassDensity",
            "Liquid mass density must be non-negative.");
    }
    m_inletLiquidMassDensity = density;
    m_inletLiquidMassFlux = Undef;
    needJacUpdate();
}

void SprayFlow1D::setLiquidMassFlux(double massFlux)
{
    if (massFlux < 0.0) {
        throw CanteraError("SprayFlow1D::setLiquidMassFlux",
            "Liquid mass flux must be non-negative.");
    }
    m_inletLiquidMassFlux = massFlux;
    m_inletLiquidMassDensity = Undef;
    needJacUpdate();
}

void SprayFlow1D::setLiquidTemperature(double temperature)
{
    if (temperature <= 0.0) {
        throw CanteraError("SprayFlow1D::setLiquidTemperature",
            "Liquid temperature must be positive.");
    }
    m_inletLiquidTemperature = temperature;
    needJacUpdate();
}

void SprayFlow1D::setDropletVelocity(double velocity)
{
    m_inletDropletVelocity = velocity;
    needJacUpdate();
}

void SprayFlow1D::setDropletSpreadRate(double spreadRate)
{
    m_inletDropletSpreadRate = spreadRate;
    needJacUpdate();
}

void SprayFlow1D::setSprayInlet(int side)
{
    if (side != 0 && side != 1) {
        throw CanteraError("SprayFlow1D::setSprayInlet",
            "Spray inlet side must be 0 (left) or 1 (right).");
    }
    m_inletSide = side;
    setSprayBounds();
    needJacUpdate();
}

void SprayFlow1D::resize(size_t components, size_t points)
{
    Flow1D::resize(components, points);
    m_sprayMassSource.resize(m_points, 0.0);
    m_sprayHeatTransfer.resize(m_points, 0.0);
    m_sprayEnergySource.resize(m_points, 0.0);
    m_sprayMomentumSource.resize(m_points, 0.0);
    m_dropletAxialDrag.resize(m_points, 0.0);
    m_dropletSpreadDrag.resize(m_points, 0.0);
    m_reynoldsNumber.resize(m_points, 0.0);
    m_nusseltNumber.resize(m_points, 2.0);
    m_sherwoodNumber.resize(m_points, 2.0);
}

void SprayFlow1D::setSprayBounds()
{
    setBounds(liquidMassDensityIndex(), 0.0, 1e20);
    setBounds(dropletMassIndex(), minimumDropletMass(), 1e20);
    if (!m_checkDropletReversal) {
        setBounds(dropletVelocityIndex(), -1e20, 1e20);
    } else if (m_inletSide == 0) {
        setBounds(dropletVelocityIndex(), 0.0, 1e20);
    } else {
        setBounds(dropletVelocityIndex(), -1e20, 0.0);
    }
    setBounds(dropletSpreadRateIndex(), -1e20, 1e20);
    double maxDropletTemperature = 2*m_thermo->maxTemp();
    const auto& liquid = m_sprayModel.liquidProperties();
    if (liquid.boilingTemperature != Undef && liquid.boilingTemperature > 0.0) {
        maxDropletTemperature = liquid.boilingTemperature;
    }
    setBounds(dropletTemperatureIndex(), MinDropletTemperature, maxDropletTemperature);

    setSteadyTolerances(1e-5, 1e-12, liquidMassDensityIndex());
    setSteadyTolerances(1e-5, 1e-18, dropletMassIndex());
    setSteadyTolerances(1e-5, 1e-9, dropletVelocityIndex());
    setSteadyTolerances(1e-5, 1e-9, dropletSpreadRateIndex());
    setSteadyTolerances(1e-5, 1e-8, dropletTemperatureIndex());

    // Liquid mass density drops to zero at the numerical dryout cutoff. Let
    // droplet mass, velocity, temperature, and gas fields drive refinement
    // rather than chasing this artificial discontinuity.
    m_refiner->setActive(liquidMassDensityIndex(), false);
    m_refiner->setActive(dropletMassIndex(), true);
    m_refiner->setActive(dropletVelocityIndex(), m_enableDropletAxialDrag);
    // Droplet spread rate is an auxiliary inlet-marched quantity and is only a
    // placeholder after dryout. Other droplet fields and gas variables provide
    // more robust refinement targets for the coupled solution.
    m_refiner->setActive(dropletSpreadRateIndex(), false);
    m_refiner->setActive(dropletTemperatureIndex(), true);
}

void SprayFlow1D::checkSprayReady() const
{
    if (m_fuelIndex == npos) {
        throw CanteraError("SprayFlow1D::checkSprayReady",
            "No spray fuel species has been specified.");
    }
    if (m_inletDropletMass == Undef) {
        throw CanteraError("SprayFlow1D::checkSprayReady",
            "No inlet droplet diameter has been specified.");
    }
    if (m_inletLiquidTemperature == Undef) {
        throw CanteraError("SprayFlow1D::checkSprayReady",
            "No inlet liquid temperature has been specified.");
    }
    if (m_inletLiquidMassDensity == Undef && m_inletLiquidMassFlux == Undef) {
        throw CanteraError("SprayFlow1D::checkSprayReady",
            "Specify either inlet liquid mass density or inlet liquid mass flux.");
    }
    if (m_inletDropletDiameter <= m_minDropletDiameter) {
        throw CanteraError("SprayFlow1D::checkSprayReady",
            "The inlet droplet diameter must be greater than the minimum droplet "
            "diameter.");
    }
}

void SprayFlow1D::resetBadValues(span<double> xg)
{
    Flow1D::resetBadValues(xg);
    span<double> x = xg.subspan(loc(), size());
    for (size_t j = 0; j < m_points; j++) {
        x[index(liquidMassDensityIndex(), j)] =
            std::max(x[index(liquidMassDensityIndex(), j)], 0.0);
        x[index(dropletMassIndex(), j)] =
            std::max(x[index(dropletMassIndex(), j)], minimumDropletMass());
        x[index(dropletTemperatureIndex(), j)] =
            std::max(x[index(dropletTemperatureIndex(), j)], MinDropletTemperature);
    }
}

void SprayFlow1D::_getInitialSoln(span<double> x)
{
    checkSprayReady();
    Flow1D::_getInitialSoln(x);

    for (size_t j = 0; j < m_points; j++) {
        x[index(dropletMassIndex(), j)] = m_inletDropletMass;
        x[index(dropletTemperatureIndex(), j)] = m_inletLiquidTemperature;
        x[index(dropletSpreadRateIndex(), j)] = m_inletDropletSpreadRate;

        double ud = inletDropletVelocity(x, j);
        x[index(dropletVelocityIndex(), j)] = ud;
        if (m_inletLiquidMassDensity != Undef) {
            x[index(liquidMassDensityIndex(), j)] = m_inletLiquidMassDensity;
        } else {
            x[index(liquidMassDensityIndex(), j)] =
                m_inletLiquidMassFlux / std::max(fabs(ud), 1e-12);
        }
    }
}

string SprayFlow1D::componentName(size_t n) const
{
    if (n == liquidMassDensityIndex()) {
        return "liquid-mass-density";
    } else if (n == dropletMassIndex()) {
        return "droplet-mass";
    } else if (n == dropletVelocityIndex()) {
        return "droplet-velocity";
    } else if (n == dropletSpreadRateIndex()) {
        return "droplet-spread-rate";
    } else if (n == dropletTemperatureIndex()) {
        return "droplet-temperature";
    }
    return Flow1D::componentName(n);
}

size_t SprayFlow1D::componentIndex(const string& name, bool checkAlias) const
{
    string normalized = checkAlias ? normalizeComponentName(name) : name;
    for (size_t n = liquidMassDensityIndex(); n < nComponents(); n++) {
        if (componentName(n) == normalized) {
            return n;
        }
    }
    return Flow1D::componentIndex(name, checkAlias);
}

bool SprayFlow1D::hasComponent(const string& name, bool checkAlias) const
{
    string normalized = checkAlias ? normalizeComponentName(name) : name;
    for (size_t n = liquidMassDensityIndex(); n < nComponents(); n++) {
        if (componentName(n) == normalized) {
            return true;
        }
    }
    return Flow1D::hasComponent(name, checkAlias);
}

bool SprayFlow1D::componentActive(size_t n) const
{
    if (n == dropletSpreadRateIndex()) {
        return isStrained();
    }
    if (n >= liquidMassDensityIndex() && n < nComponents()) {
        return true;
    }
    return Flow1D::componentActive(n);
}

AnyMap SprayFlow1D::getMeta() const
{
    AnyMap state = Flow1D::getMeta();
    AnyMap spray;
    spray["fuel-species"] = sprayFuel();
    spray["droplet-diameter"] = m_inletDropletDiameter;
    spray["minimum-droplet-diameter"] = m_minDropletDiameter;
    spray["liquid-mass-density"] = m_inletLiquidMassDensity;
    spray["liquid-mass-flux"] = m_inletLiquidMassFlux;
    spray["liquid-temperature"] = m_inletLiquidTemperature;
    spray["droplet-velocity"] = m_inletDropletVelocity;
    spray["droplet-spread-rate"] = m_inletDropletSpreadRate;
    spray["inlet-side"] = m_inletSide;
    spray["free-flow-no-slip"] = m_freeFlowNoSlip;
    spray["droplet-reversal-check"] = m_checkDropletReversal;
    spray["gas-phase-mass-source"] = m_enableGasPhaseMassSource;
    spray["gas-phase-species-source"] = m_enableGasPhaseSpeciesSource;
    spray["gas-phase-energy-source"] = m_enableGasPhaseEnergySource;
    spray["gas-phase-momentum-source"] = m_enableGasPhaseMomentumSource;
    spray["droplet-evaporation"] = m_enableDropletEvaporation;
    spray["droplet-heat-transfer"] = m_enableDropletHeatTransfer;
    spray["droplet-axial-drag"] = m_enableDropletAxialDrag;
    spray["droplet-spread-drag"] = m_enableDropletSpreadDrag;
    spray["droplet-axial-drag-multiplier"] = m_dropletAxialDragMultiplier;
    spray["droplet-spread-drag-multiplier"] = m_dropletSpreadDragMultiplier;

    const auto& liquid = m_sprayModel.liquidProperties();
    spray["liquid-density"] = liquid.density;
    spray["liquid-cp"] = liquid.cp;
    spray["latent-heat"] = liquid.latentHeat;
    spray["boiling-temperature"] = liquid.boilingTemperature;
    spray["saturation-pressure"] = liquid.saturationPressure;
    state["spray"] = spray;
    return state;
}

void SprayFlow1D::setMeta(const AnyMap& state)
{
    Flow1D::setMeta(state);
    if (!state.hasKey("spray")) {
        return;
    }
    const AnyMap& spray = state["spray"].as<AnyMap>();
    setSprayFuel(spray["fuel-species"].asString());
    setLiquidProperties(spray["liquid-density"].asDouble(),
                        spray["liquid-cp"].asDouble(),
                        spray["latent-heat"].asDouble(),
                        spray["boiling-temperature"].asDouble(),
                        spray.getDouble("saturation-pressure", OneAtm));
    setMinimumDropletDiameter(spray.getDouble("minimum-droplet-diameter", 1e-7));
    setDropletDiameter(spray["droplet-diameter"].asDouble());
    if (spray.hasKey("liquid-mass-density")
        && spray["liquid-mass-density"].asDouble() != Undef) {
        setLiquidMassDensity(spray["liquid-mass-density"].asDouble());
    }
    if (spray.hasKey("liquid-mass-flux")
        && spray["liquid-mass-flux"].asDouble() != Undef) {
        setLiquidMassFlux(spray["liquid-mass-flux"].asDouble());
    }
    setLiquidTemperature(spray["liquid-temperature"].asDouble());
    if (spray.hasKey("droplet-velocity")
        && spray["droplet-velocity"].asDouble() != Undef) {
        setDropletVelocity(spray["droplet-velocity"].asDouble());
    }
    setDropletSpreadRate(spray.getDouble("droplet-spread-rate", 0.0));
    setSprayInlet(spray.getInt("inlet-side", 0));
    m_freeFlowNoSlip = spray.getBool("free-flow-no-slip", true);
    m_checkDropletReversal = spray.getBool("droplet-reversal-check", true);
    m_enableGasPhaseMassSource = spray.getBool("gas-phase-mass-source", true);
    m_enableGasPhaseSpeciesSource = spray.getBool("gas-phase-species-source", true);
    m_enableGasPhaseEnergySource = spray.getBool("gas-phase-energy-source", true);
    m_enableGasPhaseMomentumSource = spray.getBool("gas-phase-momentum-source", true);
    m_enableDropletEvaporation = spray.getBool("droplet-evaporation", true);
    m_enableDropletHeatTransfer = spray.getBool("droplet-heat-transfer", true);
    bool dropletDrag = spray.getBool("droplet-drag", true);
    m_enableDropletAxialDrag = spray.getBool("droplet-axial-drag", dropletDrag);
    m_enableDropletSpreadDrag = spray.getBool("droplet-spread-drag", dropletDrag);
    m_dropletAxialDragMultiplier =
        spray.getDouble("droplet-axial-drag-multiplier", 1.0);
    m_dropletSpreadDragMultiplier =
        spray.getDouble("droplet-spread-drag-multiplier", 1.0);
    setSprayBounds();
    needJacUpdate();
}

void SprayFlow1D::updateProperties(size_t jg, span<const double> x,
                                   size_t jmin, size_t jmax)
{
    Flow1D::updateProperties(jg, x, jmin, jmax);
    updateSpraySources(x, jmin, jmax);
}

void SprayFlow1D::updateSpraySources(span<const double> x, size_t jmin, size_t jmax)
{
    if (m_fuelIndex == npos) {
        return;
    }
    for (size_t j = jmin; j <= jmax; j++) {
        m_sprayMassSource[j] = 0.0;
        m_sprayHeatTransfer[j] = 0.0;
        m_sprayEnergySource[j] = 0.0;
        m_sprayMomentumSource[j] = 0.0;
        m_dropletAxialDrag[j] = 0.0;
        m_dropletSpreadDrag[j] = 0.0;
        m_reynoldsNumber[j] = 0.0;
        m_nusseltNumber[j] = 2.0;
        m_sherwoodNumber[j] = 2.0;

        double mass = dropletMass(x, j);
        double rhoLiquid = liquidMassDensity(x, j);
        if (mass <= 0.0 || dropletIsDry(x, j) || dropletLoadingIsEmpty(x, j)) {
            continue;
        }

        const auto& liquid = m_sprayModel.liquidProperties();
        double dropletTemp = std::min(dropletTemperature(x, j),
                                      liquid.boilingTemperature);

        setGas(x, j);
        auto result = m_sprayModel.eval(*m_thermo, *m_trans, m_fuelIndex, mass,
            dropletTemp, dropletVelocity(x, j), u(x, j),
            dropletSpreadRate(x, j), V(x, j));

        double minMass = minimumDropletMass();
        double ud = dropletVelocity(x, j);
        if (minMass > 0.0 && result.evaporationRate > 0.0 && ud != 0.0) {
            double dz = m_dz.front();
            double upstreamMass = mass;
            if (ud > 0.0 && j > 0) {
                dz = m_dz[j - 1];
                upstreamMass = dropletMass(x, j - 1);
            } else if (ud < 0.0 && j + 1 < m_points) {
                dz = m_dz[j];
                upstreamMass = dropletMass(x, j + 1);
            } else if (j > 0) {
                dz = m_dz[j - 1];
            }
            double availableMass = std::max(upstreamMass - minMass, 0.0);
            double maxEvaporationRate = availableMass * fabs(ud) / std::max(dz, Tiny);
            if (result.evaporationRate > maxEvaporationRate) {
                double limitedEvaporationRate = result.evaporationRate
                    * maxEvaporationRate
                    / (result.evaporationRate + maxEvaporationRate);
                double scale = limitedEvaporationRate / result.evaporationRate;
                result.evaporationRate = limitedEvaporationRate;
                result.heatTransferRate *= scale;
                result.dragAxial *= scale;
                result.dragSpreadRate *= scale;
            }
        }

        double numberDensity = rhoLiquid / mass;
        double massSource = numberDensity * result.evaporationRate;

        vector<double> cpbar(m_nsp);
        m_thermo->getPartialMolarCp(span<double>(cpbar.data(), cpbar.size()));
        double cpFuel = cpbar[m_fuelIndex] / m_wt[m_fuelIndex];

        m_sprayMassSource[j] = massSource;
        m_sprayHeatTransfer[j] = numberDensity * result.heatTransferRate;
        m_sprayEnergySource[j] = m_sprayMassSource[j] * cpFuel
            * (dropletTemperature(x, j) - T(x, j)) - m_sprayHeatTransfer[j];
        m_sprayMomentumSource[j] = m_sprayMassSource[j]
            * (dropletSpreadRate(x, j) - V(x, j))
            - numberDensity * result.dragSpreadRate;
        m_dropletAxialDrag[j] = result.dragAxial;
        m_dropletSpreadDrag[j] = result.dragSpreadRate;
        m_reynoldsNumber[j] = result.reynoldsNumber;
        m_nusseltNumber[j] = result.nusseltNumber;
        m_sherwoodNumber[j] = result.sherwoodNumber;
    }
}

double SprayFlow1D::continuitySource(span<const double> x, size_t j) const
{
    if (!m_enableGasPhaseMassSource) {
        return 0.0;
    }
    return m_sprayMassSource[j];
}

double SprayFlow1D::momentumSource(span<const double> x, size_t j) const
{
    if (!isStrained() || !m_enableGasPhaseMomentumSource) {
        return 0.0;
    }
    return m_sprayMomentumSource[j];
}

double SprayFlow1D::energySource(span<const double> x, size_t j) const
{
    if (!m_enableGasPhaseEnergySource) {
        return 0.0;
    }
    return m_sprayEnergySource[j];
}

double SprayFlow1D::speciesSource(span<const double> x, size_t k, size_t j) const
{
    if (!m_enableGasPhaseSpeciesSource) {
        return 0.0;
    }
    if (k == m_fuelIndex) {
        return m_sprayMassSource[j] * (1.0 - Y(x, k, j));
    }
    return -m_sprayMassSource[j] * Y(x, k, j);
}

double SprayFlow1D::dropletMass(size_t j) const
{
    if (!m_state) {
        return m_inletDropletMass;
    }
    const double* soln = m_state->data() + m_iloc;
    return soln[index(dropletMassIndex(), j)];
}

double SprayFlow1D::dropletDerivative(span<const double> x, size_t component,
                                      size_t j) const
{
    checkDropletReversal(x, j);
    double ud = dropletVelocity(x, j);
    size_t jloc = (ud > 0.0 ? j : j + 1);
    return (x[index(component, jloc)] - x[index(component, jloc-1)])
        / m_dz[jloc-1];
}

double SprayFlow1D::dropletVelocityGradientTerm(span<const double> x, size_t j) const
{
    checkDropletReversal(x, j);
    double ud = dropletVelocity(x, j);
    size_t jloc = (ud > 0.0 ? j : j + 1);
    double uRight = dropletVelocity(x, jloc);
    double uLeft = dropletVelocity(x, jloc - 1);
    return 0.5 * (uRight * uRight - uLeft * uLeft) / m_dz[jloc - 1];
}

double SprayFlow1D::liquidMassFlux(span<const double> x, size_t j) const
{
    return liquidMassDensity(x, j) * dropletVelocity(x, j);
}

double SprayFlow1D::minimumDropletMass() const
{
    const auto& liquid = m_sprayModel.liquidProperties();
    if (m_minDropletDiameter <= 0.0 || liquid.density <= 0.0) {
        return 0.0;
    }
    return m_sprayModel.dropletMass(m_minDropletDiameter);
}

bool SprayFlow1D::dropletIsDry(span<const double> x, size_t j) const
{
    double minMass = minimumDropletMass();
    if (minMass <= 0.0) {
        return false;
    }
    double mass = dropletMass(x, j);
    if (mass <= minMass * (1.0 + 1e-8)) {
        return true;
    }
    if (dropletLoadingIsEmpty(x, j) && m_inletDropletMass != Undef) {
        double emptyDryMass = std::max(minMass * (1.0 + 1e-8),
                                       1e-4 * m_inletDropletMass);
        return mass <= emptyDryMass;
    }
    return false;
}

bool SprayFlow1D::dropletLoadingIsEmpty(span<const double> x, size_t j) const
{
    double rhoLiquid = liquidMassDensity(x, j);
    if (rhoLiquid <= 0.0) {
        return true;
    }
    double reference = 0.0;
    if (m_inletLiquidMassDensity != Undef && m_inletLiquidMassDensity > 0.0) {
        reference = m_inletLiquidMassDensity;
    } else if (m_inletLiquidMassFlux != Undef && m_inletLiquidMassFlux > 0.0) {
        reference = m_inletLiquidMassFlux;
    }
    return reference > 0.0 && rhoLiquid <= 1e-30 * reference;
}

void SprayFlow1D::checkDropletReversal(span<const double> x, size_t j) const
{
    if (!m_checkDropletReversal || dropletIsDry(x, j)
        || dropletLoadingIsEmpty(x, j)) {
        return;
    }
    double direction = m_inletSide == 0 ? 1.0 : -1.0;
    if (direction * dropletVelocity(x, j) < -m_reversalTolerance) {
        throw CanteraError("SprayFlow1D::checkDropletReversal",
            "Droplet axial velocity changed sign at grid point {} (z = {} m). "
            "The monodisperse spray equations are an inlet-value problem for the "
            "dispersed phase and cannot continue through a droplet turning point. "
            "Increase 'minimum_droplet_diameter' if droplets should dry out "
            "first, or revise the spray inlet and slip conditions.", j, z(j));
    }
}

double SprayFlow1D::inletDropletVelocity(span<const double> x, size_t j) const
{
    if (m_isFree && m_freeFlowNoSlip) {
        return u(x, j);
    }
    if (m_inletDropletVelocity != Undef) {
        return m_inletSide == 0 ? fabs(m_inletDropletVelocity)
                                : -fabs(m_inletDropletVelocity);
    }
    return u(x, j);
}

void SprayFlow1D::evalAdditionalEquations(span<const double> x, span<double> rsd,
                                          span<int> diag, double rdt, size_t jmin,
                                          size_t jmax)
{
    size_t iRho = liquidMassDensityIndex();
    size_t iMass = dropletMassIndex();
    size_t iVelocity = dropletVelocityIndex();
    size_t iSpread = dropletSpreadRateIndex();
    size_t iTemp = dropletTemperatureIndex();

    if (jmin == 0) {
        if (m_inletSide == 0) {
            if (m_inletLiquidMassDensity != Undef) {
                rsd[index(iRho, 0)] = liquidMassDensity(x, 0) - m_inletLiquidMassDensity;
            } else {
                rsd[index(iRho, 0)] = liquidMassFlux(x, 0) - m_inletLiquidMassFlux;
            }
            rsd[index(iMass, 0)] = dropletMass(x, 0) - m_inletDropletMass;
            rsd[index(iVelocity, 0)] = dropletVelocity(x, 0) - inletDropletVelocity(x, 0);
            rsd[index(iSpread, 0)] = dropletSpreadRate(x, 0) - m_inletDropletSpreadRate;
            rsd[index(iTemp, 0)] = dropletTemperature(x, 0) - m_inletLiquidTemperature;
        } else {
            rsd[index(iRho, 0)] = liquidMassDensity(x, 0) - liquidMassDensity(x, 1);
            rsd[index(iMass, 0)] = dropletMass(x, 0) - dropletMass(x, 1);
            rsd[index(iVelocity, 0)] = dropletVelocity(x, 0) - dropletVelocity(x, 1);
            rsd[index(iSpread, 0)] = dropletSpreadRate(x, 0) - dropletSpreadRate(x, 1);
            rsd[index(iTemp, 0)] = dropletTemperature(x, 0) - dropletTemperature(x, 1);
        }
        if (m_isFree && m_freeFlowNoSlip) {
            rsd[index(iVelocity, 0)] = dropletVelocity(x, 0) - u(x, 0);
        }
        diag[index(iRho, 0)] = 0;
        diag[index(iMass, 0)] = 0;
        diag[index(iVelocity, 0)] = 0;
        diag[index(iSpread, 0)] = 0;
        diag[index(iTemp, 0)] = 0;
    }

    if (jmax == m_points - 1) {
        size_t j = m_points - 1;
        if (m_inletSide == 1) {
            if (m_inletLiquidMassDensity != Undef) {
                rsd[index(iRho, j)] = liquidMassDensity(x, j) - m_inletLiquidMassDensity;
            } else {
                rsd[index(iRho, j)] = -liquidMassFlux(x, j) - m_inletLiquidMassFlux;
            }
            rsd[index(iMass, j)] = dropletMass(x, j) - m_inletDropletMass;
            rsd[index(iVelocity, j)] = dropletVelocity(x, j) - inletDropletVelocity(x, j);
            rsd[index(iSpread, j)] = dropletSpreadRate(x, j) - m_inletDropletSpreadRate;
            rsd[index(iTemp, j)] = dropletTemperature(x, j) - m_inletLiquidTemperature;
        } else {
            rsd[index(iRho, j)] = liquidMassDensity(x, j) - liquidMassDensity(x, j-1);
            rsd[index(iMass, j)] = dropletMass(x, j) - dropletMass(x, j-1);
            rsd[index(iVelocity, j)] = dropletVelocity(x, j) - dropletVelocity(x, j-1);
            rsd[index(iSpread, j)] = dropletSpreadRate(x, j) - dropletSpreadRate(x, j-1);
            rsd[index(iTemp, j)] = dropletTemperature(x, j) - dropletTemperature(x, j-1);
        }
        if (m_isFree && m_freeFlowNoSlip) {
            rsd[index(iVelocity, j)] = dropletVelocity(x, j) - u(x, j);
        }
        diag[index(iRho, j)] = 0;
        diag[index(iMass, j)] = 0;
        diag[index(iVelocity, j)] = 0;
        diag[index(iSpread, j)] = 0;
        diag[index(iTemp, j)] = 0;
    }

    size_t j0 = std::max<size_t>(jmin, 1);
    size_t j1 = std::min(jmax, m_points-2);
    for (size_t j = j0; j <= j1; j++) {
        if (dropletIsDry(x, j)) {
            size_t jup = (m_inletSide == 0) ? j - 1 : j + 1;
            double dryVelocity = (m_isFree && m_freeFlowNoSlip)
                ? u(x, j) : dropletVelocity(x, jup);
            // Once droplets reach the cutoff diameter, the dispersed phase is
            // absent. Keep the conserved dry state fixed, but carry intensive
            // droplet placeholders from upstream to avoid creating artificial
            // refinement targets at the dryout front.
            rsd[index(iRho, j)] = liquidMassDensity(x, j);
            diag[index(iRho, j)] = 0;
            rsd[index(iMass, j)] = dropletMass(x, j) - minimumDropletMass();
            diag[index(iMass, j)] = 0;
            rsd[index(iVelocity, j)] = dropletVelocity(x, j) - dryVelocity;
            diag[index(iVelocity, j)] = 0;
            rsd[index(iSpread, j)] =
                dropletSpreadRate(x, j) - dropletSpreadRate(x, jup);
            diag[index(iSpread, j)] = 0;
            rsd[index(iTemp, j)] = dropletTemperature(x, j)
                - dropletTemperature(x, jup);
            diag[index(iTemp, j)] = 0;
            continue;
        }

        if (dropletLoadingIsEmpty(x, j)) {
            // With zero number density, droplet intensive variables are
            // undefined; carry them from upstream without imposing dryout.
            size_t jup = (m_inletSide == 0) ? j - 1 : j + 1;
            double emptyVelocity = (m_isFree && m_freeFlowNoSlip)
                ? u(x, j) : dropletVelocity(x, jup);
            rsd[index(iRho, j)] = liquidMassDensity(x, j);
            diag[index(iRho, j)] = 0;
            rsd[index(iMass, j)] = dropletMass(x, j) - dropletMass(x, jup);
            diag[index(iMass, j)] = 0;
            rsd[index(iVelocity, j)] = dropletVelocity(x, j) - emptyVelocity;
            diag[index(iVelocity, j)] = 0;
            rsd[index(iSpread, j)] =
                dropletSpreadRate(x, j) - dropletSpreadRate(x, jup);
            diag[index(iSpread, j)] = 0;
            rsd[index(iTemp, j)] =
                dropletTemperature(x, j) - dropletTemperature(x, jup);
            diag[index(iTemp, j)] = 0;
            continue;
        }

        double fluxDivergence;
        if (dropletVelocity(x, j) > 0.0) {
            fluxDivergence = (liquidMassFlux(x, j) - liquidMassFlux(x, j-1))
                / m_dz[j-1];
        } else {
            fluxDivergence = (liquidMassFlux(x, j+1) - liquidMassFlux(x, j))
                / m_dz[j];
        }
        double dropletMassSource = m_enableDropletEvaporation
            ? m_sprayMassSource[j] : 0.0;
        rsd[index(iRho, j)] = -fluxDivergence
            - 2.0 * liquidMassDensity(x, j) * dropletSpreadRate(x, j)
            - dropletMassSource
            - rdt * (liquidMassDensity(x, j) - prevSoln(iRho, j));
        diag[index(iRho, j)] = 1;

        double mass = std::max(dropletMass(x, j), Tiny);
        double numberDensity = liquidMassDensity(x, j) / mass;
        double singleDropletEvaporation =
            (m_enableDropletEvaporation && numberDensity > 0.0)
            ? dropletMassSource / numberDensity : 0.0;
        rsd[index(iMass, j)] = -dropletVelocity(x, j)
            * dropletDerivative(x, iMass, j) - singleDropletEvaporation
            - rdt * (dropletMass(x, j) - prevSoln(iMass, j));
        diag[index(iMass, j)] = 1;

        if (m_isFree && m_freeFlowNoSlip) {
            rsd[index(iVelocity, j)] = dropletVelocity(x, j) - u(x, j);
            diag[index(iVelocity, j)] = 0;
        } else {
            rsd[index(iVelocity, j)] =
                (m_enableDropletAxialDrag
                 ? m_dropletAxialDragMultiplier * m_dropletAxialDrag[j] / mass
                 : 0.0)
                - dropletVelocityGradientTerm(x, j)
                - rdt * (dropletVelocity(x, j) - prevSoln(iVelocity, j));
            diag[index(iVelocity, j)] = 1;
        }

        if (isStrained()) {
            rsd[index(iSpread, j)] =
                (m_enableDropletSpreadDrag
                 ? m_dropletSpreadDragMultiplier * m_dropletSpreadDrag[j] / mass
                 : 0.0)
                - dropletVelocity(x, j) * dropletDerivative(x, iSpread, j)
                - dropletSpreadRate(x, j) * dropletSpreadRate(x, j)
                - rdt * (dropletSpreadRate(x, j) - prevSoln(iSpread, j));
            diag[index(iSpread, j)] = 1;
        } else {
            rsd[index(iSpread, j)] = dropletSpreadRate(x, j);
            diag[index(iSpread, j)] = 0;
        }

        const auto& liquid = m_sprayModel.liquidProperties();
        if (dropletTemperature(x, j) >= liquid.boilingTemperature) {
            rsd[index(iTemp, j)] = dropletTemperature(x, j)
                - liquid.boilingTemperature;
            diag[index(iTemp, j)] = 0;
        } else {
            double heatPerDroplet = (m_enableDropletHeatTransfer && numberDensity > 0.0)
                ? m_sprayHeatTransfer[j] / numberDensity : 0.0;
            rsd[index(iTemp, j)] = (heatPerDroplet
                - singleDropletEvaporation * liquid.latentHeat)
                / (mass * liquid.cp)
                - dropletVelocity(x, j) * dropletDerivative(x, iTemp, j)
                - rdt * (dropletTemperature(x, j) - prevSoln(iTemp, j));
            diag[index(iTemp, j)] = 1;
        }
    }
}

}
