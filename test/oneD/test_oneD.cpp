#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <fstream>

#include "cantera/core.h"
#include "cantera/onedim.h"
#include "cantera/oneD/DomainFactory.h"
#include "cantera/oneD/IonFlow.h"
#include "cantera/oneD/SprayFlow1D.h"

using namespace Cantera;

// This test is an exact equivalent of a clib test
// (clib::test_ctonedim.cpp::ctonedim::freeflame_from_parts)
TEST(onedim, freeflame)
{
    auto sol = newSolution("h2o2.yaml", "ohmech", "mixture-averaged");
    auto gas = sol->thermo();
    size_t nsp = gas->nSpecies();

    // reactants
    double uin = .3;
    double T = 300;
    double P = 101325;
    string X = "H2:0.65, O2:0.5, AR:2";
    gas->setState_TPX(T, P, X);
    double rho_in = gas->density();
    vector<double> yin(nsp);
    gas->getMassFractions(yin);

    // product estimate
    gas->equilibrate("HP");
    vector<double> yout(nsp);
    gas->getMassFractions(yout);
    double rho_out = gas->density();
    double Tad = gas->temperature();

    // flow
    auto flow = newFlow1D("free-flow", sol, "flow");

    // grid
    int nz = 21;
    double lz = 0.02;
    flow->setupUniformGrid(nz, lz);

    // inlet
    auto inlet = newBoundary1D("inlet", sol);
    inlet->setMoleFractions(X);
    inlet->setMdot(uin * rho_in);
    inlet->setTemperature(T);

    // outlet
    auto outlet = newBoundary1D("outlet", sol);
    double uout = inlet->mdot() / rho_out;

    // set up simulation
    vector<shared_ptr<Domain1D>> domains { inlet, flow, outlet };
    auto flame = newSim1D(domains);
    int dom = static_cast<int>(flame->domainIndex("flow"));
    ASSERT_EQ(dom, 1);

    // set up initial guess
    vector<double> locs{0.0, 0.3, 0.7, 1.0};
    vector<double> value{uin, uin, uout, uout};
    flow->setProfile("velocity", locs, value);
    value = {T, T, Tad, Tad};
    flow->setProfile("T", locs, value);
    for (size_t i = 0; i < nsp; i++) {
        value = {yin[i], yin[i], yout[i], yout[i]};
        flow->setProfile(gas->speciesName(i), locs, value);
    }

    // simulation settings
    double ratio = 15.0;
    double slope = 0.3;
    double curve = 0.5;
    flame->setRefineCriteria(dom, ratio, slope, curve);
    flame->setFixedTemperature(0.85 * T + .15 * Tad);

    // solve
    flow->solveEnergyEqn();
    bool refine_grid = false;
    int loglevel = 0;
    flame->solve(loglevel, refine_grid);
    flame->save("gtest-freeflame.yaml", "cpp", "Solution from C++ interface", true);
    if (usesHDF5()) {
        flame->save("gtest-freeflame.h5", "cpp", "Solution from C++ interface", true);
    }

    ASSERT_EQ(flow->nPoints(), static_cast<size_t>(nz + 1));
    auto Tvec = flow->values("T");
    double Tprev = Tvec[0];
    for (size_t n = 0; n < flow->nPoints(); n++) {
        T = Tvec[n];
        ASSERT_GE(T, Tprev);
        Tprev = T;
    }
}

TEST(onedim, flame_types)
{
    auto sol = newSolution("h2o2.yaml", "ohmech", "mixture-averaged");

    auto free = newDomain<Flow1D>("free-flow", sol, "flow");
    ASSERT_EQ(free->domainType(), "free-flow");
    auto symm = newDomain<Flow1D>("axisymmetric-flow", sol, "flow");
    ASSERT_EQ(symm->domainType(), "axisymmetric-flow");
    auto burner = newDomain<Flow1D>("unstrained-flow", sol, "flow");
    ASSERT_EQ(burner->domainType(), "unstrained-flow");

    ASSERT_THROW(burner->componentName(200), IndexError);
    ASSERT_THROW(burner->componentIndex("spam"), CanteraError);
}

TEST(onedim, ion_flame_types)
{
    auto sol = newSolution("ch4_ion.yaml");
    ASSERT_EQ(sol->transport()->transportModel(), "ionized-gas");

    auto free = newDomain<IonFlow>("free-flow", sol, "flow");
    ASSERT_EQ(free->domainType(), "free-ion-flow");
    auto symm = newDomain<IonFlow>("axisymmetric-flow", sol, "flow");
    ASSERT_EQ(symm->domainType(), "axisymmetric-ion-flow");
    auto burner = newDomain<IonFlow>("unstrained-flow", sol, "flow");
    ASSERT_EQ(burner->domainType(), "unstrained-ion-flow");
}

TEST(onedim, spray_flame_types)
{
    auto sol = newSolution("h2o2.yaml", "ohmech", "mixture-averaged");

    auto free = newDomain<SprayFlow1D>("spray-free-flow", sol, "flow");
    ASSERT_EQ(free->domainType(), "spray-free-flow");
    ASSERT_EQ(free->componentName(free->componentIndex("liquid-mass-density")),
              "liquid-mass-density");
    ASSERT_EQ(free->componentIndex("droplet_temperature"),
              free->componentIndex("droplet-temperature"));

    auto symm = newDomain<SprayFlow1D>("spray-axisymmetric-flow", sol, "flow");
    ASSERT_EQ(symm->domainType(), "spray-axisymmetric-flow");
    ASSERT_TRUE(symm->componentActive(symm->componentIndex("droplet-spread-rate")));

    auto burner = newDomain<SprayFlow1D>("spray-unstrained-flow", sol, "flow");
    ASSERT_EQ(burner->domainType(), "spray-unstrained-flow");
    ASSERT_FALSE(burner->componentActive(burner->componentIndex("droplet-spread-rate")));
}

TEST(onedim, spray_sources)
{
    auto sol = newSolution("h2o2.yaml", "ohmech", "mixture-averaged");
    auto gas = sol->thermo();
    gas->setState_TPX(900.0, OneAtm, "H2:1e-12, O2:0.21, N2:0.79");

    auto flow = newDomain<SprayFlow1D>("spray-free-flow", sol, "flow");
    flow->setupUniformGrid(6, 0.01);
    flow->setSprayFuel("H2");
    flow->setLiquidProperties(70.0, 9500.0, 4.5e5, 20.3);
    flow->setDropletDiameter(10e-6);
    flow->setLiquidMassDensity(1e-8);
    flow->setLiquidTemperature(20.0);

    auto inlet = newBoundary1D("inlet", sol);
    inlet->setMoleFractions("H2:1e-12, O2:0.21, N2:0.79");
    inlet->setMdot(0.2 * gas->density());
    inlet->setTemperature(900.0);

    auto outlet = newBoundary1D("outlet", sol);
    vector<shared_ptr<Domain1D>> domains { inlet, flow, outlet };
    auto sim = newSim1D(domains);
    sim->eval(0.0);

    ASSERT_GT(flow->sprayEvaporationRate(1), 0.0);
    ASSERT_LT(flow->sprayGasEnergySource(1), 0.0);
    ASSERT_GT(flow->dropletDiameter(1), 0.0);
    ASSERT_EQ(flow->values("liquid-mass-density").size(), flow->nPoints());
    ASSERT_EQ(flow->values("droplet-velocity")[1], flow->values("velocity")[1]);
}

int main(int argc, char** argv)
{
    printf("Running main() from test_oneD.cpp\n");
    testing::InitGoogleTest(&argc, argv);
    make_deprecation_warnings_fatal();
    printStackTraceOnSegfault();
    Cantera::addDataDirectory("test/data");
    Cantera::addDataDirectory("data");
    CanteraError::setStackTraceDepth(20);
    int result = RUN_ALL_TESTS();
    appdelete();
    return result;
}
