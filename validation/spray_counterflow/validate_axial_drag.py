"""One-way counterflow spray validation with axial droplet drag enabled.

The workflow is staged deliberately:

1. solve the one-way evaporation problem with drag disabled and grid refinement on
2. integrate the one-way droplet ODEs with axial drag over that gas field
3. use the integrated droplet state as the initial guess
4. solve with axial drag enabled, gas feedback disabled, spread drag disabled
5. run a drag-enabled refinement pass

Run from the Cantera repository root with:

    ../.venv-cantera/bin/python validation/spray_counterflow/validate_axial_drag.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build" / "python"))

import cantera as ct  # noqa: E402
import CoolProp.CoolProp as CP  # noqa: E402


P = ct.one_atm
TGAS = 300.0
WIDTH = 0.02
GAS_SPEED = 0.2
DROPLET_SPEED = 1.0
DIAMETER = 40e-6
D_MIN = 2e-6
LIQUID_LOADING = 1e-6


def methane_props() -> dict[str, float]:
    tref = 111.0
    return {
        "liquid_density": CP.PropsSI("Dmass", "T", tref, "P", P, "Methane"),
        "liquid_cp": CP.PropsSI("Cpmass", "T", tref, "P", P, "Methane"),
        "latent_heat": (
            CP.PropsSI("Hmass", "T", tref, "Q", 1, "Methane")
            - CP.PropsSI("Hmass", "T", tref, "Q", 0, "Methane")
        ),
        "boiling_temperature": CP.PropsSI("T", "P", P, "Q", 0, "Methane"),
        "saturation_pressure": P,
    }


PROPS = methane_props()


def air() -> ct.Solution:
    gas = ct.Solution("gri30.yaml")
    gas.TPX = TGAS, P, "O2:0.21,N2:0.79"
    gas.transport_model = "mixture-averaged"
    return gas


def configure_counterflow(sim: ct.CounterflowDiffusionFlame, rho: float) -> None:
    for inlet in (sim.fuel_inlet, sim.oxidizer_inlet):
        inlet.T = TGAS
        inlet.X = "O2:0.21,N2:0.79"
        inlet.mdot = rho * GAS_SPEED
    sim.energy_enabled = False
    sim.flame.gas_phase_spray_sources_enabled = False
    sim.set_initial_guess(mode="linear")
    sim.set_refine_criteria(ratio=3.0, slope=0.03, curve=0.06, prune=0.0)
    sim.set_max_grid_points(sim.flame, 900)


def make_spray(rho_gas: float) -> ct.MonodisperseSpray:
    return ct.MonodisperseSpray(
        fuel_species="CH4",
        diameter=DIAMETER,
        minimum_droplet_diameter=D_MIN,
        liquid_temperature=PROPS["boiling_temperature"] - 1.0,
        liquid_mass_density=LIQUID_LOADING * rho_gas,
        droplet_velocity=DROPLET_SPEED,
        **PROPS,
    )


def make_sim(drag: bool = False) -> tuple[ct.CounterflowDiffusionFlame, ct.Solution, float]:
    gas = air()
    sim = ct.CounterflowDiffusionFlame(gas, width=WIDTH, spray=make_spray(gas.density))
    configure_counterflow(sim, gas.density)
    sim.flame.droplet_axial_drag_enabled = drag
    sim.flame.droplet_spread_drag_enabled = False
    return sim, gas, gas.density


def film_correction(transfer_number: float) -> float:
    if abs(transfer_number) < 1e-8:
        return 1.0
    limited = max(transfer_number, -0.95)
    return (1.0 + limited) ** 0.7 * math.log1p(limited) / limited


def droplet_rates(
    gas: ct.Solution, droplet_mass: float, droplet_temperature: float,
    droplet_velocity: float, gas_velocity: float
) -> tuple[float, float, float]:
    rho_l = PROPS["liquid_density"]
    latent_heat = PROPS["latent_heat"]
    boiling_temperature = PROPS["boiling_temperature"]
    diameter = (6.0 * droplet_mass / (math.pi * rho_l)) ** (1.0 / 3.0)
    radius = 0.5 * diameter

    gas.TPX = TGAS, P, "O2:0.21,N2:0.79"
    i_fuel = gas.species_index("CH4")
    density = gas.density
    viscosity = gas.viscosity
    conductivity = gas.thermal_conductivity
    cp_gas = gas.cp_mass
    diff = max(gas.mix_diff_coeffs[i_fuel], 1e-300)

    slip = abs(gas_velocity - droplet_velocity)
    reynolds = density * slip * diameter / max(viscosity, 1e-300)
    prandtl = cp_gas * viscosity / max(conductivity, 1e-300)
    schmidt = viscosity / max(density * diff, 1e-300)
    nu0 = 2.0 + 0.552 * math.sqrt(reynolds) * prandtl ** (1.0 / 3.0)
    sh0 = 2.0 + 0.552 * math.sqrt(reynolds) * schmidt ** (1.0 / 3.0)

    w_fuel = gas.molecular_weights[i_fuel]
    rv = ct.gas_constant / w_fuel
    psat = PROPS["saturation_pressure"] * math.exp(
        -latent_heat / rv * (1.0 / droplet_temperature - 1.0 / boiling_temperature)
    )
    psat = min(max(psat, 0.0), 0.999 * P)
    x_surface = psat / P
    x = gas.X
    weights = gas.molecular_weights
    x_carrier = sum(x[k] for k in range(gas.n_species) if k != i_fuel)
    carrier_weight = sum(x[k] * weights[k] for k in range(gas.n_species) if k != i_fuel)
    if x_carrier > 1e-300:
        carrier_weight /= x_carrier
    else:
        carrier_weight = gas.mean_molecular_weight
    y_surface = x_surface * w_fuel / max(
        x_surface * w_fuel + (1.0 - x_surface) * carrier_weight, 1e-300
    )
    b_mass = max((y_surface - gas.Y[i_fuel]) / max(1.0 - y_surface, 1e-300), 0.0)
    sherwood = 2.0 + (sh0 - 2.0) / film_correction(b_mass)
    evaporation = 0.0
    if b_mass > 0.0:
        evaporation = math.pi * diameter * density * diff * sherwood * math.log1p(b_mass)

    cp_fuel = gas.partial_molar_cp[i_fuel] / w_fuel
    b_heat = cp_fuel * (TGAS - droplet_temperature) / max(latent_heat, 1e-300)
    nusselt = 2.0 + (nu0 - 2.0) / film_correction(b_heat)
    if abs(b_heat) > 1e-8:
        heat_correction = math.log1p(max(b_heat, -0.95)) / b_heat
    else:
        heat_correction = 1.0
    heat = (
        2.0 * math.pi * radius * conductivity * nusselt
        * (TGAS - droplet_temperature) * heat_correction
    )
    drag = 3.0 * math.pi * viscosity * diameter * (gas_velocity - droplet_velocity)
    return evaporation, heat, drag


def integrate_drag_guess(sim: ct.CounterflowDiffusionFlame, gas: ct.Solution, rho_gas: float):
    z = sim.grid
    ugas = sim.velocity
    rho_l = PROPS["liquid_density"]
    cp_l = PROPS["liquid_cp"]
    latent_heat = PROPS["latent_heat"]
    boiling_temperature = PROPS["boiling_temperature"]
    min_mass = math.pi / 6.0 * rho_l * D_MIN ** 3

    mass = np.empty_like(z)
    velocity = np.empty_like(z)
    temperature = np.empty_like(z)
    liquid_density = np.empty_like(z)

    mass[0] = math.pi / 6.0 * rho_l * DIAMETER ** 3
    velocity[0] = DROPLET_SPEED
    temperature[0] = boiling_temperature - 1.0
    liquid_density[0] = LIQUID_LOADING * rho_gas

    dry = False
    for j in range(1, len(z)):
        dz = z[j] - z[j - 1]
        m = mass[j - 1]
        u = velocity[j - 1]
        t = temperature[j - 1]
        rho_liq = liquid_density[j - 1]
        steps = max(1, int(math.ceil(dz / 1e-5)))

        for step in range(steps):
            if dry or m <= min_mass * (1.0 + 1e-8) or rho_liq <= 0.0:
                dry = True
                m = min_mass
                rho_liq = 0.0
                t = boiling_temperature
                break

            frac = (step + 0.5) / steps
            gas_velocity = ugas[j - 1] + frac * (ugas[j] - ugas[j - 1])
            evaporation, heat, drag = droplet_rates(gas, m, min(t, boiling_temperature),
                                                    u, gas_velocity)
            h = dz / steps
            inv_u = 1.0 / max(u, 1e-6)
            du_dz = drag / m * inv_u
            dm_dz = -evaporation * inv_u
            if t >= boiling_temperature:
                dt_dz = 0.0
                t = boiling_temperature
            else:
                dt_dz = (heat - evaporation * latent_heat) / (m * cp_l) * inv_u
            drho_dz = (-rho_liq / m * evaporation - rho_liq * du_dz) * inv_u

            u = max(u + h * du_dz, 1e-5)
            m = max(m + h * dm_dz, min_mass)
            t = min(max(t + h * dt_dz, 1.0), boiling_temperature)
            rho_liq = max(rho_liq + h * drho_dz, 0.0)

        mass[j] = m
        velocity[j] = u
        temperature[j] = t
        liquid_density[j] = rho_liq

    return mass, velocity, temperature, liquid_density


def apply_drag_guess(sim: ct.CounterflowDiffusionFlame, profiles) -> None:
    zrel = sim.grid / sim.grid[-1]
    for name, values in zip(
        ("droplet-mass", "droplet-velocity", "droplet-temperature", "liquid-mass-density"),
        profiles,
    ):
        sim.flame.set_profile(name, zrel, values)


def dryout_location(sim: ct.CounterflowDiffusionFlame, rho_gas: float) -> float:
    d = sim.flame.droplet_diameter
    yl = sim.flame.liquid_mass_density / rho_gas
    dry = np.where((d <= D_MIN * 1.0001) | (yl <= 1e-16))[0]
    if len(dry) == 0:
        return float("nan")
    return float(sim.grid[dry[0]])


def plot_drag_solution(sim: ct.CounterflowDiffusionFlame, rho_gas: float, outdir: Path) -> None:
    z_mm = sim.grid * 1e3
    yl = sim.flame.liquid_mass_density / rho_gas
    wet = (yl > 1e-9 * yl[0]) & (sim.flame.droplet_diameter > D_MIN * 1.0001)
    dry_z = dryout_location(sim, rho_gas)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)
    axes[0].plot(z_mm[wet], sim.flame.droplet_diameter[wet] * 1e6, label="diameter")
    axes[0].axhline(D_MIN * 1e6, color="0.5", linestyle=":", linewidth=1.0)
    axes[0].axvline(dry_z * 1e3, color="tab:red", linestyle=":", linewidth=1.0)
    axes[0].axvline(0.5 * WIDTH * 1e3, color="0.3", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("diameter [um]")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(z_mm[wet], sim.flame.droplet_velocity[wet], label="droplet")
    axes[1].plot(z_mm, sim.velocity, label="gas", linestyle="--")
    axes[1].axvline(dry_z * 1e3, color="tab:red", linestyle=":", linewidth=1.0)
    axes[1].axvline(0.5 * WIDTH * 1e3, color="0.3", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("z [mm]")
    axes[1].set_ylabel("axial velocity [m/s]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("One-way methane spray with axial drag")
    fig.tight_layout()
    fig.savefig(outdir / "axial_drag_diameter_velocity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.semilogy(z_mm[wet], (yl / yl[0])[wet])
    ax.axvline(dry_z * 1e3, color="tab:red", linestyle=":", linewidth=1.0)
    ax.axvline(0.5 * WIDTH * 1e3, color="0.3", linestyle="--", linewidth=1.0)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("liquid mass density / inlet")
    ax.set_title("Axial drag compresses the liquid phase before evaporation wins")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "axial_drag_liquid_loading.png", dpi=180)
    plt.close(fig)


def main() -> None:
    outdir = Path(__file__).resolve().parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    sim, gas, rho_gas = make_sim(drag=False)
    sim.solve(loglevel=0, refine_grid=True, auto=False)
    base_points = len(sim.grid)

    profiles = integrate_drag_guess(sim, gas, rho_gas)
    apply_drag_guess(sim, profiles)
    sim.flame.droplet_axial_drag_enabled = True
    sim.flame.droplet_spread_drag_enabled = False
    sim.solve(loglevel=0, refine_grid=False, auto=False)

    sim.set_refine_criteria(ratio=4.0, slope=0.08, curve=0.12, prune=0.0)
    sim.set_max_grid_points(sim.flame, 350)
    sim.solve(loglevel=0, refine_grid=True, auto=False)

    dry_z = dryout_location(sim, rho_gas)
    d = sim.flame.droplet_diameter
    yl = sim.flame.liquid_mass_density / rho_gas
    ud = sim.flame.droplet_velocity
    wet = (yl > 1e-9 * yl[0]) & (d > D_MIN * 1.0001)

    assert np.isfinite(dry_z)
    assert dry_z < 0.5 * WIDTH
    assert np.all(ud[wet] > 0.0)
    assert np.interp(0.002, sim.grid, ud) < DROPLET_SPEED
    assert np.interp(0.005, sim.grid, d) <= D_MIN * 1.0001
    assert len(sim.grid) > base_points
    assert np.all(np.diff(ud[wet]) <= 1e-8)
    liquid_flux = sim.flame.liquid_mass_density * ud
    flux_wet = liquid_flux[wet] / liquid_flux[wet][0]
    assert np.all(np.diff(flux_wet) <= 1e-8)

    plot_drag_solution(sim, rho_gas, outdir)
    summary = {
        "base_points": base_points,
        "drag_points": len(sim.grid),
        "dryout_mm": dry_z * 1e3,
        "droplet_velocity_2mm": float(np.interp(0.002, sim.grid, ud)),
        "droplet_velocity_5mm": float(np.interp(0.005, sim.grid, ud)),
        "liquid_loading_2mm_over_inlet": float(np.interp(0.002, sim.grid, yl) / yl[0]),
        "liquid_flux_2mm_over_inlet": float(np.interp(0.002, sim.grid, liquid_flux) / liquid_flux[0]),
    }
    with (outdir / "axial_drag_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(summary)


if __name__ == "__main__":
    main()
