"""Direct user-facing nonreacting counterflow spray validation.

This script exercises the intended Python workflow without validation-only
staging helpers: construct a spray-laden counterflow, set inlet conditions, and
call ``solve(auto=True)``. The automatic solver performs the internal source
continuation and grid refinement.

Run from the Cantera repository root with:

    ../.venv-cantera/bin/python validation/spray_counterflow/validate_auto_solve.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build" / "python"))

import cantera as ct  # noqa: E402

try:
    import CoolProp.CoolProp as CP  # noqa: E402
except ImportError:
    CP = None


P = ct.one_atm
TGAS = 300.0
WIDTH = 0.02
GAS_SPEED = 0.2
DROPLET_SPEED = 1.0
DIAMETER = 40e-6
D_MIN = 2e-6
LIQUID_LOADING = 1e-4
AIR = "O2:0.21,N2:0.79"


def methane_props() -> dict[str, float]:
    if CP is None:
        return {
            "liquid_density": 423.3,
            "liquid_cp": 3476.0,
            "latent_heat": 512040.0,
            "boiling_temperature": 111.667,
            "saturation_pressure": P,
        }
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


def solve_direct_case() -> ct.CounterflowDiffusionFlame:
    props = methane_props()

    gas = ct.Solution("gri30.yaml")
    gas.transport_model = "mixture-averaged"
    gas.TPX = TGAS, P, AIR
    rho_gas = gas.density

    spray = ct.MonodisperseSpray(
        fuel_species="CH4",
        diameter=DIAMETER,
        minimum_droplet_diameter=D_MIN,
        liquid_temperature=props["boiling_temperature"] - 1.0,
        liquid_mass_density=LIQUID_LOADING * rho_gas,
        droplet_velocity=DROPLET_SPEED,
        **props,
    )

    sim = ct.CounterflowDiffusionFlame(gas, width=WIDTH, spray=spray)
    for inlet in (sim.fuel_inlet, sim.oxidizer_inlet):
        inlet.T = TGAS
        inlet.X = AIR
        inlet.mdot = rho_gas * GAS_SPEED

    sim.solve(loglevel=0, auto=True)
    return sim


def stagnation_plane(sim: ct.CounterflowDiffusionFlame) -> float:
    z = sim.grid
    u = sim.velocity
    sign_change = np.where(np.signbit(u[:-1]) != np.signbit(u[1:]))[0]
    if len(sign_change) == 0:
        raise AssertionError("No stagnation plane found")
    j = sign_change[0]
    return float(z[j] - u[j] * (z[j + 1] - z[j]) / (u[j + 1] - u[j]))


def droplet_diameter(sim: ct.CounterflowDiffusionFlame) -> np.ndarray:
    rho_l = sim.spray.liquid_density
    mass = sim.flame.values("droplet-mass")
    return (6.0 * mass / (np.pi * rho_l)) ** (1.0 / 3.0)


def dryout_location(sim: ct.CounterflowDiffusionFlame, diameter: np.ndarray) -> float:
    liquid_density = sim.flame.values("liquid-mass-density")
    dry = np.where((diameter <= D_MIN * 1.01) | (liquid_density <= 1e-16))[0]
    if len(dry) == 0:
        return float("nan")
    return float(sim.grid[dry[0]])


def summarize(sim: ct.CounterflowDiffusionFlame) -> dict[str, float]:
    diameter = droplet_diameter(sim)
    dryout = dryout_location(sim, diameter)
    i_fuel = sim.gas.species_index("CH4")
    return {
        "points": len(sim.grid),
        "stagnation_mm": stagnation_plane(sim) * 1e3,
        "dryout_mm": dryout * 1e3,
        "inlet_diameter_um": diameter[0] * 1e6,
        "minimum_diameter_um": float(np.min(diameter)) * 1e6,
        "max_liquid_mass_density": float(np.max(sim.flame.values("liquid-mass-density"))),
        "max_ch4_mass_fraction": float(np.max(sim.Y[i_fuel])),
        "minimum_temperature": float(np.min(sim.T)),
        "minimum_energy_source": float(np.min(sim.flame.spray_gas_energy_source)),
    }


def check_solution(sim: ct.CounterflowDiffusionFlame, summary: dict[str, float]) -> None:
    diameter = droplet_diameter(sim)
    liquid_density = sim.flame.values("liquid-mass-density")
    wet = (diameter > D_MIN * 1.01) & (liquid_density > 1e-16)

    assert summary["points"] > 100
    assert abs(summary["stagnation_mm"] - 0.5 * WIDTH * 1e3) < 0.05
    assert 1.0 < summary["dryout_mm"] < 0.5 * WIDTH * 1e3
    assert abs(summary["inlet_diameter_um"] - DIAMETER * 1e6) < 1e-3
    assert abs(summary["minimum_diameter_um"] - D_MIN * 1e6) < 1e-3
    assert np.count_nonzero(wet) > 25
    assert np.all(np.diff(diameter[wet]) <= 5e-11)
    assert np.all(liquid_density >= -1e-18)
    assert summary["max_ch4_mass_fraction"] > 1e-5
    assert summary["minimum_temperature"] < TGAS - 0.05
    assert summary["minimum_energy_source"] < 0.0
    assert np.max(np.abs(np.sum(sim.Y, axis=0) - 1.0)) < 1e-7


def plot_solution(sim: ct.CounterflowDiffusionFlame, outdir: Path) -> None:
    z_mm = sim.grid * 1e3
    diameter = droplet_diameter(sim)
    liquid_density = sim.flame.values("liquid-mass-density")
    dryout = dryout_location(sim, diameter) * 1e3
    stagnation = stagnation_plane(sim) * 1e3
    i_fuel = sim.gas.species_index("CH4")

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True)
    axes[0].plot(z_mm, diameter * 1e6)
    axes[0].set_ylabel("droplet diameter [um]")

    axes[1].semilogy(z_mm, np.maximum(liquid_density / liquid_density[0], 1e-18))
    axes[1].set_ylabel("liquid density / inlet")

    axes[2].plot(z_mm, sim.flame.values("droplet-velocity"), label="droplet")
    axes[2].plot(z_mm, sim.velocity, label="gas")
    axes[2].set_ylabel("axial velocity [m/s]")
    axes[2].set_xlabel("z [mm]")
    axes[2].legend()

    for ax in axes:
        ax.axvline(stagnation, color="0.3", linestyle="--", linewidth=1.0)
        ax.axvline(dryout, color="0.5", linestyle=":", linewidth=1.0)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "auto_solve_spray_profiles.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True)
    axes[0].plot(z_mm, sim.Y[i_fuel])
    axes[0].set_ylabel("CH4 mass fraction")

    axes[1].plot(z_mm, sim.T)
    axes[1].set_ylabel("gas temperature [K]")

    axes[2].plot(z_mm, sim.flame.evaporation_rate, label="mass")
    axes[2].set_ylabel("evaporation source [kg/m3/s]")
    axes[2].set_xlabel("z [mm]")

    for ax in axes:
        ax.axvline(stagnation, color="0.3", linestyle="--", linewidth=1.0)
        ax.axvline(dryout, color="0.5", linestyle=":", linewidth=1.0)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "auto_solve_gas_feedback.png", dpi=180)
    plt.close(fig)


def main() -> None:
    outdir = Path(__file__).resolve().parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    sim = solve_direct_case()
    summary = summarize(sim)
    check_solution(sim, summary)
    plot_solution(sim, outdir)

    with (outdir / "auto_solve_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

    print("Direct counterflow spray auto-solve validation passed.")
    for key, value in summary.items():
        print(f"{key}: {value:.8g}")


if __name__ == "__main__":
    main()
