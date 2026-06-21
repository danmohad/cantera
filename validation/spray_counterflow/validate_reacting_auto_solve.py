"""Direct user-facing reacting counterflow spray validation.

This script exercises a lit methane-air counterflow diffusion flame with a
small liquid methane spray injected from the fuel side. The setup intentionally
uses the normal Python interface and calls ``solve(auto=True)`` without manual
staging.

Run from the Cantera repository root with:

    ../.venv-cantera/bin/python validation/spray_counterflow/validate_reacting_auto_solve.py
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
from spray_diagnostics import (  # noqa: E402
    check_counterflow_spray_diagnostics,
    summarize_counterflow_spray,
)

try:
    import CoolProp.CoolProp as CP  # noqa: E402
except ImportError:
    CP = None


P = ct.one_atm
WIDTH = 0.02
FUEL = "CH4:1"
AIR = "O2:0.21,N2:0.79"
TFUEL = 300.0
TOX = 300.0
FUEL_MDOT = 0.12
OX_MDOT = 0.24
DROPLET_DIAMETER = 20e-6
MIN_DIAMETER = 2e-6
DROPLET_SPEED = 0.2
LIQUID_LOADING = 1e-4


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
    gas.TPX = TFUEL, P, FUEL
    fuel_density = gas.density

    spray = ct.MonodisperseSpray(
        fuel_species="CH4",
        diameter=DROPLET_DIAMETER,
        minimum_droplet_diameter=MIN_DIAMETER,
        liquid_temperature=props["boiling_temperature"] - 1.0,
        liquid_mass_density=LIQUID_LOADING * fuel_density,
        droplet_velocity=DROPLET_SPEED,
        **props,
    )

    sim = ct.CounterflowDiffusionFlame(gas, width=WIDTH, spray=spray)
    sim.fuel_inlet.T = TFUEL
    sim.fuel_inlet.X = FUEL
    sim.fuel_inlet.mdot = FUEL_MDOT
    sim.oxidizer_inlet.T = TOX
    sim.oxidizer_inlet.X = AIR
    sim.oxidizer_inlet.mdot = OX_MDOT

    sim.solve(loglevel=0, auto=True)
    return sim


def dryout_location(sim: ct.CounterflowDiffusionFlame) -> float:
    diameter = sim.flame.droplet_diameter
    dry = np.where(diameter <= MIN_DIAMETER * 1.01)[0]
    if len(dry) == 0:
        return float("nan")
    return float(sim.grid[dry[0]])


def summarize(sim: ct.CounterflowDiffusionFlame) -> dict[str, float]:
    i_fuel = sim.gas.species_index("CH4")
    dryout = dryout_location(sim)
    return {
        "points": len(sim.grid),
        "tmax": float(np.max(sim.T)),
        "tmin": float(np.min(sim.T)),
        "flame_location_mm": float(sim.grid[np.argmax(sim.T)] * 1e3),
        "dryout_mm": dryout * 1e3,
        "max_heat_release_rate": float(np.max(sim.heat_release_rate)),
        "max_ch4_mass_fraction": float(np.max(sim.Y[i_fuel])),
        "minimum_diameter_um": float(np.min(sim.flame.droplet_diameter) * 1e6),
        "species_sum_error": float(np.max(np.abs(np.sum(sim.Y, axis=0) - 1.0))),
    }


def check_solution(summary: dict[str, float]) -> None:
    assert summary["points"] > 80
    assert summary["tmax"] > 1800.0
    assert summary["tmin"] < TFUEL
    assert 0.0 < summary["dryout_mm"] < summary["flame_location_mm"]
    assert abs(summary["minimum_diameter_um"] - MIN_DIAMETER * 1e6) < 1e-3
    assert summary["max_heat_release_rate"] > 1e7
    assert summary["max_ch4_mass_fraction"] > 0.9
    assert summary["species_sum_error"] < 1e-7


def plot_solution(sim: ct.CounterflowDiffusionFlame, outdir: Path) -> None:
    z_mm = sim.grid * 1e3
    dryout_mm = dryout_location(sim) * 1e3
    flame_mm = sim.grid[np.argmax(sim.T)] * 1e3

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True)
    axes[0].plot(z_mm, sim.T)
    axes[0].set_ylabel("gas temperature [K]")

    axes[1].plot(z_mm, sim.flame.droplet_diameter * 1e6)
    axes[1].set_ylabel("droplet diameter [um]")

    axes[2].semilogy(
        z_mm,
        np.maximum(sim.flame.liquid_mass_density / sim.flame.liquid_mass_density[0],
                   1e-18),
    )
    axes[2].set_ylabel("liquid density / inlet")
    axes[2].set_xlabel("z [mm]")

    for ax in axes:
        ax.axvline(dryout_mm, color="0.5", linestyle=":", linewidth=1.0)
        ax.axvline(flame_mm, color="0.25", linestyle="--", linewidth=1.0)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "reacting_auto_spray_profiles.png", dpi=180)
    plt.close(fig)

    i_fuel = sim.gas.species_index("CH4")
    i_o2 = sim.gas.species_index("O2")
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True)
    axes[0].plot(z_mm, sim.Y[i_fuel], label="CH4")
    axes[0].plot(z_mm, sim.Y[i_o2], label="O2")
    axes[0].legend()
    axes[0].set_ylabel("mass fraction")

    axes[1].plot(z_mm, sim.heat_release_rate)
    axes[1].set_ylabel("heat release [W/m3]")

    axes[2].plot(z_mm, sim.flame.droplet_velocity, label="droplet")
    axes[2].plot(z_mm, sim.velocity, label="gas")
    axes[2].legend()
    axes[2].set_ylabel("axial velocity [m/s]")
    axes[2].set_xlabel("z [mm]")

    for ax in axes:
        ax.axvline(dryout_mm, color="0.5", linestyle=":", linewidth=1.0)
        ax.axvline(flame_mm, color="0.25", linestyle="--", linewidth=1.0)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "reacting_auto_gas_flame.png", dpi=180)
    plt.close(fig)


def main() -> None:
    outdir = Path(__file__).resolve().parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    sim = solve_direct_case()
    summary = summarize(sim)
    check_solution(summary)
    check_counterflow_spray_diagnostics(
        summarize_counterflow_spray(sim, "CH4"),
        min_wet_points=20,
    )
    plot_solution(sim, outdir)

    with (outdir / "reacting_auto_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary)

    print("Direct reacting counterflow spray auto-solve validation passed.")
    for key, value in summary.items():
        print(f"{key}: {value:.8g}")


if __name__ == "__main__":
    main()
