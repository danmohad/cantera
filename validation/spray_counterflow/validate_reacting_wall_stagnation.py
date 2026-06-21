"""Direct reacting wall-stagnation spray validation.

This script validates a premixed methane-air impinging-jet flame with a dilute
liquid methane spray. It uses the public Python interface only:
``ct.ImpingingJet(..., spray=spray)`` followed by ``solve(auto=True)``.

Run from the Cantera repository root with:

    PYTHONPATH=build/python ../.venv-cantera/bin/python \
        validation/spray_counterflow/validate_reacting_wall_stagnation.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cantera as ct  # noqa: E402
from spray_diagnostics import (  # noqa: E402
    check_counterflow_spray_diagnostics,
    summarize_counterflow_spray,
)
from validate_nonreacting_sweep import (  # noqa: E402
    MIN_DIAMETER,
    P,
    methane_props,
)


WIDTH = 0.02
TIN = 300.0
TSURF = 500.0
PHI = 0.8
INLET_MDOT = 0.04
DROPLET_DIAMETER = 80e-6
DROPLET_SPEED = 0.2
LIQUID_LOADINGS = (1e-7, 1e-6, 1e-5)

SUMMARY_FIELDS = (
    "name",
    "elapsed_s",
    "liquid_loading",
    "points",
    "wet_points",
    "dryout_mm",
    "tmax",
    "flame_location_mm",
    "max_heat_release_rate",
    "max_evaporation_rate",
    "minimum_energy_source",
    "minimum_diameter_um",
    "max_ch4_mass_fraction",
    "gas_continuity_abs",
    "gas_continuity_rel",
    "liquid_continuity_abs",
    "liquid_continuity_rel",
    "droplet_mass_abs",
    "droplet_mass_rel",
    "species_sum_error",
)


def make_gas() -> ct.Solution:
    gas = ct.Solution("gri30.yaml")
    gas.transport_model = "mixture-averaged"
    gas.TP = TIN, P
    gas.set_equivalence_ratio(PHI, "CH4:1", "O2:1,N2:3.76")
    return gas


def solve_case(liquid_loading: float, props: dict[str, float]) -> ct.ImpingingJet:
    gas = make_gas()
    rho_gas = gas.density
    inlet_y = gas.Y.copy()
    spray = ct.MonodisperseSpray(
        fuel_species="CH4",
        diameter=DROPLET_DIAMETER,
        minimum_droplet_diameter=MIN_DIAMETER,
        liquid_temperature=props["boiling_temperature"] - 1.0,
        liquid_mass_density=liquid_loading * rho_gas,
        droplet_velocity=DROPLET_SPEED,
        **props,
    )

    sim = ct.ImpingingJet(gas, width=WIDTH, spray=spray)
    sim.inlet.T = TIN
    sim.inlet.Y = inlet_y
    sim.inlet.mdot = INLET_MDOT
    sim.surface.T = TSURF
    sim.set_refine_criteria(ratio=4.0, slope=0.16, curve=0.24, prune=0.0)
    sim.solve(loglevel=0, auto=True)
    sim.eval()
    return sim


def summarize(name: str, loading: float, sim: ct.ImpingingJet, elapsed: float):
    diagnostics = summarize_counterflow_spray(sim, "CH4")
    i_fuel = sim.gas.species_index("CH4")
    return {
        "name": name,
        "elapsed_s": elapsed,
        "liquid_loading": loading,
        "points": len(sim.grid),
        "wet_points": diagnostics.wet_points,
        "dryout_mm": diagnostics.first_dryout_mm,
        "tmax": float(np.max(sim.T)),
        "flame_location_mm": float(sim.grid[np.argmax(sim.T)] * 1e3),
        "max_heat_release_rate": float(np.max(sim.heat_release_rate)),
        "max_evaporation_rate": diagnostics.max_evaporation_rate,
        "minimum_energy_source": diagnostics.min_wet_energy_source,
        "minimum_diameter_um": diagnostics.minimum_diameter_um,
        "max_ch4_mass_fraction": float(np.max(sim.Y[i_fuel])),
        "gas_continuity_abs": diagnostics.max_gas_continuity_abs,
        "gas_continuity_rel": diagnostics.max_gas_continuity_rel,
        "liquid_continuity_abs": diagnostics.max_liquid_continuity_abs,
        "liquid_continuity_rel": diagnostics.max_liquid_continuity_rel,
        "droplet_mass_abs": diagnostics.max_droplet_mass_abs,
        "droplet_mass_rel": diagnostics.max_droplet_mass_rel,
        "species_sum_error": diagnostics.max_species_sum_error,
    }, diagnostics


def check_solution(row: dict[str, float], diagnostics) -> None:
    assert row["points"] >= 80
    assert row["wet_points"] >= 40
    assert row["tmax"] > 1200.0
    assert row["max_heat_release_rate"] > 1e7
    assert 0.0 < row["dryout_mm"] < row["flame_location_mm"]
    assert abs(row["minimum_diameter_um"] - MIN_DIAMETER * 1e6) < 1e-3
    assert row["max_ch4_mass_fraction"] > 0.02
    check_counterflow_spray_diagnostics(diagnostics, min_wet_points=40)


def check_sweep(rows: list[dict[str, float]]) -> None:
    evaporations = np.array([row["max_evaporation_rate"] for row in rows])
    cooling = np.array([-row["minimum_energy_source"] for row in rows])
    assert np.all(np.diff(evaporations) > 0.0)
    assert np.all(np.diff(cooling) > 0.0)


def plot_results(solutions: dict[str, ct.ImpingingJet], outdir: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(7.4, 9.4), sharex=True)
    for name, sim in solutions.items():
        z_mm = sim.grid * 1e3
        label = name.replace("_", " ")
        axes[0].plot(z_mm, sim.T, label=label)
        axes[1].plot(z_mm, sim.flame.droplet_diameter * 1e6, label=label)
        liquid = sim.flame.liquid_mass_density
        axes[2].semilogy(z_mm, np.maximum(liquid / liquid[0], 1e-18),
                         label=label)
        axes[3].plot(z_mm, sim.heat_release_rate, label=label)
    axes[0].set_ylabel("gas temperature [K]")
    axes[1].set_ylabel("diameter [um]")
    axes[2].set_ylabel("liquid density / inlet")
    axes[3].set_ylabel("heat release [W/m3]")
    axes[3].set_xlabel("z [mm]")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "reacting_wall_profiles.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.2), sharex=True)
    for name, sim in solutions.items():
        z_mm = sim.grid * 1e3
        label = name.replace("_", " ")
        axes[0].plot(z_mm, sim.flame.evaporation_rate, label=label)
        axes[1].plot(z_mm, sim.flame.spray_gas_energy_source, label=label)
        axes[2].plot(z_mm, sim.flame.spray_gas_momentum_source, label=label)
    axes[0].set_ylabel("evaporation [kg/m3/s]")
    axes[1].set_ylabel("gas energy source [W/m3]")
    axes[2].set_ylabel("gas momentum source")
    axes[2].set_xlabel("z [mm]")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "reacting_wall_sources.png", dpi=180)
    plt.close(fig)


def main() -> None:
    outdir = Path(__file__).resolve().parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    props = methane_props()

    rows: list[dict[str, float]] = []
    solutions: dict[str, ct.ImpingingJet] = {}
    for loading in LIQUID_LOADINGS:
        name = f"loading_{loading:.0e}"
        print(f"solving {name}...", flush=True)
        start = time.perf_counter()
        sim = solve_case(loading, props)
        elapsed = time.perf_counter() - start
        row, diagnostics = summarize(name, loading, sim, elapsed)
        check_solution(row, diagnostics)
        rows.append(row)
        solutions[name] = sim
        print(
            f"  passed in {elapsed:.1f}s: "
            f"Tmax={row['tmax']:.1f} K, "
            f"dryout={row['dryout_mm']:.2f} mm, "
            f"evap={row['max_evaporation_rate']:.3e}, "
            f"points={row['points']}",
            flush=True,
        )

    check_sweep(rows)
    plot_results(solutions, outdir)

    with (outdir / "reacting_wall_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print("Reacting wall-stagnation spray validation passed.")


if __name__ == "__main__":
    main()
