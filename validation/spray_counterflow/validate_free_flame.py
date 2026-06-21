"""Direct no-slip freely propagating spray flame validation.

This script exercises the intended public Python workflow for freely
propagating premixed methane-air flames with a dilute liquid methane spray. In
free flames, the spray uses the no-slip model: droplet axial velocity is
constrained to the gas velocity, and no independent axial momentum equation is
claimed for the gas.

Run from the Cantera repository root with:

    PYTHONPATH=build/python ../.venv-cantera/bin/python \
        validation/spray_counterflow/validate_free_flame.py
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

import cantera as ct  # noqa: E402
from spray_diagnostics import (  # noqa: E402
    check_free_flame_spray_diagnostics,
    summarize_free_flame_spray,
)

try:
    import CoolProp.CoolProp as CP  # noqa: E402
except ImportError:
    CP = None


P = ct.one_atm
TIN = 300.0
PHI = 0.9
WIDTH = 0.03
FUEL = "CH4:1"
OXIDIZER = "O2:1,N2:3.76"
DROPLET_DIAMETER = 500e-6
MIN_DIAMETER = 2e-6
LIQUID_LOADINGS = (1e-6, 1e-5, 1e-4)

SUMMARY_FIELDS = (
    "name",
    "elapsed_s",
    "liquid_loading",
    "points",
    "flame_speed",
    "tmax",
    "flame_location_mm",
    "dryout_mm",
    "wet_points",
    "max_evaporation_rate",
    "minimum_energy_source",
    "minimum_diameter_um",
    "max_no_slip_error",
    "max_ch4_mass_fraction",
    "gas_continuity_abs",
    "gas_continuity_rel",
    "liquid_continuity_abs",
    "liquid_continuity_rel",
    "droplet_mass_abs",
    "droplet_mass_rel",
    "species_sum_error",
)


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


def make_gas() -> ct.Solution:
    gas = ct.Solution("gri30.yaml")
    gas.transport_model = "mixture-averaged"
    gas.TP = TIN, P
    gas.set_equivalence_ratio(PHI, FUEL, OXIDIZER)
    return gas


def set_refinement(sim: ct.FreeFlame) -> None:
    sim.set_refine_criteria(ratio=4.0, slope=0.16, curve=0.24, prune=0.0)


def solve_unsprayed() -> ct.FreeFlame:
    gas = make_gas()
    sim = ct.FreeFlame(gas, width=WIDTH)
    set_refinement(sim)
    sim.solve(loglevel=0, auto=True)
    return sim


def solve_spray_case(
    liquid_loading: float,
    props: dict[str, float],
) -> ct.FreeFlame:
    gas = make_gas()
    rho_gas = gas.density
    spray = ct.MonodisperseSpray(
        fuel_species="CH4",
        diameter=DROPLET_DIAMETER,
        minimum_droplet_diameter=MIN_DIAMETER,
        liquid_temperature=props["boiling_temperature"] - 1.0,
        liquid_mass_density=liquid_loading * rho_gas,
        **props,
    )
    sim = ct.FreeFlame(gas, width=WIDTH, spray=spray)
    set_refinement(sim)
    sim.solve(loglevel=0, auto=True)
    sim.eval()
    return sim


def summarize(name: str, loading: float, sim: ct.FreeFlame, elapsed: float):
    diagnostics = summarize_free_flame_spray(sim, "CH4")
    i_fuel = sim.gas.species_index("CH4")
    no_slip = np.max(np.abs(sim.flame.droplet_velocity - sim.velocity))
    return {
        "name": name,
        "elapsed_s": elapsed,
        "liquid_loading": loading,
        "points": len(sim.grid),
        "flame_speed": float(sim.velocity[0]),
        "tmax": float(np.max(sim.T)),
        "flame_location_mm": float(sim.grid[np.argmax(sim.T)] * 1e3),
        "dryout_mm": diagnostics.first_dryout_mm,
        "wet_points": diagnostics.wet_points,
        "max_evaporation_rate": diagnostics.max_evaporation_rate,
        "minimum_energy_source": diagnostics.min_wet_energy_source,
        "minimum_diameter_um": diagnostics.minimum_diameter_um,
        "max_no_slip_error": float(no_slip),
        "max_ch4_mass_fraction": float(np.max(sim.Y[i_fuel])),
        "gas_continuity_abs": diagnostics.max_gas_continuity_abs,
        "gas_continuity_rel": diagnostics.max_gas_continuity_rel,
        "liquid_continuity_abs": diagnostics.max_liquid_continuity_abs,
        "liquid_continuity_rel": diagnostics.max_liquid_continuity_rel,
        "droplet_mass_abs": diagnostics.max_droplet_mass_abs,
        "droplet_mass_rel": diagnostics.max_droplet_mass_rel,
        "species_sum_error": diagnostics.max_species_sum_error,
    }, diagnostics


def check_solution(summary: dict[str, float], diagnostics) -> None:
    assert summary["points"] >= 60
    assert summary["flame_speed"] > 0.1
    assert summary["tmax"] > 1800.0
    assert 0.0 < summary["dryout_mm"] < summary["flame_location_mm"]
    assert abs(summary["minimum_diameter_um"] - MIN_DIAMETER * 1e6) < 1e-3
    assert summary["max_no_slip_error"] < 1e-10
    assert summary["max_ch4_mass_fraction"] > 0.04
    check_free_flame_spray_diagnostics(diagnostics, min_wet_points=8)


def check_sweep(rows: list[dict[str, float]]) -> None:
    evaporations = np.array([row["max_evaporation_rate"] for row in rows])
    cooling = np.array([-row["minimum_energy_source"] for row in rows])
    speeds = np.array([row["flame_speed"] for row in rows])
    assert np.all(np.diff(evaporations) > 0.0)
    assert np.all(np.diff(cooling) > 0.0)
    assert np.all(np.diff(speeds) > 0.0)


def plot_results(
    unsprayed: ct.FreeFlame,
    solutions: dict[str, ct.FreeFlame],
    outdir: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True)
    axes[0].plot(unsprayed.grid * 1e3, unsprayed.T, color="0.35",
                 linestyle="--", label="gas only")
    for name, sim in solutions.items():
        z_mm = sim.grid * 1e3
        label = name.replace("_", " ")
        axes[0].plot(z_mm, sim.T, label=label)
        axes[1].plot(z_mm, sim.flame.droplet_diameter * 1e6, label=label)
        liquid = sim.flame.liquid_mass_density
        axes[2].semilogy(z_mm, np.maximum(liquid / liquid[0], 1e-18),
                         label=label)
    axes[0].set_ylabel("gas temperature [K]")
    axes[1].set_ylabel("droplet diameter [um]")
    axes[2].set_ylabel("liquid density / inlet")
    axes[2].set_xlabel("z [mm]")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "free_flame_profiles.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True)
    for name, sim in solutions.items():
        z_mm = sim.grid * 1e3
        label = name.replace("_", " ")
        axes[0].plot(z_mm, sim.flame.evaporation_rate, label=label)
        axes[1].plot(z_mm, sim.flame.spray_gas_energy_source, label=label)
        axes[2].plot(z_mm, sim.flame.droplet_velocity - sim.velocity,
                     label=label)
    axes[0].set_ylabel("evaporation [kg/m3/s]")
    axes[1].set_ylabel("gas energy source [W/m3]")
    axes[2].set_ylabel("droplet-gas velocity [m/s]")
    axes[2].set_xlabel("z [mm]")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "free_flame_sources.png", dpi=180)
    plt.close(fig)


def main() -> None:
    outdir = Path(__file__).resolve().parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    unsprayed = solve_unsprayed()
    props = methane_props()
    rows: list[dict[str, float]] = []
    solutions: dict[str, ct.FreeFlame] = {}
    for loading in LIQUID_LOADINGS:
        name = f"loading_{loading:.0e}"
        print(f"solving {name}...", flush=True)
        start = time.perf_counter()
        sim = solve_spray_case(loading, props)
        elapsed = time.perf_counter() - start
        row, diagnostics = summarize(name, loading, sim, elapsed)
        check_solution(row, diagnostics)
        rows.append(row)
        solutions[name] = sim
        print(
            f"  passed in {elapsed:.1f}s: "
            f"Su={row['flame_speed']:.4f} m/s, "
            f"Tmax={row['tmax']:.1f} K, "
            f"dryout={row['dryout_mm']:.2f} mm, "
            f"evap={row['max_evaporation_rate']:.3e}",
            flush=True,
        )

    check_sweep(rows)
    plot_results(unsprayed, solutions, outdir)

    with (outdir / "free_flame_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print("No-slip freely propagating spray flame validation passed.")


if __name__ == "__main__":
    main()
