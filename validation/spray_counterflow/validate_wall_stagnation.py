"""Direct nonreacting wall-stagnation spray validation.

This script exercises the intended public workflow for an impinging jet:
construct ``ct.ImpingingJet(..., spray=spray)``, set inlet and wall boundary
conditions, and call ``solve(auto=True)``. The case is nonreacting air with a
dilute liquid methane spray, so failures are attributable to spray transport,
dryout, source coupling, or wall-stagnation staging rather than chemistry.

Run from the Cantera repository root with:

    PYTHONPATH=build/python ../.venv-cantera/bin/python \
        validation/spray_counterflow/validate_wall_stagnation.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cantera as ct  # noqa: E402
from spray_diagnostics import (  # noqa: E402
    check_counterflow_spray_diagnostics,
    summarize_counterflow_spray,
)
from validate_nonreacting_sweep import (  # noqa: E402
    AIR,
    MIN_DIAMETER,
    P,
    TGAS,
    methane_props,
)


WIDTH = 0.02
GAS_SPEED = 0.2
DROPLET_SPEED = 0.8
DIAMETER = 25e-6
LIQUID_LOADING = 1e-4


def solve_direct_case() -> ct.ImpingingJet:
    props = methane_props()
    gas = ct.Solution("gri30.yaml")
    gas.transport_model = "mixture-averaged"
    gas.TPX = TGAS, P, AIR
    rho_gas = gas.density

    spray = ct.MonodisperseSpray(
        fuel_species="CH4",
        diameter=DIAMETER,
        minimum_droplet_diameter=MIN_DIAMETER,
        liquid_temperature=props["boiling_temperature"] - 1.0,
        liquid_mass_density=LIQUID_LOADING * rho_gas,
        droplet_velocity=DROPLET_SPEED,
        **props,
    )

    sim = ct.ImpingingJet(gas, width=WIDTH, spray=spray)
    sim.inlet.T = TGAS
    sim.inlet.X = AIR
    sim.inlet.mdot = rho_gas * GAS_SPEED
    sim.surface.T = TGAS
    sim.energy_enabled = True
    sim.solve(loglevel=0, auto=True)
    return sim


def wet_mask(sim: ct.ImpingingJet) -> np.ndarray:
    liquid_density = sim.flame.liquid_mass_density
    diameter = sim.flame.droplet_diameter
    return (
        (liquid_density > np.max(liquid_density) * 1e-12)
        & (diameter > sim.flame.minimum_droplet_diameter * (1.0 + 1e-8))
    )


def dryout_location(sim: ct.ImpingingJet, wet: np.ndarray) -> float:
    dry = np.where(~wet)[0]
    if dry.size == 0:
        return float("nan")
    return float(sim.grid[dry[0]])


def summarize(sim: ct.ImpingingJet) -> dict[str, float]:
    wet = wet_mask(sim)
    dryout = dryout_location(sim, wet)
    diagnostics = summarize_counterflow_spray(sim, "CH4")
    i_fuel = sim.gas.species_index("CH4")
    return {
        "points": len(sim.grid),
        "wet_points": int(np.count_nonzero(wet)),
        "dryout_mm": dryout * 1e3,
        "inlet_diameter_um": float(sim.flame.droplet_diameter[0] * 1e6),
        "minimum_diameter_um": float(np.min(sim.flame.droplet_diameter) * 1e6),
        "max_ch4_mass_fraction": float(np.max(sim.Y[i_fuel])),
        "minimum_temperature": float(np.min(sim.T)),
        "minimum_energy_source": float(np.min(sim.flame.spray_gas_energy_source)),
        "gas_continuity_rel": diagnostics.max_gas_continuity_rel,
        "liquid_continuity_rel": diagnostics.max_liquid_continuity_rel,
        "droplet_mass_abs": diagnostics.max_droplet_mass_abs,
        "droplet_mass_rel": diagnostics.max_droplet_mass_rel,
        "species_sum_error": diagnostics.max_species_sum_error,
    }


def check_solution(sim: ct.ImpingingJet, summary: dict[str, float]) -> None:
    wet = wet_mask(sim)
    diagnostics = summarize_counterflow_spray(sim, "CH4")
    check_counterflow_spray_diagnostics(
        diagnostics,
        min_wet_points=12,
        require_energy_cooling=True,
    )
    assert summary["points"] >= 30
    assert summary["wet_points"] >= 12
    assert 0.0 < summary["dryout_mm"] < 0.5 * WIDTH * 1e3
    assert abs(summary["inlet_diameter_um"] - DIAMETER * 1e6) < 1e-3
    assert abs(summary["minimum_diameter_um"] - MIN_DIAMETER * 1e6) < 1e-3
    assert np.all(np.diff(sim.flame.droplet_diameter[wet]) <= 5e-11)
    assert np.all(sim.flame.liquid_mass_density >= -1e-18)
    assert summary["max_ch4_mass_fraction"] > 1e-4
    assert summary["minimum_temperature"] < TGAS - 0.05


def plot_solution(sim: ct.ImpingingJet, outdir: Path) -> None:
    z_mm = sim.grid * 1e3
    wet = wet_mask(sim)
    dryout = dryout_location(sim, wet) * 1e3
    i_fuel = sim.gas.species_index("CH4")

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 9.2), sharex=True)
    axes[0].plot(z_mm, sim.flame.droplet_diameter * 1e6)
    axes[0].set_ylabel("diameter [um]")

    liquid_density = sim.flame.liquid_mass_density
    axes[1].semilogy(z_mm, np.maximum(liquid_density / liquid_density[0], 1e-18))
    axes[1].set_ylabel("liquid density / inlet")

    axes[2].plot(z_mm, sim.Y[i_fuel])
    axes[2].set_ylabel("CH4 mass fraction")

    axes[3].plot(z_mm, sim.T)
    axes[3].set_ylabel("gas T [K]")
    axes[3].set_xlabel("z [mm]")

    for ax in axes:
        ax.axvline(dryout, color="0.5", linestyle=":", linewidth=1.0)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "wall_stagnation_profiles.png", dpi=180)
    plt.close(fig)


def main() -> None:
    outdir = Path(__file__).resolve().parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    sim = solve_direct_case()
    summary = summarize(sim)
    check_solution(sim, summary)
    plot_solution(sim, outdir)

    with (outdir / "wall_stagnation_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary)

    print("Direct wall-stagnation spray validation passed.")
    for key, value in summary.items():
        print(f"{key}: {value:.8g}")


if __name__ == "__main__":
    main()
