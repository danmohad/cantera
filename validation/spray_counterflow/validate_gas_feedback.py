"""Nonreacting counterflow spray validation with staged gas feedback.

The droplet equations are solved with full one-way drag first. Gas feedback is
then enabled source-by-source:

1. gas continuity source only
2. gas continuity plus species sources
3. gas continuity, species, and radial momentum sources
4. gas energy equation with spray energy source off
5. gas energy equation with spray energy source on

Chemistry is left enabled but inactive at this 300 K nonreacting condition.

Run from the Cantera repository root with:

    ../.venv-cantera/bin/python validation/spray_counterflow/validate_gas_feedback.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import validate_axial_drag as axial
import validate_droplet_drag as drag


def axisymmetric_continuity_residual(sim, include_source: bool) -> np.ndarray:
    z = sim.grid
    rho_u = sim.density * sim.velocity
    rho = sim.density
    spread = sim.spread_rate
    source = sim.flame.evaporation_rate if include_source else np.zeros_like(z)
    residual = []
    for j in range(1, len(z) - 1):
        residual.append(
            -(rho_u[j + 1] - rho_u[j]) / (z[j + 1] - z[j])
            - (rho[j + 1] * spread[j + 1] + rho[j] * spread[j])
            + source[j]
        )
    return np.array(residual)


def solve_stages():
    sim, rho_gas, base_points = drag.solve_case(include_spread_drag=True)
    one_way = snapshot(sim, rho_gas, "one-way", base_points)

    sim.flame.gas_phase_spray_sources_enabled = False
    sim.flame.gas_phase_spray_mass_source_enabled = True
    sim.flame.gas_phase_spray_species_source_enabled = False
    sim.flame.gas_phase_spray_energy_source_enabled = False
    sim.flame.gas_phase_spray_momentum_source_enabled = False
    sim.solve(loglevel=0, refine_grid=False, auto=False)
    sim.solve(loglevel=0, refine_grid=True, auto=False)
    mass_only = snapshot(sim, rho_gas, "mass", base_points)

    sim.flame.gas_phase_spray_species_source_enabled = True
    sim.solve(loglevel=0, refine_grid=False, auto=False)
    sim.solve(loglevel=0, refine_grid=True, auto=False)
    mass_species = snapshot(sim, rho_gas, "mass+species", base_points)

    sim.flame.gas_phase_spray_momentum_source_enabled = True
    sim.solve(loglevel=0, refine_grid=False, auto=False)
    sim.solve(loglevel=0, refine_grid=True, auto=False)
    mass_species_momentum = snapshot(sim, rho_gas, "mass+species+momentum",
                                     base_points)

    sim.energy_enabled = True
    sim.flame.gas_phase_spray_energy_source_enabled = False
    sim.solve(loglevel=0, refine_grid=False, auto=False)
    sim.solve(loglevel=0, refine_grid=True, auto=False)
    energy_source_off = snapshot(sim, rho_gas, "energy-source-off",
                                 base_points)

    sim.flame.gas_phase_spray_energy_source_enabled = True
    sim.solve(loglevel=0, refine_grid=False, auto=False)
    sim.solve(loglevel=0, refine_grid=True, auto=False)
    energy_source_on = snapshot(sim, rho_gas, "energy-source-on", base_points)

    return (
        one_way,
        mass_only,
        mass_species,
        mass_species_momentum,
        energy_source_off,
        energy_source_on,
    )


def snapshot(sim, rho_gas: float, label: str, base_points: int) -> dict:
    i_fuel = sim.gas.species_index("CH4")
    liquid_flux = sim.flame.liquid_mass_density * sim.flame.droplet_velocity
    momentum_source = getattr(
        sim.flame, "spray_gas_momentum_source", np.zeros_like(sim.grid)
    )
    return {
        "label": label,
        "grid": sim.grid.copy(),
        "density": sim.density.copy(),
        "velocity": sim.velocity.copy(),
        "spread_rate": sim.spread_rate.copy(),
        "temperature": sim.T.copy(),
        "fuel_y": sim.Y[i_fuel].copy(),
        "y_sum": np.sum(sim.Y, axis=0),
        "evaporation_rate": sim.flame.evaporation_rate.copy(),
        "energy_source": sim.flame.spray_gas_energy_source.copy(),
        "momentum_source": momentum_source.copy(),
        "liquid_flux": liquid_flux.copy(),
        "base_points": base_points,
        "points": len(sim.grid),
        "dryout": axial.dryout_location(sim, rho_gas),
        "continuity_without_source": axisymmetric_continuity_residual(
            sim, include_source=False
        ),
        "continuity_with_source": axisymmetric_continuity_residual(
            sim, include_source=True
        ),
    }


def summarize(stage: dict) -> dict[str, float | str]:
    z = stage["grid"]
    return {
        "case": stage["label"],
        "points": stage["points"],
        "dryout_mm": stage["dryout"] * 1e3,
        "max_y_ch4": float(np.max(stage["fuel_y"])),
        "y_ch4_2mm": float(np.interp(0.002, z, stage["fuel_y"])),
        "gas_velocity_2mm": float(np.interp(0.002, z, stage["velocity"])),
        "gas_spread_rate_2mm": float(np.interp(0.002, z, stage["spread_rate"])),
        "temperature_2mm": float(np.interp(0.002, z, stage["temperature"])),
        "min_temperature": float(np.min(stage["temperature"])),
        "min_energy_source": float(np.min(stage["energy_source"])),
        "momentum_source_2mm": float(np.interp(0.002, z, stage["momentum_source"])),
        "continuity_no_source_norm": float(
            np.max(np.abs(stage["continuity_without_source"]))
        ),
        "continuity_with_source_norm": float(
            np.max(np.abs(stage["continuity_with_source"]))
        ),
    }


def check_stages(one_way: dict, mass_only: dict, mass_species: dict,
                 momentum: dict, energy_off: dict, energy_on: dict) -> None:
    assert one_way["points"] > one_way["base_points"]
    assert mass_only["points"] >= one_way["points"]
    assert mass_species["points"] >= mass_only["points"]
    assert momentum["points"] >= mass_species["points"]
    assert energy_off["points"] >= momentum["points"]
    assert energy_on["points"] >= energy_off["points"]

    one_way_no_source = np.max(np.abs(one_way["continuity_without_source"]))
    one_way_with_source = np.max(np.abs(one_way["continuity_with_source"]))
    mass_with_source = np.max(np.abs(mass_only["continuity_with_source"]))
    species_with_source = np.max(np.abs(mass_species["continuity_with_source"]))
    momentum_with_source = np.max(np.abs(momentum["continuity_with_source"]))
    energy_with_source = np.max(np.abs(energy_on["continuity_with_source"]))
    assert one_way_no_source < 1e-5
    assert one_way_with_source > 1e-4
    assert mass_with_source < 1e-5
    assert species_with_source < 1e-5
    assert momentum_with_source < 1e-5
    assert energy_with_source < 1e-5

    assert np.max(np.abs(mass_only["fuel_y"])) < 1e-10
    assert np.max(mass_species["fuel_y"]) > 1e-6
    assert np.interp(0.002, mass_species["grid"], mass_species["fuel_y"]) > 1e-6
    assert np.max(np.abs(mass_species["y_sum"] - 1.0)) < 1e-7
    assert np.max(np.abs(momentum["y_sum"] - 1.0)) < 1e-7

    one_way_u2 = np.interp(0.002, one_way["grid"], one_way["velocity"])
    mass_u2 = np.interp(0.002, mass_only["grid"], mass_only["velocity"])
    assert mass_u2 > one_way_u2

    species_v2 = np.interp(0.002, mass_species["grid"], mass_species["spread_rate"])
    momentum_v2 = np.interp(0.002, momentum["grid"], momentum["spread_rate"])
    momentum_source_2mm = np.interp(0.002, momentum["grid"],
                                    momentum["momentum_source"])
    assert momentum_source_2mm < 0.0
    assert momentum_v2 < species_v2

    assert np.max(np.abs(energy_off["temperature"] - axial.TGAS)) < 1e-8
    assert np.min(energy_on["energy_source"]) < 0.0
    assert np.min(energy_on["temperature"]) < axial.TGAS - 1e-4
    assert np.interp(0.002, energy_on["grid"], energy_on["temperature"]) < axial.TGAS


def plot_stages(stages: list[dict], outdir: Path) -> None:
    one_way, mass_only, mass_species, momentum, energy_off, energy_on = stages
    stag = 0.5 * axial.WIDTH * 1e3
    dry = momentum["dryout"] * 1e3

    source_limit = dry + 0.5

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.6))
    for stage in (mass_only, mass_species):
        reference = np.interp(stage["grid"], one_way["grid"], one_way["velocity"])
        source_region = stage["grid"] * 1e3 <= source_limit
        axes[0].plot(
            stage["grid"][source_region] * 1e3,
            (stage["velocity"][source_region] - reference[source_region]) * 1e6,
            label=stage["label"],
        )
    axes[0].set_ylabel("delta gas axial velocity [um/s]")
    axes[0].set_xlabel("z [mm]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(mass_only["grid"] * 1e3, mass_only["fuel_y"], label="mass only")
    axes[1].plot(mass_species["grid"] * 1e3, mass_species["fuel_y"],
                 label="mass + species")
    axes[1].set_xlabel("z [mm]")
    axes[1].set_ylabel("CH4 mass fraction")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    for ax in axes:
        ax.axvline(dry, color="tab:red", linestyle=":", linewidth=1.0)
    axes[1].axvline(stag, color="0.3", linestyle="--", linewidth=1.0)
    fig.suptitle("Gas mass and species feedback")
    fig.tight_layout()
    fig.savefig(outdir / "gas_feedback_mass_species.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.6))
    reference = np.interp(momentum["grid"], mass_species["grid"],
                          mass_species["spread_rate"])
    source_region = momentum["grid"] * 1e3 <= source_limit
    axes[0].plot(
        momentum["grid"][source_region] * 1e3,
        (momentum["spread_rate"][source_region] - reference[source_region]) * 1e6,
    )
    axes[0].set_ylabel("delta gas spread rate [1e-6 1/s]")
    axes[0].set_xlabel("z [mm]")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(momentum["grid"] * 1e3, momentum["momentum_source"])
    axes[1].set_xlabel("z [mm]")
    axes[1].set_ylabel("radial momentum source [N/m^3]")
    axes[1].grid(True, alpha=0.25)
    for ax in axes:
        ax.axvline(dry, color="tab:red", linestyle=":", linewidth=1.0)
    axes[1].axvline(stag, color="0.3", linestyle="--", linewidth=1.0)
    fig.suptitle("Gas momentum feedback")
    fig.tight_layout()
    fig.savefig(outdir / "gas_feedback_momentum.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.6))
    axes[0].plot(
        energy_off["grid"] * 1e3,
        (axial.TGAS - energy_off["temperature"]) * 1e3,
        label="energy source off",
    )
    axes[0].plot(
        energy_on["grid"] * 1e3,
        (axial.TGAS - energy_on["temperature"]) * 1e3,
        label="energy source on",
    )
    axes[0].set_ylabel("gas cooling [mK]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(energy_on["grid"] * 1e3, energy_on["energy_source"])
    axes[1].set_xlabel("z [mm]")
    axes[1].set_ylabel("gas energy source [W/m^3]")
    axes[1].grid(True, alpha=0.25)
    for ax in axes:
        ax.axvline(dry, color="tab:red", linestyle=":", linewidth=1.0)
        ax.axvline(stag, color="0.3", linestyle="--", linewidth=1.0)
    fig.suptitle("Gas energy feedback")
    fig.tight_layout()
    fig.savefig(outdir / "gas_feedback_energy.png", dpi=180)
    plt.close(fig)


def main() -> None:
    outdir = Path(__file__).resolve().parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    stages = list(solve_stages())
    check_stages(*stages)
    plot_stages(stages, outdir)

    rows = [summarize(stage) for stage in stages]
    with (outdir / "gas_feedback_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
