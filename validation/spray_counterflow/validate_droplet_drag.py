"""One-way counterflow spray validation with axial and spread-rate drag.

This extends ``validate_axial_drag.py`` by enabling the particle spread-rate
drag equation in a strained counterflow. Gas-phase spray feedback remains
disabled so the validation isolates droplet drag.

Run from the Cantera repository root with:

    ../.venv-cantera/bin/python validation/spray_counterflow/validate_droplet_drag.py
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
import validate_axial_drag as axial  # noqa: E402


STEP_MAX = 1e-6


def integrate_spread_guess(sim: ct.CounterflowDiffusionFlame, gas: ct.Solution,
                           rho_gas: float, profiles):
    """Add a spread-rate profile and update liquid density for radial divergence."""
    mass, velocity, temperature, _ = profiles
    z = sim.grid
    gas_spread = sim.spread_rate
    mu = gas.viscosity
    rho_l = axial.PROPS["liquid_density"]
    min_mass = math.pi / 6.0 * rho_l * axial.D_MIN ** 3

    spread = np.zeros_like(z)
    dry = False
    for j in range(1, len(z)):
        if dry or mass[j - 1] <= min_mass * (1.0 + 1e-8):
            dry = True
            spread[j] = 0.0
            continue

        dz = z[j] - z[j - 1]
        steps = max(1, int(math.ceil(dz / STEP_MAX)))
        vd = spread[j - 1]
        for step in range(steps):
            frac = (step + 0.5) / steps
            m = mass[j - 1] + frac * (mass[j] - mass[j - 1])
            if m <= min_mass * (1.0 + 1e-8):
                dry = True
                vd = 0.0
                break
            ud = velocity[j - 1] + frac * (velocity[j] - velocity[j - 1])
            vg = gas_spread[j - 1] + frac * (gas_spread[j] - gas_spread[j - 1])
            diameter = (6.0 * m / (math.pi * rho_l)) ** (1.0 / 3.0)
            relaxation_rate = 3.0 * math.pi * mu * diameter / max(m, 1e-300)
            d_spread_dz = (
                relaxation_rate * (vg - vd) - vd * vd
            ) / max(ud, 1e-6)
            vd += dz / steps * d_spread_dz
            if not math.isfinite(vd):
                raise AssertionError("Non-finite droplet spread-rate initial guess")
        spread[j] = vd

    liquid_density = np.empty_like(z)
    liquid_density[0] = axial.LIQUID_LOADING * rho_gas
    dry = False
    for j in range(1, len(z)):
        if dry or mass[j - 1] <= min_mass * (1.0 + 1e-8) or liquid_density[j - 1] <= 0.0:
            dry = True
            liquid_density[j] = 0.0
            continue

        dz = z[j] - z[j - 1]
        steps = max(1, int(math.ceil(dz / STEP_MAX)))
        rho = liquid_density[j - 1]
        dm_dz = (mass[j] - mass[j - 1]) / dz
        du_dz = (velocity[j] - velocity[j - 1]) / dz
        for step in range(steps):
            frac = (step + 0.5) / steps
            m = mass[j - 1] + frac * (mass[j] - mass[j - 1])
            if m <= min_mass * (1.0 + 1e-8):
                dry = True
                rho = 0.0
                break
            ud = velocity[j - 1] + frac * (velocity[j] - velocity[j - 1])
            vd = spread[j - 1] + frac * (spread[j] - spread[j - 1])
            drho_dz = (
                rho / max(m, 1e-300) * dm_dz
                - 2.0 * rho * vd / max(ud, 1e-6)
                - rho * du_dz / max(ud, 1e-6)
            )
            rho = max(rho + dz / steps * drho_dz, 0.0)
        liquid_density[j] = rho

    return mass, velocity, temperature, liquid_density, spread


def apply_profiles(sim: ct.CounterflowDiffusionFlame, profiles) -> None:
    zrel = sim.grid / sim.grid[-1]
    names = (
        "droplet-mass",
        "droplet-velocity",
        "droplet-temperature",
        "liquid-mass-density",
        "droplet-spread-rate",
    )
    for name, values in zip(names, profiles):
        sim.flame.set_profile(name, zrel, values)


def solve_case(include_spread_drag: bool):
    sim, gas, rho_gas = axial.make_sim(drag=False)
    sim.solve(loglevel=0, refine_grid=True, auto=False)
    base_points = len(sim.grid)

    profiles = axial.integrate_drag_guess(sim, gas, rho_gas)
    if include_spread_drag:
        profiles = integrate_spread_guess(sim, gas, rho_gas, profiles)
    else:
        profiles = (*profiles, np.zeros_like(sim.grid))
    apply_profiles(sim, profiles)

    sim.flame.droplet_axial_drag_enabled = True
    sim.flame.droplet_spread_drag_enabled = include_spread_drag
    sim.solve(loglevel=0, refine_grid=False, auto=False)

    sim.set_refine_criteria(ratio=4.0, slope=0.08, curve=0.12, prune=0.0)
    sim.set_max_grid_points(sim.flame, 450)
    sim.solve(loglevel=0, refine_grid=True, auto=False)
    return sim, rho_gas, base_points


def wet_mask(sim: ct.CounterflowDiffusionFlame, rho_gas: float) -> np.ndarray:
    yl = sim.flame.liquid_mass_density / rho_gas
    return (yl > 1e-9 * yl[0]) & (
        sim.flame.droplet_diameter > axial.D_MIN * 1.0001
    )


def summarize_case(label: str, sim: ct.CounterflowDiffusionFlame, rho_gas: float,
                   base_points: int) -> dict[str, float | str]:
    z = sim.grid
    yl = sim.flame.liquid_mass_density / rho_gas
    liquid_flux = sim.flame.liquid_mass_density * sim.flame.droplet_velocity
    return {
        "case": label,
        "base_points": base_points,
        "drag_points": len(z),
        "dryout_mm": axial.dryout_location(sim, rho_gas) * 1e3,
        "droplet_velocity_2mm": float(np.interp(0.002, z, sim.flame.droplet_velocity)),
        "droplet_spread_rate_2mm": float(np.interp(0.002, z, sim.flame.droplet_spread_rate)),
        "gas_spread_rate_2mm": float(np.interp(0.002, z, sim.spread_rate)),
        "liquid_loading_2mm_over_inlet": float(np.interp(0.002, z, yl) / yl[0]),
        "liquid_flux_2mm_over_inlet": float(np.interp(0.002, z, liquid_flux) / liquid_flux[0]),
    }


def check_case(sim: ct.CounterflowDiffusionFlame, rho_gas: float,
               base_points: int, include_spread_drag: bool) -> None:
    z = sim.grid
    wet = wet_mask(sim, rho_gas)
    dry_z = axial.dryout_location(sim, rho_gas)
    ud = sim.flame.droplet_velocity
    vd = sim.flame.droplet_spread_rate
    liquid_flux = sim.flame.liquid_mass_density * ud
    flux_wet = liquid_flux[wet] / liquid_flux[wet][0]

    assert np.isfinite(dry_z)
    assert dry_z < 0.5 * axial.WIDTH
    assert len(z) > base_points
    assert np.all(ud[wet] > 0.0)
    assert np.all(np.diff(ud[wet]) <= 1e-8)
    assert np.all(np.diff(flux_wet) <= 1e-8)

    if include_spread_drag:
        gas_spread_2mm = np.interp(0.002, z, sim.spread_rate)
        droplet_spread_2mm = np.interp(0.002, z, vd)
        assert np.all(vd[wet] >= -1e-10)
        assert droplet_spread_2mm > 0.0
        assert droplet_spread_2mm < gas_spread_2mm
    else:
        assert np.max(np.abs(vd[wet])) < 1e-10


def plot_drag_comparison(axial_sim: ct.CounterflowDiffusionFlame,
                         full_sim: ct.CounterflowDiffusionFlame,
                         rho_gas: float, outdir: Path) -> None:
    z_full = full_sim.grid * 1e3
    wet_full = wet_mask(full_sim, rho_gas)
    dry_full = axial.dryout_location(full_sim, rho_gas)
    stag = 0.5 * axial.WIDTH * 1e3

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.0), sharex=True)
    axes[0].plot(
        z_full[wet_full],
        full_sim.flame.droplet_diameter[wet_full] * 1e6,
        label="droplet",
    )
    axes[0].axhline(axial.D_MIN * 1e6, color="0.5", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("diameter [um]")

    axes[1].plot(
        z_full[wet_full],
        full_sim.flame.droplet_velocity[wet_full],
        label="droplet",
    )
    axes[1].plot(z_full, full_sim.velocity, label="gas", linestyle="--")
    axes[1].set_ylabel("axial velocity [m/s]")
    axes[1].legend()

    axes[2].plot(
        z_full[wet_full],
        full_sim.flame.droplet_spread_rate[wet_full],
        label="droplet",
    )
    axes[2].plot(z_full, full_sim.spread_rate, label="gas", linestyle="--")
    axes[2].set_xlabel("z [mm]")
    axes[2].set_ylabel("spread rate [1/s]")
    axes[2].legend()

    for ax in axes:
        ax.axvline(dry_full * 1e3, color="tab:red", linestyle=":", linewidth=1.0)
        ax.axvline(stag, color="0.3", linestyle="--", linewidth=1.0)
        ax.grid(True, alpha=0.25)
    fig.suptitle("One-way methane spray with full droplet drag")
    fig.tight_layout()
    fig.savefig(outdir / "droplet_drag_velocity_spread.png", dpi=180)
    plt.close(fig)

    z_axial = axial_sim.grid * 1e3
    wet_axial = wet_mask(axial_sim, rho_gas)
    yl_axial = axial_sim.flame.liquid_mass_density / rho_gas
    yl_full = full_sim.flame.liquid_mass_density / rho_gas

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.semilogy(z_axial[wet_axial], (yl_axial / yl_axial[0])[wet_axial],
                label="axial drag")
    ax.semilogy(z_full[wet_full], (yl_full / yl_full[0])[wet_full],
                label="axial + spread drag")
    ax.axvline(dry_full * 1e3, color="tab:red", linestyle=":", linewidth=1.0)
    ax.axvline(stag, color="0.3", linestyle="--", linewidth=1.0)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("liquid mass density / inlet")
    ax.set_title("Spread drag adds radial dilution of the liquid phase")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "droplet_drag_liquid_loading.png", dpi=180)
    plt.close(fig)


def main() -> None:
    outdir = Path(__file__).resolve().parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    axial_sim, rho_gas, axial_base_points = solve_case(include_spread_drag=False)
    full_sim, _, full_base_points = solve_case(include_spread_drag=True)

    check_case(axial_sim, rho_gas, axial_base_points, include_spread_drag=False)
    check_case(full_sim, rho_gas, full_base_points, include_spread_drag=True)

    axial_summary = summarize_case("axial", axial_sim, rho_gas, axial_base_points)
    full_summary = summarize_case("axial+spread", full_sim, rho_gas, full_base_points)
    assert full_summary["droplet_spread_rate_2mm"] > axial_summary["droplet_spread_rate_2mm"]
    assert full_summary["liquid_loading_2mm_over_inlet"] < axial_summary[
        "liquid_loading_2mm_over_inlet"
    ]
    assert full_summary["liquid_flux_2mm_over_inlet"] < axial_summary[
        "liquid_flux_2mm_over_inlet"
    ]

    plot_drag_comparison(axial_sim, full_sim, rho_gas, outdir)

    summaries = [axial_summary, full_summary]
    with (outdir / "droplet_drag_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
