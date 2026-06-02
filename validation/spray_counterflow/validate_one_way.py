"""One-way counterflow spray validation for the local Cantera build.

This script intentionally validates the simplest robust subset first:

* symmetric nonreacting air-air counterflow, energy equation disabled
* monodisperse liquid methane spray injected from the left
* gas-phase spray feedback disabled
* droplet drag disabled
* droplet heat transfer and evaporation enabled

Run from the Cantera repository root with:

    ../.venv-cantera/bin/python validation/spray_counterflow/validate_one_way.py
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
D_MIN = 2e-6
LIQUID_LOADING = 1e-6


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
    sim.set_initial_guess(mode="linear")
    sim.set_refine_criteria(ratio=3.0, slope=0.03, curve=0.06, prune=0.0)
    sim.set_max_grid_points(sim.flame, 900)


def stagnation_plane(sim: ct.CounterflowDiffusionFlame) -> float:
    z = sim.grid
    u = sim.velocity
    sign_change = np.where(np.signbit(u[:-1]) != np.signbit(u[1:]))[0]
    if len(sign_change) == 0:
        raise AssertionError("No stagnation plane found")
    j = sign_change[0]
    return z[j] - u[j] * (z[j + 1] - z[j]) / (u[j + 1] - u[j])


def solve_gas_only() -> ct.CounterflowDiffusionFlame:
    gas = air()
    sim = ct.CounterflowDiffusionFlame(gas, width=WIDTH)
    configure_counterflow(sim, gas.density)
    sim.solve(loglevel=0, refine_grid=True, auto=False)
    z_stag = stagnation_plane(sim)
    assert abs(z_stag - 0.5 * WIDTH) < 1e-8
    return sim


def make_spray(d0: float, props: dict[str, float], rho_gas: float) -> ct.MonodisperseSpray:
    return ct.MonodisperseSpray(
        fuel_species="CH4",
        diameter=d0,
        minimum_droplet_diameter=D_MIN,
        liquid_temperature=props["boiling_temperature"] - 1.0,
        liquid_mass_density=LIQUID_LOADING * rho_gas,
        droplet_velocity=DROPLET_SPEED,
        **props,
    )


def solve_spray(d0: float, props: dict[str, float]) -> ct.CounterflowDiffusionFlame:
    gas = air()
    spray = make_spray(d0, props, gas.density)
    sim = ct.CounterflowDiffusionFlame(gas, width=WIDTH, spray=spray)
    configure_counterflow(sim, gas.density)
    sim.flame.gas_phase_spray_sources_enabled = False
    sim.flame.droplet_drag_enabled = False
    sim.solve(loglevel=0, refine_grid=True, auto=False)
    return sim


def d2_fit_r2(z: np.ndarray, d: np.ndarray) -> float:
    wet = d > D_MIN * 1.05
    window = wet & (z > 1e-3)
    if np.count_nonzero(window) < 5:
        return float("nan")
    x = z[window]
    y = d[window] ** 2
    p = np.polyfit(x, y, 1)
    yhat = np.polyval(p, x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot


def summarize_case(sim: ct.CounterflowDiffusionFlame, d0: float, rho_gas: float) -> dict[str, float]:
    z = sim.grid
    d = sim.flame.droplet_diameter
    yl = sim.flame.liquid_mass_density / rho_gas
    wet = yl > 1e-16
    dry_candidates = np.where((d <= D_MIN * 1.0001) | ~wet)[0]
    dry_z = float(z[dry_candidates[0]]) if len(dry_candidates) else float("nan")

    active = wet & (d > D_MIN * 1.0001)
    if np.count_nonzero(active) > 1:
        assert np.all(np.diff(d[active]) <= 1e-10)
        assert np.all(np.diff(yl[active]) <= 1e-13)

    return {
        "d0_um": d0 * 1e6,
        "points": len(z),
        "dryout_mm": dry_z * 1e3 if np.isfinite(dry_z) else float("nan"),
        "diameter_end_um": d[-1] * 1e6,
        "diameter_5mm_um": np.interp(0.005, z, d) * 1e6,
        "liquid_loading_5mm_over_inlet": np.interp(0.005, z, yl) / yl[0],
        "max_evaporation_rate": float(np.max(sim.flame.evaporation_rate)),
        "d2_fit_r2": d2_fit_r2(z, d),
    }


def plot_profiles(cases: dict[float, ct.CounterflowDiffusionFlame], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for d0, sim in cases.items():
        ax.plot(sim.grid * 1e3, sim.flame.droplet_diameter * 1e6, label=f"{d0 * 1e6:.0f} um")
    ax.axvline(0.5 * WIDTH * 1e3, color="0.3", linestyle="--", linewidth=1.0, label="stagnation")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("droplet diameter [um]")
    ax.set_title("One-way methane spray evaporation, drag off")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "one_way_diameter_profiles.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for d0, sim in cases.items():
        yl = sim.flame.liquid_mass_density / sim.flame.liquid_mass_density[0]
        ax.semilogy(sim.grid * 1e3, np.maximum(yl, 1e-18), label=f"{d0 * 1e6:.0f} um")
    ax.axvline(0.5 * WIDTH * 1e3, color="0.3", linestyle="--", linewidth=1.0)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("liquid mass density / inlet")
    ax.set_title("Resolved liquid loading decay")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "one_way_liquid_loading_profiles.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for d0, sim in cases.items():
        yl = sim.flame.liquid_mass_density / sim.flame.liquid_mass_density[0]
        wet = (yl > 1e-9) & (sim.flame.droplet_diameter > D_MIN * 1.0001)
        ax.plot(
            sim.grid[wet] * 1e3,
            sim.flame.droplet_temperature[wet],
            label=f"{d0 * 1e6:.0f} um",
        )
    ax.axvline(0.5 * WIDTH * 1e3, color="0.3", linestyle="--", linewidth=1.0)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("droplet temperature [K]")
    ax.set_title("Droplet thermal relaxation")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "one_way_droplet_temperature_profiles.png", dpi=180)
    plt.close(fig)


def plot_dryout(summary: list[dict[str, float]], outdir: Path) -> None:
    finite = [row for row in summary if np.isfinite(row["dryout_mm"])]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(
        [row["d0_um"] for row in finite],
        [row["dryout_mm"] for row in finite],
        marker="o",
    )
    ax.axhline(0.5 * WIDTH * 1e3, color="0.3", linestyle="--", linewidth=1.0)
    ax.set_xlabel("inlet diameter [um]")
    ax.set_ylabel("dryout location [mm]")
    ax.set_title("Dryout moves downstream with inlet diameter")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "one_way_dryout_scaling.png", dpi=180)
    plt.close(fig)


def main() -> None:
    outdir = Path(__file__).resolve().parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    gas_only = solve_gas_only()
    print(f"gas-only points={len(gas_only.grid)} stagnation={stagnation_plane(gas_only) * 1e3:.6f} mm")

    props = methane_props()
    cases: dict[float, ct.CounterflowDiffusionFlame] = {}
    summary: list[dict[str, float]] = []
    rho_gas = air().density
    for d0_um in (30.0, 40.0, 50.0, 60.0, 70.0):
        sim = solve_spray(d0_um * 1e-6, props)
        cases[d0_um * 1e-6] = sim
        row = summarize_case(sim, d0_um * 1e-6, rho_gas)
        summary.append(row)
        print(row)

    dry = [row["dryout_mm"] for row in summary if np.isfinite(row["dryout_mm"])]
    assert dry == sorted(dry), "Dryout location should increase with inlet diameter"
    assert summary[0]["dryout_mm"] < 0.5 * WIDTH * 1e3
    assert summary[3]["dryout_mm"] > 0.5 * WIDTH * 1e3
    assert not np.isfinite(summary[4]["dryout_mm"])
    assert min(row["d2_fit_r2"] for row in summary[:4]) > 0.95

    plot_profiles(cases, outdir)
    plot_dryout(summary, outdir)

    with (outdir / "one_way_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
