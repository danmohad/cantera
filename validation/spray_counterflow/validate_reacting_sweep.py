"""Direct reacting counterflow spray auto-solve sweep.

This script exercises a small envelope of dilute, gas-assisted methane spray
counterflow flames using the public Python interface only. Each case constructs
``ct.MonodisperseSpray``, passes it to ``ct.CounterflowDiffusionFlame``, sets
the inlet boundary conditions, and calls ``solve(auto=True)``.

Run from the Cantera repository root with:

    PYTHONPATH=build/python ../.venv-cantera/bin/python \
        validation/spray_counterflow/validate_reacting_sweep.py
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
    check_counterflow_spray_diagnostics,
    summarize_counterflow_spray,
)

try:
    import CoolProp.CoolProp as CP  # noqa: E402
except ImportError:
    CP = None


P = ct.one_atm
WIDTH = 0.02
AIR = "O2:0.21,N2:0.79"
TFUEL = 300.0
TOX = 300.0
BASE_FUEL_MDOT = 0.12
BASE_OX_MDOT = 0.24
BASE_DROPLET_DIAMETER = 20e-6
MIN_DIAMETER = 2e-6
DROPLET_SPEED = 0.2


class Case:
    __slots__ = (
        "name", "fuel", "fuel_mdot", "oxidizer_mdot", "liquid_loading",
        "droplet_diameter", "expected_failure", "failure_class",
    )

    def __init__(
        self,
        name: str,
        *,
        fuel: str = "CH4:1",
        fuel_mdot: float = BASE_FUEL_MDOT,
        oxidizer_mdot: float = BASE_OX_MDOT,
        liquid_loading: float = 1e-4,
        droplet_diameter: float = BASE_DROPLET_DIAMETER,
        expected_failure: bool = False,
        failure_class: str = "",
    ):
        self.name = name
        self.fuel = fuel
        self.fuel_mdot = fuel_mdot
        self.oxidizer_mdot = oxidizer_mdot
        self.liquid_loading = liquid_loading
        self.droplet_diameter = droplet_diameter
        self.expected_failure = expected_failure
        self.failure_class = failure_class


CASES = (
    Case("baseline"),
    Case("loading_1e-3", liquid_loading=1e-3),
    Case("loading_3e-3", liquid_loading=3e-3),
    Case("diameter_15um", droplet_diameter=15e-6),
    Case("diameter_30um", droplet_diameter=30e-6),
    Case("lower_strain", fuel_mdot=0.08, oxidizer_mdot=0.16),
    Case("higher_strain", fuel_mdot=0.16, oxidizer_mdot=0.32),
    Case("fuel_20pct_n2", fuel="CH4:0.8,N2:0.2"),
)


SUMMARY_FIELDS = (
    "name",
    "status",
    "expected_failure",
    "failure_class",
    "failure_reason",
    "elapsed_s",
    "points",
    "liquid_loading",
    "droplet_diameter_um",
    "liquid_to_fuel_mass_flux_ratio",
    "fuel_mdot",
    "oxidizer_mdot",
    "fuel_ch4_mass_fraction",
    "tmax",
    "tmin",
    "flame_location_mm",
    "dryout_mm",
    "wet_points",
    "max_heat_release_rate",
    "max_ch4_mass_fraction",
    "minimum_diameter_um",
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


def make_solution(case: Case, props: dict[str, float]) -> ct.CounterflowDiffusionFlame:
    gas = ct.Solution("gri30.yaml")
    gas.transport_model = "mixture-averaged"
    gas.TPX = TFUEL, P, case.fuel
    fuel_density = gas.density

    spray = ct.MonodisperseSpray(
        fuel_species="CH4",
        diameter=case.droplet_diameter,
        minimum_droplet_diameter=MIN_DIAMETER,
        liquid_temperature=props["boiling_temperature"] - 1.0,
        liquid_mass_density=case.liquid_loading * fuel_density,
        droplet_velocity=DROPLET_SPEED,
        **props,
    )

    sim = ct.CounterflowDiffusionFlame(gas, width=WIDTH, spray=spray)
    sim.fuel_inlet.T = TFUEL
    sim.fuel_inlet.X = case.fuel
    sim.fuel_inlet.mdot = case.fuel_mdot
    sim.oxidizer_inlet.T = TOX
    sim.oxidizer_inlet.X = AIR
    sim.oxidizer_inlet.mdot = case.oxidizer_mdot
    sim.solve(loglevel=0, auto=True)
    return sim


def dryout_location(sim: ct.CounterflowDiffusionFlame) -> float:
    diameter = sim.flame.droplet_diameter
    dry = np.where(diameter <= MIN_DIAMETER * 1.01)[0]
    if len(dry) == 0:
        return float("nan")
    return float(sim.grid[dry[0]])


def summarize(
    case: Case,
    sim: ct.CounterflowDiffusionFlame,
    elapsed: float,
) -> dict[str, float | str]:
    i_fuel = sim.gas.species_index("CH4")
    sim.gas.TPX = TFUEL, P, case.fuel
    fuel_ch4_y = float(sim.gas.Y[i_fuel])
    fuel_density = sim.gas.density
    dryout = dryout_location(sim)
    liquid_density = sim.flame.liquid_mass_density
    diameter = sim.flame.droplet_diameter
    wet = (liquid_density > liquid_density[0] * 1e-12) & (
        diameter > MIN_DIAMETER * (1.0 + 1e-8)
    )
    return {
        "name": case.name,
        "status": "pass",
        "expected_failure": case.expected_failure,
        "failure_class": case.failure_class,
        "failure_reason": "",
        "elapsed_s": elapsed,
        "points": len(sim.grid),
        "liquid_loading": case.liquid_loading,
        "droplet_diameter_um": case.droplet_diameter * 1e6,
        "liquid_to_fuel_mass_flux_ratio": (
            case.liquid_loading * fuel_density * DROPLET_SPEED / case.fuel_mdot
        ),
        "fuel_mdot": case.fuel_mdot,
        "oxidizer_mdot": case.oxidizer_mdot,
        "fuel_ch4_mass_fraction": fuel_ch4_y,
        "tmax": float(np.max(sim.T)),
        "tmin": float(np.min(sim.T)),
        "flame_location_mm": float(sim.grid[np.argmax(sim.T)] * 1e3),
        "dryout_mm": dryout * 1e3,
        "wet_points": int(np.count_nonzero(wet)),
        "max_heat_release_rate": float(np.max(sim.heat_release_rate)),
        "max_ch4_mass_fraction": float(np.max(sim.Y[i_fuel])),
        "minimum_diameter_um": float(np.min(diameter) * 1e6),
        "species_sum_error": float(np.max(np.abs(np.sum(sim.Y, axis=0) - 1.0))),
    }


def check_summary(summary: dict[str, float | str]) -> None:
    assert summary["points"] >= 80
    assert summary["tmax"] > 1800.0
    assert summary["tmin"] < TFUEL
    assert summary["max_heat_release_rate"] > 1e7
    assert summary["species_sum_error"] < 1e-7
    assert summary["wet_points"] >= 8
    assert 0.5 < summary["dryout_mm"] < summary["flame_location_mm"]
    assert abs(summary["minimum_diameter_um"] - MIN_DIAMETER * 1e6) < 1e-3
    assert (
        summary["max_ch4_mass_fraction"]
        > 0.95 * summary["fuel_ch4_mass_fraction"]
    )


def failed_summary(case: Case, elapsed: float, reason: str) -> dict[str, float | str]:
    if not reason:
        reason = "post-solve physical assertion failed"
    return {
        "name": case.name,
        "status": "expected_fail" if case.expected_failure else "fail",
        "expected_failure": case.expected_failure,
        "failure_class": case.failure_class,
        "failure_reason": reason.replace("\n", " ")[:240],
        "elapsed_s": elapsed,
        "points": "",
        "liquid_loading": case.liquid_loading,
        "droplet_diameter_um": case.droplet_diameter * 1e6,
        "liquid_to_fuel_mass_flux_ratio": "",
        "fuel_mdot": case.fuel_mdot,
        "oxidizer_mdot": case.oxidizer_mdot,
        "fuel_ch4_mass_fraction": "",
        "tmax": "",
        "tmin": "",
        "flame_location_mm": "",
        "dryout_mm": "",
        "wet_points": "",
        "max_heat_release_rate": "",
        "max_ch4_mass_fraction": "",
        "minimum_diameter_um": "",
        "species_sum_error": "",
    }


def plot_results(
    solutions: dict[str, ct.CounterflowDiffusionFlame],
    summaries: list[dict[str, float | str]],
    outdir: Path,
) -> None:
    passed = [row for row in summaries if row["status"] == "pass"]
    if not passed:
        return

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.4), sharex=True)
    for row in passed:
        sim = solutions[row["name"]]
        z_mm = sim.grid * 1e3
        label = row["name"].replace("_", " ")
        axes[0].plot(z_mm, sim.T, label=label)
        axes[1].plot(z_mm, sim.flame.droplet_diameter * 1e6, label=label)
    axes[0].set_ylabel("gas temperature [K]")
    axes[1].set_ylabel("droplet diameter [um]")
    axes[1].set_xlabel("z [mm]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "reacting_sweep_profiles.png", dpi=180)
    plt.close(fig)

    labels = [row["name"].replace("_", "\n") for row in passed]
    x = np.arange(len(passed))
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.2), sharex=True)
    axes[0].bar(x, [float(row["tmax"]) for row in passed])
    axes[0].set_ylabel("Tmax [K]")

    axes[1].plot(x, [float(row["flame_location_mm"]) for row in passed],
                 "o-", label="flame")
    axes[1].plot(x, [float(row["dryout_mm"]) for row in passed],
                 "o-", label="dryout")
    axes[1].set_ylabel("location [mm]")
    axes[1].legend()

    axes[2].bar(x, [float(row["liquid_to_fuel_mass_flux_ratio"]) for row in passed])
    axes[2].set_ylabel("liquid/fuel\nmass flux")
    axes[2].set_xticks(x, labels, rotation=0)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "reacting_sweep_metrics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    outdir = Path(__file__).resolve().parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    props = methane_props()

    summaries: list[dict[str, float | str]] = []
    solutions: dict[str, ct.CounterflowDiffusionFlame] = {}
    for case in CASES:
        print(f"solving {case.name}...", flush=True)
        start = time.perf_counter()
        try:
            sim = make_solution(case, props)
            elapsed = time.perf_counter() - start
            summary = summarize(case, sim, elapsed)
            check_summary(summary)
            check_counterflow_spray_diagnostics(
                summarize_counterflow_spray(sim, "CH4"),
                min_wet_points=8,
            )
            solutions[case.name] = sim
            summaries.append(summary)
            print(
                f"  passed in {elapsed:.1f}s: "
                f"Tmax={summary['tmax']:.1f} K, "
                f"dryout={summary['dryout_mm']:.2f} mm, "
                f"flame={summary['flame_location_mm']:.2f} mm, "
                f"points={summary['points']}",
                flush=True,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            summary = failed_summary(case, elapsed, str(exc))
            summaries.append(summary)
            print(f"  failed in {elapsed:.1f}s: {exc}", flush=True)

    with (outdir / "reacting_sweep_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)

    plot_results(solutions, summaries, outdir)

    failed = [row for row in summaries if row["status"] == "fail"]
    if failed:
        names = ", ".join(str(row["name"]) for row in failed)
        raise SystemExit(f"Reacting spray auto-solve sweep failed: {names}")

    expected = [row for row in summaries if row["status"] == "expected_fail"]
    if expected:
        names = ", ".join(str(row["name"]) for row in expected)
        print(f"Reacting counterflow spray sweep passed with expected gaps: {names}")
    else:
        print("Reacting counterflow spray auto-solve sweep passed.")


if __name__ == "__main__":
    main()
