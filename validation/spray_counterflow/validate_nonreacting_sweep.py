"""Direct nonreacting counterflow spray robustness sweep.

Each case uses the public Python interface, calls ``solve(auto=True)``, and then
applies source-balance diagnostics. The matrix is intentionally nonreacting so
that convergence failures are easier to attribute to spray transport, dryout,
or two-way nonreacting feedback rather than chemistry.

Run from the Cantera repository root with:

    PYTHONPATH=build/python ../.venv-cantera/bin/python \
        validation/spray_counterflow/validate_nonreacting_sweep.py
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

try:
    import CoolProp.CoolProp as CP  # noqa: E402
except ImportError:
    CP = None


P = ct.one_atm
TGAS = 300.0
WIDTH = 0.02
AIR = "O2:0.21,N2:0.79"
BASE_GAS_SPEED = 0.2
BASE_DROPLET_SPEED = 1.0
BASE_DIAMETER = 40e-6
MIN_DIAMETER = 2e-6
BASE_LOADING = 1e-3


class Case:
    __slots__ = (
        "name", "liquid_loading", "droplet_diameter", "droplet_speed",
        "gas_speed", "energy_enabled", "spray_inlet", "expected_failure",
        "failure_class",
    )

    def __init__(
        self,
        name: str,
        *,
        liquid_loading: float = BASE_LOADING,
        droplet_diameter: float = BASE_DIAMETER,
        droplet_speed: float = BASE_DROPLET_SPEED,
        gas_speed: float = BASE_GAS_SPEED,
        energy_enabled: bool = True,
        spray_inlet: str = "fuel",
        expected_failure: bool = False,
        failure_class: str = "",
    ):
        self.name = name
        self.liquid_loading = liquid_loading
        self.droplet_diameter = droplet_diameter
        self.droplet_speed = droplet_speed
        self.gas_speed = gas_speed
        self.energy_enabled = energy_enabled
        self.spray_inlet = spray_inlet
        self.expected_failure = expected_failure
        self.failure_class = failure_class


CASES = (
    Case("baseline"),
    Case("loading_1e-4", liquid_loading=1e-4),
    Case("loading_3e-3", liquid_loading=3e-3),
    Case("diameter_25um", droplet_diameter=25e-6),
    Case("diameter_60um", droplet_diameter=60e-6, expected_failure=True,
         failure_class="droplet reversal before dryout"),
    Case("droplet_speed_0p4", droplet_speed=0.4),
    Case("droplet_speed_1p6", droplet_speed=1.6),
    Case("gas_speed_0p1", gas_speed=0.1),
    Case("gas_speed_0p3", gas_speed=0.3),
    Case("energy_off", energy_enabled=False),
    Case("right_injection", spray_inlet="oxidizer"),
)


SUMMARY_FIELDS = (
    "name",
    "status",
    "expected_failure",
    "failure_class",
    "failure_reason",
    "elapsed_s",
    "points",
    "spray_inlet",
    "energy_enabled",
    "liquid_loading",
    "droplet_diameter_um",
    "droplet_speed",
    "gas_speed",
    "wet_points",
    "stagnation_mm",
    "dryout_mm",
    "dryout_distance_from_inlet_mm",
    "max_ch4_mass_fraction",
    "minimum_temperature",
    "minimum_energy_source",
    "gas_continuity_rel",
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


def stagnation_plane(sim: ct.CounterflowDiffusionFlame) -> float:
    z = sim.grid
    u = sim.velocity
    sign_change = np.where(np.signbit(u[:-1]) != np.signbit(u[1:]))[0]
    if len(sign_change) == 0:
        raise AssertionError("No stagnation plane found")
    j = sign_change[0]
    return float(z[j] - u[j] * (z[j + 1] - z[j]) / (u[j + 1] - u[j]))


def solve_case(case: Case, props: dict[str, float]) -> ct.CounterflowDiffusionFlame:
    gas = ct.Solution("gri30.yaml")
    gas.transport_model = "mixture-averaged"
    gas.TPX = TGAS, P, AIR
    rho_gas = gas.density

    spray = ct.MonodisperseSpray(
        fuel_species="CH4",
        diameter=case.droplet_diameter,
        minimum_droplet_diameter=MIN_DIAMETER,
        liquid_temperature=props["boiling_temperature"] - 1.0,
        liquid_mass_density=case.liquid_loading * rho_gas,
        droplet_velocity=case.droplet_speed,
        **props,
    )

    sim = ct.CounterflowDiffusionFlame(
        gas, width=WIDTH, spray=spray, spray_inlet=case.spray_inlet
    )
    for inlet in (sim.fuel_inlet, sim.oxidizer_inlet):
        inlet.T = TGAS
        inlet.X = AIR
        inlet.mdot = rho_gas * case.gas_speed
    sim.energy_enabled = case.energy_enabled
    sim.solve(loglevel=0, auto=True)
    return sim


def summarize(case: Case, sim: ct.CounterflowDiffusionFlame, elapsed: float):
    diagnostics = summarize_counterflow_spray(sim, "CH4")
    check_counterflow_spray_diagnostics(
        diagnostics,
        min_wet_points=8,
        require_energy_cooling=case.energy_enabled,
    )
    stagnation = stagnation_plane(sim)
    dryout = diagnostics.first_dryout_mm / 1e3
    if case.spray_inlet in ("oxidizer", "right"):
        dryout_distance = WIDTH - dryout
        assert stagnation < dryout < WIDTH
    else:
        dryout_distance = dryout
        assert 0.0 < dryout < stagnation
    assert diagnostics.max_fuel_mass_fraction > 1e-4
    if case.energy_enabled:
        assert np.min(sim.T) < TGAS - 0.05
    else:
        assert np.max(np.abs(sim.T - TGAS)) < 1e-7

    return {
        "name": case.name,
        "status": "pass",
        "expected_failure": case.expected_failure,
        "failure_class": case.failure_class,
        "failure_reason": "",
        "elapsed_s": elapsed,
        "points": len(sim.grid),
        "spray_inlet": case.spray_inlet,
        "energy_enabled": case.energy_enabled,
        "liquid_loading": case.liquid_loading,
        "droplet_diameter_um": case.droplet_diameter * 1e6,
        "droplet_speed": case.droplet_speed,
        "gas_speed": case.gas_speed,
        "wet_points": diagnostics.wet_points,
        "stagnation_mm": stagnation * 1e3,
        "dryout_mm": dryout * 1e3,
        "dryout_distance_from_inlet_mm": dryout_distance * 1e3,
        "max_ch4_mass_fraction": diagnostics.max_fuel_mass_fraction,
        "minimum_temperature": float(np.min(sim.T)),
        "minimum_energy_source": float(np.min(sim.flame.spray_gas_energy_source)),
        "gas_continuity_rel": diagnostics.max_gas_continuity_rel,
        "liquid_continuity_rel": diagnostics.max_liquid_continuity_rel,
        "droplet_mass_abs": diagnostics.max_droplet_mass_abs,
        "droplet_mass_rel": diagnostics.max_droplet_mass_rel,
        "species_sum_error": diagnostics.max_species_sum_error,
    }


def failed_summary(case: Case, elapsed: float, reason: str):
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
        "spray_inlet": case.spray_inlet,
        "energy_enabled": case.energy_enabled,
        "liquid_loading": case.liquid_loading,
        "droplet_diameter_um": case.droplet_diameter * 1e6,
        "droplet_speed": case.droplet_speed,
        "gas_speed": case.gas_speed,
        "wet_points": "",
        "stagnation_mm": "",
        "dryout_mm": "",
        "dryout_distance_from_inlet_mm": "",
        "max_ch4_mass_fraction": "",
        "minimum_temperature": "",
        "minimum_energy_source": "",
        "gas_continuity_rel": "",
        "liquid_continuity_rel": "",
        "droplet_mass_abs": "",
        "droplet_mass_rel": "",
        "species_sum_error": "",
    }


def plot_summaries(rows, outdir: Path) -> None:
    passed = [row for row in rows if row["status"] == "pass"]
    if not passed:
        return
    labels = [row["name"].replace("_", "\n") for row in passed]
    x = np.arange(len(passed))

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.4), sharex=True)
    axes[0].bar(x, [float(row["wet_points"]) for row in passed])
    axes[0].set_ylabel("wet cells")

    axes[1].plot(x, [float(row["dryout_distance_from_inlet_mm"]) for row in passed],
                 "o-")
    axes[1].set_ylabel("dryout distance\nfrom inlet [mm]")

    axes[2].semilogy(x, [float(row["liquid_continuity_rel"]) for row in passed],
                     "o-", label="liquid")
    axes[2].semilogy(x, [float(row["gas_continuity_rel"]) for row in passed],
                     "o-", label="gas")
    axes[2].semilogy(x, [float(row["droplet_mass_rel"]) for row in passed],
                     "o-", label="droplet mass")
    axes[2].set_ylabel("max relative\ncontinuity residual")
    axes[2].legend()
    axes[2].set_xticks(x, labels)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "nonreacting_sweep_metrics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    outdir = Path(__file__).resolve().parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    props = methane_props()
    rows = []

    for case in CASES:
        print(f"solving {case.name}...", flush=True)
        start = time.perf_counter()
        try:
            sim = solve_case(case, props)
            elapsed = time.perf_counter() - start
            row = summarize(case, sim, elapsed)
            rows.append(row)
            print(
                f"  passed in {elapsed:.1f}s: wet={row['wet_points']}, "
                f"dryout={row['dryout_mm']:.2f} mm, "
                f"gas_rel={row['gas_continuity_rel']:.2e}, "
                f"liquid_rel={row['liquid_continuity_rel']:.2e}",
                flush=True,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            rows.append(failed_summary(case, elapsed, str(exc)))
            print(f"  failed in {elapsed:.1f}s: {exc}", flush=True)

    with (outdir / "nonreacting_sweep_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    plot_summaries(rows, outdir)

    failed = [row for row in rows if row["status"] == "fail"]
    if failed:
        names = ", ".join(str(row["name"]) for row in failed)
        raise SystemExit(f"Nonreacting spray auto-solve sweep failed: {names}")
    expected = [row for row in rows if row["status"] == "expected_fail"]
    if expected:
        names = ", ".join(str(row["name"]) for row in expected)
        print(f"Nonreacting counterflow spray sweep passed with expected gaps: {names}")
    else:
        print("Nonreacting counterflow spray auto-solve sweep passed.")


if __name__ == "__main__":
    main()
