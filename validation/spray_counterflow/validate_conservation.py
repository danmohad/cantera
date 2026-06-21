"""Conservation and source-term diagnostics for counterflow spray cases.

This validation runs direct user-facing nonreacting and reacting methane spray
counterflow cases and checks the finite-difference source balances used by the
C++ equations.

Run from the Cantera repository root with:

    PYTHONPATH=build/python ../.venv-cantera/bin/python \
        validation/spray_counterflow/validate_conservation.py
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

from spray_diagnostics import (  # noqa: E402
    check_counterflow_spray_diagnostics,
    gas_continuity_residual,
    liquid_continuity_residual,
    summarize_counterflow_spray,
)
from validate_auto_solve import solve_direct_case as solve_nonreacting  # noqa: E402
from validate_reacting_auto_solve import solve_direct_case as solve_reacting  # noqa: E402


def plot_residuals(cases, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=False)
    for name, sim in cases.items():
        gas_residual, gas_scale = gas_continuity_residual(sim)
        gas_relative = np.abs(gas_residual) / np.maximum(gas_scale, 1e-300)
        axes[0].semilogy(
            sim.grid[1:-1] * 1e3,
            np.maximum(gas_relative, 1e-18),
            label=name,
        )

        liquid_residual, liquid_scale = liquid_continuity_residual(sim)
        liquid_relative = np.abs(liquid_residual) / np.maximum(liquid_scale, 1e-300)
        axes[1].semilogy(
            np.arange(len(liquid_relative)),
            np.maximum(liquid_relative, 1e-18),
            "o-",
            label=name,
        )

    axes[0].set_ylabel("gas continuity\nrelative residual")
    axes[0].set_xlabel("z [mm]")
    axes[1].set_ylabel("liquid continuity\nrelative residual")
    axes[1].set_xlabel("wet-cell index")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "conservation_residuals.png", dpi=180)
    plt.close(fig)


def main() -> None:
    outdir = Path(__file__).resolve().parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    cases = {
        "nonreacting_direct": solve_nonreacting(),
        "reacting_direct": solve_reacting(),
    }

    rows = []
    for name, sim in cases.items():
        diagnostics = summarize_counterflow_spray(sim, "CH4")
        check_counterflow_spray_diagnostics(
            diagnostics,
            min_wet_points=20,
            require_evaporation=True,
            require_energy_cooling=True,
        )
        row = {"case": name}
        row.update(diagnostics.as_dict())
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with (outdir / "conservation_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    plot_residuals(cases, outdir)

    print("Counterflow spray conservation validation passed.")
    for row in rows:
        print(
            f"{row['case']}: wet={row['wet_points']}, "
            f"gas_rel={row['max_gas_continuity_rel']:.3e}, "
            f"liquid_rel={row['max_liquid_continuity_rel']:.3e}, "
            f"species_sum={row['max_species_sum_error']:.3e}"
        )


if __name__ == "__main__":
    main()
