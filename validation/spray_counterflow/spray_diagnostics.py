"""Physical diagnostics for one-dimensional spray validation cases."""

from __future__ import annotations

import numpy as np


class SprayDiagnostics:
    __slots__ = (
        "points",
        "wet_points",
        "dry_points",
        "max_gas_continuity_abs",
        "max_gas_continuity_rel",
        "max_liquid_continuity_abs",
        "max_liquid_continuity_rel",
        "max_droplet_mass_abs",
        "max_droplet_mass_rel",
        "max_species_sum_error",
        "max_dry_evaporation_rate",
        "max_dry_heat_transfer_rate",
        "max_dry_energy_source",
        "max_dry_momentum_source",
        "min_wet_energy_source",
        "min_momentum_damping_product",
        "max_evaporation_rate",
        "max_fuel_mass_fraction",
        "minimum_diameter_um",
        "first_dryout_mm",
    )

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs[name])

    def as_dict(self) -> dict[str, float | int]:
        return {
            "points": self.points,
            "wet_points": self.wet_points,
            "dry_points": self.dry_points,
            "max_gas_continuity_abs": self.max_gas_continuity_abs,
            "max_gas_continuity_rel": self.max_gas_continuity_rel,
            "max_liquid_continuity_abs": self.max_liquid_continuity_abs,
            "max_liquid_continuity_rel": self.max_liquid_continuity_rel,
            "max_droplet_mass_abs": self.max_droplet_mass_abs,
            "max_droplet_mass_rel": self.max_droplet_mass_rel,
            "max_species_sum_error": self.max_species_sum_error,
            "max_dry_evaporation_rate": self.max_dry_evaporation_rate,
            "max_dry_heat_transfer_rate": self.max_dry_heat_transfer_rate,
            "max_dry_energy_source": self.max_dry_energy_source,
            "max_dry_momentum_source": self.max_dry_momentum_source,
            "min_wet_energy_source": self.min_wet_energy_source,
            "min_momentum_damping_product": self.min_momentum_damping_product,
            "max_evaporation_rate": self.max_evaporation_rate,
            "max_fuel_mass_fraction": self.max_fuel_mass_fraction,
            "minimum_diameter_um": self.minimum_diameter_um,
            "first_dryout_mm": self.first_dryout_mm,
        }


def _max_or_zero(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(values))


def _min_or_zero(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.min(values))


def _safe_max_relative(residual: np.ndarray, scale: np.ndarray) -> float:
    if residual.size == 0:
        return 0.0
    return float(np.max(np.abs(residual) / np.maximum(scale, 1e-300)))


def _wet_and_dry_masks(sim):
    liquid_density = np.asarray(sim.flame.liquid_mass_density)
    diameter = np.asarray(sim.flame.droplet_diameter)
    min_diameter = sim.flame.minimum_droplet_diameter
    density_reference = max(float(np.max(liquid_density)), 1e-300)
    wet = (
        (liquid_density > density_reference * 1e-12)
        & (diameter > min_diameter * (1.0 + 1e-8))
    )
    dry = (
        (diameter <= min_diameter * (1.0 + 1e-8))
        | (liquid_density <= density_reference * 1e-30)
    )
    return wet, dry


def gas_continuity_residual(sim) -> tuple[np.ndarray, np.ndarray]:
    """Return axisymmetric gas continuity residual and scale at interior points."""
    z = np.asarray(sim.grid)
    j = np.arange(1, len(z) - 1)
    if j.size == 0:
        return np.array([]), np.array([])
    dz = np.diff(z)
    rho = np.asarray(sim.density)
    rho_u = rho * np.asarray(sim.velocity)
    spread = np.asarray(sim.spread_rate)
    source = np.asarray(sim.flame.evaporation_rate)

    flux_divergence = (rho_u[j + 1] - rho_u[j]) / dz[j]
    radial_divergence = rho[j + 1] * spread[j + 1] + rho[j] * spread[j]
    residual = -flux_divergence - radial_divergence + source[j]
    scale = np.abs(flux_divergence) + np.abs(radial_divergence) + np.abs(source[j])
    return residual, scale


def free_flame_gas_continuity_residual(sim) -> tuple[np.ndarray, np.ndarray]:
    """Return planar free-flame gas continuity residual away from the anchor."""
    z = np.asarray(sim.grid)
    j = np.arange(1, len(z) - 1)
    if j.size == 0:
        return np.array([]), np.array([])
    dz = np.diff(z)
    rho_u = np.asarray(sim.density) * np.asarray(sim.velocity)
    source = np.asarray(sim.flame.evaporation_rate)
    zfixed = sim.fixed_temperature_location

    residual = []
    scale = []
    for jloc in j:
        if np.isclose(z[jloc], zfixed, rtol=0.0, atol=1e-14):
            continue
        if z[jloc] > zfixed:
            flux_divergence = (rho_u[jloc] - rho_u[jloc - 1]) / dz[jloc - 1]
        else:
            flux_divergence = (rho_u[jloc + 1] - rho_u[jloc]) / dz[jloc]
        residual.append(-flux_divergence + source[jloc])
        scale.append(abs(flux_divergence) + abs(source[jloc]))
    return np.asarray(residual), np.asarray(scale)


def liquid_continuity_residual(sim) -> tuple[np.ndarray, np.ndarray]:
    """Return dispersed-phase continuity residual and scale in wet interior cells."""
    z = np.asarray(sim.grid)
    if len(z) < 3:
        return np.array([]), np.array([])
    dz = np.diff(z)
    wet, _ = _wet_and_dry_masks(sim)
    liquid_density = np.asarray(sim.flame.liquid_mass_density)
    droplet_velocity = np.asarray(sim.flame.droplet_velocity)
    if getattr(sim.flame, "type", "") == "spray-free-flow":
        droplet_spread_rate = np.zeros_like(liquid_density)
    else:
        droplet_spread_rate = np.asarray(sim.flame.droplet_spread_rate)
    source = np.asarray(sim.flame.evaporation_rate)
    flux = liquid_density * droplet_velocity

    residual = []
    scale = []
    for j in range(1, len(z) - 1):
        if not wet[j]:
            continue
        if droplet_velocity[j] > 0.0:
            flux_divergence = (flux[j] - flux[j - 1]) / dz[j - 1]
        else:
            flux_divergence = (flux[j + 1] - flux[j]) / dz[j]
        radial_divergence = 2.0 * liquid_density[j] * droplet_spread_rate[j]
        residual.append(-flux_divergence - radial_divergence - source[j])
        scale.append(
            abs(flux_divergence) + abs(radial_divergence) + abs(source[j])
        )
    return np.asarray(residual), np.asarray(scale)


def droplet_mass_residual(sim) -> tuple[np.ndarray, np.ndarray]:
    """Return single-droplet mass-equation residual and scale in wet cells."""
    z = np.asarray(sim.grid)
    if len(z) < 3:
        return np.array([]), np.array([])
    dz = np.diff(z)
    wet, _ = _wet_and_dry_masks(sim)
    liquid_density = np.asarray(sim.flame.liquid_mass_density)
    droplet_velocity = np.asarray(sim.flame.droplet_velocity)
    droplet_mass = np.asarray(sim.flame.values("droplet-mass"))
    source = np.asarray(sim.flame.evaporation_rate)

    residual = []
    scale = []
    for j in range(1, len(z) - 1):
        if not wet[j]:
            continue
        jloc = j if droplet_velocity[j] > 0.0 else j + 1
        derivative = (droplet_mass[jloc] - droplet_mass[jloc - 1]) / dz[jloc - 1]
        number_density = liquid_density[j] / max(droplet_mass[j], 1e-300)
        single_droplet_evaporation = (
            source[j] / number_density if number_density > 0.0 else 0.0
        )
        advective_term = droplet_velocity[j] * derivative
        residual.append(-advective_term - single_droplet_evaporation)
        scale.append(abs(advective_term) + abs(single_droplet_evaporation))
    return np.asarray(residual), np.asarray(scale)


def summarize_counterflow_spray(sim, fuel_species: str) -> SprayDiagnostics:
    sim.eval()
    wet, dry = _wet_and_dry_masks(sim)
    gas_residual, gas_scale = gas_continuity_residual(sim)
    liquid_residual, liquid_scale = liquid_continuity_residual(sim)
    mass_residual, mass_scale = droplet_mass_residual(sim)

    evaporation = np.asarray(sim.flame.evaporation_rate)
    heat = np.asarray(sim.flame.spray_heat_transfer_rate)
    energy = np.asarray(sim.flame.spray_gas_energy_source)
    momentum = np.asarray(sim.flame.spray_gas_momentum_source)
    spread_slip = np.asarray(sim.flame.droplet_spread_rate) - np.asarray(
        sim.spread_rate
    )
    diameter = np.asarray(sim.flame.droplet_diameter)
    dry_indices = np.where(dry)[0]
    first_dryout = float("nan")
    if dry_indices.size:
        if getattr(sim.flame, "spray_inlet", "left") == "right":
            first_dryout = float(sim.grid[dry_indices[-1]] * 1e3)
        else:
            first_dryout = float(sim.grid[dry_indices[0]] * 1e3)

    i_fuel = sim.gas.species_index(fuel_species)
    return SprayDiagnostics(
        points=len(sim.grid),
        wet_points=int(np.count_nonzero(wet)),
        dry_points=int(np.count_nonzero(dry)),
        max_gas_continuity_abs=_max_or_zero(np.abs(gas_residual)),
        max_gas_continuity_rel=_safe_max_relative(gas_residual, gas_scale),
        max_liquid_continuity_abs=_max_or_zero(np.abs(liquid_residual)),
        max_liquid_continuity_rel=_safe_max_relative(liquid_residual, liquid_scale),
        max_droplet_mass_abs=_max_or_zero(np.abs(mass_residual)),
        max_droplet_mass_rel=_safe_max_relative(mass_residual, mass_scale),
        max_species_sum_error=float(np.max(np.abs(np.sum(sim.Y, axis=0) - 1.0))),
        max_dry_evaporation_rate=_max_or_zero(np.abs(evaporation[dry])),
        max_dry_heat_transfer_rate=_max_or_zero(np.abs(heat[dry])),
        max_dry_energy_source=_max_or_zero(np.abs(energy[dry])),
        max_dry_momentum_source=_max_or_zero(np.abs(momentum[dry])),
        min_wet_energy_source=_min_or_zero(energy[wet]),
        min_momentum_damping_product=_min_or_zero((momentum * spread_slip)[wet]),
        max_evaporation_rate=_max_or_zero(evaporation),
        max_fuel_mass_fraction=float(np.max(sim.Y[i_fuel])),
        minimum_diameter_um=float(np.min(diameter) * 1e6),
        first_dryout_mm=first_dryout,
    )


def summarize_free_flame_spray(sim, fuel_species: str) -> SprayDiagnostics:
    sim.eval()
    wet, dry = _wet_and_dry_masks(sim)
    gas_residual, gas_scale = free_flame_gas_continuity_residual(sim)
    liquid_residual, liquid_scale = liquid_continuity_residual(sim)
    mass_residual, mass_scale = droplet_mass_residual(sim)

    evaporation = np.asarray(sim.flame.evaporation_rate)
    heat = np.asarray(sim.flame.spray_heat_transfer_rate)
    energy = np.asarray(sim.flame.spray_gas_energy_source)
    momentum = np.asarray(sim.flame.spray_gas_momentum_source)
    diameter = np.asarray(sim.flame.droplet_diameter)
    dry_indices = np.where(dry)[0]
    first_dryout = float("nan")
    if dry_indices.size:
        first_dryout = float(sim.grid[dry_indices[0]] * 1e3)

    i_fuel = sim.gas.species_index(fuel_species)
    return SprayDiagnostics(
        points=len(sim.grid),
        wet_points=int(np.count_nonzero(wet)),
        dry_points=int(np.count_nonzero(dry)),
        max_gas_continuity_abs=_max_or_zero(np.abs(gas_residual)),
        max_gas_continuity_rel=_safe_max_relative(gas_residual, gas_scale),
        max_liquid_continuity_abs=_max_or_zero(np.abs(liquid_residual)),
        max_liquid_continuity_rel=_safe_max_relative(liquid_residual, liquid_scale),
        max_droplet_mass_abs=_max_or_zero(np.abs(mass_residual)),
        max_droplet_mass_rel=_safe_max_relative(mass_residual, mass_scale),
        max_species_sum_error=float(np.max(np.abs(np.sum(sim.Y, axis=0) - 1.0))),
        max_dry_evaporation_rate=_max_or_zero(np.abs(evaporation[dry])),
        max_dry_heat_transfer_rate=_max_or_zero(np.abs(heat[dry])),
        max_dry_energy_source=_max_or_zero(np.abs(energy[dry])),
        max_dry_momentum_source=_max_or_zero(np.abs(momentum[dry])),
        min_wet_energy_source=_min_or_zero(energy[wet]),
        min_momentum_damping_product=0.0,
        max_evaporation_rate=_max_or_zero(evaporation),
        max_fuel_mass_fraction=float(np.max(sim.Y[i_fuel])),
        minimum_diameter_um=float(np.min(diameter) * 1e6),
        first_dryout_mm=first_dryout,
    )


def check_counterflow_spray_diagnostics(
    diagnostics: SprayDiagnostics,
    *,
    min_wet_points: int,
    require_evaporation: bool = True,
    require_energy_cooling: bool = True,
) -> None:
    assert diagnostics.wet_points >= min_wet_points
    # The largest residuals occur at the final wet cell next to the dryout
    # cutoff, where source limiting and diameter clamping deliberately replace
    # a smooth continuum droplet field. The bulk wet-region residuals are much
    # smaller; these gates catch sign/coupling regressions without treating the
    # cutoff front as physical smoothness.
    assert (
        diagnostics.max_gas_continuity_rel < 1e-4
        or diagnostics.max_gas_continuity_abs < 1e-6
    )
    assert (
        diagnostics.max_liquid_continuity_rel < 1e-2
        or diagnostics.max_liquid_continuity_abs < 1e-6
    )
    assert (
        diagnostics.max_droplet_mass_rel < 1e-2
        or diagnostics.max_droplet_mass_abs < 1e-12
    )
    assert diagnostics.max_species_sum_error < 1e-7
    assert diagnostics.max_dry_evaporation_rate < 1e-20
    assert diagnostics.max_dry_heat_transfer_rate < 1e-16
    assert diagnostics.max_dry_energy_source < 1e-12
    assert diagnostics.max_dry_momentum_source < 1e-16
    assert diagnostics.min_momentum_damping_product > -1e-12
    if require_evaporation:
        assert diagnostics.max_evaporation_rate > 0.0
    if require_energy_cooling:
        assert diagnostics.min_wet_energy_source < 0.0


def check_free_flame_spray_diagnostics(
    diagnostics: SprayDiagnostics,
    *,
    min_wet_points: int,
    require_evaporation: bool = True,
    require_energy_cooling: bool = True,
) -> None:
    assert diagnostics.wet_points >= min_wet_points
    assert (
        diagnostics.max_gas_continuity_rel < 1e-4
        or diagnostics.max_gas_continuity_abs < 1e-6
    )
    assert (
        diagnostics.max_liquid_continuity_rel < 1e-2
        or diagnostics.max_liquid_continuity_abs < 1e-6
    )
    assert (
        diagnostics.max_droplet_mass_rel < 1e-2
        or diagnostics.max_droplet_mass_abs < 1e-12
    )
    assert diagnostics.max_species_sum_error < 1e-7
    assert diagnostics.max_dry_evaporation_rate < 1e-20
    assert diagnostics.max_dry_heat_transfer_rate < 1e-16
    assert diagnostics.max_dry_energy_source < 1e-12
    assert diagnostics.max_dry_momentum_source < 1e-16
    if require_evaporation:
        assert diagnostics.max_evaporation_rate > 0.0
    if require_energy_cooling:
        assert diagnostics.min_wet_energy_source < 0.0
