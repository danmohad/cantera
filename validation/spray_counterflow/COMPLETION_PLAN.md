# Spray Flamelets Completion Plan

This file is the working execution plan for bringing the dilute monodisperse
spray flamelet implementation to a PR-ready state. The definition of done is:

- full two-way coupled mass, gas species, gas momentum, and gas energy feedback
  in strained configurations
- physically consistent no-slip spray coupling for freely propagating flames
- direct Python setup with `solve(auto=True)` for expected user workflows
- robust convergence over a documented dilute operating envelope
- numerical and physical verification tests that fail on conservation, source
  sign, dryout, or coupling regressions
- clear documentation of model scope and known limits

The formulation remains a dilute Eulerian-Eulerian monodisperse spray model.
It is not a pure liquid-fuel flame model and it is not expected to represent
dense sprays, droplet crossing, or post-turning-point droplet trajectories.

## Acceptance Gates

### Gate 1: Source-Term And Conservation Verification

Add reusable diagnostics and tests that verify, on converged solutions:

- liquid continuity residual is small in every wet cell:
  `d(rho_l u_l)/dz + 2 rho_l V_l + m_evap = 0`
- integrated gas mass source matches integrated liquid mass loss within the
  discretization tolerance
- species source terms preserve total gas species normalization
- spray fuel source has the correct sign and is localized to the wet/evaporating
  region
- gas energy source is negative for cold evaporating droplets and vanishes after
  dryout
- radial gas momentum source has the expected sign relative to droplet/gas
  spread-rate relaxation
- droplet axial velocity does not reverse before dryout unless the user disables
  the reversal guard explicitly
- dryout sets liquid density to zero, clamps diameter to the configured minimum,
  and does not create active gas or droplet source terms

Required artifacts:

- a fast Python test in `test/python/test_onedim.py`
- a validation diagnostic script in `validation/spray_counterflow`
- a CSV/plot artifact summarizing physical residuals

### Gate 2: Counterflow Nonreacting Robustness

Expand the nonreacting direct `solve(auto=True)` matrix over:

- liquid loading
- droplet diameter
- droplet injection velocity
- gas strain rate
- left/right spray injection where physically admissible
- energy equation on/off

Pass criteria:

- every case converges without validation-only staging helpers
- wet regions are resolved by multiple grid cells
- dryout is not a first-cell artifact
- source signs, integral balances, and species sums pass Gate 1 diagnostics
- failures, if any, are classified as physical/model-limit/solver-limit

### Gate 3: Counterflow Reacting Robustness

Expand the gas-assisted methane-air reacting matrix over:

- liquid loading beyond the current `3e-3` liquid-to-gas density ratio
- droplet diameter and velocity
- fuel dilution and oxidizer preheat
- strain rate and flame-position shifts
- cases with droplets drying well before the flame, near the flame, and as close
  as the dilute model can robustly support

Pass criteria:

- direct `solve(auto=True)` reaches a lit flame or a clearly classified physical
  extinction/model-limit case
- two-way mass/species/momentum/energy feedback remains enabled for accepted
  strained cases
- conservation diagnostics remain within documented tolerances
- flame temperature, heat-release location, dryout location, and source
  magnitudes vary smoothly across continuation parameters

### Gate 4: Wall-Stagnation Workflow

Bring `ct.ImpingingJet(..., spray=spray)` to the same user-facing standard:

- add or reuse spray-aware auto staging
- validate nonreacting and reacting wall-stagnation cases
- verify source balances, dryout behavior, and wall-adjacent robustness
- document any boundary-specific limitations

### Gate 5: Free-Flame Workflow

Complete free-flame support under the no-slip assumption:

- droplet velocity follows the gas axial velocity
- no independent gas axial momentum feedback is claimed
- mass/species/energy feedback are verified
- automatic solve handles zero, tiny, and finite dilute loadings
- tests document the modeling limitation relative to strained two-way momentum
  coupling

### Gate 6: API, Documentation, And PR Readiness

Before PR:

- add user-facing docs and examples
- add concise C++/Python API documentation for spray parameters and switches
- consolidate validation scripts so they are reproducible but not burdensome
  for the normal test suite
- add release-note/changelog material if required by Cantera guidelines
- split the work into logical commits
- update `spray_flamelets_pr.md` with scope, limitations, tests, and validation
  results

## Execution Order

1. Implement Gate 1 diagnostics and run them on existing nonreacting and
   reacting counterflow cases.
2. Harden direct counterflow auto-solve wherever Gate 1 exposes residual,
   source, or dryout failures.
3. Expand nonreacting and reacting counterflow sweeps until the robust envelope
   is useful and well characterized.
4. Generalize spray auto-staging to wall-stagnation and validate it.
5. Finish free-flame no-slip validation and document its momentum limitation.
6. Clean up API/docs/tests and prepare the PR.

## Current Status

As of the latest validation pass, the current PR-scope implementation has a
working direct Python path for counterflow, wall-stagnation, and no-slip free
flames. The full validation battery in `validation/spray_counterflow` passes:
staged one-way evaporation/drag/gas-feedback checks, direct nonreacting
counterflow and wall-stagnation checks, a nonreacting counterflow sweep, reacting
counterflow and wall-stagnation sweeps, and a no-slip free-flame loading sweep.

The remaining work is PR mechanics: commit/push cleanup and final PR text.
Broader envelope expansion beyond the accepted dilute range is follow-up work,
not a blocker for the current validated scope.

Gate 1 is closed for the current counterflow, wall-stagnation, and free-flame
validation scope:

- `spray_diagnostics.py` provides reusable gas/liquid continuity, dryout source
  shutoff, single-droplet mass-equation, species normalization, spray cooling,
  and radial momentum damping diagnostics.
- `validate_conservation.py` runs those diagnostics on the direct nonreacting
  and reacting baseline counterflow cases.
- `validate_nonreacting_sweep.py`, `validate_reacting_sweep.py`,
  `validate_wall_stagnation.py`, and `validate_free_flame.py` apply the same
  source-sign, dryout, and residual checks across the current validation
  envelope.
- `test/python/test_onedim.py::TestMonodisperseSpray` includes a fast
  H2-based gas continuity, left- and right-injection liquid continuity,
  droplet-mass, and dryout-source regression check.

Future envelope widening should add corresponding residual/source diagnostics
for any newly accepted regimes.

Gate 2 is closed for the current nonreacting counterflow envelope:

- `validate_nonreacting_sweep.py` runs direct `solve(auto=True)` nonreacting
  counterflow cases with conservation diagnostics.
- The current accepted nonreacting envelope covers liquid density ratios
  `1e-4` to `3e-3`, diameters `25-40 um`, droplet injection speeds
  `0.4-1.6 m/s`, gas speeds `0.1-0.3 m/s`, left- and right-side injection,
  and both energy-enabled and fixed-temperature solves.
- The current named nonreacting model-limit case is `60 um` droplets, which
  trigger the droplet-reversal guard before dryout.

The previously blocking low-strain, high-strain, and right-side injection
failures have been fixed. Additional cases can widen the documented envelope,
but the current direct nonreacting gate is passing.

Gate 3 is closed for the current reacting counterflow envelope:

- `validate_reacting_sweep.py` applies the same diagnostics to direct
  gas-assisted reacting methane-air counterflow cases.
- The current accepted reacting envelope covers liquid density ratios `1e-4` to
  `3e-3`, droplet diameters `15-30 um`, lower/baseline/higher strain flow
  rates, and a `CH4:0.8,N2:0.2` gaseous fuel dilution case.
- The previous lower-strain stale-source gap is fixed: the accepted case reports
  nonzero evaporation, a resolved wet region, and dryout upstream of the flame.

Future reacting-counterflow work should widen this envelope to include droplet
velocity variation, oxidizer preheat, and cases where droplets dry closer to the
flame.

Gate 4 is closed for the current wall-stagnation scope:

- `ct.ImpingingJet(..., spray=spray).solve(auto=True)` now uses spray-aware
  staged source continuation.
- `validate_wall_stagnation.py` validates a direct nonreacting wall-stagnation
  methane-spray case with gas/liquid/droplet residual diagnostics and plot/CSV
  artifacts.
- `validate_reacting_wall_stagnation.py` validates direct premixed methane-air
  reacting wall-stagnation cases over liquid loadings `1e-7` to `1e-5`, with
  lit flames, resolved dryout upstream of the heat-release region, and the same
  source-balance diagnostics.
- `test/python/test_onedim.py::TestMonodisperseSpray` includes a lightweight
  impinging-jet spray auto-solve regression.

Future wall-stagnation work should widen this envelope to include more
strain/wall-temperature/loading variation, but the current direct
wall-stagnation gate is passing.

Gate 5 is closed for the current no-slip free-flame scope:

- `ct.FreeFlame(..., spray=spray).solve(auto=True)` bootstraps from the
  zero-loading flame and solves the finite no-slip spray on the refined gas
  flame grid to avoid unbounded dryout-front refinement.
- `validate_free_flame.py` validates premixed methane-air flames at `phi=0.9`
  with liquid methane loadings `1e-6`, `1e-5`, and `1e-4`.
- The accepted free-flame cases enforce no-slip velocity to roundoff, dry out
  upstream of the temperature peak, preserve species normalization, and show
  monotone evaporation/cooling response with loading.

Gate 5 remains intentionally limited to no-slip mass/species/energy coupling;
full axial two-way momentum coupling is outside Cantera's current free-flame
equation set.

Gate 6 status:

- shared Python staging helpers are consolidated in `_SprayAutoSolveMixin`
- validation artifacts are generated by the scripts in this directory
- `README.md` and `spray_flamelets_pr.md` describe the validated scope
- focused unit tests, the full spray validation battery, and the broader
  upstream `scons test -j8` suite pass locally
- logical commits have been created and pushed to `fork/spray-flamelets`
