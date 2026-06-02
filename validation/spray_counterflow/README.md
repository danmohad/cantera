# Spray Counterflow Validation

This directory records the staged validation status for the spray flamelet work.

## Current validated stages

`validate_one_way.py` validates one-way, nonreacting counterflow evaporation:

- symmetric air-air counterflow at 1 atm and 300 K
- liquid methane droplets injected from the left
- gas-phase spray feedback disabled
- droplet drag disabled
- droplet heat transfer and evaporation enabled
- energy equation disabled

The checks are intentionally physical:

- gas-only stagnation plane is at the midplane
- droplet diameter and liquid loading decrease monotonically while droplets are wet
- dryout moves downstream as inlet droplet diameter increases
- the wet-region diameter follows an approximate `D^2` trend away from startup/dryout

`validate_axial_drag.py` validates one-way axial droplet drag over the same
counterflow gas field:

- the no-drag evaporation case is solved first on a refined grid
- the one-way droplet ODEs are integrated over that gas field to initialize
  mass, temperature, axial velocity, and liquid density together
- axial drag is enabled, gas-phase feedback remains disabled, spread-rate drag
  remains disabled
- the drag-on solution is converged and then refined dynamically

The checks are physical:

- droplets decelerate monotonically while wet
- liquid mass flux decreases monotonically even though local liquid density can
  initially rise from axial compression
- droplets dry out before the stagnation plane
- the drag-enabled refinement adds grid points around the droplet relaxation and
  dryout layer

`validate_droplet_drag.py` validates full one-way droplet drag in the strained
counterflow equations:

- axial-only drag and axial-plus-spread drag cases are solved from staged
  droplet initial guesses
- the spread-rate equation is enabled while gas-phase feedback remains disabled
- dynamic refinement is active for both droplet axial velocity and droplet
  spread rate
- droplet spread rate is positive on the injector side and remains below the
  gas spread rate while droplets are wet
- spread drag reduces liquid loading and liquid flux relative to axial-only drag

`validate_gas_feedback.py` validates nonreacting gas feedback over the full
droplet-drag counterflow solution:

- gas continuity feedback is enabled first, with species, energy, and gas
  momentum feedback disabled
- species feedback is then enabled and produces a smooth methane vapor profile
  while preserving species normalization
- radial momentum feedback is enabled last and reduces the gas spread rate in
  the injector-side region where the spray momentum source is negative
- the gas energy equation is then enabled with spray energy feedback disabled,
  followed by a final solve with spray energy feedback enabled

The checks are conservative:

- without gas mass feedback, the axisymmetric gas continuity residual is small
  only when the evaporation source is omitted
- with gas mass feedback, the same residual is small only when the evaporation
  source is included
- mass-only feedback leaves methane absent from the gas phase
- species feedback gives a positive methane vapor profile
- momentum feedback changes gas spread rate with the sign implied by the radial
  spray momentum source
- with energy enabled and spray energy feedback disabled, the gas remains at the
  300 K boundary temperature
- enabling spray energy feedback gives a negative gas energy source and a
  resolved gas-temperature depression in the wet source region

`validate_auto_solve.py` validates the intended user-facing workflow:

- construct `ct.MonodisperseSpray`
- construct `ct.CounterflowDiffusionFlame(..., spray=spray)`
- set symmetric air inlet boundary conditions
- call `solve(auto=True)` with no validation-only staging helpers

The case is a fully coupled nonreacting methane spray in air with droplet drag,
gas mass/species/momentum feedback, the gas energy equation, and spray energy
feedback enabled. The checks verify that the stagnation plane remains centered,
the droplets dry out before the stagnation plane, the dryout layer is resolved
on a dynamically refined grid, methane vapor and gas cooling are produced, and
species remain normalized.

## Not yet validated

Reacting spray flames are still open. The current validation covers nonreacting
evaporation, drag, gas feedback, nonreacting energy feedback, and the direct
Python automatic-solve workflow.
