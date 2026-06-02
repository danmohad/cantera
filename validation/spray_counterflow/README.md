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

## Not yet validated

Gas-phase two-way feedback, energy-coupled gaseous flames, and the final
user-facing automatic staging are still open. The current drag validations use
explicit staged initial guesses; that is a validation harness, not yet the
intended Python-only workflow.
