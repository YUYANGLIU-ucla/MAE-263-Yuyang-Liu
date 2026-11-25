# Snap-through design of a bistable buckled beam for a fast soft gripper

MAE 263F (Fall 2026) course project, Yuyang Liu.

A thin strip is clamped at both ends, buckled by end shortening, and switched between
its two stable arches by a short actuator patch that changes its natural curvature.
The code simulates the strip as a planar discrete elastic beam (course notes,
"Discrete Simulation of Slender Structures"), validates the solver, and uses it to
choose the end shortening `eps`, the patch position `s_a` and the patch length `l_a`.

## Requirements

Python 3.9+ with NumPy and Matplotlib (`pip install numpy matplotlib`). No other package.

## How to run

Run from this folder, in this order.

| Command | What it does | Time* |
|---|---|---|
| `python beam.py` | compares the vectorized energy derivatives with the course routines | 1 s |
| `python validate.py` | validation items 1-5, writes `results/validation.*`, `figures/val_*` | 3 min |
| `python design.py grid` | evaluates 357 designs, writes `results/grid.csv` | 12 min |
| `python design.py search` | Nelder-Mead search, writes `results/search.json` | 6 min |
| `python make_figures.py` | design figures `figures/design_*` | 1 min |
| `python convergence.py` | time-step and mesh convergence of the selected design, writes `results/convergence.*`, `figures/val_convergence` | 1 min |

*on a 2-core machine. `design.py grid` uses two processes.

## Files

| File | Content |
|---|---|
| `beam.py` | `Beam` (geometry, material, DOF bookkeeping); elastic energy, gradient and Hessian; static Newton solver with stability check; implicit Euler step (Algorithm 1 of the proposal) with step halving; trapezoidal step for cross-checks |
| `snap.py` | buckling to the open state; point-force loading; quasi-static trigger of the actuator patch; snap dynamics; settle and release |
| `theory.py` | reference results: elastica, point-force snap-through, vibration frequency |
| `validate.py` | validation against the references |
| `design.py` | design evaluation (Algorithm 3 of the midterm report), grid, constrained Nelder-Mead search |
| `convergence.py` | time-step and mesh convergence of the selected design |
| `make_figures.py`, `plotstyle.py` | figures |
| `course/` | unchanged course routines `gradEb`, `hessEb`, `gradEs`, `hessEs`, `crossMat`, used only to check `beam.py` |
| `results/`, `figures/` | generated output |

## Model

* Nodes `x_k = (x_k, y_k)`; `N` free edges of length `dl = L/N` plus one clamped edge
  at each end, so `N + 3` nodes. The two nodes of each clamped edge are prescribed
  (8 fixed DOF, `2(N-1)` free DOF). The right clamp is moved by `eps * L`.
* Energy `E = sum 1/2 EA dl (|e|/dl - 1)^2 + sum 1/2 EI/l_k (kappa_k - kappa0_k)^2`,
  `kappa_k = 2 tan(phi_k/2)`.
* Actuator: nodes inside the patch get the natural curvature
  `kappa0_k = kbar * (length of the node's Voronoi cell inside the patch)`.
* Strip: L = 100 mm, b = 10 mm, h = 0.5 mm, E = 3 GPa, rho = 1380 kg/m^3
  (design choices, not measured). Time scale `t* = L^2 sqrt(rho b h / EI)` = 47 ms.

## Performance measures

* `kcrit`: commanded curvature (times L) at which the open arch loses stability, found
  by static continuation and the smallest eigenvalue of the free Hessian.
* `W_act`: actuator work up to `kcrit`.
* `F_hold`: midpoint force at which the closed arch snaps back (depends on `eps` only).
* `t_snap`: after a step command of `1.1 kcrit`, time until the midpoint first reaches
  the height it has when the strip has settled with the command on.
* `strain`: largest bending strain `h |kappa_k| / (2 l_k)` during the ramp.

## What differs from the proposal

1. **Clamp nodes.** The bending spring at the two clamp nodes uses the Voronoi length
   `dl/2`, because the half cell inside the clamp cannot bend. With `dl` the error of
   the buckling load is first order in `dl` (5% at N = 40); with `dl/2` it is 0.2%.
2. **Trigger.** The trigger curvature is found by static solves with a stability check
   instead of a slow damped ramp. It is faster and gives a sharp threshold.
3. **Actuator work.** Eq. (3) of the proposal can be negative for patches near the
   inflection point of the arch, because the strip gives energy back to part of the
   patch. `W_act` counts only the increments where the actuator supplies energy
   (`W_net` is eq. (3) as written and is also stored).
4. **Snap time.** With the command on, the settled midpoint is at about -0.6 w0, so the
   threshold "-0.9 w0" of the proposal is not always reached. `t_snap` uses the settled
   height instead.
5. **Sign of the command.** Both signs are tried and the one that switches the arch
   with less work is kept. (The first version chose the sign to oppose the
   patch-averaged curvature of the open arch; that rule fails for a patch centred on
   the inflection point `s_a = 0.25`, where the average is zero.) In practice only one
   sign works for each design: negative for `s_a <= 0.25`, positive above.
6. **Trapezoidal rule.** Added as a cross-check, because implicit Euler removes a
   visible part of the energy during the snap.

## Results

### Validation (`results/validation.txt`)

| Check | Result |
|---|---|
| Gradient and Hessian vs course routines | agree to 1e-16 |
| Gradient and Hessian vs finite differences | 3e-8 |
| Buckled height and axial force vs elastica, eps = 0.2-5%, N = 80 | within 0.03% and 0.04% |
| Midpoint snap-through force vs Pandey et al. (eq. 3, constant 129.53), eps = 1% | +0.29% |
| Slope of the asymmetric branch vs Pandey et al. (207.75) | +0.4% |
| Critical force at 7 positions vs Yan et al., eps = 1% and 3% | within 2.8% |
| Lowest vibration frequency of the arch vs beam theory (44.36/t*) | -0.2% (N = 80) |
| Snap time vs overdrive, exponent (Gomez et al.: -1/4) | -0.27 |
| Snap time, implicit Euler (t*/4000) vs trapezoidal rule | 0.5% apart |
| Energy during a snap | trapezoidal rule conserves it to 0.05%; implicit Euler at t*/4000 removes 38% of the released energy |
| Selected design, time step (`results/convergence.txt`) | trapezoidal rule converges to t_snap = 4.66 ms; implicit Euler at t*/4000 is 1.8% high |
| Selected design, mesh N = 40 vs 160 | kcrit, F_hold, t_snap within 1%; W_act within 2.5% |

### Design (`results/grid.csv`, `results/search.json`)

Requirements: `F_hold >= 0.5 N`, `t_snap <= 10 ms`, bending strain `<= 1%`.

| Design | eps | s_a | l_a | kcrit | W_act | F_hold | strain | t_snap |
|---|---|---|---|---|---|---|---|---|
| Search optimum (N = 40) | 1.53% | 0.269 | 0.136 | 3.39 | 0.197 mJ | 0.500 N | 1.00% | 4.7 ms |
| Same, N = 80, half time step | | | | 3.39 | 0.203 mJ | 0.503 N | 0.99% | 4.7 ms |
| Best grid point | 2% | 0.275 | 0.2 | 2.70 | 0.487 mJ | 0.574 N | 0.82% | 5.2 ms |
| Patch at 0.35, same eps and l_a | 1.53% | 0.35 | 0.136 | 2.21 | 0.638 mJ | 0.500 N | 0.68% | 7.2 ms |
| Centered patch, same eps and l_a | 1.53% | 0.5 | 0.136 | 4.23 | 3.30 mJ | 0.500 N | 0.75% | 3.6 ms |

Findings:

* At the quarter point (`s_a = 0.25`, an inflection point of the arch) the patch makes
  the open state unstable, but for `eps <= 1.5%` the strip returns to the open state
  when the actuator is turned off; for larger `eps` it switches but exceeds 1% strain.
* The least work is found just past the quarter point. The optimum needs 17 times less
  work than a centered patch of the same length.
* The holding force depends only on `eps` and follows
  `F_hold L^2/EI = 129.53 sqrt(eps - 80.76 S)` (Pandey et al.). The holding-force and strain limits are
  both active at the optimum. The time limit is not.
* `t_snap` is the first arrival. With a damping ratio of 0.02 the strip keeps ringing
  for more than 65 ms; with 0.10 it settles within 10% of the stroke after 19 ms.

## Attribution

The discrete beam energies and their derivatives follow the MAE 263F course notes and
the course repository https://github.com/StructuresComp/mae-263F (folder `appendix`).
The files in `course/` are unchanged copies. Generative AI tools were used to help
write the code, as permitted by the syllabus.
