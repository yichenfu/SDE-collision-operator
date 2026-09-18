# Implementation review against the revised PRE manuscript

## Follow-up fixes

The five requested issues have now been addressed in the solver controls and isotropization benchmark:

- `T_iso.py` calls the new `get_ensemble_intra_explicit` helper and uses the manuscript's isotropization time. Its launcher is behind a main guard, closes its process pool, and propagates worker errors.
- Each species is reshuffled when any enabled collision operator needs multiple groups.
- Default group counts are calculated independently for each call and validated against particle counts. Caller-provided lists are copied; disabled collision operators do not calculate unused weights. Odd particle counts and inter-only one-particle species are covered.
- Both solvers expose `step_history` and `time_history`, record regular completed-step intervals, and include the final state. Use `time_history` when plotting, since the final interval may be shorter than `N_record * dt`.
- `special_record` now uses cumulative completed-step numbers: `0` means the initial state, `1` means after one advance. Continued calls retain cumulative step numbers and times. Stored velocity arrays are copied to avoid later shuffles changing earlier records; array index zero still does not track a fixed particle identity across regrouping.

All 14 regression tests in `tests/test_solver_controls.py` pass. Run them from the repository root with `python -B -m unittest discover -s tests -v` in an environment containing NumPy and SciPy. Tests use synthetic particle states and temporary output directories inside the workspace; no benchmark initial-condition files are required.

All 12 existing noise-generation, matrix-assembly, and single-step collision methods, and the `ParticleSpecies` methods, were checked against the original revision and are unchanged. Four seeded comparisons (intra- and two-species, grouped and ungrouped) produced bit-for-bit identical final velocities for previously valid settings. Mixed-group conservation tests also pass.

The missing benchmark input files remain deferred. `T_relax_m.py` and the numerical kernel's domain assumptions were not changed. The review below is the **original assessment before these fixes**; its source line numbers refer to that earlier version.

## Original review

Reviewed on 2026-09-17 against `../Landau_collision_paper/PRE_submission/1st_revised_submission/main.tex`. Equation numbers below were checked against its accompanying `main.aux`.

**Verdict:** The core single-step collision operators implement the paper's modified midpoint equations correctly for equally weighted particles with nonzero pairwise relative velocities. The benchmark scripts and some solver options have defects, so the repository does not currently provide a working, fully matching reproduction of the paper's experiments.

The three original Python files were reviewed without modification. Numerical checks used Python 3.13.12, NumPy 2.5.3, and SciPy 1.17.0 from the existing `py313` environment.

## What matches

| Paper requirement | Implementation | Assessment |
| --- | --- | --- |
| Collision coefficient \(L_{ab}=e_a^2e_b^2\ln\Lambda/(4\pi\epsilon_0^2)\) | `BothLandauCollision.py:123,313–315` | Correct with \(\epsilon_0=1\). The unit convention should be documented alongside \(k_B=1\). |
| Independent pair noise, \(\Delta W^{ij}\sim N(0,I\Delta t)\), antisymmetric for intra-species collisions | `BothLandauCollision.py:161–173,377–386,524–542` | Correct. Only one triangle supplies each intra-species pair's noise; subtraction therefore does not double its variance. |
| \(\Omega^{ij}=(u^{ij}\times\Delta W^{ij})|u^{ij}|^{\gamma/2-1}\) and Eq. (17) | `BothLandauCollision.py:177–236,546–605` | Correct, including cross-product orientation, inverse mass, and the factor of one half from midpoint averaging. |
| Opposite inter-species impulses with separate \(1/m_a\), \(1/m_b\) factors, Eq. (18) | `BothLandauCollision.py:391–519` | Correct, including unequal masses and unequal particle counts. |
| Cayley solution in Appendix C.2 | `BothLandauCollision.py:233,510,602` | Correct: \((I+Q)^{-1}(I-Q)\). For unequal masses, \(HQ+Q^T H=0\), where \(H\) contains the particle masses. This preserves the mass-weighted kinetic energy. |
| Self-interaction correction \(w=n/(N-1)\) and grouped weights in Appendix C.5 | `BothLandauCollision.py:255,627–629` | Correct coefficients: \(w_{ab}=n_aN_g/N_a\), \(w_{aa}=n_a/(N_a/N_g-1)\). Group membership has a separate defect below. |
| Inter-species benchmark parameters | `T_relax_m.py:23–59` | Temperatures, masses, densities, counts, and \(\Delta t=10^{-3}\tau_{11,0}\) match the paper. Charge \(e_2=+1\) instead of \(-1\) is harmless here because all collision coefficients use squared charges. |

`BothExplicit.explicit_solve` applies the intra-species updates and then the inter-species update. Each substep conserves the same total energy and momentum, so their composition also conserves them. This is an operator splitting implementation; the checks here do not establish a convergence order for that combined solver.

## Confirmed defects and discrepancies

### 1. Mixed grouping settings do not reshuffle particles

**High priority for grouped simulations.** [BothLandauCollision.py:635](/Users/yichenfu/Documents/Researchs/2024/SDE_Collision/SDE-collision-operator/BothLandauCollision.py:635) uses:

```python
if np.all(np.array(N_groups) > 1):
```

For supported-looking settings such as `[2,1,1]` or `[1,2,2]`, particles are never reshuffled. Appendix C.5 explicitly requires new random groups each step (`main.tex:1310`). With fixed groups, the grouped operators repeatedly collide the same subsets; in particular, an inter-only run with `[2,1,1]` cannot exchange momentum or energy between its two groups. Increasing the within-group weight does not restore the omitted interactions.

Reproduced by counting calls to `numpy.random.shuffle`: over three steps, `[2,1,1]` and `[1,2,2]` each make zero calls, whereas `[2,2,2]` makes six.

**Correction:** reshuffle whenever an enabled collision operator uses multiple groups. Preserve particle identity separately if recording individual trajectories. The paper's ungrouped benchmark setting `[1,1,1]` is unaffected.

### 2. The isotropization benchmark calls the wrong ensemble helper

**High priority for reproducing Fig. 1(a,b).** [T_iso.py:71](/Users/yichenfu/Documents/Researchs/2024/SDE_Collision/SDE-collision-operator/T_iso.py:71) calls `get_ensemble_explicit` with one species and keyword `N_group`. The only helper of that name, at [BothLandauCollision.py:690](/Users/yichenfu/Documents/Researchs/2024/SDE_Collision/SDE-collision-operator/BothLandauCollision.py:690), requires `s2`, accepts `N_groups`, and constructs `BothExplicit`.

Calling it with the script's arguments raises:

```text
TypeError: get_ensemble_explicit() got an unexpected keyword argument 'N_group'
```

Changing the keyword alone would still leave the required second species missing. `IntraExplicit` exists, but no matching one-species ensemble helper is supplied.

**Correction:** provide a one-species ensemble helper that constructs `IntraExplicit`, and call that helper from `T_iso.py`.

### 3. The isotropization time normalization differs by a factor of three

**Medium priority; affects the benchmark time axis and step size.** [T_iso.py:24](/Users/yichenfu/Documents/Researchs/2024/SDE_Collision/SDE-collision-operator/T_iso.py:24) evaluates, for the script's unit charge,

\[
\nu_{\mathrm{code}}=\frac{3n\ln\Lambda}{8\pi^{3/2}\sqrt m\,T_\parallel^{3/2}}A^{-2}f(A)
=3\tau_{\mathrm{iso,paper}}^{-1}.
\]

The paper's expression at `main.tex:365–369` has no leading factor of three. At \(T_\perp=4,T_\parallel=1\), the code returns `0.004696193728960844`; the manuscript gives `0.001565397909653615`.

Consequently, the script runs with \(\Delta t=(0.01/3)\tau_{\mathrm{iso,paper}}\) and duration \((10/3)\tau_{\mathrm{iso,paper}}\), despite using names suggesting the paper's \(\tau_\mathrm{iso}\). The factor of three *does* describe the decay rate of \(T_\perp-T_\parallel\), since its derivative is \(-3\tau_\mathrm{iso}^{-1}(T_\perp-T_\parallel)\). It could be intentional as an alternative time convention, but that convention does not match the manuscript's stated time step.

**Correction:** remove the factor of three for the paper's convention, or explicitly rename the time scale and convert the benchmark parameters. This is not a factor-of-three error in the collision kernel itself.

### 4. Recorded histories have an irregular first interval and can omit the final state

**Medium priority for plotting relaxation curves.** [BothLandauCollision.py:274](/Users/yichenfu/Documents/Researchs/2024/SDE_Collision/SDE-collision-operator/BothLandauCollision.py:274) and [BothLandauCollision.py:672](/Users/yichenfu/Documents/Researchs/2024/SDE_Collision/SDE-collision-operator/BothLandauCollision.py:672) check `i % N_record == 0` *after* advancing step `i`. The constructor has already recorded the initial state.

For `Nt=9, N_record=3`, the actual saved step numbers are `[0,1,4,7]`, rather than `[0,3,6,9]`. No time-history array accompanies these values. Thus, assigning times with `arange(len(history))*N_record*dt` gives an incorrect time axis.

With the present inter-species benchmark parameters, `Nt=22823` and `N_record=11`; stored steps begin `[0,1,12,23]` and end at `22815`.

**Correction:** record at `(i+1) % N_record == 0`, store actual times, and explicitly include the final state. Define `special_record` indices consistently; currently key `0` refers to the state after the first advance.

### 5. Default grouping is mutable and can generate invalid group sizes

**Medium priority for reuse of the public solver.** [BothLandauCollision.py:611](/Users/yichenfu/Documents/Researchs/2024/SDE_Collision/SDE-collision-operator/BothLandauCollision.py:611) uses the mutable default `N_groups=[-1,-1,-1]`, then modifies it in place. A first call with four particles in each species permanently changes that default to `[4,2,2]`; a later six-particle solver reuses those values and fails during reshaping. The ensemble helper also has a mutable default that is passed into this method.

Even with a fresh list, `min(N1,N2)` need not divide both counts: `(N1,N2)=(6,10)` fails with the default grouping. A one-particle-per-species, inter-only run also raises `ZeroDivisionError` because the code computes intra-species weights despite `intra_collision=False`.

**Correction:** use `None` defaults and copy caller-provided lists, validate divisibility and minimum group sizes, choose a common divisor for inter-species grouping, and calculate weights only for enabled operators.

### 6. Benchmarks are not self-contained and the supplied ensemble count differs

- Both scripts require initial-condition pickle files absent from this checkout (`T_iso.py:51`, `T_relax_m.py:48`). No generator for these files is supplied.
- Both create multiprocessing pools at module scope (`T_iso.py:63`, `T_relax_m.py:61`), without a `__main__` guard. On the current macOS spawn-based multiprocessing environment, workers re-import the script and attempt to start pools again. This is an additional blocker after supplying initial conditions.
- `T_relax_m.py:13` schedules 512 realizations, whereas `main.tex:382` specifies \(2^{11}=2048\). This changes sampling statistics, not the collision equations.
- Both scripts wait on asynchronous jobs with `.wait()` rather than collecting them with `.get()`. Worker errors are printed by the error callback, but the parent can finish normally after failed jobs.

**Correction:** supply or generate the initial conditions, put execution behind a main guard, set the intended ensemble count, and propagate worker failures.

### 7. Shuffling can mutate previously saved velocity records

[BothLandauCollision.py:156](/Users/yichenfu/Documents/Researchs/2024/SDE_Collision/SDE-collision-operator/BothLandauCollision.py:156) appends `self.s1.v[0]`, which is a view. Similar code is used in `BothExplicit`; `specific_v1` and `specific_v2` also store arrays without copying. The next in-place shuffle changes these earlier records. A two-step grouped run reproduced changes to both a saved snapshot and the previous first-particle history entry.

**Correction:** copy recorded arrays. For a genuine sample path of a particular particle, also track identity through each permutation or restore the original ordering. Merely copying does not make index zero refer to the same particle after regrouping. Reordering a snapshot alone does not change its empirical distribution, but it does invalidate particle-index comparisons.

## Assumptions and numerical limitations

- **Equal particle weights are essential.** The constructor only prints a warning for unequal `n/N` (`BothLandauCollision.py:319`), then continues with an inter-species coefficient derived from species 1 alone. In a one-step test with weights `1/4` and `1/2`, the physical weighted energy changed by about `4.65e-4` relative and the weighted momentum changed by `0.0474` in norm. This input lies outside the manuscript's assumptions; rejecting it would prevent misleading results.
- **Diagnostics omit the macro-particle weight.** `get_statistics` computes \(m\sum_i v_i\) and \(m\sum_i|v_i|^2/2\). Multiply by the common physical weight `n/N` to compare their absolute values with the manuscript's \(P,E\). Under equal weights, this does not change relative conservation errors or temperatures. Do not use the grouping-adjusted collision weight to rescale global diagnostics.
- **Exactly coincident velocities produce NaNs.** Only intra-species diagonal entries are protected against zero relative speed. Distinct particles with identical velocities produce `0 * infinity` at `BothLandauCollision.py:193,402,562`; tests reproduced nonfinite results in both operators. The Coulomb kernel in the paper is itself singular there, so this is an unresolved numerical domain issue rather than a disagreement with a prescribed regularization. Any guard or regularization should be explicit.
- **Manuscript inconsistency:** in Appendix C.2, the definitions of the inter-species blocks \(M_2^{ij}\) and \(N_2^{ij}\) say \(i\ne j\) (`main.tex:930,939`). Cross-species particles with equal index labels still interact under Eq. (18). The code includes these interactions, correctly following Eq. (18). Those restrictions in the appendix appear to be typographical errors.

## Numerical validation performed

The single-step residuals were evaluated directly from the paper's cross-product equations, independently of the implementation's block-matrix assembly. Tests used \(\gamma=-3,0,1\), time steps `0.01` and `1`, grouped and ungrouped configurations, unequal masses, and unequal species counts. Seeds were fixed (`7241` for the numerical equation/conservation checks).

| Check | Result |
| --- | --- |
| Eq. (17), 24 cases covering both intra-species implementations | Maximum absolute residual `1.08e-15` |
| Eq. (18), 12 cases | Maximum absolute residual `1.17e-15` |
| Single-step relative kinetic-energy error | Below `4.8e-16` |
| Single-step absolute momentum-component error | Below `7.8e-15` |
| Inter-species mass-weighted skew identity | Maximum residual `5.6e-17` |
| 250 combined Coulomb steps, 8 and 16 particles, masses 1 and 5, groups `[1,1,1]` | Maximum relative energy drift `4.76e-15`; absolute momentum-component drift `3.20e-14` |
| Same duration with groups `[4,2,4]` | Maximum relative energy drift `1.02e-15`; absolute momentum-component drift `9.77e-15` |
| Intra-species noise antisymmetry | Exact zero residual |
| Empirical noise variance divided by \(\Delta t\), 8128 independent pair samples | Component values `1.015`, `1.008`, `1.001` |

These checks support the correctness of the implemented modified midpoint kernels and their conservation properties. They do **not** reproduce the full relaxation figures, establish a strong-convergence rate, or validate grouped relaxation statistics. The benchmark data dependencies and defects above prevent treating this checkout as a complete reproduction of the paper.
