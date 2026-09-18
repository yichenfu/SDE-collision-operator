# SDE collision operator

Python implementations of the energy- and momentum-conserving stochastic particle method for the homogeneous Landau–Fokker–Planck equation. The code evolves three-dimensional particle velocities through self-collisions within a species and collisions between two species.

The algorithm follows **Energy-Momentum-Conserving Stochastic Differential Equations and Algorithms for Nonlinear Landau-Fokker-Planck Equation**, by Yichen Fu, Justin R. Angus, Hong Qin, and Vasily I. Geyko. The numerical updates correspond to Eqs. (17)–(18), the Cayley solution to Appendix C.2, and particle grouping to Appendix C.5 of the revised manuscript.

## Setup

Run examples from the repository root, where `BothLandauCollision.py` can be imported. Install the two runtime dependencies in your Python environment:

```bash
python -m pip install numpy scipy
```

The examples have been checked with Python 3.13, NumPy 2.5.3, and SciPy 1.17.0. There is no build or installation step for the local module.

| File | Purpose |
| --- | --- |
| [BothLandauCollision.py](BothLandauCollision.py) | Particle data, collision solvers, diagnostics, and ensemble workers |
| [T_iso.py](T_iso.py) | Temperature-isotropization benchmark and its analytical reference rate |
| [T_relax_m.py](T_relax_m.py) | Two-species temperature-relaxation benchmark |

## Units and input conventions

The implementation sets Boltzmann's constant $k_B=1$ and vacuum permittivity $\epsilon_0=1$. Temperatures are in energy units, and a Maxwellian velocity component has standard deviation $\sqrt{T/m}$. Physical quantities must be expressed consistently in these units.

`ParticleSpecies(m, e, N, n)` stores mass, charge, macro-particle count, and number density. Its velocity array `v` has shape `(N, 3)`, ordered as Cartesian components `(x, y, z)`. The physical weight per macro-particle is

$$
w_0=\frac{n}{N}.
$$

For two species, choose counts proportional to their densities so that $n_1/N_1=n_2/N_2$. Both species may have different masses and temperatures. Use `gamma=-3` for Coulomb collisions; the solver constructors also accept other kernel exponents. Both solvers copy the supplied species, so their evolving velocities are available as `solver.s1.v` and, when present, `solver.s2.v`.

## Example: temperature isotropization

This example samples $T_x=T_y=4$ and $T_z=1$, evolves self-collisions, and checks conservation. It needs no initial-condition files.

```python
import numpy as np
from BothLandauCollision import ParticleSpecies, IntraExplicit

np.random.seed(42)
particles = ParticleSpecies(m=1.0, e=1.0, N=32, n=1.0)
particles.initialize(vt=np.sqrt(np.array([4.0, 4.0, 1.0]) / particles.m))

solver = IntraExplicit(particles, coulomb_log=1.0, dt=0.01, gamma=-3)
solver.explicit_solve(
    Nt=100,
    N_group=1,             # All particles interact within one group.
    N_record=10,
    special_record=[0, 100],
)

times = np.asarray(solver.time_history)
temperatures = np.asarray(solver.T1_direction_history)  # (records, 3)
energy = np.asarray(solver.E1_total_history)
momentum = np.asarray(solver.P1_history)

print("Final time:", times[-1])
print("Final directional temperatures:", temperatures[-1])
print("Maximum relative energy change:", np.max(np.abs(energy / energy[0] - 1)))
print("Maximum momentum-component change:", np.max(np.abs(momentum - momentum[0])))
```

Finite samples will not have exactly the requested initial temperatures. `initialize(..., T_tolerance=...)` optionally resamples components until the measured **thermal speed** meets a relative tolerance. The example above uses the default single Gaussian draw per component. A mean flow can be supplied as `u=np.array([ux, uy, uz])`.

The short run demonstrates the API and conservation diagnostics; it does not reproduce the full ensemble relaxation benchmark. Choose the timestep and duration relative to the relevant collision time and check timestep sensitivity for quantitative relaxation results.

## Example: collisions between two species

Here $m_1=1$, $m_2=5$, $T_1=4$, $T_2=1$, and the common particle weight is $1/16$. Both self-collisions and inter-species collisions are enabled.

```python
import numpy as np
from BothLandauCollision import ParticleSpecies, BothExplicit

np.random.seed(43)
s1 = ParticleSpecies(m=1.0, e=2.0, N=16, n=1.0)
s2 = ParticleSpecies(m=5.0, e=-1.0, N=32, n=2.0)
s1.initialize(vt=np.full(3, np.sqrt(4.0 / s1.m)))
s2.initialize(vt=np.full(3, np.sqrt(1.0 / s2.m)))

solver = BothExplicit(s1, s2, coulomb_log=1.0, dt=0.01, gamma=-3)
solver.explicit_solve(
    Nt=100,
    N_groups=[1, 1, 1],    # [inter-species, species 1, species 2]
    N_record=10,
    special_record=[0, 100],
    intra_collision=True,
    inter_collision=True,
)

times = np.asarray(solver.time_history)
T1 = np.asarray(solver.T1_history)
T2 = np.asarray(solver.T2_history)
w0 = s1.n / s1.N
energy = w0 * (np.asarray(solver.E1_total_history) + solver.E2_total_history)
momentum = w0 * (np.asarray(solver.P1_history) + solver.P2_history)

print("Final temperatures:", T1[-1], T2[-1])
print("Maximum relative total-energy change:", np.max(np.abs(energy / energy[0] - 1)))
print("Maximum total-momentum-component change:", np.max(np.abs(momentum - momentum[0])))
```

Set `intra_collision=False` for inter-species collisions alone, or `inter_collision=False` to evolve the two species independently through self-collisions. Within a combined step, the code applies both self-collision updates first, then the inter-species update using those new velocities. The same `coulomb_log` is used for all three collision coefficients.

## Grouping, recording, and saved runs

### Grouping controls

Grouping limits collision partners to randomly selected subsets while adjusting the collision coefficient to represent the full density. Each participating species is reshuffled at every step when an enabled operator uses multiple groups.

| Solver argument | Meaning |
| --- | --- |
| `IntraExplicit.explicit_solve(N_group=1)` | All self-collision partners in a single group |
| `N_group=-1` (default) | Largest valid number of equal-sized groups, with at least two particles each |
| `BothExplicit.explicit_solve(N_groups=[G12, G1, G2])` | Separate counts for inter-species and each self-collision operator |
| `N_groups=None` (default) | Resolve each entry as `-1`, independently for that call |
| `N_groups=[1, 1, 1]` | All partners included for each enabled operator |

An inter-species group count must divide both particle counts. A self-collision group count must divide its species' count and leave at least two particles per group. Disabled operators use a group count of one internally. Invalid enabled group sizes raise `ValueError` before advancing. For example, counts `(16, 32)` permit `N_groups=[4, 2, 4]` and the mixed setting `[4, 1, 1]`.

The kernels assemble dense matrices and invert one matrix per group. For inter-species collisions its dimension is $3(N_1+N_2)/G_{12}$; for self-collisions it is $3N_s/G_s$. Grouping reduces these matrix sizes, but its statistical and finite-timestep effects still need to be assessed for the intended simulation.

### Histories and snapshots

`explicit_solve` advances **an additional** `Nt` steps and returns `None`. Repeated calls continue the current state and cumulative step/time counters. The constructor records the initial moments; subsequent calls record every `N_record` cumulative steps and always include their final state. Use `time_history` as the plotting axis because a final recording interval can be shorter than usual.

| Attribute | Contents |
| --- | --- |
| `step_history`, `time_history` | Completed step numbers and actual times for every moment record |
| `T1_history`, `T2_history` | Mean temperatures, one scalar per record |
| `T1_direction_history`, `T2_direction_history` | Three directional temperatures per record |
| `E1_total_history`, `E2_total_history` | Total kinetic-energy sums, including mean flow |
| `E1_thermal_history`, `E2_thermal_history` | Kinetic-energy sums with the sample mean flow removed |
| `P1_history`, `P2_history` | Three-component momentum sums |
| `v1_avg_history`, `v2_avg_history` | Mean flow velocities |
| `specific_v1`, `specific_v2` | Copied velocity arrays keyed by requested completed-step number |

Species-2 attributes only exist on `BothExplicit`. `special_record=[0, 100]` stores the initial state and the state after 100 steps, provided those steps are reached; `[]` disables snapshots, while `None` defaults to `[1]`. Snapshots can be requested independently of `N_record`.

The `E*` and `P*` histories omit the physical macro-particle weight. Multiply them by the common $w_0=n_s/N_s$ for the energy and momentum in the manuscript. For conservation checks, sum both species' **total** energies and momenta; their individual values and thermal energies can change. Temperatures already have the correct per-particle normalization:

$$
T_{s,d}=\frac{m_s}{N_s}\sum_i(v^{s,i}_d-\bar v_{s,d})^2,
\qquad T_s=\frac{T_{s,x}+T_{s,y}+T_{s,z}}{3}.
$$

`v1_1_history` and `v2_1_history` store velocities at array index zero. Grouping shuffles particle indices, so these do not track a fixed particle's trajectory when grouping is enabled.

### Saving an ensemble realization

The top-level helpers run and pickle **one** realization per call. `get_ensemble_intra_explicit` takes one species and `N_group`; `get_ensemble_explicit` takes `s1`, `s2`, and `N_groups`. Both use `gamma=-3`, seed NumPy's collision draws with `sub_id`, and write the complete solver to `<save_dir>/<sub_id>.pickle`. An existing file with that name is replaced. The initial distribution is supplied by the caller, so initialize separate species objects if independent initial samples are needed across realizations.

```python
import pickle
from pathlib import Path
import numpy as np
from BothLandauCollision import ParticleSpecies, get_ensemble_intra_explicit

np.random.seed(44)          # Seed the initial distribution separately.
particles = ParticleSpecies(m=1.0, e=1.0, N=16, n=1.0)
particles.initialize(vt=np.array([2.0, 2.0, 1.0]))
output_dir = Path("data/example")

get_ensemble_intra_explicit(
    dt=0.01,
    total_t=0.2,
    s1=particles,
    save_dir=output_dir,
    sub_id=0,             # Collision seed and output filename.
    N_group=1,
    N_record=5,
    special_record=[0],
    print_status=False,
)

with (output_dir / "0.pickle").open("rb") as handle:
    saved_solver = pickle.load(handle)
print("Saved times:", saved_solver.time_history)
```

Helpers take `int(total_t / dt)` full steps and do not add a fractional final step. Read the saved `time_history` for the realized end time. For process-based ensembles, submit one helper call per distinct `sub_id` and place the launcher inside `if __name__ == "__main__":`.

For controlled-noise studies, `BothExplicit.explicit_solve(dWg_given=...)` accepts inter-species increments shaped `(Nt, G12, N1/G12, N2/G12, 3)`, already scaled with variance `dt`. Self-collision noise is still sampled internally, and grouped runs also require control of the random permutations to compare the same particle paths.

## Mathematical model and numerical update

### Landau coefficients and particle SDEs

For relative velocity $\mathbf u=\mathbf v-\mathbf v'$, define

$$
\Pi(\mathbf u)=I_3-\frac{\mathbf u\mathbf u^T}{|\mathbf u|^2},\qquad
a(\mathbf u)=|\mathbf u|^{2+\gamma}\Pi(\mathbf u),\qquad
b(\mathbf u)=-2|\mathbf u|^\gamma\mathbf u.
$$

The pair diffusion factor and collision constant are

$$
\sigma(\mathbf u)=|\mathbf u|^{1+\gamma/2}\Pi(\mathbf u),\qquad
\sigma\sigma^T=a,\qquad
L_{\alpha\beta}=\frac{e_\alpha^2e_\beta^2\ln\Lambda}{4\pi}
\quad (\epsilon_0=1).
$$

These correspond to the Landau–Fokker–Planck equation

$$
\left(\frac{\partial f_\alpha}{\partial t}\right)_{\alpha\beta}
=-\nabla_{\mathbf v}\cdot\left[\mu_{\alpha\beta}f_\alpha
-\frac12\nabla_{\mathbf v}\cdot(D_{\alpha\beta}f_\alpha)\right],
$$

$$
D_{\alpha\beta}=\frac{L_{\alpha\beta}}{m_\alpha^2}\int a(\mathbf v-\mathbf v')f_\beta(\mathbf v')\,d\mathbf v',
\qquad
\mu_{\alpha\beta}=\frac{L_{\alpha\beta}}{2m_\alpha}
\left(\frac1{m_\alpha}+\frac1{m_\beta}\right)
\int b(\mathbf v-\mathbf v')f_\beta(\mathbf v')\,d\mathbf v'.
$$

The code represents the distribution by particles rather than evaluating these integrals on a velocity grid. In the manuscript's Stratonovich formulation, the self-collision SDE is

$$
d\mathbf v^i=\frac{\sqrt{w_cL}}{m}\sum_{j\ne i}
\sigma(\mathbf v^i-\mathbf v^j)\circ d\mathbf W^{ij},
\qquad d\mathbf W^{ij}=-d\mathbf W^{ji}.
$$

For inter-species collisions, the same pair increment drives both partners:

$$
\begin{aligned}
d\mathbf v^{\alpha,i}&=\frac{\sqrt{w_cL_{\alpha\beta}}}{m_\alpha}
\sum_j\sigma(\mathbf v^{\alpha,i}-\mathbf v^{\beta,j})\circ d\mathbf W^{ij},\\
d\mathbf v^{\beta,j}&=-\frac{\sqrt{w_cL_{\alpha\beta}}}{m_\beta}
\sum_i\sigma(\mathbf v^{\alpha,i}-\mathbf v^{\beta,j})\circ d\mathbf W^{ij}.
\end{aligned}
$$

The opposite impulses enforce Newton's third law. The Stratonovich equations contain no explicit drag term; the corresponding Itô drift is already accounted for by the stochastic interpretation. Here $w_c$ denotes the **collision weight**, to distinguish it from the physical weight $w_0$ used in diagnostics:

| Operator | Collision weight $w_c$ |
| --- | --- |
| Self-collisions, one group | $n/(N-1)$, excluding the particle itself |
| Self-collisions, $G$ groups | $n/(N/G-1)$ |
| Inter-species, one group | $w_0=n_\alpha/N_\alpha=n_\beta/N_\beta$ |
| Inter-species, $G$ groups | $G w_0$ |

In grouped updates the sums include only partners in the current group.

### Modified midpoint scheme

For each pair sample $\Delta\mathbf W^{ij}\sim\mathcal N(0,I_3\Delta t)$ and form

$$
\boldsymbol\Omega_k^{ij}
=\frac{\mathbf u_k^{ij}\times\Delta\mathbf W^{ij}}
{|\mathbf u_k^{ij}|^{1-\gamma/2}},\qquad
\mathbf u_{k+1/2}^{ij}=\frac{\mathbf u_k^{ij}+\mathbf u_{k+1}^{ij}}2.
$$

The intra-species update (Eq. 17) is

$$
\mathbf v^i_{k+1}-\mathbf v^i_k
=\frac{\sqrt{w_cL}}m\sum_{j\ne i}
\boldsymbol\Omega_k^{ij}\times\mathbf u_{k+1/2}^{ij}.
$$

The inter-species update (Eq. 18) is

$$
\begin{aligned}
\mathbf v^{\alpha,i}_{k+1}-\mathbf v^{\alpha,i}_k
&=\frac{\sqrt{w_cL_{\alpha\beta}}}{m_\alpha}\sum_j
\boldsymbol\Omega_k^{ij}\times\mathbf u_{k+1/2}^{ij},\\
\mathbf v^{\beta,j}_{k+1}-\mathbf v^{\beta,j}_k
&=-\frac{\sqrt{w_cL_{\alpha\beta}}}{m_\beta}\sum_i
\boldsymbol\Omega_k^{ij}\times\mathbf u_{k+1/2}^{ij}.
\end{aligned}
$$

The coefficient $\boldsymbol\Omega_k$ uses old velocities; only the final relative velocity uses a midpoint. Therefore the unknown new velocities enter linearly. With $\widehat{\boldsymbol\omega}\mathbf x=\boldsymbol\omega\times\mathbf x$, the code assembles the block system and applies its Cayley transform:

$$
(I+Q)\mathbf V_{k+1}=(I-Q)\mathbf V_k,\qquad
\mathbf V_{k+1}=(I+Q)^{-1}(I-Q)\mathbf V_k.
$$

The intra-species matrix is called `M` in the code. Its pair blocks include $\sqrt{w_cL}/(2m)$; for inter-species collisions, `omega_hat_g_ij_inter` includes $\sqrt{w_cL}/2$ and the species block builders subsequently divide by the appropriate mass. The factor $1/2$ comes from midpoint averaging, while the timestep is already contained in $\Delta\mathbf W$.

For equal masses $Q$ is skew-symmetric. For unequal masses it satisfies $HQ+Q^TH=0$, with $H=\mathrm{diag}(m_\alpha I,m_\beta I)$. Its Cayley transform preserves the mass-weighted kinetic energy. Pair impulses cancel in the momentum sum, and the energy cancellation follows from $\mathbf u_{k+1/2}\cdot(\boldsymbol\Omega_k\times\mathbf u_{k+1/2})=0$. Thus, for equal physical particle weights, the scheme preserves

$$
\mathbf P=\sum_{s,i}w_0m_s\mathbf v^{s,i},\qquad
E=\frac12\sum_{s,i}w_0m_s|\mathbf v^{s,i}|^2
$$

up to floating-point roundoff.

| Code methods | Mathematical role |
| --- | --- |
| `generate_dW_g_ij_intra`, `generate_dW_g_ij_inter` | Sample pair Wiener increments |
| `M_g_ij_intra` | Assemble the self-collision block matrix |
| `omega_hat_g_ij_inter` | Form cross-product blocks from $\boldsymbol\Omega_k$ |
| `M_g_ij_inter`, `N_g_ij_inter`, `Q_g_inter` | Apply mass factors and assemble the two-species matrix |
| `intra_one_step_explicit`, `inter_one_step_explicit` | Apply the Cayley update |
| `explicit_solve` | Select groups, apply enabled substeps, and record diagnostics |

## Benchmark scripts and current limits

The example snippets above generate their own initial states. The larger benchmark scripts expect external pickle files:

- `T_iso.py`: `initial_conditions/N(256)_T(4,1)_m(1)_n(1).pickle`, an ensemble indexed by realization, each entry a `ParticleSpecies`. It schedules 2048 realizations. Once that file is supplied, run `python T_iso.py`.
- `T_relax_m.py`: `data/initial_conditions/N(64,128)_T(4,1)_m(1,5)_n(1,2).pickle`, with species accessible as `ensemble[i][1]` and `ensemble[i][2]`. It schedules 512 realizations. This legacy launcher still executes at import time and lacks a multiprocessing main guard, so it requires launcher changes for spawn-based multiprocessing, including the macOS default. Use the direct two-species API above in the meantime.

`T_iso.get_nu_iso` returns the coefficient in $dT_\perp/dt=\tau_{\rm iso}^{-1}(T_\parallel-T_\perp)$ for the $T_\perp>T_\parallel$ benchmark. The rate for the temperature difference itself is $3\tau_{\rm iso}^{-1}$. `T_relax_m.get_nu_relax` gives the coefficient in $dT_1/dt=\tau_{12}^{-1}(T_2-T_1)$; importing that legacy script also runs its top-level code.

The equal-weight condition is essential for physical conservation; the current constructor only warns about unequal weights. The Coulomb kernel is singular for exactly coincident velocities of distinct particles and has no regularization in this implementation. Conservation alone does not establish timestep accuracy, particularly for pairs with very small relative speeds. The optional Sobol initializer is used only with `T_tolerance`, requires a power-of-two particle count, and currently ignores the requested mean flow.

## License

[MIT](LICENSE).
