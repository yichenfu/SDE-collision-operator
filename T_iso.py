"""
Calculate intra-species temperature isotropization using modified midpoint method.

This ensemble benchmark expects pre-generated ParticleSpecies objects in the
initial_conditions pickle named below. Each worker writes one solver pickle.
For an example that creates its own initial velocities, see README.md.
"""
import numpy as np
from BothLandauCollision import get_ensemble_intra_explicit
import pickle
import time
import multiprocessing as mp


# ==================== script specific setting ====================
N_group = 1
save_dir = './data/T_relax_iso_N(256)_T(4,1)_m(1)_n(1)/({})_0.01'.format(N_group)
job_per_script = 2048
Process_list = range(job_per_script)
N_process = 16


# ==================== Define collision time ====================
def get_nu_iso(T_perp, T_para, m, n, coulomb_log):
    """Return tau_iso**(-1) for the paper's T_perp > T_para benchmark.

    Temperatures use k_B=1; m and n are mass and number density. The charge
    and epsilon_0 are fixed to 1. With A=T_perp/T_para-1, the rate is
    n*ln(Lambda)*[(A+3)*atan(sqrt(A))/sqrt(A)-3] /
    [8*pi**(3/2)*sqrt(m)*T_para**(3/2)*A**2].

    This coefficient multiplies (T_para-T_perp) in dT_perp/dt. The decay
    coefficient for their difference is three times larger. The helper does
    not implement the A=0 limit or the opposite-anisotropy analytic branch.
    """

    A = np.abs(T_perp/T_para - 1.)
    f = np.arctan( A**0.5 ) / A**0.5

    return n * coulomb_log * A**(-2) * ((A+3)*f - 3) / (8. * np.pi**1.5 * m**0.5 * T_para**1.5)



def main():
    """Load initial states and launch the configured isotropization ensemble.

    Use dt=0.01*tau_iso,0 and total_t=10*tau_iso,0, following the script's
    benchmark convention. Process_list selects realizations from the input;
    sub_id sets each worker's collision seed and output filename. Worker
    failures propagate through AsyncResult.get instead of being discarded.
    """
    # ==================== Define system ====================
    T1_perp=4.
    T1_para=1.

    m1=1.
    n1=1.
    N1=256
    coulomb_log=1.

    # Fix the timestep from the initial temperature distribution's time scale.
    tau = 1./get_nu_iso(T1_perp, T1_para, m1, n1, coulomb_log)

    # ==================== Load initial conditions ====================
    with open('./initial_conditions/N({})_T({},{})_m({})_n({}).pickle'.format(N1,int(T1_perp),int(T1_para),int(m1),int(n1)), 'rb') as handle:
        s_ensemble = pickle.load(handle)

    # ==================== calculate ensemble ====================
    start_time = time.time()

    dt = 0.01 * tau
    total_t = 10. * tau
    Nt = int(total_t/dt)
    # Limit the moment history to roughly 2000 regular samples plus endpoints.
    N_record = max(int(Nt/2000),1)

    with mp.Pool(processes=N_process) as pool:
        job_status = {}
        for i in Process_list:
            # Initial-condition sampling is separate from each worker's SDE seed.
            job_status[i] = pool.apply_async(
                func=get_ensemble_intra_explicit,
                kwds={
                    'dt': dt,
                    'total_t': total_t,
                    's1': s_ensemble[i],
                    'coulomb_log': coulomb_log,
                    'save_dir': save_dir,
                    'sub_id': i,
                    'N_group': N_group,
                    'N_record': N_record,
                    'special_record': [0],
                    'print_status': i % N_process == 0,
                },
            )
        for i in Process_list:
            job_status[i].get()

    time_used = time.time() - start_time
    print('total time = {:.3f} s = {:.3f} mins = {:.3f} hours.'.format(
        time_used, time_used/60, time_used/3600))


if __name__ == '__main__':
    # Spawned workers import this module; only the parent launches a process pool.
    main()
