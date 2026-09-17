"""
Calculate the temperature relaxation with mass ratio, using modified midpoint method
"""
import numpy as np
from BothLandauCollision import *
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

    A = np.abs(T_perp/T_para - 1.)
    f = np.arctan( A**0.5 ) / A**0.5

    return 3. * 2. * np.pi**0.5 * n * coulomb_log * A**(-2) * ( (A+3)*f - 3 ) / ( (4.*np.pi)**2. * m**0.5 * T_para**1.5 )



# ==================== Define system ====================
T1_perp=4.
T1_para=1.

m1=1.

v1xt = np.sqrt(T1_perp/m1)
v1yt = np.sqrt(T1_perp/m1)
v1zt = np.sqrt(T1_para/m1)

n1=1.
e1=1.
N1 = 256

s1 = ParticleSpecies(m=m1, N=N1, n=n1, e=e1)
s1.initialize(vt = np.array([v1xt,v1yt,v1zt]), T_tolerance=1.e-3)

coulomb_log = 1.

tau = 1./get_nu_iso(T1_perp, T1_para, m1, n1, coulomb_log)


# ==================== Load initial conditions ====================
with open('./initial_conditions/N({})_T({},{})_m({})_n({}).pickle'.format(N1,int(T1_perp),int(T1_para),int(m1),int(n1)), 'rb') as handle:
    s_ensemble = pickle.load(handle)


# ==================== calculate ensemble ====================
start_time = time.time()

dt = 0.01 * tau
total_t = 10. * tau
Nt = int(total_t/dt)
N_record = max(int(Nt/2000),1)

pool = mp.Pool(processes=N_process)
job_status = {}

for i in Process_list:
    
    if i%N_process == 0: print_status = True 
    else: print_status = False
    
    job_status[i] = pool.apply_async(func=get_ensemble_explicit, 
                                     kwds={
                                         'dt':dt, 
                                         'total_t':total_t, 
                                         's1':s_ensemble[i], 
                                         'coulomb_log':coulomb_log, 
                                         'save_dir':save_dir, 
                                         'sub_id':i,
                                         'N_group':N_group, 
                                         'N_record':N_record, 
                                         'special_record':[0],
                                         'print_status':print_status}, 
                                     error_callback=print)

for i in Process_list:
    job_status[i].wait()

time_used = time.time() - start_time

print('total time = {:.3f} s = {:.3f} mins = {:.3f} hours.'.format(
    time_used, time_used/60, time_used/3600))
