"""
Calculate the temperature relaxation with mass ratio, using modified midpoint method
"""
import numpy as np
from BothLandauCollision import *
import time
import multiprocessing as mp


# ==================== script specific setting ====================
save_dir = './data/T_relax_T(4,1)_m(1,5)_n(1,2)/(64,128)_0.001_ee_1'
# save_dir = './data/T_relax_T(2,1)_m(1,20)_n(1,2)/test'
Process_list = range(512)
N_process = 16


# ==================== Define collision time ====================
def get_nu_relax(T1, T2, m1, m2, e1, e2, n2, coulomb_log):
    return n2 * e1**2. * e2**2. * coulomb_log * ( (T1/m1) + (T2/m2) )**(-1.5) / ( 3. * np.sqrt(2.*np.pi**3.) * m1 * m2 )


# ==================== Define system ====================
T1=4.
T2=1.

m1=1.
m2=5.

v1t = np.sqrt(T1/m1)
v2t = np.sqrt(T2/m2)

n1=1.
n2=2.

e1=2.
e2=1.

N1 = 64
N2 = 128

coulomb_log = 1.

tau = 1./get_nu_relax(T1, T2, m1, m2, e1, e2, n2, coulomb_log)
tau_ee = 1./get_nu_relax(T1, T1, m1, m1, e1, e1, n1, coulomb_log)


# ==================== Load initial conditions ====================
with open('./data/initial_conditions/N({},{})_T({},{})_m({},{})_n({},{}).pickle'.format(N1,N2,int(T1),int(T2),int(m1),int(m2),int(n1),int(n2)), 'rb') as handle:
    s_ensemble = pickle.load(handle)


# ==================== calculate ensemble ====================
start_time = time.time()

dt = 1.e-3 * tau_ee
total_t = 6. * tau
Nt = int(total_t/dt)
N_record = max(int(Nt/2000),1)
N_groups = [1,1,1]

pool = mp.Pool(processes=N_process)
job_status = {}

for i in Process_list:
    
    if i%N_process == 0: print_status = True 
    else: print_status = False
    
    job_status[i] = pool.apply_async(func=get_ensemble_explicit, 
                                     kwds={
                                         'dt':dt, 
                                         'total_t':total_t, 
                                         's1':s_ensemble[i][1], 
                                         's2':s_ensemble[i][2], 
                                         'coulomb_log':coulomb_log, 
                                         'save_dir':save_dir, 
                                         'sub_id':i,
                                         'N_groups':N_groups, 
                                         'N_record':N_record, 
                                         'special_record':[0],
                                         'print_status':print_status,
                                         'intra_collision':True, 
                                         'inter_collision':True}, 
                                     error_callback=print)

for i in Process_list:
    job_status[i].wait()

time_used = time.time() - start_time

print('total time = {:.3f} s = {:.3f} mins = {:.3f} hours.'.format(
    time_used, time_used/60, time_used/3600))
