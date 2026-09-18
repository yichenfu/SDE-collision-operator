"""Energy- and momentum-conserving Landau collision solvers in velocity space.

ParticleSpecies stores an (N, 3) velocity array and its moments. IntraExplicit
advances one species; BothExplicit composes intra- and inter-species updates.
Both use the modified midpoint/Cayley method from Eqs. (17)-(18) of the paper.
Units set k_B = epsilon_0 = 1, and gamma = -3 gives Coulomb collisions.

Array conventions in the kernels are g = group, i/j = particle within a group,
and the final length-3 axes = Cartesian components. Collision weights account
for the sampled group size; physical macro-particle weights remain n / N.
See README.md for equations, assumptions, and complete examples.
"""

import numpy as np 
import os, pickle, copy, time, copy
import multiprocessing as mp
from math import gcd, isqrt
from scipy.stats import norm
from scipy.stats.qmc import Sobol


def _resolve_group_count(value, counts, min_particles, name):
    """Return a valid integer group count for an enabled collision operator.

    ``counts`` contains the participating species' particle counts. A value of
    -1 selects the largest common divisor that leaves ``min_particles`` in
    each group: one for inter-species collisions, two for intra-species.
    Explicit counts must obey the same divisibility and size requirements.
    ``name`` identifies the argument in any ValueError raised to the caller.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be a positive integer or -1')
    if any(count < min_particles for count in counts):
        raise ValueError(f'{name} requires at least {min_particles} particles per species')
    if value == -1:
        common = gcd(*counts)
        limit = min(count // min_particles for count in counts)
        value = 1
        # Search paired divisors up to sqrt(common), including odd counts.
        for divisor in range(1, isqrt(common) + 1):
            if common % divisor == 0:
                for candidate in (divisor, common // divisor):
                    if candidate <= limit:
                        value = max(value, candidate)
    if value < 1 or any(count % value or count // value < min_particles for count in counts):
        raise ValueError(
            f'{name} must divide the particle counts and leave at least '
            f'{min_particles} particles per group'
        )
    return int(value)


def _validate_recording(Nt, N_record):
    """Reject invalid step counts or recording intervals before advancing."""
    for name, value, minimum in [('Nt', Nt, 0), ('N_record', N_record, 1)]:
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < minimum):
            raise ValueError(f'{name} must be an integer >= {minimum}')


class ParticleSpecies:
    """Store one homogeneous species and sample moments of its velocities.

    Attributes m, e, n, and N are mass, charge, number density, and particle
    count. The physical weight of each macro-particle is n / N. Call initialize
    or assign an (N, 3) floating-point array to v before starting a simulation.
    """


    def __init__(self, m = 1., e = 1., N = 1, n=1.):
        """Allocate zero velocities and diagnostic fields for N particles.

        Use positive m, n, and integer N; either charge sign is allowed.
        The initial zero array is a placeholder, not a sampled distribution.
        """
        
        self.N = N
        self.m = m
        self.e = e
        self.n = n
        self.v = np.zeros((self.N,3))
        
        self.P = np.zeros(3)
        self.v_avg = np.zeros(3)
        self.E_total = 0.
        self.E_thermal = 0.
        self.T = 0.
        self.T_direction = np.zeros(3)
        
        
    
    def get_statistics(self):
        """Refresh momentum, mean flow, kinetic energy, and temperatures.

        P = m * sum(v) and E_total = m * sum(|v|**2) / 2 omit the physical
        particle weight n / N. E_thermal subtracts the sample mean flow first.
        T_direction[d] = m * mean((v[d] - v_avg[d])**2), with k_B = 1;
        T is the average of these three directional temperatures. This method
        updates attributes in place and does not return a separate result.
        """
        
        self.P = self.m * np.sum(self.v, axis=0)
        self.v_avg = np.average(self.v, axis=0)
        
        self.E_total = 0.5 * self.m * np.sum(self.v**2.)
        self.E_thermal = 0.5 * self.m * np.sum((self.v-self.v_avg)**2.)
        
        self.T_direction = self.m * np.sum((self.v-self.v_avg)**2., axis=0) / self.N 
        self.T = np.average(self.T_direction)
        


    def initialize(self,
                   vt=np.array([1., 1., 1.]), 
                   u=np.array([0., 0., 0.]),
                   T_tolerance = None,
                   sobol = False):
        """Sample Cartesian Gaussian velocities and update the diagnostics.

        vt and u are length-3 arrays of thermal standard deviations and mean
        flow velocities. For desired temperatures use vt[d] = sqrt(T[d] / m).
        With T_tolerance=None, each component is sampled once. Otherwise,
        components are resampled until the relative error in the measured
        thermal speed (not temperature) meets the tolerance.

        The optional scrambled Sobol branch is used only when T_tolerance is
        supplied; it requires N to be a power of two and currently ignores u.
        Positive vt components are expected when using the tolerance check.
        """

        # Sample components independently, allowing an anisotropic Maxwellian.
        for i in range(3):

            if T_tolerance is not None:
                
                if sobol is False:
                
                    # Reject samples whose thermal speed misses the target.
                    while np.abs(np.sqrt(self.T_direction[i]/self.m) - vt[i]) / np.abs(vt[i]) > T_tolerance:
                        self.v[:,i] = np.random.normal(u[i], vt[i], self.N)
                        self.get_statistics()
                        
                else:
                    
                    m = int(np.log2(self.N))
                    
                    # Apply the same speed tolerance to inverse-CDF samples.
                    while np.abs(np.sqrt(self.T_direction[i]/self.m) - vt[i]) / np.abs(vt[i]) > T_tolerance:
                        # Map low-discrepancy uniform samples to Gaussian speeds.
                        thermal_dis = norm(loc=0., scale=vt[i])
                        sampler = Sobol(d=1, scramble=True)
                        sample = sampler.random_base2(m=m).reshape((-1))
                        self.v[:,i] = thermal_dis.ppf(sample)
                        self.get_statistics()
                    
            else:
                
                self.v[:,i] = np.random.normal(u[i], vt[i], self.N)
                self.get_statistics()
            
            
       
       
       



class IntraExplicit:
    """Advance self-collisions of one species with the modified midpoint rule.

    The pair noises are antisymmetric, and the coupled velocity update is a
    Cayley transform. This implements Eq. (17) and conserves energy and momentum
    to roundoff for finite, nonsingular inputs. explicit_solve handles grouping
    and recording; the lower-level methods build and apply one collision step.
    """
    
    def __init__(self, s1: ParticleSpecies, coulomb_log, dt, gamma=-3):
        """Copy s1, set collision parameters, and record the initial state.

        coulomb_log is ln(Lambda), dt is the step size, and gamma is the kernel
        exponent (-3 for Coulomb collisions). Evolution modifies self.s1, not
        the supplied ParticleSpecies. L11 uses the convention epsilon_0 = 1.
        """ 
        
        self.s1 = copy.deepcopy(s1)
        self.m1 = s1.m 
        self.e1 = s1.e 
        self.N1 = s1.N 
        self.n1 = s1.n
        
        self.coulomb_log = coulomb_log
        self.L11 = self.e1**4. * coulomb_log / ( 4.*np.pi )
        
        self.gamma = gamma        
        self.dt = dt
        
        # time history of physical quantities
        self.step = 0
        self.time = 0.
        self.step_history = []
        self.time_history = []
        self.P1_history = []
        self.v1_avg_history = []
        self.E1_total_history = []
        self.E1_thermal_history = []
        self.T1_direction_history = []
        self.T1_history = []
        self.v1_1_history = [] # value of the velocity of the first particle
        self.specific_v1 = {} # distribution of v1 at specific time 
        
        self.record_physics()
        
        
        
        
        
    def record_physics(self):
        """Append current time, completed step count, and species-1 moments.

        Velocity index zero is copied, but shuffling changes which particle
        occupies that index. Its history is not a fixed-particle trajectory
        when grouping is enabled. E/P histories use unweighted particle sums.
        """
        
        self.step_history.append(self.step)
        self.time_history.append(self.time)
        self.s1.get_statistics()
        self.P1_history.append(self.s1.P)
        self.v1_avg_history.append(self.s1.v_avg)
        self.E1_total_history.append(self.s1.E_total)
        self.E1_thermal_history.append(self.s1.E_thermal)
        self.T1_direction_history.append(self.s1.T_direction)
        self.T1_history.append(self.s1.T)
        self.v1_1_history.append(self.s1.v[0].copy())
         
    
    
    
    def generate_dW_g_ij_intra(self, dt=0.1, N_group1=1):
        """Return antisymmetric Wiener increments for all intra-species pairs.

        With G = N_group1 and p = N1 / G, the shape is (G, p, p, 3).
        Each upper-triangle entry has covariance dt * I3; lower entries are its
        negative and diagonal entries are zero. Different unordered pairs and
        groups use independent samples. The public solver passes self.dt.
        """
    
        N_particle = int(self.N1/N_group1)
        dW = np.random.normal(0., np.sqrt(dt), (N_group1, N_particle, N_particle, 3))
        # Retain one independent draw per unordered pair before antisymmetrizing.
        # Subtracting two full random arrays would incorrectly double variance.
        i_indices, j_indices = np.tril_indices(N_particle)
        dW[:,i_indices, j_indices,:] = 0.
        dW1 = dW - dW.transpose((0,2,1,3))
            
        return dW1
    
    
    
    def M_g_ij_intra(self, vg, m, N, w, L, dWg, N_group):
        """
        Build the intra-species block matrix for (I + M) v_new = (I - M) v.

        For G = N_group and p = N / G, vg has shape (G, p, 3), dWg has
        shape (G, p, p, 3), and the returned array has shape (G, 3*p, 3*p).
        w is the collision weight n / (p - 1), L is the collision coefficient,
        and m is the species mass. Each off-diagonal block is the cross-product
        matrix of sqrt(w*L) * Omega_ij / (2*m); each diagonal is minus its row
        sum. The factor 1/2 comes from averaging old and new velocities.
        """
        
        # calculate u = vi - vj
        u = vg[:,:, np.newaxis, :] - vg[:,np.newaxis, :, :]
        u_abs = np.linalg.norm(u, axis=3)
        
        # Exclude i == j: the numerator is zero, so replace its denominator by 1.
        # This does not regularize coincident velocities of distinct particles.
        i_j_indices = np.arange(int(N/N_group))
        i_indices = np.tile(i_j_indices, N_group)
        j_indices = np.tile(i_j_indices, N_group)
        u_abs[:,i_indices,j_indices] = 1.
        
        # Fold sqrt(w*L)/(2*m) into the paper's Omega before forming its hat map.
        omega = ((w*L)**0.5 / (2.*m)) * (u_abs**(0.5*self.gamma-1.))[:,:,:,np.newaxis] * np.cross(u, dWg)
        
        # Hat map: omega_hat @ x = omega cross x for any 3-vector x.
        omega_hat = np.zeros((N_group, int(N/N_group), int(N/N_group), 3, 3))
        omega_hat[:,:,:,0,1] = - omega[:,:,:,2]
        omega_hat[:,:,:,0,2] = omega[:,:,:,1]
        omega_hat[:,:,:,1,0] = omega[:,:,:,2]
        omega_hat[:,:,:,1,2] = - omega[:,:,:,0]
        omega_hat[:,:,:,2,0] = - omega[:,:,:,1]
        omega_hat[:,:,:,2,1] = omega[:,:,:,0]
        
        # Zero block-row sums preserve a common velocity (and total momentum).
        M = copy.deepcopy(omega_hat)
        omega_hat_sum = np.sum(omega_hat, axis=2)
        diag_index = np.arange(int(N/N_group), dtype=int)
        M[:,diag_index,diag_index,:,:] = - omega_hat_sum[:,diag_index,:,:]
        
        # Interleave particle and Cartesian indices into ordinary matrix axes.
        return (M.transpose((0,1,3,2,4))).reshape((N_group, int(3*N/N_group), int(3*N/N_group)))
    
    
    
    
    def intra_one_step_explicit(self, v, m, N, w, L, dWg, N_group):
        """Return an (N, 3) velocity array after one intra-species step.

        Inputs v and dWg are the old velocities and already-scaled Wiener
        increments; dt enters through dWg, not an additional factor here.
        Group sizes must be valid and velocities already ordered by group.
        Batched dense Cayley transforms solve Eq. (17) without nonlinear
        iteration. The input v is not modified by this method.
        """
        
        # group up velocities
        vg = v.reshape((N_group, int(N/N_group), 3))
        vg_vec = vg.reshape((N_group, int(3*N/N_group)))
            
        # generate big matrix M
        M = self.M_g_ij_intra(vg, m, N, w, L, dWg, N_group)
        
        # generate big identity matrix
        I = np.zeros((N_group, int(3*N/N_group), int(3*N/N_group)))
        I[np.arange(N_group), :, :] = np.identity(int(3*N/N_group))
        
        # M is skew-symmetric: its Cayley transform preserves the squared norm.
        Cayley = np.einsum('pij,pjk->pik', np.linalg.inv(I + M), I - M)
        
        # calculate new vg_vec
        return (np.einsum('pij,pj->pi', Cayley, vg_vec)).reshape((N, 3))
    
    
    
    
    # final solver
    def explicit_solve(self, Nt=10, N_group=-1, N_record=1, print_status=False,
                       special_record=None):
        """
        Advance self.s1 by Nt additional steps, updating velocities in place.

        N_group=1 includes every other particle as a collision partner;
        -1 chooses the largest valid number of equal-sized groups, with at
        least two particles per group. Multiple groups are shuffled each step.
        Smaller partner samples use the corrected weight n / (N/N_group - 1).

        Record moments every N_record cumulative steps and at the end of this
        call. time_history includes any shorter final interval. special_record
        selects cumulative completed-step numbers for full velocity snapshots:
        0 is the initial state, None defaults to [1], and [] disables snapshots.
        print_status prints progress at recording steps. This method returns
        None; results are stored on the solver, including across repeated calls.
        """
        
        _validate_recording(Nt, N_record)
        N_group = _resolve_group_count(N_group, (self.N1,), 2, 'N_group')
        special_record = {1} if special_record is None else set(special_record)

        self.N_group = N_group
        self.N_record = N_record
        
        # Collision normalization counts field particles only, excluding self.
        w1 = self.s1.n / ( self.s1.N / N_group - 1 )
        if self.step in special_record:
            self.specific_v1[self.step] = self.s1.v.copy()
        
        # main loop
        for i in range(Nt):
            
            # shuffle velocities if N_group > 1
            if N_group>1: np.random.shuffle(self.s1.v)
                
            # calculate one step advance for intra-species collision
            dWg1 = self.generate_dW_g_ij_intra(dt=self.dt, N_group1=N_group)
            self.s1.v = self.intra_one_step_explicit(v=self.s1.v, 
                                                        m=self.s1.m, 
                                                        N=self.s1.N, 
                                                        w=w1, 
                                                        L=self.L11,
                                                        dWg=dWg1, 
                                                        N_group=N_group)
            
            self.step += 1
            self.time += self.dt
            # Record regular intervals and always include the final state.
            if self.step % N_record == 0 or i == Nt - 1:
                self.record_physics()
                if print_status: print('Status = {} over {}.'.format(i + 1, Nt), end='\r')
            
            # record specific distribution function
            if self.step in special_record:
                self.specific_v1[self.step] = self.s1.v.copy()

    


     
            
            

class BothExplicit:
    """Advance collisions within and between two equally weighted species.

    Each time step first advances both intra-species operators, then the
    inter-species operator (Eq. (18)). Either type can be disabled. All enabled
    substeps preserve the same total energy and momentum. Initial species are
    copied, and diagnostics/snapshots are retained on this solver instance.
    """
    
    def __init__(self, s1: ParticleSpecies, s2:ParticleSpecies, coulomb_log, dt, gamma=-3):
        """Copy species alpha/beta, initialize coefficients, and record t=0.

        s1 and s2 must satisfy s1.n / s1.N == s2.n / s2.N for the paper's
        inter-species model. Unequal weights currently emit a warning rather
        than being rejected. A single coulomb_log is used for L11, L22, and
        L12. dt is the step size and gamma=-3 selects the Coulomb kernel.
        """ 
        
        self.s1 = copy.deepcopy(s1)
        self.m1 = s1.m 
        self.e1 = s1.e 
        self.N1 = s1.N 
        self.n1 = s1.n
        
        self.s2 = copy.deepcopy(s2)
        self.m2 = s2.m 
        self.e2 = s2.e 
        self.N2 = s2.N 
        self.n2 = s2.n
        
        self.coulomb_log = coulomb_log
        self.L11 = self.e1**4. * coulomb_log / ( 4.*np.pi )
        self.L22 = self.e2**4. * coulomb_log / ( 4.*np.pi )
        self.L12 = self.e1**2. * self.e2**2. * coulomb_log / ( 4.*np.pi )
        
        self.gamma = gamma
        
        if (self.n1/self.N1) != (self.n2/self.N2):
            print('Warning, unequal particle weight!')
        
        self.dt = dt
        
        # time history of physical quantities
        self.step = 0
        self.time = 0.
        self.step_history = []
        self.time_history = []
        self.P1_history = []
        self.v1_avg_history = []
        self.E1_total_history = []
        self.E1_thermal_history = []
        self.T1_direction_history = []
        self.T1_history = []
        self.v1_1_history = [] # value of the velocity of the first particle
        self.specific_v1 = {} # distribution of v1 at specific time 
        
        self.P2_history = []
        self.v2_avg_history = []
        self.E2_total_history = []
        self.E2_thermal_history = []
        self.T2_direction_history = []
        self.T2_history = []
        self.v2_1_history = [] # value of the velocity of the first particle
        self.specific_v2 = {} # distribution of v2 at specific time 
        
        self.record_physics()
        
        
        
        
        
    def record_physics(self):
        """Append the current step, time, and both species' sample moments.

        Energy and momentum histories omit the common physical weight n / N.
        Sum both species to check conservation. Copies of index-zero velocities
        are also recorded; after regrouping, that index can be a different
        particle, so these are not fixed-particle trajectories.
        """
        
        self.step_history.append(self.step)
        self.time_history.append(self.time)
        self.s1.get_statistics()
        self.P1_history.append(self.s1.P)
        self.v1_avg_history.append(self.s1.v_avg)
        self.E1_total_history.append(self.s1.E_total)
        self.E1_thermal_history.append(self.s1.E_thermal)
        self.T1_direction_history.append(self.s1.T_direction)
        self.T1_history.append(self.s1.T)
        self.v1_1_history.append(self.s1.v[0].copy())
        
        self.s2.get_statistics()
        self.P2_history.append(self.s2.P)
        self.v2_avg_history.append(self.s2.v_avg)
        self.E2_total_history.append(self.s2.E_total)
        self.E2_thermal_history.append(self.s2.E_thermal)
        self.T2_direction_history.append(self.s2.T_direction)
        self.T2_history.append(self.s2.T)
        self.v2_1_history.append(self.s2.v[0].copy())
        
        
        
        
        
        
    def generate_dW_g_ij_inter(self, dt=0.1, N_group=1):
        """Sample independent inter-species pair increments with variance dt.

        Return shape (G, N1/G, N2/G, 3), where G = N_group. Unlike the
        intra-species array, this array is not antisymmetrized: its two particle
        indices belong to different species. The same draw drives both partners
        with opposite mass-weighted impulses in the assembled matrix Q.
        """
        
        N_particle1 = int(self.N1/N_group)
        N_particle2 = int(self.N2/N_group)
            
        return np.random.normal(0., np.sqrt(dt), (N_group, N_particle1, N_particle2, 3))
    
    
    
    
    def omega_hat_g_ij_inter(self, v1g, v2g, w, L, dWg, N_group):
        """
        Form the 3-by-3 cross-product block for each inter-species pair.

        v1g/v2g have shapes (G, N1/G, 3)/(G, N2/G, 3); dWg has shape
        (G, N1/G, N2/G, 3). w is the group's collision weight and L = L12.
        Return shape (G, N1/G, N2/G, 3, 3), representing sqrt(w*L)*Omega/2.
        Species masses are applied later in M_g_ij_inter and N_g_ij_inter.
        Each final 3-by-3 block is skew-symmetric in Cartesian indices.
        """
        
        # calculate u = vi - vj
        u = v1g[:,:, np.newaxis, :] - v2g[:,np.newaxis, :, :]
        u_abs = np.linalg.norm(u, axis=3)
        
        # Freeze the relative-velocity coefficient at the beginning of the step.
        # The 1/2 is the midpoint factor; do not multiply dWg by dt again.
        omega = ((w*L)**0.5 / 2.) * (u_abs**(0.5*self.gamma-1.))[:,:,:,np.newaxis] * np.cross(u, dWg)
        
        # create the matrix omega_hat
        omega_hat = np.zeros((N_group, int(self.N1/N_group), int(self.N2/N_group), 3, 3))
        omega_hat[:,:,:,0,1] = - omega[:,:,:,2]
        omega_hat[:,:,:,0,2] = omega[:,:,:,1]
        omega_hat[:,:,:,1,0] = omega[:,:,:,2]
        omega_hat[:,:,:,1,2] = - omega[:,:,:,0]
        omega_hat[:,:,:,2,0] = - omega[:,:,:,1]
        omega_hat[:,:,:,2,1] = omega[:,:,:,0]
        
        return omega_hat


    
    
    
    def M_g_ij_inter(self, omega_hat, N_group):
        """
        Return the species-alpha block row of Q in Appendix C.2.

        For p = N1/G and q = N2/G, M1 has shape (G, p, p, 3, 3) and
        M2 has shape (G, p, q, 3, 3). The diagonal M1 blocks contain the
        negative sum over beta partners; M2 couples alpha to every beta
        particle, including pairs whose integer index labels happen to match.
        Both blocks divide the shared omega_hat coefficient by mass m1.
        """
        
        omega_hat_sum = np.sum(omega_hat, axis=2)
        diag_index = np.arange(int(self.N1/N_group), dtype=int)
        M1 = np.zeros((N_group, int(self.N1/N_group), int(self.N1/N_group), 3, 3))
        M1[:,diag_index,diag_index,:,:] = - omega_hat_sum[:,diag_index,:,:] / self.m1
        
        M2 = omega_hat / self.m1
        
        return M1, M2
    
    
    
    
    def N_g_ij_inter(self, omega_hat, N_group):
        """
        Return the species-beta block row of Q in Appendix C.2.

        For p = N1/G and q = N2/G, N1 has shape (G, q, q, 3, 3),
        and N2_T has shape (G, q, p, 3, 3). The diagonal sums over alpha
        partners and both arrays divide by m2. N2_T swaps particle indices
        only: transposing the Cartesian block too would reverse its sign.
        """
        
        omega_hat_sum = np.sum(omega_hat, axis=1)
        diag_index = np.arange(int(self.N2/N_group), dtype=int)
        N1 = np.zeros((N_group, int(self.N2/N_group), int(self.N2/N_group), 3, 3))
        N1[:,diag_index,diag_index,:,:] = - omega_hat_sum[:,diag_index,:,:] / self.m2
        
        N2_T = omega_hat.transpose((0,2,1,3,4)) / self.m2
        
        return N1, N2_T
    
    
    
    def Q_g_inter(self, omega_hat, N_group):
        """
        Assemble Q = [[M1, M2], [N2_T, N1]] for each group.

        Return shape (G, 3*(N1+N2)/G, 3*(N1+N2)/G). Flattened velocities
        place all alpha particles before beta particles, with xyz contiguous
        for each particle. For H = diag(m1*I, m2*I), H@Q + Q.T@H = 0;
        this mass-weighted skew identity underlies kinetic-energy conservation.
        """
        
        M1, M2 = self.M_g_ij_inter(omega_hat, N_group)
        N1, N2_T = self.N_g_ij_inter(omega_hat, N_group)
        
        # Convert (particle row, particle column, xyz row, xyz column) blocks
        # into a dense matrix whose row/column order matches flattened velocities.
        M1_big = (np.transpose(M1,(0,1,3,2,4))).reshape((N_group, int(3*self.N1/N_group), int(3*self.N1/N_group)))
        M2_big = (np.transpose(M2,(0,1,3,2,4))).reshape((N_group, int(3*self.N1/N_group), int(3*self.N2/N_group)))
        
        N1_big = (np.transpose(N1,(0,1,3,2,4))).reshape((N_group, int(3*self.N2/N_group), int(3*self.N2/N_group)))
        N2_T_big = (np.transpose(N2_T,(0,1,3,2,4))).reshape((N_group, int(3*self.N2/N_group), int(3*self.N1/N_group)))
        
        Q = np.zeros((N_group, int(3*(self.N1+self.N2)/N_group), int(3*(self.N1+self.N2)/N_group)))
        
        Q[:,:int(3*self.N1/N_group), :int(3*self.N1/N_group)] = M1_big
        Q[:,:int(3*self.N1/N_group), int(3*self.N1/N_group):] = M2_big
        Q[:,int(3*self.N1/N_group):, :int(3*self.N1/N_group)] = N2_T_big 
        Q[:,int(3*self.N1/N_group):, int(3*self.N1/N_group):] = N1_big
        
        return Q
    
    
    
    
    
    def inter_one_step_explicit(self, v1, v2, w, L, dWg, N_group):
        """Return both species' velocities after one inter-species step.

        v1/v2 have shapes (N1, 3)/(N2, 3) and must already be ordered by
        group. w is the grouped collision weight; dWg contains the sampled
        Wiener increments, including sqrt(dt). The Cayley transform solves
        Eq. (18) simultaneously for both species. Inputs are not modified;
        the returned pair of arrays has the same shapes as v1 and v2.
        """
        
        # group up velocities
        v1g = v1.reshape((N_group, int(self.N1/N_group), 3))
        v2g = v2.reshape((N_group, int(self.N2/N_group), 3))
        v1g_vec = v1g.reshape((N_group, int(3*self.N1/N_group)))
        v2g_vec = v2g.reshape((N_group, int(3*self.N2/N_group)))
        
        # get the total vector 
        vg_vec = np.zeros((N_group, int(3*(self.N1+self.N2)/N_group)))
        vg_vec[:,:int(3*self.N1/N_group)] = v1g_vec
        vg_vec[:,int(3*self.N1/N_group):] = v2g_vec
              
        
        # generate omega hat matrix
        omega_hat = self.omega_hat_g_ij_inter(v1g, v2g, w, L, dWg, N_group)
        
        # generate big matrix Q
        Q = self.Q_g_inter(omega_hat, N_group)
        
        # generate big identity matrix
        I = np.zeros((N_group, int(3*(self.N1+self.N2)/N_group), int(3*(self.N1+self.N2)/N_group)))
        I[np.arange(N_group), :, :] = np.identity(int(3*(self.N1+self.N2)/N_group))
        
        # Q need not be skew for unequal masses; its Cayley transform preserves H.
        Cayley = np.einsum('pij,pjk->pik', np.linalg.inv(I + Q), I - Q)
        
        # calculate new vg_vec
        vg_vec_new = np.einsum('pij,pj->pi', Cayley, vg_vec)
        
        v1_new = vg_vec_new[:,:int(3*self.N1/N_group)].reshape((int(self.N1), 3))
        v2_new = vg_vec_new[:,int(3*self.N1/N_group):].reshape((int(self.N2), 3))
        
        # return v1_new, v2_new
        return v1_new, v2_new
    
    
    
    
    def generate_dW_g_ij_intra(self, dt=0.1, N_group1=1, N_group2=1):
        """Return independent antisymmetric pair noises for each species.

        The two shapes are (G1, N1/G1, N1/G1, 3) and
        (G2, N2/G2, N2/G2, 3). For each species, diagonal entries are zero
        and dW[g,i,j] = -dW[g,j,i]. Each unordered pair has covariance dt*I3;
        the two species use independent draws even if their group sizes match.
        """
    
        N_particle = int(self.N1/N_group1)
        dW = np.random.normal(0., np.sqrt(dt), (N_group1, N_particle, N_particle, 3))
        # Use only the upper triangle, avoiding double variance on subtraction.
        i_indices, j_indices = np.tril_indices(N_particle)
        dW[:,i_indices, j_indices,:] = 0.
        dW1 = dW - dW.transpose((0,2,1,3))
        
        N_particle = int(self.N2/N_group2)
        dW = np.random.normal(0., np.sqrt(dt), (N_group2, N_particle, N_particle, 3))
        i_indices, j_indices = np.tril_indices(N_particle)
        dW[:,i_indices, j_indices,:] = 0.
        dW2 = dW - dW.transpose((0,2,1,3))
            
        return dW1, dW2
    
    
    
    def M_g_ij_intra(self, vg, m, N, w, L, dWg, N_group):
        """
        Build the intra-species matrix M for either species (Eq. (17)).

        vg has shape (G, N/G, 3) and dWg has shape (G, N/G, N/G, 3).
        Pass that species' mass m, count N, corrected weight w=n/(N/G-1),
        and self-collision coefficient L. Return shape (G, 3*N/G, 3*N/G).
        This is the same block construction as IntraExplicit.M_g_ij_intra:
        off-diagonal cross-product blocks and negative block-row sums.
        """
        
        # calculate u = vi - vj
        u = vg[:,:, np.newaxis, :] - vg[:,np.newaxis, :, :]
        u_abs = np.linalg.norm(u, axis=3)
        
        # Self-pairs have zero noise; protect only their zero denominators.
        i_j_indices = np.arange(int(N/N_group))
        i_indices = np.tile(i_j_indices, N_group)
        j_indices = np.tile(i_j_indices, N_group)
        u_abs[:,i_indices,j_indices] = 1.
        
        # Combine the paper's Omega with the midpoint and mass coefficients.
        omega = ((w*L)**0.5 / (2.*m)) * (u_abs**(0.5*self.gamma-1.))[:,:,:,np.newaxis] * np.cross(u, dWg)
        
        # A skew 3-by-3 block implements the cross product with omega.
        omega_hat = np.zeros((N_group, int(N/N_group), int(N/N_group), 3, 3))
        omega_hat[:,:,:,0,1] = - omega[:,:,:,2]
        omega_hat[:,:,:,0,2] = omega[:,:,:,1]
        omega_hat[:,:,:,1,0] = omega[:,:,:,2]
        omega_hat[:,:,:,1,2] = - omega[:,:,:,0]
        omega_hat[:,:,:,2,0] = - omega[:,:,:,1]
        omega_hat[:,:,:,2,1] = omega[:,:,:,0]
        
        # The diagonal balances each block row, preserving a common velocity.
        M = copy.deepcopy(omega_hat)
        omega_hat_sum = np.sum(omega_hat, axis=2)
        diag_index = np.arange(int(N/N_group), dtype=int)
        M[:,diag_index,diag_index,:,:] = - omega_hat_sum[:,diag_index,:,:]
        
        # Flatten with each particle's xyz components adjacent in the vector.
        return (M.transpose((0,1,3,2,4))).reshape((N_group, int(3*N/N_group), int(3*N/N_group)))
    
    
    
    
    def intra_one_step_explicit(self, v, m, N, w, L, dWg, N_group):
        """Return one species' new (N, 3) velocities for its self-collisions.

        The mass, particle count, weight, and L select which species is evolved.
        Already-grouped v and Wiener increments dWg determine a batched Cayley
        update; dt is included in dWg. This matches the standalone IntraExplicit
        single-step method and leaves the input velocity array unchanged.
        """
        
        # group up velocities
        vg = v.reshape((N_group, int(N/N_group), 3))
        vg_vec = vg.reshape((N_group, int(3*N/N_group)))
            
        # generate big matrix M
        M = self.M_g_ij_intra(vg, m, N, w, L, dWg, N_group)
        
        # generate big identity matrix
        I = np.zeros((N_group, int(3*N/N_group), int(3*N/N_group)))
        I[np.arange(N_group), :, :] = np.identity(int(3*N/N_group))
        
        # A skew-symmetric M gives a norm-preserving Cayley transform per group.
        Cayley = np.einsum('pij,pjk->pik', np.linalg.inv(I + M), I - M)
        
        # calculate new vg_vec
        return (np.einsum('pij,pj->pi', Cayley, vg_vec)).reshape((N, 3))
    
    
    
    
    # final solver
    def explicit_solve(self, Nt=10, N_groups=None, N_record=1, print_status=False,
                       special_record=None, dWg_given=None,
                       intra_collision = True, inter_collision = True):
        """
        Advance both species by Nt additional steps, with optional collisions.

        N_groups = [G_inter, G_species1, G_species2]; 1 means all partners,
        -1 selects the largest valid group count, and None defaults all entries
        to -1. Disabled operators use 1. Each species is shuffled whenever an
        enabled operator groups it. intra_collision and inter_collision select
        substeps; self-collisions are applied before inter-species collisions.

        Moments are recorded every N_record cumulative steps and at this call's
        end; time_history stores the actual times. special_record selects full
        velocity snapshots by cumulative completed-step number (0 is initial,
        None defaults to [1], [] disables snapshots). print_status prints progress.

        dWg_given optionally supplies inter-species noise with shape
        (Nt, G_inter, N1/G_inter, N2/G_inter, 3), already scaled by sqrt(dt).
        Intra-species noise is still drawn internally. For repeatable paths,
        grouping permutations must also be controlled. This method returns None;
        velocities, times, and histories are updated on the solver instance.
        """
        
        _validate_recording(Nt, N_record)
        N_groups = [-1, -1, -1] if N_groups is None else list(N_groups)
        if len(N_groups) != 3:
            raise ValueError('N_groups must contain three group counts')
        N_groups[0] = (_resolve_group_count(N_groups[0], (self.N1, self.N2), 1, 'N_groups[0]')
                       if inter_collision else 1)
        N_groups[1] = (_resolve_group_count(N_groups[1], (self.N1,), 2, 'N_groups[1]')
                       if intra_collision else 1)
        N_groups[2] = (_resolve_group_count(N_groups[2], (self.N2,), 2, 'N_groups[2]')
                       if intra_collision else 1)
        special_record = {1} if special_record is None else set(special_record)
        
        self.N_groups = N_groups
        self.N_record = N_record
        
        if inter_collision:
            # Every beta partner counts: equal physical weights give w_inter=w0*G.
            w_inter = self.s1.n / ( self.s1.N / N_groups[0] )
        if intra_collision:
            # Each species excludes the test particle from its field sample.
            w1 = self.s1.n / ( self.s1.N / N_groups[1] - 1 )
            w2 = self.s2.n / ( self.s2.N / N_groups[2] - 1 )
        if self.step in special_record:
            self.specific_v1[self.step] = self.s1.v.copy()
            self.specific_v2[self.step] = self.s2.v.copy()
        
        # main loop
        for i in range(Nt):
            
            # Regroup each species whenever an enabled operator needs it.
            if (inter_collision and N_groups[0] > 1) or (intra_collision and N_groups[1] > 1):
                np.random.shuffle(self.s1.v)
            if (inter_collision and N_groups[0] > 1) or (intra_collision and N_groups[2] > 1):
                np.random.shuffle(self.s2.v)
                
            # First substep: each species self-collides using independent noise.
            if intra_collision:
                dWg1, dWg2 = self.generate_dW_g_ij_intra(dt=self.dt, N_group1=N_groups[1], N_group2=N_groups[2])
                self.s1.v = self.intra_one_step_explicit(v=self.s1.v, 
                                                         m=self.s1.m, 
                                                         N=self.s1.N, 
                                                         w=w1, 
                                                         L=self.L11,
                                                         dWg=dWg1, 
                                                         N_group=N_groups[1])
                self.s2.v = self.intra_one_step_explicit(v=self.s2.v, 
                                                         m=self.s2.m, 
                                                         N=self.s2.N, 
                                                         w=w2,
                                                         L=self.L22,
                                                         dWg=dWg2, 
                                                         N_group=N_groups[2])
            
            # Second substep: inter-species collisions use the updated velocities.
            if inter_collision:
                if dWg_given is None:
                    dWg = self.generate_dW_g_ij_inter(dt=self.dt, N_group=N_groups[0])
                else:
                    dWg = dWg_given[i]
                    
                self.s1.v, self.s2.v = self.inter_one_step_explicit(v1=self.s1.v, 
                                                                    v2=self.s2.v, 
                                                                    w=w_inter, 
                                                                    L=self.L12,
                                                                    dWg=dWg, 
                                                                    N_group=N_groups[0])
            
            self.step += 1
            self.time += self.dt
            # Record regular intervals and always include the final state.
            if self.step % N_record == 0 or i == Nt - 1:
                self.record_physics()
                if print_status: print('Status = {} over {}.'.format(i + 1, Nt), end='\r')
            
            # record specific distribution function
            if self.step in special_record:
                self.specific_v1[self.step] = self.s1.v.copy()
                self.specific_v2[self.step] = self.s2.v.copy()

    
    
    
    
    
    
    

        
def get_ensemble_explicit(dt, 
                          total_t, 
                          s1:ParticleSpecies, 
                          s2:ParticleSpecies, 
                          save_dir, 
                          sub_id,
                          coulomb_log=1., 
                          N_groups=None,
                          N_record=1, 
                          special_record=None,
                          print_status=True,
                          intra_collision=True, 
                          inter_collision=True):
    """Compute and pickle one two-species realization for an ensemble.

    Seed NumPy's collision RNG with sub_id, copy the supplied initial species
    into BothExplicit, and advance int(total_t/dt) full steps with gamma=-3.
    No fractional final step is taken. Initial velocities are supplied by the
    caller; this function does not independently resample the distribution.

    Grouping, recording, and collision flags are forwarded to explicit_solve.
    Save the complete solver to save_dir/<sub_id>.pickle, replacing that file
    if it exists, and return None. This top-level function can be submitted to
    a multiprocessing pool; each realization should use a distinct sub_id.
    """
    
    # Seeding here makes a realization independent of worker scheduling order.
    np.random.seed(sub_id)
    
    # create folder if not exist
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except: 
            pass
    
    # Only complete time steps are evolved; inspect saved time_history for times.
    Nt = int(total_t/dt)
    
    solver = BothExplicit(s1=s1, s2=s2, coulomb_log=coulomb_log, gamma=-3, dt=dt)
    solver.explicit_solve(Nt=Nt, 
                          N_groups=N_groups, 
                          N_record=N_record, 
                          print_status=print_status,
                          special_record=special_record, 
                          intra_collision=intra_collision, 
                          inter_collision=inter_collision)
        
    with open('{}/{}.pickle'.format(save_dir, sub_id), 'wb') as handle:
        pickle.dump(solver, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_ensemble_intra_explicit(dt, total_t, s1: ParticleSpecies, save_dir, sub_id,
                              coulomb_log=1., N_group=-1, N_record=1,
                              special_record=None, print_status=True):
    """Compute and pickle one self-collision realization using IntraExplicit.

    The single-species counterpart of get_ensemble_explicit. Seed collision
    draws with sub_id, use the caller's initial s1, and take int(total_t/dt)
    steps with gamma=-3. N_group, N_record, special_record, and print_status
    are passed to explicit_solve. Save the solver in save_dir/<sub_id>.pickle
    (replacing any existing file) and return None; the input s1 is unchanged.
    """
    np.random.seed(sub_id)
    os.makedirs(save_dir, exist_ok=True)
    solver = IntraExplicit(s1=s1, coulomb_log=coulomb_log, gamma=-3, dt=dt)
    solver.explicit_solve(Nt=int(total_t / dt), N_group=N_group,
                          N_record=N_record, print_status=print_status,
                          special_record=special_record)
    with open(os.path.join(save_dir, f'{sub_id}.pickle'), 'wb') as handle:
        pickle.dump(solver, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
