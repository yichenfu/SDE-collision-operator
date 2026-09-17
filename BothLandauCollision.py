import numpy as np 
import os, pickle, copy, time, copy
import multiprocessing as mp
from scipy.stats import norm
from scipy.stats.qmc import Sobol
""" 
Define a class describing Landau collision, regardless of same or different species

The Boltzmann constant k_B = 1.
"""


class ParticleSpecies:
    """ 
    Define a container for all data for one species.
    """


    def __init__(self, m = 1., e = 1., N = 1, n=1.):
        """ 
        Initialize all data, including:
        1. Particle number;
        2. Each particle's mass (m), charge, and velocity (N,3).
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
        """ 
        Calculate the energy, momentum, and temperature in the system
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
        """ 
        Initialize particle velocities of N particles with:
        1. thermal velocity vt = (vtx, vty, vtz);
        2. flow speed u = (ux, uy, uz).
        """

        # initalize velocity
        for i in range(3):

            if T_tolerance is not None:
                
                if sobol is False:
                
                    # make sure initial temperature not differ too much
                    while np.abs(np.sqrt(self.T_direction[i]/self.m) - vt[i]) / np.abs(vt[i]) > T_tolerance:
                        self.v[:,i] = np.random.normal(u[i], vt[i], self.N)
                        self.get_statistics()
                        
                else:
                    
                    m = int(np.log2(self.N))
                    
                    # make sure initial temperature not differ too much
                    while np.abs(np.sqrt(self.T_direction[i]/self.m) - vt[i]) / np.abs(vt[i]) > T_tolerance:
                        # use sobol sequence for low discrepency
                        thermal_dis = norm(loc=0., scale=vt[i])
                        sampler = Sobol(d=1, scramble=True)
                        sample = sampler.random_base2(m=m).reshape((-1))
                        self.v[:,i] = thermal_dis.ppf(sample)
                        self.get_statistics()
                    
            else:
                
                self.v[:,i] = np.random.normal(u[i], vt[i], self.N)
                self.get_statistics()
            
            
       
       
       



class IntraExplicit:
    """ 
    Define the modified midpoint method for both inter and intra species collision that is explicitly solvable
    """
    
    def __init__(self, s1: ParticleSpecies, coulomb_log, dt, gamma=-3):
        """
        Define the initial status of one species and some constant.
        s1: species alpha; s2 species beta. 
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
        """ 
        Record the current status of self.species
        """
        
        self.s1.get_statistics()
        self.P1_history.append(self.s1.P)
        self.v1_avg_history.append(self.s1.v_avg)
        self.E1_total_history.append(self.s1.E_total)
        self.E1_thermal_history.append(self.s1.E_thermal)
        self.T1_direction_history.append(self.s1.T_direction)
        self.T1_history.append(self.s1.T)
        self.v1_1_history.append(self.s1.v[0])
         
    
    
    
    def generate_dW_g_ij_intra(self, dt=0.1, N_group1=1):
        """ 
        Generate anti-symmetric Gaussian random variable Delta W^{ij},
        shape = (N_group, N, N, 3).
        """
    
        N_particle = int(self.N1/N_group1)
        dW = np.random.normal(0., np.sqrt(dt), (N_group1, N_particle, N_particle, 3))
        i_indices, j_indices = np.tril_indices(N_particle)
        dW[:,i_indices, j_indices,:] = 0.
        dW1 = dW - dW.transpose((0,2,1,3))
            
        return dW1
    
    
    
    def M_g_ij_intra(self, vg, m, N, w, L, dWg, N_group):
        """
        Generate the matrix M with indices [g,i,j], shape = (N_group, 3*N/N_group,  3*N/N_group)
        """
        
        # calculate u = vi - vj
        u = vg[:,:, np.newaxis, :] - vg[:,np.newaxis, :, :]
        u_abs = np.linalg.norm(u, axis=3)
        
        # remove zero diagnal terms
        i_j_indices = np.arange(int(N/N_group))
        i_indices = np.tile(i_j_indices, N_group)
        j_indices = np.tile(i_j_indices, N_group)
        u_abs[:,i_indices,j_indices] = 1.
        
        # create the vector omega
        omega = ((w*L)**0.5 / (2.*m)) * (u_abs**(0.5*self.gamma-1.))[:,:,:,np.newaxis] * np.cross(u, dWg)
        
        # create the matrix omega_hat
        omega_hat = np.zeros((N_group, int(N/N_group), int(N/N_group), 3, 3))
        omega_hat[:,:,:,0,1] = - omega[:,:,:,2]
        omega_hat[:,:,:,0,2] = omega[:,:,:,1]
        omega_hat[:,:,:,1,0] = omega[:,:,:,2]
        omega_hat[:,:,:,1,2] = - omega[:,:,:,0]
        omega_hat[:,:,:,2,0] = - omega[:,:,:,1]
        omega_hat[:,:,:,2,1] = omega[:,:,:,0]
        
        # generate matrix M
        M = copy.deepcopy(omega_hat)
        omega_hat_sum = np.sum(omega_hat, axis=2)
        diag_index = np.arange(int(N/N_group), dtype=int)
        M[:,diag_index,diag_index,:,:] = - omega_hat_sum[:,diag_index,:,:]
        
        # return the big matrix
        return (M.transpose((0,1,3,2,4))).reshape((N_group, int(3*N/N_group), int(3*N/N_group)))
    
    
    
    
    def intra_one_step_explicit(self, v, m, N, w, L, dWg, N_group):
        """ 
        Return the one step calculation in modified midpoint method.
        """
        
        # group up velocities
        vg = v.reshape((N_group, int(N/N_group), 3))
        vg_vec = vg.reshape((N_group, int(3*N/N_group)))
            
        # generate big matrix M
        M = self.M_g_ij_intra(vg, m, N, w, L, dWg, N_group)
        
        # generate big identity matrix
        I = np.zeros((N_group, int(3*N/N_group), int(3*N/N_group)))
        I[np.arange(N_group), :, :] = np.identity(int(3*N/N_group))
        
        # calculate the Cayley matrix
        Cayley = np.einsum('pij,pjk->pik', np.linalg.inv(I + M), I - M)
        
        # calculate new vg_vec
        return (np.einsum('pij,pj->pi', Cayley, vg_vec)).reshape((N, 3))
    
    
    
    
    # final solver
    def explicit_solve(self, Nt=10, N_group=-1, N_record=1, print_status=False,
                       special_record = [1]):
        """
        Using modified midpoint method to calculate the time evolution explicitly.
        N_groups = [N_group_inter, N_group_species1, N_group_species2]
        """
        
        # group -1 means max group
        if N_group == -1: N_group = int(self.N1/2)

        self.N_group = N_group
        self.N_record = N_record
        
        w1 = self.s1.n / ( self.s1.N / N_group - 1 )
        
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
            
            # recording
            if i%N_record == 0: 
                self.record_physics()
                if print_status: print('Status = {} over {}.'.format(i, Nt), end='\r')
            
            # record specific distribution function
            if i in special_record:
                self.specific_v1[i] = self.s1.v 

    


     
            
            

class BothExplicit:
    """ 
    Define the modified midpoint method for both inter and intra species collision that is explicitly solvable
    """
    
    def __init__(self, s1: ParticleSpecies, s2:ParticleSpecies, coulomb_log, dt, gamma=-3):
        """
        Define the initial status of one species and some constant.
        s1: species alpha; s2 species beta. 
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
        """ 
        Record the current status of self.species
        """
        
        self.s1.get_statistics()
        self.P1_history.append(self.s1.P)
        self.v1_avg_history.append(self.s1.v_avg)
        self.E1_total_history.append(self.s1.E_total)
        self.E1_thermal_history.append(self.s1.E_thermal)
        self.T1_direction_history.append(self.s1.T_direction)
        self.T1_history.append(self.s1.T)
        self.v1_1_history.append(self.s1.v[0])
        
        self.s2.get_statistics()
        self.P2_history.append(self.s2.P)
        self.v2_avg_history.append(self.s2.v_avg)
        self.E2_total_history.append(self.s2.E_total)
        self.E2_thermal_history.append(self.s2.E_thermal)
        self.T2_direction_history.append(self.s2.T_direction)
        self.T2_history.append(self.s2.T)
        self.v2_1_history.append(self.s2.v[0])
        
        
        
        
        
        
    def generate_dW_g_ij_inter(self, dt=0.1, N_group=1):
        """ 
        Generate Gaussian random variable Delta W^{ij},
        shape = (N_group, N1/N_group, N2/N_group, 3).
        """
        
        N_particle1 = int(self.N1/N_group)
        N_particle2 = int(self.N2/N_group)
            
        return np.random.normal(0., np.sqrt(dt), (N_group, N_particle1, N_particle2, 3))
    
    
    
    
    def omega_hat_g_ij_inter(self, v1g, v2g, w, L, dWg, N_group):
        """
        Generate anti-symmetric matrix omega hat. 
        shape = (g, N1/N_group, N2/N_group, 3, 3)
        """
        
        # calculate u = vi - vj
        u = v1g[:,:, np.newaxis, :] - v2g[:,np.newaxis, :, :]
        u_abs = np.linalg.norm(u, axis=3)
        
        # create the vector omega
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
        Matrix M1, shape = (N_group, N1/N_group, N1/N_group, 3, 3)
        Matrix M2, shape = (N_group, N1/N_group, N2/N_group, 3, 3)
        """
        
        omega_hat_sum = np.sum(omega_hat, axis=2)
        diag_index = np.arange(int(self.N1/N_group), dtype=int)
        M1 = np.zeros((N_group, int(self.N1/N_group), int(self.N1/N_group), 3, 3))
        M1[:,diag_index,diag_index,:,:] = - omega_hat_sum[:,diag_index,:,:] / self.m1
        
        M2 = omega_hat / self.m1
        
        return M1, M2
    
    
    
    
    def N_g_ij_inter(self, omega_hat, N_group):
        """
        Matrix N1, shape = (N_group, N2/N_group, N1/N_group, 3, 3)
        Matrix N2_T, shape = (N_group, N2/N_group, N1/N_group, 3, 3)
        """
        
        omega_hat_sum = np.sum(omega_hat, axis=1)
        diag_index = np.arange(int(self.N2/N_group), dtype=int)
        N1 = np.zeros((N_group, int(self.N2/N_group), int(self.N2/N_group), 3, 3))
        N1[:,diag_index,diag_index,:,:] = - omega_hat_sum[:,diag_index,:,:] / self.m2
        
        N2_T = omega_hat.transpose((0,2,1,3,4)) / self.m2
        
        return N1, N2_T
    
    
    
    def Q_g_inter(self, omega_hat, N_group):
        """
        Matrix Q 
        shape = (N_group, 3*(N1+N2)/N_group, 3*(N1+N2)/N_group)
        """
        
        M1, M2 = self.M_g_ij_inter(omega_hat, N_group)
        N1, N2_T = self.N_g_ij_inter(omega_hat, N_group)
        
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
        """ 
        Return the one step calculation in modified midpoint method.
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
        
        # calculate the Cayley matrix
        Cayley = np.einsum('pij,pjk->pik', np.linalg.inv(I + Q), I - Q)
        
        # calculate new vg_vec
        vg_vec_new = np.einsum('pij,pj->pi', Cayley, vg_vec)
        
        v1_new = vg_vec_new[:,:int(3*self.N1/N_group)].reshape((int(self.N1), 3))
        v2_new = vg_vec_new[:,int(3*self.N1/N_group):].reshape((int(self.N2), 3))
        
        # return v1_new, v2_new
        return v1_new, v2_new
    
    
    
    
    def generate_dW_g_ij_intra(self, dt=0.1, N_group1=1, N_group2=1):
        """ 
        Generate anti-symmetric Gaussian random variable Delta W^{ij},
        shape = (N_group, N, N, 3).
        """
    
        N_particle = int(self.N1/N_group1)
        dW = np.random.normal(0., np.sqrt(dt), (N_group1, N_particle, N_particle, 3))
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
        Generate the matrix M with indices [g,i,j], shape = (N_group, 3*N/N_group,  3*N/N_group)
        """
        
        # calculate u = vi - vj
        u = vg[:,:, np.newaxis, :] - vg[:,np.newaxis, :, :]
        u_abs = np.linalg.norm(u, axis=3)
        
        # remove zero diagnal terms
        i_j_indices = np.arange(int(N/N_group))
        i_indices = np.tile(i_j_indices, N_group)
        j_indices = np.tile(i_j_indices, N_group)
        u_abs[:,i_indices,j_indices] = 1.
        
        # create the vector omega
        omega = ((w*L)**0.5 / (2.*m)) * (u_abs**(0.5*self.gamma-1.))[:,:,:,np.newaxis] * np.cross(u, dWg)
        
        # create the matrix omega_hat
        omega_hat = np.zeros((N_group, int(N/N_group), int(N/N_group), 3, 3))
        omega_hat[:,:,:,0,1] = - omega[:,:,:,2]
        omega_hat[:,:,:,0,2] = omega[:,:,:,1]
        omega_hat[:,:,:,1,0] = omega[:,:,:,2]
        omega_hat[:,:,:,1,2] = - omega[:,:,:,0]
        omega_hat[:,:,:,2,0] = - omega[:,:,:,1]
        omega_hat[:,:,:,2,1] = omega[:,:,:,0]
        
        # generate matrix M
        M = copy.deepcopy(omega_hat)
        omega_hat_sum = np.sum(omega_hat, axis=2)
        diag_index = np.arange(int(N/N_group), dtype=int)
        M[:,diag_index,diag_index,:,:] = - omega_hat_sum[:,diag_index,:,:]
        
        # return the big matrix
        return (M.transpose((0,1,3,2,4))).reshape((N_group, int(3*N/N_group), int(3*N/N_group)))
    
    
    
    
    def intra_one_step_explicit(self, v, m, N, w, L, dWg, N_group):
        """ 
        Return the one step calculation in modified midpoint method.
        """
        
        # group up velocities
        vg = v.reshape((N_group, int(N/N_group), 3))
        vg_vec = vg.reshape((N_group, int(3*N/N_group)))
            
        # generate big matrix M
        M = self.M_g_ij_intra(vg, m, N, w, L, dWg, N_group)
        
        # generate big identity matrix
        I = np.zeros((N_group, int(3*N/N_group), int(3*N/N_group)))
        I[np.arange(N_group), :, :] = np.identity(int(3*N/N_group))
        
        # calculate the Cayley matrix
        Cayley = np.einsum('pij,pjk->pik', np.linalg.inv(I + M), I - M)
        
        # calculate new vg_vec
        return (np.einsum('pij,pj->pi', Cayley, vg_vec)).reshape((N, 3))
    
    
    
    
    # final solver
    def explicit_solve(self, Nt=10, N_groups=[-1,-1,-1], N_record=1, print_status=False,
                       special_record = [1], dWg_given=None,
                       intra_collision = True, inter_collision = True):
        """
        Using modified midpoint method to calculate the time evolution explicitly.
        N_groups = [N_group_inter, N_group_species1, N_group_species2]
        """
        
        # group -1 means max group
        if N_groups[0] == -1: N_groups[0] = min(self.N1, self.N2)
        if N_groups[1] == -1: N_groups[1] = int(self.N1/2)
        if N_groups[2] == -1: N_groups[2] = int(self.N2/2)
        
        self.N_groups = N_groups
        self.N_record = N_record
        
        w_inter = self.s1.n / ( self.s1.N / N_groups[0] )
        w1 = self.s1.n / ( self.s1.N / N_groups[1] - 1 )
        w2 = self.s2.n / ( self.s2.N / N_groups[2] - 1 )
        
        # main loop
        for i in range(Nt):
            
            # shuffle velocities if N_group > 1
            if np.all(np.array(N_groups)>1): 
                np.random.shuffle(self.s1.v)
                np.random.shuffle(self.s2.v)
                
            # calculate one step advance for intra-species collision
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
            
            # calculate one step advance for inter-species collision
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
            
            # recording
            if i%N_record == 0: 
                self.record_physics()
                if print_status: print('Status = {} over {}.'.format(i, Nt), end='\r')
            
            # record specific distribution function
            if i in special_record:
                self.specific_v1[i] = self.s1.v 
                self.specific_v2[i] = self.s2.v

    
    
    
    
    
    
    

        
def get_ensemble_explicit(dt, 
                          total_t, 
                          s1:ParticleSpecies, 
                          s2:ParticleSpecies, 
                          save_dir, 
                          sub_id,
                          coulomb_log=1., 
                          N_groups=[-1,-1,-1], 
                          N_record=1, 
                          special_record = [1],
                          print_status=True,
                          intra_collision=True, 
                          inter_collision=True):
    """ 
    Calculate an ensemble of sample pathes using modified midpoint method.
    """
    
    np.random.seed(sub_id)
    
    # create folder if not exist
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except: 
            pass
    
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
        
