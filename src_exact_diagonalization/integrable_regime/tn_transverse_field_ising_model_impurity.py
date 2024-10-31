import numpy as np
import quimb as qu
from quimb import *
from scipy.integrate import ode
import argparse
import os
from pathlib import Path
import time

# folder where to save data
folder = "./"

parser = argparse.ArgumentParser(description='Parameters Hamiltonian spin')

# Hamiltonian parameters
parser.add_argument('--N', type = int, default = 10, help = 'Number of sites in the system')
parser.add_argument('--h', type = float, default = 0.5, help = 'transversal magnetic field')
parser.add_argument('--gamma', type = float, default = 0, help = 'dissipation rate')
parser.add_argument('--Tness',type=float, default = 40., help = 'time before computing autocorrelator')
parser.add_argument('--T', type = float, default = 200, help = 'total time after reaching NESS')
parser.add_argument('--dt', type = float, default = 0.01, help = 'time step')

args = vars(parser.parse_args())

N = args['N']
J = 1.
h = args['h']
gamma = args['gamma']
Tness = args['Tness']
T = args['T']
dt = args['dt']

# ----------------------------------------


def dot_C(t, C, args):

    H = args['H']
    gamma = args['gamma']
    C = np.reshape(C,H.shape)

    Delta1 = np.zeros(H.shape)
    Delta1[0,0] = 1.

    # nb H = i A -> A = -i H 
    # We have dC = 4[A,C] + ...
        
    dC = -4j * (H @ C - C @ H) - 2 * gamma * (Delta1 @ C + C @ Delta1 ) + 4 * gamma * Delta1 @ C @ Delta1

    return np.ndarray.flatten(dC)


def dot_C_autocorrelator(t, C, args):

    H = args['H']
    gamma = args['gamma']
    C = np.reshape(C,H.shape)

    # Delta1 = np.zeros(H.shape,dtype=np.dtype(np.float))
    # Delta1[0,0] = 1.

    # equivalent to Delta1 - Id

    D1 = -np.eye(H.shape[0],dtype=complex)
    D1[0,0] = 0
    
    dC = ( -4j * H - 2 * gamma * D1 ) @ C
    return np.ndarray.flatten(dC)


# single particle Hamiltonian transverse field ising model
# in Majorana fermions basis

def H_TFIC(N,J,h):

    h_pot = np.eye(N) * (-h/2) 
    h_kin = [J/2 for _ in range(N-1)]

    H_block = h_pot + np.diag(h_kin,k=-1)

    H = np.zeros((2*N,2*N),dtype=complex)
    H[:N,N:] =  H_block
    H[N:,:N] = -np.transpose(H_block)

    return 1j*H

def H_ising_BdG(L,J,h,sparse=False):

    if type(h) is not list:
        print('converting in a list')
        h = np.full(shape=[L],fill_value=h)
    if type(J) is not list:
        J = np.full(shape=[L-1],fill_value=J)

    A = np.diag(h) + np.diag(-J/2,k=1) + np.diag(-J/2,k=-1)
    B = np.diag(-J/2 , k=1) + np.diag(J/2,k=-1)

    H =  np.zeros(shape=(2*L,2*L),dtype=float)
    H[:L,:L] =  A
    H[L:,L:] = -A
    H[:L,L:] =  B
    H[L:,:L] = -B

    return H



if dt > 0.1:
    nsave = 1
else:
    nsave = int(0.1/dt)

tness = np.linspace(dt,Tness,num=int(Tness/dt))
ts    = np.linspace(dt,T,num=int(T/dt))

##################################################
# INITIALIZING THE STATE

# single-particle Hamiltonian

H_dirac = H_ising_BdG(N,J,h)
H_maj   = H_TFIC(N,J,h)

# transformation from Dirac fermions to Majorana
Om = np.zeros((2*N,2*N),dtype=complex)
Om[:N,:N] = np.eye(N)
Om[N:,N:] = -1j*np.eye(N)
Om[:N,N:] = np.eye(N)
Om[N:,:N] =  1j*np.eye(N)
Omd = np.conj(np.transpose(Om)) 
print('Diagonalizating')
eigenvalues , U = eigh(H_dirac)
U = U[:,::-1]
Ud = np.conj(np.transpose(U))
print('Diagonalization done')
# covariance matrix ground state
# It is C_t0 = [[I-C,F^dag],[F,C]]
# where C_ij = <c_i^dag c_j> 
#       F_ij = <c_i^dag c_j^dag>
# In the normal modes we have the vacuum: which implies the correlation matrix
# C_t0 = [[I,0],[0,0]]
C_dirac = np.zeros((2*N,2*N))
C_dirac[:N,:N] = np.eye(N)

# Then, we rotate back to the JW fermions via the unitary which enables to find the normal modes.
# Otherwise, we can work with the normal modes, but it would be more tricky to compute the 
# quantities of interest as the magnetization, so we rotate the state back.
C_dirac = U @ C_dirac @ Ud
# we rotate in the Majorana basis
# C_d = <\Psi \Psi^d>
# C_maj = < \eta \eta^d> = Om <\Psi \Psi^d> Omd
# where \eta = Om \Psi
C_maj   = Om @ C_dirac @ Omd

# let us compare some observcables
mz_m = np.diag(1j*C_maj[:N,N:])
mz_d = 1 - 2 * np.diag(C_dirac[N:,N:])

################################################################
# Dynamics
print('Starting dynamics')

# Evolution up to t=40 to reach NESS near the impurity site

args = {}
args['H'] = H_maj # should be 4??? In the americans we have a factor 4 here...
args['gamma'] = gamma

t_maj = []
integrator_maj = ode(dot_C).set_integrator('zvode', method='bdf',rtol=1E-12)
integrator_maj.set_f_params(args)
integrator_maj.set_initial_value(np.ndarray.flatten(C_maj))

mt_ness = []
t_ness = []
start_time = time.time()

for idx, tau in enumerate(tness):
    print(tau)
    ct = integrator_maj.integrate(tau)      
    if idx % nsave == 0:
        ct = np.reshape(ct,(2*N,2*N))
        m = -np.imag(np.diag(ct[:N,N:]))

        mt_ness += [m]
        t_ness += [tau]


mt_ness = np.array(mt_ness)
t_ness = np.array(t_ness)

folder = f"{folder}TFIC_dephaing_N{N}_h{h:.3f}_gamma{gamma:.3f}_Tness{Tness:.1f}"

if not os.path.isdir(folder):
    Path(folder).mkdir()

np.save(f'{folder}/t_ness.npy',t_ness)
np.save(f'{folder}/mt_ness.npy',mt_ness)

################################################
# Autocorrelator once the NESS is reached

ct = np.array(ct)

integrator_autocorr = ode(dot_C_autocorrelator).set_integrator('zvode', method='bdf',rtol=1E-12)
integrator_autocorr.set_f_params(args)
integrator_autocorr.set_initial_value(np.ndarray.flatten(ct))
zt_z0 = []
t = []
for idx, tau in enumerate(ts):

    ct_tau = integrator_autocorr.integrate(tau)     

    if idx % nsave == 0:
        zt_z0 += [ct_tau[0]]
        t += [tau]


print("--- %s seconds ---" % (time.time() - start_time))

zt_z0 = np.array(zt_z0,dtype=complex)
np.save(f'{folder}/zt_z0.npy',zt_z0)
np.save(f'{folder}/t.npy',t)




