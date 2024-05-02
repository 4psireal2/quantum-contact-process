"""
Checking the total polarization of the steady state
"""

import numpy as np

import scikit_tt.tensor_train as tt
from scikit_tt.solvers.evp import als

import src.models.diss_ising_model as diss_ising_model
import src.utilities.utils as utils

L = 10
GAMMA = 1.0
OMEGA = 1.5
V = 5.0
# DELTAS = np.linspace(-4, 4, 9)
DELTA = 0.0

bond_dim = 16
conv_eps = 1e-6

print(f"{L=}")
print(f"{DELTA=}")
print(f"{bond_dim=}")

ising_lindblad = diss_ising_model.construct_lindblad(gamma=GAMMA, V=V, omega=OMEGA, delta=DELTA, L=L)
ising_lindblad_dag = diss_ising_model.construct_lindblad_dag(gamma=GAMMA, V=V, omega=OMEGA, delta=DELTA, L=L)
ising_lindblad_hermitian = ising_lindblad_dag @ ising_lindblad

mps = tt.ones(row_dims=L * [4], col_dims=L * [1], ranks=bond_dim)
mps = mps.ortho()
mps = (1 / mps.norm()**2) * mps
eigenvalue, eigentensor, _ = als(ising_lindblad_hermitian, mps, number_ev=1, repeats=10, conv_eps=conv_eps, sigma=0)

gs_mps = utils.canonicalize_mps(eigentensor)
gs_mps_dag = gs_mps.transpose(conjugate=True)

print(f"expVal_0, eigenvalue_0: {utils.compute_expVal(gs_mps, ising_lindblad_hermitian)}, {eigenvalue}")

# compute non-Hermitian part of mps
non_hermit_mps = (1 / 2) * (gs_mps - gs_mps_dag)
print(f"The norm of the non-Hermitian part: {non_hermit_mps.norm()**2}")

# compute Hermitian part of mps
hermit_mps = (1 / 2) * (gs_mps + gs_mps_dag)

print("Compute overlap with dark state")
basis_0 = np.array([1, 0])
dark_state = utils.construct_basis_mps(L, basis=[np.kron(basis_0, basis_0)] * L)
print(f"{utils.compute_overlap(dark_state, gs_mps)=}")

print("Compute local polarization")
sigmaz = 0.5 * np.array([[1, 0], [0, -1]])
sigmaz = np.kron(sigmaz, np.eye(2)) + np.kron(np.eye(2), sigmaz)
an_op = tt.TT(sigmaz)
an_op.cores[0] = an_op.cores[0].reshape(2, 2, 2, 2)
polarizations = utils.compute_site_expVal_vMPO(hermit_mps, an_op)
print(f"Local polarization of ground state: {polarizations}")
print(f"Local polarization of dark state: {utils.compute_site_expVal_vMPO(dark_state, an_op)}")

print("Compute two-site correlation")

correlations = diss_ising_model.commpute_half_chain_corr(hermit_mps)
print(f"Magnetization correlation of ground state: {diss_ising_model.commpute_half_chain_corr(hermit_mps)}")
print(f"Magnetization correlation of dark state: {diss_ising_model.commpute_half_chain_corr(dark_state)}")

# import matplotlib.pyplot as plt

# plt.figure()
# plt.plot(list(range(L//2+1, L)), correlations)
# plt.show()
