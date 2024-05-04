"""
Do static simulations for first excited state
"""

import logging
import numpy as np
import time
from datetime import datetime

import scikit_tt.tensor_train as tt
from scikit_tt.solvers.evp import als

from src.models.contact_process_model import (construct_lindblad, construct_num_op)
from src.utilities.utils import (canonicalize_mps, compute_correlation, compute_dens_dens_corr, compute_purity,
                                 compute_site_expVal, compute_expVal, compute_entanglement_spectrum,
                                 construct_basis_mps, compute_overlap)

logger = logging.getLogger(__name__)
log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(filename="/scratch/nguyed99/qcp-1d/logging/" + log_filename, level=logging.INFO)

# path for results
PATH = "/scratch/nguyed99/qcp-1d/results/"

# system parameters
L = 50
d = 2
OMEGAS = np.linspace(0.5, 10, 10)

# TN algorithm parameters
bond_dims = np.array([50, 70])
conv_eps = 1e-6

### Stationary simulation
print("Stationary simulation")
eval_0 = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))
eval_1 = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))
evp_residual = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))
spectral_gaps = np.zeros(len(OMEGAS))
n_s = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))
entanglement_spectrum = np.zeros((OMEGAS.shape[0], bond_dims[-1] * d**2))  # d² * bond_dim
purities = np.zeros(len(OMEGAS))
correlations = np.zeros((len(OMEGAS), L - 1))
dens_dens_corr = np.zeros((len(OMEGAS), L - 1))

for i, OMEGA in enumerate(OMEGAS):
    for j, bond_dim in enumerate(bond_dims):
        print(f"Run ALS for {L=}, {OMEGA=} and {bond_dim=}")
        lindblad = construct_lindblad(gamma=1.0, omega=OMEGA, L=L)
        lindblad_hermitian = lindblad.transpose(conjugate=True) @ lindblad

        mps = tt.ones(row_dims=L * [4], col_dims=L * [1], ranks=bond_dim)
        mps = mps.ortho()
        mps = (1 / mps.norm()**2) * mps
        time1 = time.time()
        eigenvalues, eigentensors, _ = als(lindblad_hermitian, mps, number_ev=2, repeats=10, conv_eps=conv_eps, sigma=0)
        time2 = time.time()
        print(f"Elapsed time: {time2 - time1} seconds")

        evp_residual[j, i] = (lindblad_hermitian @ eigentensors[1] - eigenvalues[1] * eigentensors[1]).norm()**2
        eval_0[j, i] = eigenvalues[0]
        eval_1[j, i] = eigenvalues[1]
        print(f"Eigensolver error for first excited state: {evp_residual[j, i]}")

        print(f"Ground state energy per site E: {eigenvalues[1]/L}")
        print(f"Norm of first excited state: {eigentensors[1].norm()**2}")

        mps = canonicalize_mps(eigentensors[1])
        mps_dag = mps.transpose(conjugate=True)

        # compute non-Hermitian part of mps
        non_hermit_mps = (1 / 2) * (mps - mps_dag)
        print(f"The norm of the non-Hermitian part: {non_hermit_mps.norm()**2}")

        print(f"expVal_1, eigenvalue_1: {compute_expVal(mps, lindblad_hermitian)}, {eigenvalues[1]}")

        # compute Hermitian part of mps
        hermit_mps = (1 / 2) * (mps + mps_dag)

        # compute observables
        print("Compute particle numbers")
        particle_nums = compute_site_expVal(hermit_mps, construct_num_op(L))
        print(f"Particle number/site: {particle_nums}")
        n_s[j, i] = np.mean(particle_nums)
        print(f"Mean Particle number: {n_s[j, i]}")

        if bond_dim == bond_dims[-1]:
            print(f"{bond_dim=}")
            print("Compute spectral gap of L†L for largest bond dimension")
            spectral_gaps[i] = abs(eigenvalues[1] - eigenvalues[0])

            print("Compute purity of state for largest bond dimension")
            purities[i] = compute_purity(mps)
            print(f"Purity: {purities[-1]}")

            print("Compute two-point correlation for largest bond dimension")
            an_op = construct_num_op(1)
            for k in range(L - 1):
                correlations[i, k] = compute_correlation(mps, an_op, r0=0, r1=k + 1)

            print("Compute density-density correlation for largest bond dimension")
            for k in range(L - 1):
                dens_dens_corr[i, k] = compute_dens_dens_corr(mps, an_op, r=k + 1)

            print("Compute half-chain entanglement spectrum for largest bond dimension")
            entanglement_spectrum[i, :] = compute_entanglement_spectrum(mps)

            basis_0 = np.array([1, 0])
            dark_state = construct_basis_mps(L, basis=[np.kron(basis_0, basis_0)] * L)
            print(f"Overlap with dark state: {compute_overlap(dark_state, mps)}")  #NOTE: negative?

time3 = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())

# save result arrays
np.savetxt(PATH + f"eval_0_exc_L_{L}_{time3}.txt", eval_0, delimiter=',')
np.savetxt(PATH + f"eval_1_exc_L_{L}_{time3}.txt", eval_1, delimiter=',')
np.savetxt(PATH + f"evp_residual_exc_L_{L}_{time3}.txt", evp_residual, delimiter=',')
np.savetxt(PATH + f"spectral_gaps_exc_L_{L}_{time3}.txt", spectral_gaps, delimiter=',')
np.savetxt(PATH + f"n_s_L_exc_{L}_{time3}.txt", n_s, delimiter=',')
np.savetxt(PATH + f"entanglement_spectrum_exc_L_{L}_{time3}.txt", entanglement_spectrum, delimiter=',')
np.savetxt(PATH + f"purities_exc_L_{L}_{time3}.txt", purities, delimiter=',')
np.savetxt(PATH + f"correlations_exc_L_{L}_{time3}.txt", correlations, delimiter=',')
np.savetxt(PATH + f"dens_dens_corr_exc_L_{L}_{time3}.txt", dens_dens_corr, delimiter=',')
