import numpy as np
import time
from datetime import datetime

import scikit_tt.tensor_train as tt
from scikit_tt.solvers.evp import als

from src.models.contact_process_model import (construct_lindblad, construct_num_op)
from src.utilities.utils import (canonicalize_mps, compute_correlation_vMPO, compute_purity, compute_site_expVal_vMPO,
                                 compute_expVal, compute_entanglement_spectrum, construct_basis_mps, compute_overlap)

# path for results
PATH = "/home/psireal42/study/quantum-contact-process-1D/results/"

# system parameters
L = 10
# OMEGAS = np.linspace(0, 10, 10)
OMEGAS = np.array([0.0])

# TN algorithm parameters
# bond_dims = np.array([8, 16, 20])
bond_dims = np.array([8])
conv_eps = 1e-6

### Stationary simulation
print("Stationary simulation")
d = 2
eval_0 = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))
eval_1 = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))
evp_residual = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))
spectral_gaps = np.zeros(len(OMEGAS))
n_s = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))
ent_ent_spectrum = np.zeros((OMEGAS.shape[0], bond_dims[-1] * d**2))  # d² * bond_dim
purities = np.zeros(len(OMEGAS))
correlations = np.zeros((len(OMEGAS), L - 1))

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

        evp_residual[j, i] = (lindblad_hermitian @ eigentensors[0] - eigenvalues[0] * eigentensors[0]).norm()**2
        eval_0[j, i] = eigenvalues[0]
        eval_1[j, i] = eigenvalues[1]
        print(f"Eigensolver error: {evp_residual[j, i]}")

        print(f"Ground state energy per site E: {eigenvalues/L}")
        print(f"Norm of ground state: {eigentensors[0].norm()**2}")

        gs_mps = canonicalize_mps(eigentensors[0])
        gs_mps_dag = gs_mps.transpose(conjugate=True)

        # compute non-Hermitian part of mps
        non_hermit_mps = (1 / 2) * (gs_mps - gs_mps_dag)
        print(f"The norm of the non-Hermitian part: {non_hermit_mps.norm()**2}")

        print(f"expVal_0, eigenvalue_0: {compute_expVal(gs_mps, lindblad_hermitian)}, {eigenvalues[0]}")

        # compute Hermitian part of mps
        hermit_mps = (1 / 2) * (gs_mps + gs_mps_dag)

        # compute observables
        print("Compute particle numbers")
        particle_nums = compute_site_expVal_vMPO(hermit_mps, construct_num_op(L))
        print(f"Particle number/site: {particle_nums}")
        n_s[j, i] = np.mean(particle_nums)
        print(f"Mean Particle number: {n_s[j, i]}")

        if bond_dim == bond_dims[-1]:
            print(f"{bond_dim=}")
            print("Compute spectral gap of L†L for largest bond dimension")
            spectral_gaps[i] = abs(eigenvalues[1] - eigenvalues[0])

            print("Compute purity of state for largest bond dimension")
            purities[i] = compute_purity(gs_mps)
            print(f"Purity: {purities[-1]}")

            print("Compute density-density correlation for largest bond dimension")
            an_op = construct_num_op(1)
            for k in range(L - 1):
                correlations[i, k] = abs(compute_correlation_vMPO(gs_mps, an_op, r0=0,
                                                                  r1=k + 1))  #NOTE: absolute value or not??

            print("Compute half-chain entanglement entropy spectrum for largest bond dimension")
            ent_ent_spectrum[i, :] = compute_entanglement_spectrum(gs_mps)

            print("Compute overlap with dark state")
            basis_0 = np.array([1, 0])
            dark_state = construct_basis_mps(L, basis=[np.kron(basis_0, basis_0)] * L)
            print(f"{compute_overlap(dark_state, gs_mps)=}")  #TODO: negative?

time3 = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())

# save result arrays
# np.savetxt(PATH + f"eval_0_L_{L}_{time3}.txt", eval_0, delimiter=',')
# np.savetxt(PATH + f"eval_1_L_{L}_{time3}.txt", eval_1, delimiter=',')
# np.savetxt(PATH + f"evp_residual_L_{L}_{time3}.txt", evp_residual, delimiter=',')
# np.savetxt(PATH + f"spectral_gaps_L_{L}_{time3}.txt", spectral_gaps, delimiter=',')
# np.savetxt(PATH + f"n_s_L_{L}_{time3}.txt", n_s, delimiter=',')
# np.savetxt(PATH + f"ent_ent_spectrum_L_{L}_{time3}.txt", ent_ent_spectrum, delimiter=',')
# np.savetxt(PATH + f"purities_L_{L}_{time3}.txt", purities, delimiter=',')
# np.savetxt(PATH + f"correlations_L_{L}_{time3}.txt", correlations, delimiter=',')
