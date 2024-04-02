import logging
import numpy as np
import time
from copy import deepcopy

import scikit_tt.tensor_train as tt
from scikit_tt.solvers.evp import als

from src.models.contact_process_model import (compute_site_expVal, construct_lindblad, construct_num_op)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# path for results
PATH = "/home/psireal42/study/quantum-contact-process-1D/results"

# system parameters
L = 10
GAMMA = 1
OMEGAS = np.linspace(0, 10, 20)

# TN algorithm parameters
bond_dims = np.array([8, 16, 20])
conv_eps = 1e-6

### Stationary simulation
spectral_gaps = []
n_stag_s = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))
n_s = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))

for i, OMEGA in enumerate(OMEGAS):
    for j, bond_dim in enumerate(bond_dims):
        logger.info(f"Run ALS for {L=}, {OMEGA=} and {bond_dim=}")
        lindblad = construct_lindblad(gamma=GAMMA, omega=OMEGA, L=L)
        lindblad_hermitian = lindblad.transpose(conjugate=True) @ lindblad

        mps = tt.ones(row_dims=L * [4], col_dims=L * [1], ranks=bond_dim)
        mps = mps.ortho()
        mps = (1 / mps.norm()) * mps
        time1 = time.time()
        eigenvalues, eigentensors, _ = als(lindblad_hermitian, mps, number_ev=2, repeats=10, conv_eps=conv_eps, sigma=0)
        time2 = time.time()
        logger.info(f"Elapsed time: {time2 - time1} seconds")
        logger.info(f"Ground state energy per site E = {eigenvalues/L}")
        logger.info(f"Norm of ground state: {eigentensors[0].norm()}")

        if bond_dim == bond_dims[-1]:
            logger.info("Compute spectral gap of L†L for largest bond dimension")
            spectral_gaps.append(abs(eigenvalues[1] - eigenvalues[0]))

        logger.info("Reshape MPS of ground state")
        gs_mps = eigentensors[0]

        first, _, _, last = gs_mps.cores[0].shape
        gs_mps.cores[0] = gs_mps.cores[0].reshape(first, 2, 2, last)
        for k in range(1, L - 1):
            first, _, _, last = gs_mps.cores[k].shape
            gs_mps.cores[k] = gs_mps.cores[k].reshape(first, 2, 2, last)
        first, _, _, last = gs_mps.cores[-1].shape
        gs_mps.cores[-1] = gs_mps.cores[-1].reshape(first, 2, 2, last)

        logger.info("Compute expectation value")

        # compute Hermitian part of mps
        hermit_mps = deepcopy(gs_mps)
        gs_mps_dag = gs_mps.transpose(conjugate=True)
        for k in range(L):
            hermit_mps.cores[k] = (gs_mps.cores[k] + gs_mps_dag.cores[k]) / 2

        logger.info("Compute particle numbers")
        particle_nums = compute_site_expVal(hermit_mps, construct_num_op(L))
        n_s[j, i] = np.mean(particle_nums)
        logger.info(f"Mean particle number = {n_s[j, i]}")

        # logger.info("Compute staggered particle numbers")  #XXX: Does it even make sense?
        # signs = np.array([-1 if i % 2 == 0 else 1 for i in range(L)])
        # stag_particle_nums = np.sqrt(np.mean(signs * particle_nums**2))
        # n_stag_s[j, i] = stag_particle_nums
        # logger.info(f"Mean staggered particle number = {n_stag_s[j, i]}")

        logger.info("Compute purity of state")
        logger.info("Compute density-density correlation")

# # plot spectral gap
# plt.figure()
# plt.plot(OMEGAS, spectral_gaps, 'o', label=f"{L=}")
# plt.legend()
# plt.tight_layout()
# plt.savefig(PATH + "/spectral_gaps.png")

# # plot particle numbers
# plt.figure()
# for i, bond_dim in enumerate(bond_dims):
#     plt.plot(OMEGAS, n_s[i, :], label=f"$\Chi=${bond_dim}")
# plt.xlabel(r"$\Omega$")
# plt.ylabel(r"$n_s$")
# plt.legend()
# plt.tight_layout()
# plt.savefig(PATH + "/stationary_density.png")

# # plot staggered particle numbers
# plt.figure()
# for i, bond_dim in enumerate(bond_dims):
#     plt.plot(OMEGAS, n_stag_s[i, :], label=f"$\Chi=${bond_dim}")
# plt.xlabel(r"$\Omega$")
# plt.ylabel(r"$n_{staggered\_s}$")
# plt.legend()
# plt.tight_layout()
# plt.savefig(PATH + "/staggered_stationary_density.png")

### Dynamical simulation

# system parameters
# L=51
# OMEGA = 6.0

# # dynamics parameter
# step_size = 1
# number_of_steps = 10
# max_rank = 100

# lindblad = construct_lindblad(gamma=GAMMA, omega=OMEGA, L=L)
# lindblad_hermitian = lindblad.transpose(conjugate=True) @ lindblad
# num_op = construct_num_op(L)
# mps = tt.unit(L * [4], inds=25 * [0] + [3] + 25 * [0])
# mps_t = hod(lindblad_hermitian, mps, step_size=step_size, number_of_steps=number_of_steps, normalize=2, max_rank=max_rank)

# t = np.linspace(0,step_size*number_of_steps,number_of_steps)
# particle_num_t = np.zeros((len(t), L))

# for j in range(len(t)):
#     first, _, _, last = mps_t[j].cores[0].shape
#     mps_t[j].cores[0] = mps_t[j].cores[0].reshape(first, 2, 2, last)
#     for i in range(1, L - 1):
#         first, _, _, last = mps_t[j].cores[i].shape
#         mps_t[j].cores[i] = mps_t[j].cores[i].reshape(first, 2, 2, last)
#     first, _, _, last = mps_t[j].cores[-1].shape
#     mps_t[j].cores[-1] = mps_t[j].cores[-1].reshape(first, 2, 2, last)

#     particle_num_t[j,:] = compute_site_expVal(mps_t[j], num_op)

# compute active-site density

# plot result
# for i in range(len(solution)):
#     probs[i, :] = 1 / np.linalg.norm(probs[i, :], ord=1) * probs[i, :]

# plt.figure()
# plt.imshow(probs)
# plt.colorbar()
# plt.tight_layout()
# plt.savefig("density_plot_L_51_Chi_200_critical.png")
