import logging
import numpy as np
import time
from copy import deepcopy
from datetime import datetime

import scikit_tt.tensor_train as tt
from scikit_tt.solvers.evp import als

from src.models.contact_process_model import (construct_lindblad, construct_num_op)
from src.utilities.utils import (canonicalize_mps, compute_correlation_1, compute_purity, compute_site_expVal_1)

logger = logging.getLogger(__name__)
log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(filename="/home/psireal42/study/quantum-contact-process-1D/playground/logging/" + log_filename,
                    level=logging.INFO)

# path for results
PATH = "/home/psireal42/study/quantum-contact-process-1D/results/"

# system parameters
L = 10
GAMMA = 1
# OMEGAS = np.linspace(0, 10, 10)
OMEGAS = np.array([3.0])

# TN algorithm parameters
# bond_dims = np.array([8, 16, 20])
bond_dims = np.array([16])
conv_eps = 1e-6

### Stationary simulation
logger.info("Stationary simulation")
spectral_gaps = np.zeros(len(OMEGAS))
n_s = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))
particle_nums_left = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))
particle_nums_right = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))
evp_residual = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))
eval_0 = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))
eval_1 = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))
purities = np.zeros(len(OMEGAS))
correlations = np.zeros((len(OMEGAS), L - 1))

for i, OMEGA in enumerate(OMEGAS):
    for j, bond_dim in enumerate(bond_dims):
        logger.info(f"Run ALS for {L=}, {OMEGA=} and {bond_dim=}")
        lindblad = construct_lindblad(gamma=GAMMA, omega=OMEGA, L=L)
        lindblad_hermitian = lindblad.transpose(conjugate=True) @ lindblad

        mps = tt.ones(row_dims=L * [4], col_dims=L * [1], ranks=bond_dim)
        mps = mps.ortho()
        mps = (1 / mps.norm()**2) * mps
        time1 = time.time()
        eigenvalues, eigentensors, _ = als(lindblad_hermitian, mps, number_ev=2, repeats=10, conv_eps=conv_eps, sigma=0)
        evp_residual[j, i] = (lindblad_hermitian @ eigentensors[0] - eigenvalues[0] * eigentensors[0]).norm()**2
        eval_0[j, i] = eigenvalues[0]
        eval_1[j, i] = eigenvalues[1]

        logger.info(f"Residual of eigensolver: {evp_residual[j, i]}")
        time2 = time.time()
        logger.info(f"Elapsed time: {time2 - time1} seconds")
        logger.info(f"Ground state energy per site E = {eigenvalues/L}")
        logger.info(f"Norm of ground state: {eigentensors[0].norm()**2}")

        logger.info("Reshape MPS of ground state")
        gs_mps = eigentensors[0]
        gs_mps = canonicalize_mps(gs_mps)

        logger.info("Compute expectation value")
        # compute Hermitian part of mps
        hermit_mps = deepcopy(gs_mps)
        gs_mps_dag = gs_mps.transpose(conjugate=True)
        for k in range(L):
            hermit_mps.cores[k] = (gs_mps.cores[k] + gs_mps_dag.cores[k]) / 2

        logger.info("Compute particle numbers")
        particle_nums = compute_site_expVal_1(hermit_mps, construct_num_op(L))
        n_s[j, i] = np.mean(particle_nums)
        logger.info(f"Mean particle number = {n_s[j, i]}")

        # logger.info("Compute particle numbers (right leg)")
        # num_op_r = [None] * L
        # for k in range(L):
        #     number_op_r = np.kron(np.eye(2), np.array([[0, 0], [0, 1]]))
        #     num_op_r[k] = np.zeros([2, 2, 2, 2], dtype=complex)
        #     num_op_r[k] = number_op_r.reshape(2, 2, 2, 2)
        # num_op_r = tt.TT(num_op_r)
        # particle_nums_right[j, i] = np.mean(compute_site_expVal(hermit_mps, num_op_r))

        # logger.info("Compute particle numbers (left leg)")
        # num_op_l = [None] * L
        # for k in range(L):
        #     number_op_l = np.kron(np.array([[0, 0], [0, 1]]), np.eye(2))
        #     num_op_l[k] = np.zeros([2, 2, 2, 2], dtype=complex)
        #     num_op_l[k] = number_op_l.reshape(2, 2, 2, 2)
        # num_op_l = tt.TT(num_op_l)
        # particle_nums_left[j, i] = np.mean(compute_site_expVal(hermit_mps, num_op_l))

        if bond_dim == bond_dims[-1]:
            logger.info(f"{bond_dim=}")
            logger.info("Compute spectral gap of L†L for largest bond dimension")
            spectral_gaps[i] = abs(eigenvalues[1] - eigenvalues[0])

            logger.info("Compute purity of state for largest bond dimension")
            purities[i] = compute_purity(gs_mps)
            logger.info(f"Purity = {purities[-1]}")

            logger.info("Compute half-chain density correlation for largest bond dimension")
            an_op = construct_num_op(1)
            for k in range(L - 1):
                correlations[i, k] = abs(compute_correlation_1(gs_mps, an_op, r0=0, r1=k + 1))

time3 = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
# save result arrays
# np.savetxt(PATH + f"eval_0_L_{L}_{time3}.txt", eval_0, delimiter=',')
# np.savetxt(PATH + f"eval_1_L_{L}_{time3}.txt", eval_1, delimiter=',')
# np.savetxt(PATH + f"evp_residual_L_{L}_{time3}.txt", evp_residual, delimiter=',')
# np.savetxt(PATH + f"particle_nums_left_L_{L}_{time3}.txt", particle_nums_left, delimiter=',')
# np.savetxt(PATH + f"particle_nums_right_L_{L}_{time3}.txt", particle_nums_right, delimiter=',')
# np.savetxt(PATH + f"spectral_gaps_L_{L}_{time3}.txt", spectral_gaps, delimiter=',')
# np.savetxt(PATH + f"n_s_L_{L}_{time3}.txt", n_s, delimiter=',')
# np.savetxt(PATH + f"purities_L_{L}_{time3}.txt", purities, delimiter=',')
# np.savetxt(PATH + f"correlations_L_{L}_{time3}.txt", correlations, delimiter=',')

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
