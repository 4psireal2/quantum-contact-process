import numpy as np
import matplotlib.pyplot as plt
import time

import scikit_tt.tensor_train as tt
from scikit_tt.solvers.evp import als

from src.models.contact_process_model import (compute_site_expVal, construct_lindblad, construct_num_op)

# path for results
PATH = "/home/psireal42/study/quantum-contact-process-1D/results"

# system parameters
L = 50
GAMMA = 1
OMEGA = 6

# tensor network parameters
bond_dim = 8

# dynamics parameter
step_size = 1  # How to choose this?
number_of_steps = 10
max_rank = 200  # Bond dimension?

OMEGAS = np.linspace(0, 10, 20)
bond_dims = np.array([10, 50, 100])

### Stationary state calculations
spectral_gaps = []
n_stag_s = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))
n_s = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))

for i, OMEGA in enumerate(OMEGAS):
    for j, bond_dim in enumerate(bond_dims):
        lindblad = construct_lindblad(gamma=GAMMA, omega=OMEGA, L=L)
        lindblad_hermitian = lindblad.transpose(conjugate=True) @ lindblad

        psi = tt.uniform(L * [4], ranks=bond_dim)
        time1 = time.time()
        eigenvalues, eigentensors, _ = als(lindblad_hermitian, psi, number_ev=2, repeats=10, conv_eps=1e-6, sigma=0)
        time2 = time.time()
        print(f"Elapsed time: {time2 - time1} seconds")
        print(f"Ground state energy per site E = {eigenvalues/L}")
        assert abs(eigenvalues[0]) < 1e-5

        # compute spectral gap of L†L for biggest bond dimension
        if bond_dim == bond_dims[-1]:
            spectral_gaps.append(abs(eigenvalues[1] - eigenvalues[0]))

        # compute staggered particle numbers
        particle_nums = compute_site_expVal(eigentensors[0], construct_num_op(L))
        n_s[j, i] = np.mean(particle_nums)

        # compute staggered particle numbers
        signs = np.array([-1 if i % 2 == 0 else 1 for i in range(L)])
        stag_particle_nums = np.sqrt(np.mean(signs * particle_nums**2))
        n_stag_s[j, i] = stag_particle_nums

# plot spectral gap
plt.figure()
plt.plot(OMEGAS, spectral_gaps, 'o', label=f"{L=}")
plt.legend()
plt.tight_layout()
plt.savefig(PATH + "/spectral_gaps.png")

# plot particle numbers
plt.figure()
for i, bond_dim in enumerate(bond_dims):
    plt.plot(OMEGAS, n_s[i, :], label=f"$\Chi=${bond_dim}")
plt.xlabel(r"$\Omega$")
plt.ylabel(r"$n_s$")
plt.legend()
plt.tight_layout()
plt.savefig(PATH + "/stationary_density.png")

# plot staggered particle numbers
plt.figure()
for i, bond_dim in enumerate(bond_dims):
    plt.plot(OMEGAS, n_stag_s[i, :], label=f"$\Chi=${bond_dim}")
plt.xlabel(r"$\Omega$")
plt.ylabel(r"$n_{staggered\_s}$")
plt.legend()
plt.tight_layout()
plt.savefig(PATH + "/staggered_stationary_density.png")

# basis_0 = np.array([1, 0])
# basis_1 = np.array([0, 1])
# psi = construct_basis_mps(L, basis=[np.kron(basis_1, basis_1)] * L)
# particle_nums = compute_site_expVal(psi, construct_num_op(L))

# compute exponential decay

# simulate quantum contact process
# psi = tt.unit(L * [4], inds=25 * [0] + [3] + 25 * [0])
# solution = hod(op, psi, step_size=step_size, number_of_steps=number_of_steps, normalize=2, max_rank=max_rank)

# compute active-site density
# probs = np.zeros([len(solution), L])
# for i in range(len(solution)):
#     for j in range(L):
#         sol = solution[i].copy()
#         for k in range(L):
#             if k != j:
#                 sol.cores[k] = (sol.cores[k][:, 0, :, :] + sol.cores[k][:, 3, :, :]).reshape(
#                     [sol.ranks[k], 1, 1, sol.ranks[k + 1]])
#             else:
#                 sol.cores[k] = sol.cores[k][:, 3, :, :].reshape([sol.ranks[k], 1, 1, sol.ranks[k + 1]])
#             sol.row_dims[k] = 1
#         probs[i, j] = np.abs(sol.matricize()[0])

# # plot result
# for i in range(len(solution)):
#     probs[i, :] = 1 / np.linalg.norm(probs[i, :], ord=1) * probs[i, :]

# plt.figure()
# plt.imshow(probs)
# plt.colorbar()
# plt.tight_layout()
# plt.savefig("density_plot_L_51_Chi_200_critical.png")
