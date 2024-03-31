import numpy as np

from src.models.contact_process_model import (compute_site_expVal, construct_num_op, construct_basis_mps)

# path for results
PATH = "/home/psireal42/study/quantum-contact-process-1D/results"

# system parameters
L = 10
GAMMA = 1
OMEGA = 6

# tensor network parameters
bond_dim = 8

# dynamics parameter
step_size = 1  # How to choose this?
number_of_steps = 10
max_rank = 200  # Bond dimension?

# compute spectral gap of L†L
# Ls = list(range(10,35,5))
# bond_dims = [8, 16, 32, 40, 50]
# for bond_dim in bond_dims:
#     # print(f"{L=}")
#     print(f"{bond_dim=}")
#     lindblad = construct_lindblad(gamma=GAMMA, omega=OMEGA, L=10)
#     lindblad_hermitian = lindblad.transpose(conjugate=True) @ lindblad

#     psi = tt.uniform(10 * [4], ranks=bond_dim)
#     time1 = time.time()
#     eigenvalues, eigentensors, _ = als(lindblad_hermitian, psi, number_ev=1, repeats=10, conv_eps=1e-6, sigma=0)
#     time2 = time.time()
#     print(f"Elapsed time: {time2 - time1} seconds")
#     print(f"Ground state energy per site E = {eigenvalues/10}")
# assert abs(eigenvalues[0]) < 1e-5

# plt.figure()
# plt.plot(eigenvalues, 'o', label=f"{L=}")
# plt.legend()
# plt.tight_layout()
# plt.savefig(PATH + f"/eigenvalues_spectrum_{L=}.png")

# compute particle numbers
basis_0 = np.array([1, 0])
basis_1 = np.array([0, 1])
psi = construct_basis_mps(L, basis=[np.kron(basis_1, basis_1)] * L)
particle_nums = compute_site_expVal(psi, construct_num_op(L))

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
