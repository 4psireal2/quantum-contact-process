import numpy as np
from scikit_tt.tensor_train import (TT)
import scikit_tt.tensor_train as tt
from scikit_tt.solvers.evp import als

# parameters
L = 25
GAMMA = 1
OMEGA = 6
step_size = 1  # How to choose this?
number_of_steps = 10
max_rank = 200  # Bond dimension?

# arrays
identity = np.eye(2)
sigmax = 0.5 * np.array([[0, 1], [1, 0]])
annihilation_op = np.array([[0, 1], [0, 0]])
number_op = np.array([[0, 0], [0, 1]])

# core components
S = GAMMA * np.kron(annihilation_op,
                    annihilation_op) - 0.5 * (np.kron(number_op, identity) + np.kron(identity, number_op))
L_1 = np.kron(sigmax, identity)
L_2 = np.kron(identity, sigmax)
L_3 = np.kron(number_op, identity)
L_4 = np.kron(identity, number_op)
Id = np.eye(4)
M_1 = -1j * OMEGA * np.kron(number_op, identity)
M_2 = -1j * OMEGA * np.kron(identity, number_op)
M_3 = -1j * OMEGA * np.kron(sigmax, identity)
M_4 = -1j * OMEGA * np.kron(identity, sigmax)

# construct core
op_cores = [None] * L
op_cores[0] = np.zeros([1, 4, 4, 6], dtype=complex)
op_cores[0][0, :, :, 0] = S
op_cores[0][0, :, :, 1] = L_1
op_cores[0][0, :, :, 2] = L_2
op_cores[0][0, :, :, 3] = L_3
op_cores[0][0, :, :, 4] = L_4
op_cores[0][0, :, :, 5] = Id
for i in range(1, L - 1):
    op_cores[i] = np.zeros([6, 4, 4, 6], dtype=complex)
    op_cores[i][0, :, :, 0] = Id
    op_cores[i][1, :, :, 0] = M_1
    op_cores[i][2, :, :, 0] = M_2
    op_cores[i][3, :, :, 0] = M_3
    op_cores[i][4, :, :, 0] = M_4
    op_cores[i][5, :, :, 0] = S
    op_cores[i][5, :, :, 1] = L_1
    op_cores[i][5, :, :, 2] = L_2
    op_cores[i][5, :, :, 3] = L_3
    op_cores[i][5, :, :, 4] = L_4
    op_cores[i][5, :, :, 5] = Id
op_cores[-1] = np.zeros([6, 4, 4, 1], dtype=complex)
op_cores[-1][0, :, :, 0] = Id
op_cores[-1][1, :, :, 0] = M_1
op_cores[-1][2, :, :, 0] = M_2
op_cores[-1][3, :, :, 0] = M_3
op_cores[-1][4, :, :, 0] = M_4
op_cores[-1][5, :, :, 0] = S
op = TT(op_cores)

# compute ground state
psi = tt.uniform(L * [4], 6)
# eigenvalues, eigentensors, _ = als(op, psi, number_ev = 1, repeats = 10, sigma=0)
eigenvalues, eigentensors, _ = als(op.transpose(conjugate=True) @ op, psi, number_ev=1, repeats=10, sigma=0)

print(f"{eigenvalues=}")

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
