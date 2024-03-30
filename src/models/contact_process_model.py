import numpy as np
from scikit_tt.tensor_train import TT


def create_mpo(gamma: float, omega: float, L: int) -> TT:
    """
    Construct MPO for the Lindbladian of the contact process
    """

    # define operators
    identity = np.eye(2)
    sigmax = 0.5 * np.array([[0, 1], [1, 0]])
    annihilation_op = np.array([[0, 1], [0, 0]])
    number_op = np.array([[0, 0], [0, 1]])

    # core components
    S = gamma * np.kron(annihilation_op,
                        annihilation_op) - 0.5 * (np.kron(number_op, identity) + np.kron(identity, number_op))
    L_1 = np.kron(sigmax, identity)
    L_2 = np.kron(identity, sigmax)
    L_3 = np.kron(number_op, identity)
    L_4 = np.kron(identity, number_op)
    Id = np.eye(4)
    M_1 = -1j * omega * np.kron(number_op, identity)
    M_2 = -1j * omega * np.kron(identity, number_op)
    M_3 = -1j * omega * np.kron(sigmax, identity)
    M_4 = -1j * omega * np.kron(identity, sigmax)

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

    return TT(op_cores)
