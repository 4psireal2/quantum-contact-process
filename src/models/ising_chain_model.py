import numpy as np
from scikit_tt.tensor_train import TT


def create_mpo(gamma: float, V: float, omega: float, delta: float, L: int) -> TT:
    """
    Construct MPO for the Lindbladian of the dissipative Ising chain
    """

    # define operators
    identity = np.eye(2)
    sigmax = 0.5 * np.array([[0, 1], [1, 0]])
    sigmaz = 0.5 * np.array([[1, 0], [0, -1]])
    annihilation_op = np.array([[0, 1], [0, 0]])
    creation_op = np.array([[0, 0], [1, 0]])
    an_op = annihilation_op * creation_op
    creation_op * annihilation_op

    # core components
    S = -1j * (omega / 2) * (np.kron(identity, sigmax) + np.kron(sigmax, identity)) + 1j * (
        V - delta) / 2 * (np.kron(identity, sigmaz) + np.kron(sigmaz, identity)) + gamma * np.kron(
            creation_op, creation_op) - (1 / 2) * gamma * (np.kron(identity, an_op) + np.kron(an_op, identity))
    L_1 = np.kron(sigmaz, identity)
    L_2 = np.kron(identity, sigmaz)
    Id = np.eye(4)
    M_1 = -1j * V / 4 * np.kron(sigmaz, identity)
    M_2 = -1j * V / 4 * np.kron(identity, sigmaz)

    # construct core
    op_cores = [None] * L
    op_cores[0] = np.zeros([1, 4, 4, 4], dtype=complex)
    op_cores[0][0, :, :, 0] = S + (V / 4) * (np.kron(identity, sigmaz) + np.kron(sigmaz, identity))
    op_cores[0][0, :, :, 1] = L_1
    op_cores[0][0, :, :, 2] = L_2
    op_cores[0][0, :, :, 3] = Id
    for i in range(1, L - 1):
        op_cores[i] = np.zeros([4, 4, 4, 4], dtype=complex)
        op_cores[i][0, :, :, 0] = Id
        op_cores[i][1, :, :, 0] = M_1
        op_cores[i][2, :, :, 0] = M_2
        op_cores[i][3, :, :, 0] = S
        op_cores[i][3, :, :, 1] = L_1
        op_cores[i][3, :, :, 2] = L_2
        op_cores[i][3, :, :, 3] = Id
    op_cores[-1] = np.zeros([4, 4, 4, 1], dtype=complex)
    op_cores[-1][0, :, :, 0] = Id
    op_cores[-1][1, :, :, 0] = M_1
    op_cores[-1][2, :, :, 0] = M_2
    op_cores[-1][3, :, :, 0] = S + (V / 4) * (np.kron(identity, sigmaz) + np.kron(sigmaz, identity))

    return TT(op_cores)
