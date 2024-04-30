"""
For 1D dissipative Ising chain. See Ref: 10.1103/PhysRevLett.114.220601
"""

import numpy as np
from scikit_tt.tensor_train import TT


def construct_lindblad(gamma: float, V: float, omega: float, delta: float, L: int) -> TT:
    """
    Construct MPO for the Lindbladian of the dissipative Ising chain
    """

    # define operators
    identity = np.eye(2)
    sigmax = 0.5 * np.array([[0, 1], [1, 0]])
    sigmax_L = np.kron(sigmax, identity)
    sigmax_R = np.kron(identity, sigmax)

    sigmaz = 0.5 * np.array([[1, 0], [0, -1]])
    sigmaz_L = np.kron(sigmaz, identity)
    sigmaz_R = np.kron(identity, sigmaz)

    annihilation_op = np.array([[0, 1], [0, 0]])
    creation_op = np.array([[0, 0], [1, 0]])
    an_op = creation_op * annihilation_op
    an_op_L = np.kron(an_op, identity)
    an_op_R = np.kron(identity, an_op)

    # core components
    S = -1j * (omega / 2) * (sigmax_L - sigmax_R) + 1j * (V - delta) / 2 * (sigmaz_L - sigmaz_R) + gamma * np.kron(
        annihilation_op, annihilation_op) - (1 / 2) * gamma * (an_op_L + an_op_R)
    L_1 = sigmaz_L
    L_2 = sigmaz_R
    Id = np.eye(4)
    M_1 = -1j * V / 4 * sigmaz_L
    M_2 = -1j * V / 4 * -sigmaz_R

    # construct core
    op_cores = [None] * L
    op_cores[0] = np.zeros([1, 4, 4, 4], dtype=complex)
    op_cores[0][0, :, :, 0] = S + -1j * (V / 4) * (sigmaz_L - sigmaz_R)
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
    op_cores[-1][3, :, :, 0] = S + -1j * (V / 4) * (sigmaz_L - sigmaz_R)

    return TT(op_cores)


def construct_lindblad_dag(gamma: float, V: float, omega: float, delta: float, L: int) -> TT:
    """
    Construct MPO for the Lindbladian of the dissipative Ising chain
    """

    # define operators
    identity = np.eye(2)
    sigmax = 0.5 * np.array([[0, 1], [1, 0]])
    sigmax_L = np.kron(sigmax, identity)
    sigmax_R = np.kron(identity, sigmax)

    sigmaz = 0.5 * np.array([[1, 0], [0, -1]])
    sigmaz_L = np.kron(sigmaz, identity)
    sigmaz_R = np.kron(identity, sigmaz)

    annihilation_op = np.array([[0, 1], [0, 0]])
    creation_op = np.array([[0, 0], [1, 0]])
    an_op = creation_op * annihilation_op
    an_op_L = np.kron(an_op, identity)
    an_op_R = np.kron(identity, an_op)

    # core components
    S = 1j * (omega / 2) * (sigmax_L - sigmax_R) - 1j * (V - delta) / 2 * (sigmaz_L - sigmaz_R) + gamma * np.kron(
        creation_op, creation_op) - (1 / 2) * gamma * (an_op_L + an_op_R)
    L_1 = sigmaz_L
    L_2 = sigmaz_R
    Id = np.eye(4)
    M_1 = 1j * V / 4 * sigmaz_L
    M_2 = -1j * V / 4 * sigmaz_R

    # construct core
    op_cores = [None] * L
    op_cores[0] = np.zeros([1, 4, 4, 4], dtype=complex)
    op_cores[0][0, :, :, 0] = S + 1j * (V / 4) * (sigmaz_L - sigmaz_R)
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
    op_cores[-1][3, :, :, 0] = S + 1j * (V / 4) * (sigmaz_L - sigmaz_R)

    return TT(op_cores)
