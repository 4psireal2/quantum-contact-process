"""
For 1D dissipative Ising chain. See Ref: 10.1103/PhysRevLett.114.220601
"""

import numpy as np
import scikit_tt.tensor_train as tt
from src.utilities.utils import compute_correlation


def construct_lindblad(gamma: float, V: float, omega: float, delta: float, L: int) -> tt.TT:
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
    an_op = annihilation_op @ creation_op

    an_op_L = np.kron(an_op, identity)
    an_op_R = np.kron(identity, an_op)

    # core components
    S = -1j * (omega / 2) * (sigmax_L - sigmax_R) + 1j * (V - delta) / 2 * (sigmaz_L - sigmaz_R) + gamma * np.kron(
        creation_op, creation_op) - (1 / 2) * gamma * (an_op_L + an_op_R)
    L_1 = sigmaz_R
    L_2 = sigmaz_L
    Id = np.eye(4)
    M_1 = 1j * V / 4 * sigmaz_R
    M_2 = -1j * V / 4 * sigmaz_L

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

    return tt.TT(op_cores)


def construct_lindblad_dag(gamma: float, V: float, omega: float, delta: float, L: int) -> tt.TT:
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
    an_op = annihilation_op @ creation_op

    an_op_L = np.kron(an_op, identity)
    an_op_R = np.kron(identity, an_op)

    # core components
    S = 1j * (omega / 2) * (sigmax_L - sigmax_R) - 1j * (V - delta) / 2 * (sigmaz_L - sigmaz_R) + gamma * np.kron(
        annihilation_op, annihilation_op) - (1 / 2) * gamma * (an_op_L + an_op_R)
    L_1 = sigmaz_R
    L_2 = sigmaz_L
    Id = np.eye(4)
    M_1 = -1j * V / 4 * sigmaz_R
    M_2 = 1j * V / 4 * sigmaz_L

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

    return tt.TT(op_cores)


def commpute_half_chain_corr(mps: tt.TT) -> np.ndarray:
    L = mps.order
    r0 = L // 2
    corr = []

    sigmaz = 0.5 * np.array([[1, 0], [0, -1]])
    an_op = tt.TT(sigmaz)
    an_op.cores[0] = an_op.cores[0].reshape(1, 2, 2, 1)

    for i in range(r0 + 1, L):
        corr.append(abs(compute_correlation(mps, an_op, r0=r0, r1=i)))
    return np.array(corr)


def compute_staggered_mag_1(mps: tt.TT) -> float:

    sigmaz = 0.5 * np.array([[1, 0], [0, -1]])
    sigmaz_sq = sigmaz @ sigmaz
    staggered_mag = tt.TT(sigmaz_sq)
    staggered_mag.cores[0] = staggered_mag.cores[0].reshape(1, 2, 2, 1)

    site_vals = np.zeros(mps.order, dtype=float)
    for i in range(mps.order):
        left_boundary = np.ones((1, 1))
        right_boundary = np.ones((1, 1))

        for j in range(mps.order):
            if j == i:
                contraction = np.tensordot(mps.cores[j], staggered_mag.cores[0], axes=([1, 2], [2, 1]))
            else:
                contraction = np.tensordot(mps.cores[j], np.eye(2).reshape(1, 2, 2, 1), axes=([1, 2], [2, 1]))

            left_boundary = np.tensordot(left_boundary, contraction, axes=([0, 1], [0, 2]))

        if (left_boundary @ right_boundary).item().imag < 1e-12:
            site_vals[i] = (-1)**(i + 1) * (left_boundary @ right_boundary).item().real
        else:
            raise ValueError("Complex expectation value is found.")

    return np.sqrt(np.mean(site_vals))


def compute_staggered_mag_2(mps: tt.TT) -> float:

    sigmaz = 0.5 * np.array([[1, 0], [0, -1]])
    staggered_mag = tt.TT(sigmaz)
    staggered_mag.cores[0] = staggered_mag.cores[0].reshape(1, 2, 2, 1)

    left_boundary = np.ones((1, 1, 1))
    right_boundary = np.ones((1, 1, 1))

    site_vals = []
    for i in range(mps.order):
        for j in range(i, mps.order):
            for k in range(mps.order):
                if k == i == j:
                    contraction = np.tensordot(staggered_mag.cores[0], staggered_mag.cores[0], axes=([1], [2]))
                elif k != i and j != j:
                    contraction = np.tensordot(np.eye(2).reshape(1, 2, 2, 1),
                                               np.eye(2).reshape(1, 2, 2, 1),
                                               axes=([1], [2]))
                else:
                    contraction = np.tensordot(staggered_mag.cores[0], np.eye(2).reshape(1, 2, 2, 1), axes=([1], [2]))

                contraction = np.tensordot(mps.cores[k], contraction, axes=([1, 2], [1, 4]))

                left_boundary = np.tensordot(left_boundary, contraction, axes=([0, 1, 2], [0, 2, 4]))

            if (left_boundary @ right_boundary).item().imag < 1e-12:
                site_vals.append((-1)**(i + j) * (left_boundary @ right_boundary).item().real)
            else:
                raise ValueError("Complex expectation value is found.")
    return np.sqrt((1 / (mps.order)**2) * 2 * np.sum(site_vals))
