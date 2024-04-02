import numpy as np
import scikit_tt.tensor_train as tt


def construct_lindblad(gamma: float, omega: float, L: int) -> tt.TT:
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

    return tt.TT(op_cores)


def construct_basis_mps(L: int, basis: list[np.ndarray]) -> tt.TT:
    mps_cores = [None] * L

    for i in range(L):
        mps_cores[i] = np.zeros([1, 2, 2, 1], dtype=complex)
        mps_cores[i] = basis[i].reshape(1, 2, 2, 1)

    return tt.TT(mps_cores)


def construct_uniform_mps(L: int, bond_dim: int) -> tt.TT:
    mps = tt.uniform(L * [4], ranks=bond_dim)

    mps.cores[0] = mps.cores[0].reshape(1, 2, 2, bond_dim)
    for i in range(1, L - 1):
        mps.cores[i] = mps.cores[i].reshape(bond_dim, 2, 2, bond_dim)
    mps.cores[-1] = mps.cores[-1].reshape(bond_dim, 2, 2, 1)

    return mps


def construct_num_op(L: int) -> tt.TT:
    # construct core
    op_cores = [None] * L

    for i in range(L):
        number_op = np.kron(np.array([[0, 0], [0, 1]]), np.eye(2)) + np.kron(np.eye(2), np.array([[0, 0], [0, 1]]))
        op_cores[i] = np.zeros([2, 2, 2, 2], dtype=complex)
        op_cores[i] = number_op.reshape(2, 2, 2, 2)

    return tt.TT(op_cores)


def compute_site_expVal(mps: tt.TT, mpo: tt.TT) -> np.ndarray:
    assert mps.order == mpo.order
    exp_vals = np.zeros(mps.order)

    mps_dag = mps.transpose(conjugate=True)
    for i in range(mps.order):
        left = np.tensordot(mps_dag.cores[i], mpo.cores[i], axes=([1, 2], [0, 1]))
        left = np.transpose(left, (0, 2, 3, 1))
        right = np.tensordot(left, mps.cores[i], axes=([0, 1, 2, 3], [0, 1, 2, 3]))
        exp_vals[i] = float(right.squeeze())

    return exp_vals
