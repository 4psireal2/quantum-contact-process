import numpy as np
import scikit_tt.tensor_train as tt


def construct_lindblad(L: int, g: float, J: float) -> tt.TT:

    # define operators
    sigma_x = 0.5 * np.array([[0, 1], [1, 0]])
    sigma_z = 0.5 * np.array([[1, 0], [0, -1]])
    sigma_xL = np.kron(np.eye(2), sigma_x)
    sigma_xR = np.kron(sigma_x, np.eye(2))
    sigma_zL = np.kron(np.eye(2), sigma_z)
    sigma_zR = np.kron(sigma_z, np.eye(2))
    Id = np.eye(4)

    # core component
    S = 1j * g * (sigma_zL + sigma_zR)

    op_cores = [None] * L
    op_cores[0] = np.zeros([1, 4, 4, 4], dtype=complex)
    op_cores[0][0, :, :, 0] = S
    op_cores[0][0, :, :, 1] = sigma_xR
    op_cores[0][0, :, :, 2] = sigma_xL
    op_cores[0][0, :, :, 3] = Id
    for i in range(1, L - 1):
        op_cores[i] = np.zeros([4, 4, 4, 4], dtype=complex)
        op_cores[i][0, :, :, 0] = Id
        op_cores[i][1, :, :, 0] = 1j * sigma_xR
        op_cores[i][2, :, :, 0] = 1j * sigma_xL
        op_cores[i][3, :, :, 0] = S
        op_cores[i][3, :, :, 1] = sigma_xR
        op_cores[i][3, :, :, 2] = sigma_xL
        op_cores[i][3, :, :, 3] = Id
    op_cores[-1] = np.zeros([4, 4, 4, 1], dtype=complex)
    op_cores[-1][0, :, :, 0] = Id
    op_cores[-1][1, :, :, 0] = 1j * sigma_xR
    op_cores[-1][2, :, :, 0] = 1j * sigma_xL
    op_cores[-1][3, :, :, 0] = S

    return tt.TT(op_cores)
