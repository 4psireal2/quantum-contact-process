from copy import deepcopy

import numpy as np
from scipy import linalg

import scikit_tt.tensor_train as tt


def construct_basis_mps(L: int, basis: list[np.ndarray]) -> tt.TT:
    mps_cores = [None] * L

    for i in range(L):
        mps_cores[i] = np.zeros([1, 2, 2, 1], dtype=complex)
        mps_cores[i] = basis[i].reshape(1, 2, 2, 1)

    return tt.TT(mps_cores)


def orthogonalize_mps(mps: tt.TT, ortho_center: int) -> tt.TT:
    orthog_mps = deepcopy(mps)
    orthog_mps = orthog_mps.ortho_left(start_index=0, end_index=ortho_center - 1)
    orthog_mps = orthog_mps.ortho_right(start_index=mps.order - 1, end_index=ortho_center + 1)

    return orthog_mps


def orthonormalize_mps(mps: tt.TT, ortho_center: int = 0) -> tt.TT:
    """
    Returns:
    - orthon_mps: canonicalized orthonormalized mps
    """
    orthon_mps = orthogonalize_mps(mps, ortho_center)

    orthon_mps_dag = orthon_mps.transpose(conjugate=True)
    orthon_mps = canonicalize_mps(orthon_mps)
    orthon_mps_dag = canonicalize_mps(orthon_mps_dag)

    orthon_mps.cores[ortho_center] = 1 / orthon_mps.norm() * orthon_mps.cores[ortho_center]

    return orthon_mps


def canonicalize_mps(mps: tt.TT) -> tt.TT:
    """
    Bring mps into a vectorized form in Tensorkit's notation
    """
    cores = [None] * mps.order
    for k in range(mps.order):
        r1 = mps.ranks[k]
        r2 = mps.ranks[k + 1]
        cores[k] = mps.cores[k].reshape([r1, 2, 2, r2])

    return tt.TT(cores)


def compute_purity(mps: tt.TT) -> float:
    """
    Compute purity = Tr(ρ.ρ)
    Args:
    - mps: canonicalized mps
    """
    left_boundary = np.ones((1, 1))

    for i in range(mps.order):
        contraction = np.tensordot(mps.cores[i], mps.cores[i], axes=([1, 2], [2, 1]))
        left_boundary = np.tensordot(left_boundary, contraction, axes=([0, 1], [0, 2]))

    right_boundary = np.ones((1, 1))
    return np.trace(left_boundary @ right_boundary).item()


def compute_site_expVal_mpo(mps: tt.TT, mpo) -> np.ndarray:
    """
    Compute Tr(ρ A_k) for each k
    """

    site_vals = np.zeros(mps.order, dtype=float)

    for i in range(mps.order):
        left_boundary = np.ones(1)
        right_boundary = np.ones(1)

        for j in range(mps.order):

            if j == i:
                contraction = np.tensordot(mps.cores[j], mpo, axes=([1, 2], [1, 0]))
            else:
                contraction = np.tensordot(mps.cores[j], np.eye(2), axes=([1, 2], [1, 0]))

            left_boundary = np.tensordot(left_boundary, contraction, axes=([0], [0]))

        if (left_boundary @ right_boundary).item().imag < 1e-12:
            site_vals[i] = (left_boundary @ right_boundary).item().real
        else:
            raise ValueError("Complex expectation value is found.")

    return site_vals


def compute_site_expVal_mps(mps: tt.TT, mpo: np.ndarray) -> np.ndarray:

    site_vals = np.zeros(mps.order, dtype=float)
    mps_dag = mps.transpose(conjugate=True)

    np.ones((1, 1))
    np.ones((1, 1))

    for i in range(mps.order):

        contraction = np.tensordot(mps_dag.cores[i], mpo, axes=([1], [1]))
        contraction = np.tensordot(contraction, mps.cores[i], axes=([0, 3, 1, 2], [0, 2, 1, 3]))

    if contraction.item().imag < 1e-7:
        site_vals[i] = contraction
    else:
        raise ValueError("Complex expectation value is found.")

    return site_vals


def compute_expVal(mps: tt.TT, mpo: tt.TT) -> float:
    """
    Compute ⟨Φ(ρ)∣ Lˆ†Lˆ∣Φ(ρ)⟩ 
    """
    left_boundary = np.ones((1, 1, 1))
    right_boundary = np.ones((1, 1, 1))

    mps_dag = mps.transpose(conjugate=True)
    for i in range(mps.order):
        r1, r2 = mpo.ranks[i], mpo.ranks[i + 1]
        mpo_core = mpo.cores[i].reshape(r1, 2, 2, 2, 2, r2)
        contraction = np.tensordot(mps.cores[i], mpo_core, axes=([1, 2], [3, 4]))
        contraction = np.tensordot(contraction, mps_dag.cores[i], axes=([3, 4], [1, 2]))
        left_boundary = np.tensordot(left_boundary, contraction, axes=([0, 1, 2], [0, 2, 4]))

    exp_val = np.trace(left_boundary @ right_boundary).item()
    if exp_val.imag < 1e-7:
        return exp_val.real
    else:
        raise ValueError("Complex expectation value is found.")


def compute_correlation(mps: tt.TT, mpo: tt.TT, r0: int, r1: int) -> float:
    """
    Compute  <O_{r0} . O_{r1}> - <O_{r0}><O_{r1}> with trace

    Args:
    - mps: canonicalized mps
    - mpo: 1 core with shape (2,2,2,2)
    - r0, r1: first, second index of sites on the TT 
    """

    left_boundary = np.ones((1, 1))
    right_boundary = np.ones((1, 1))

    # compute <O_{r0} . O_{r1}>
    for j in range(mps.order):
        if j != r0 and j != r1:
            contraction = np.tensordot(mps.cores[j], np.eye(2).reshape(1, 2, 2, 1), axes=([1, 2], [2, 1]))
        else:
            contraction = np.tensordot(mps.cores[j], mpo.cores[0], axes=([1, 2], [2, 1]))

        left_boundary = np.tensordot(left_boundary, contraction, axes=([0, 1], [0, 2]))

    mean_product = (left_boundary @ right_boundary).item()

    # compute <O_{r0}><O_{r1}>
    # compute <O_{r0}>

    left_boundary = np.ones((1, 1))
    right_boundary = np.ones((1, 1))

    for j in range(mps.order):
        if j == r0:
            contraction = np.tensordot(mps.cores[j], mpo.cores[0], axes=([1, 2], [2, 1]))
        else:
            contraction = np.tensordot(mps.cores[j], np.eye(2).reshape(1, 2, 2, 1), axes=([1, 2], [2, 1]))

        left_boundary = np.tensordot(left_boundary, contraction, axes=([0, 1], [0, 2]))

    product_mean = (left_boundary @ right_boundary).item()

    # compute <O_{r1}>
    left_boundary = np.ones((1, 1))
    right_boundary = np.ones((1, 1))

    for j in range(mps.order):
        if j == r1:
            contraction = np.tensordot(mps.cores[j], mpo.cores[0], axes=([1, 2], [2, 1]))
        else:
            contraction = np.tensordot(mps.cores[j], np.eye(2).reshape(1, 2, 2, 1), axes=([1, 2], [2, 1]))

        left_boundary = np.tensordot(left_boundary, contraction, axes=([0, 1], [0, 2]))

    product_mean *= (left_boundary @ right_boundary).item()

    return mean_product - product_mean


def compute_dens_dens_corr(mps: tt.TT, mpo: tt.TT, r: int) -> float:
    """
    Compute  <O_{r} . O_{0}> - <O_{0}>^2 with trace

    Args:
    - mps: canonicalized mps
    - mpo: 1 core with shape (2,2,2,2)
    """

    left_boundary = np.ones((1, 1))
    right_boundary = np.ones((1, 1))

    # compute <O_{0} . O_{r}>
    for j in range(mps.order):
        if j != 0 and j != r:
            contraction = np.tensordot(mps.cores[j], np.eye(2).reshape(1, 2, 2, 1), axes=([1, 2], [2, 1]))
        else:
            contraction = np.tensordot(mps.cores[j], mpo.cores[0], axes=([1, 2], [2, 1]))

        left_boundary = np.tensordot(left_boundary, contraction, axes=([0, 1], [0, 2]))

    mean_product = (left_boundary @ right_boundary).item()

    # compute <O_{0}>²
    left_boundary = np.ones((1, 1))
    right_boundary = np.ones((1, 1))

    for j in range(mps.order):
        if j == 0:
            contraction = np.tensordot(mps.cores[j], mpo.cores[0], axes=([1, 2], [2, 1]))
        else:
            contraction = np.tensordot(mps.cores[j], np.eye(2).reshape(1, 2, 2, 1), axes=([1, 2], [2, 1]))

        left_boundary = np.tensordot(left_boundary, contraction, axes=([0, 1], [0, 2]))

    product_mean = ((left_boundary @ right_boundary).item())**2
    return mean_product - product_mean


def compute_entanglement_spectrum(mps: tt.TT) -> np.ndarray:
    half_chain_index = mps.order // 2
    ortho_mps = orthogonalize_mps(mps, half_chain_index)
    two_site_tensor = np.tensordot(ortho_mps.cores[half_chain_index],
                                   ortho_mps.cores[half_chain_index + 1],
                                   axes=([3], [0]))
    r1, r2, r3, r4, r5, r6 = two_site_tensor.shape
    two_site_matrix = two_site_tensor.reshape(r1 * r2 * r3, r4 * r5 * r6)
    [_, s, _] = linalg.svd(two_site_matrix,
                           full_matrices=False,
                           overwrite_a=True,
                           check_finite=False,
                           lapack_driver='gesvd')
    return s**2


def compute_overlap(mps_1: tt.TT, mps_2: tt.TT) -> float:

    left_boundary = np.ones((1, 1))
    right_boundary = np.ones((1, 1))

    mps_1_dag = mps_1.transpose(conjugate=True)
    for i in range(mps_1.order):
        contraction = np.tensordot(mps_1_dag.cores[i], mps_2.cores[i], axes=([1, 2], [1, 2]))
        left_boundary = np.tensordot(left_boundary, contraction, axes=([0, 1], [0, 2]))

    return np.trace(left_boundary @ right_boundary).item()
