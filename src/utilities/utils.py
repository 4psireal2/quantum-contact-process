from copy import deepcopy

import numpy as np
from scipy import linalg

import scikit_tt.tensor_train as tt

import src.utilities.utils as utils


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


def compute_site_expVal(mps: tt.TT, mpo: tt.TT) -> np.ndarray:
    """
    Compute the expectation value < Ψ | onsiteOp | Ψ > for each site of the MPS 

    Args:
    - mps: canonicalized mps
    - mpo: cores that have dimension (2,2,2,2)
    """
    assert mps.order == mpo.order
    exp_vals = np.zeros(mps.order)

    mps_dag = mps.transpose(conjugate=True)
    for i in range(mps.order):
        tensor_norm = np.tensordot(mps.cores[i], mps_dag.cores[i], axes=([0, 1, 2, 3], [0, 1, 2, 3]))
        tensor_norm = float((tensor_norm.squeeze()).real)

        contraction = np.tensordot(mps.cores[i], mpo.cores[i], axes=([1, 2], [0, 1]))
        contraction = contraction.transpose(0, 2, 3, 1)
        contraction = np.tensordot(contraction, mps_dag.cores[i], axes=([0, 1, 2, 3], [0, 1, 2, 3]))

        exp_vals[i] = float((contraction.squeeze()).real) / tensor_norm

    return exp_vals


def compute_site_expVal_vMPO(mps: tt.TT, mpo: tt.TT) -> np.ndarray:
    """
    Compute Tr(ρ A_k) for each k
    """

    site_vals = np.zeros(mps.order, dtype=float)

    for i in range(mps.order):
        left_boundary = np.ones(1)
        right_boundary = np.ones(1)

        for j in range(mps.order):
            if j == i:
                contraction = np.tensordot(mps.cores[j], mpo.cores[0], axes=([1, 2], [0, 1]))
                contraction = np.einsum('ijlk->ij', contraction)  # tracing over the physical indices
            else:
                contraction = np.einsum('ikjl->il', mps.cores[j])  # tracing over the physical indices

            left_boundary = np.tensordot(left_boundary, contraction, axes=([0], [0]))

        if (left_boundary @ right_boundary).imag < 1e-12:
            site_vals[i] = (left_boundary @ right_boundary).real
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


def compute_correlation(mps: tt.TT, mpo: tt.TT, r: tt.TT) -> float:
    """
    Compute C(r) =  <O_{L/2} . O_{L/2+r}> - <O_{L/2}><O_{L/2+r}>

    Args:
    - mps: canonicalized mps
    - mpo: 1 core with shape (2,2,2,2)
    """
    assert r < mps.order // 2

    # bring mps to mixed-canonical form with orthogonal center at L//2
    mps = utils.orthonormalize_mps(mps, ortho_center=mps.order // 2)
    assert np.isclose(mps.norm()**2, 1.0)
    mps_dag = mps.transpose(conjugate=True)

    dim_mid_0, _, _, _ = mps.cores[mps.order // 2].shape
    left_boundary = np.ones((dim_mid_0, dim_mid_0))
    _, _, _, dim_r_3 = mps.cores[mps.order // 2 + r].shape
    right_boundary = np.ones((dim_r_3, dim_r_3))

    # compute <O_{L/2} . O_{L/2+r}>
    for i in range(mps.order // 2, mps.order // 2 + r + 1):
        if i != mps.order // 2 and i != (mps.order // 2 + r):
            contraction = np.tensordot(mps.cores[i], mps_dag.cores[i], axes=([1, 2], [1, 2]))
        else:
            contraction = np.tensordot(mps.cores[i], mpo.cores[0], axes=([1, 2], [0, 1]))
            contraction = contraction.transpose(0, 2, 3, 1)  # permute legs
            contraction = np.tensordot(contraction, mps_dag.cores[i], axes=([1, 2], [1, 2]))
            if r == 0:  # autocorrelation case
                contraction = np.tensordot(mps.cores[i], mpo.cores[0], axes=([1, 2], [0, 1]))
                contraction = contraction.transpose(0, 2, 3, 1)
                contraction = np.tensordot(contraction, mpo.cores[0], axes=([1, 2], [0, 1]))
                contraction = contraction.transpose(0, 2, 3, 1)
                contraction = np.tensordot(contraction, mps_dag.cores[i], axes=([1, 2], [1, 2]))

        left_boundary = np.tensordot(left_boundary, contraction, axes=([0, 1], [0, 2]))

    mean_product = np.trace(left_boundary @ right_boundary).item()

    # compute <O_{L/2}><O_{L/2+r}>
    contraction_1 = np.tensordot(mps.cores[mps.order // 2], mpo.cores[0], axes=([1, 2], [0, 1]))
    contraction_1 = contraction_1.transpose(0, 2, 3, 1)
    contraction_1 = np.tensordot(contraction_1, mps_dag.cores[mps.order // 2], axes=([0, 1, 2, 3], [0, 1, 2, 3]))

    mps_ortho_r = utils.orthonormalize_mps(mps, ortho_center=mps.order // 2 + r)
    assert np.isclose(mps_ortho_r.norm()**2, 1.0)
    mps_ortho_r_dag = mps_ortho_r.transpose(conjugate=True)

    contraction_2 = np.tensordot(mps_ortho_r.cores[mps.order // 2 + r], mpo.cores[0], axes=([1, 2], [0, 1]))
    contraction_2 = contraction_2.transpose(0, 2, 3, 1)
    contraction_2 = np.tensordot(contraction_2,
                                 mps_ortho_r_dag.cores[mps.order // 2 + r],
                                 axes=([0, 1, 2, 3], [0, 1, 2, 3]))

    product_mean = contraction_1 * contraction_2

    return mean_product - product_mean


def compute_correlation_vMPO(mps: tt.TT, mpo: tt.TT, r0: int, r1: int) -> float:
    """
    Compute  <O_{r0} . O_{r1}> - <O_{r0}><O_{r1}> with trace

    Args:
    - mps: canonicalized mps
    - mpo: 1 core with shape (2,2,2,2)
    - r0, r1: first, second index of sites on the TT 
    """

    left_boundary = np.ones(1)
    right_boundary = np.ones(1)

    # compute <O_{r0} . O_{r1}>
    for i in range(mps.order):
        if i != r0 and i != r1:
            contraction = np.einsum('ikjl->il', mps.cores[i])
        else:
            contraction = np.tensordot(mps.cores[i], mpo.cores[0], axes=([1, 2], [0, 1]))
            contraction = np.einsum('ijlk->ij', contraction)  # tracing over the physical indices
        left_boundary = np.tensordot(left_boundary, contraction, axes=([0], [0]))

    mean_product = left_boundary @ right_boundary

    # compute <O_{r0}><O_{r1}>
    # compute <O_{r0}>
    left_boundary = np.ones(1)
    right_boundary = np.ones(1)

    for j in range(mps.order):
        if j == r0:
            contraction = np.tensordot(mps.cores[j], mpo.cores[0], axes=([1, 2], [0, 1]))
            contraction = np.einsum('ijlk->ij', contraction)  # tracing over the physical indices
        else:
            contraction = np.einsum('ikjl->il', mps.cores[j])  # tracing over the physical indices

        left_boundary = np.tensordot(left_boundary, contraction, axes=([0], [0]))

    product_mean = left_boundary @ right_boundary

    # compute <O_{r1}>
    left_boundary = np.ones(1)
    right_boundary = np.ones(1)

    for j in range(mps.order):
        if j == r1:
            contraction = np.tensordot(mps.cores[j], mpo.cores[0], axes=([1, 2], [0, 1]))
            contraction = np.einsum('ijlk->ij', contraction)  # tracing over the physical indices
        else:
            contraction = np.einsum('ikjl->il', mps.cores[j])  # tracing over the physical indices

        left_boundary = np.tensordot(left_boundary, contraction, axes=([0], [0]))

    product_mean *= left_boundary @ right_boundary

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

    return s


def compute_overlap(mps_1: tt.TT, mps_2: tt.TT) -> float:

    left_boundary = np.ones((1, 1))
    right_boundary = np.ones((1, 1))

    mps_2_dag = mps_2.transpose(conjugate=True)
    for i in range(mps_1.order):

        contraction = np.tensordot(mps_1.cores[i], mps_2_dag.cores[i], axes=([1, 2], [1, 2]))
        left_boundary = np.tensordot(left_boundary, contraction, axes=([0, 1], [0, 2]))

    return np.trace(left_boundary @ right_boundary).item()
