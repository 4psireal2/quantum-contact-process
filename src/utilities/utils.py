from copy import deepcopy

import numpy as np

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

    norm = np.tensordot(orthon_mps.cores[ortho_center],
                        orthon_mps_dag.cores[ortho_center],
                        axes=([0, 1, 2, 3], [0, 1, 2, 3]))
    orthon_mps.cores[ortho_center] = 1 / np.sqrt(norm) * orthon_mps.cores[ortho_center]

    return orthon_mps


def canonicalize_mps(mps: tt.TT) -> tt.TT:
    """
    Bring mps into a vectorized form in Tensorkit's notation
    """
    for k in range(mps.order):
        first, _, _, last = mps.cores[k].shape
        mps.cores[k] = mps.cores[k].reshape(first, 2, 2, last)

    return mps


def compute_site_expVal(mps: tt.TT, mpo: tt.TT) -> np.ndarray:
    """
    Args:
    - mps: canonicalized mps
    - mpo: cores that have dimension (2,2,2,2)
    """
    assert mps.order == mpo.order
    exp_vals = np.zeros(mps.order)

    mps_dag = mps.transpose(conjugate=True)
    for i in range(mps.order):
        tensor_norm = np.tensordot(mps.cores[i], mps_dag.cores[i], axes=([0, 1, 2, 3], [0, 1, 2, 3]))
        tensor_norm = float(tensor_norm.squeeze())

        contraction = np.tensordot(mps.cores[i], mpo.cores[i], axes=([1, 2], [0, 1]))
        contraction = contraction.transpose(0, 2, 3, 1)
        contraction = np.tensordot(contraction, mps_dag.cores[i], axes=([0, 1, 2, 3], [0, 1, 2, 3]))

        exp_vals[i] = float(contraction.squeeze()) / tensor_norm

    return exp_vals


def compute_norm(mps: tt.TT) -> float:
    """
    Compute norm = Tr(ρ.ρ^†)
    Args:
    - mps: canonicalized mps
    """
    mps_dag = mps.transpose(conjugate=True)

    left_boundary = np.ones((1, 1))

    for i in range(mps_dag.order):
        contraction = np.tensordot(mps.cores[i], mps_dag.cores[i], axes=([1, 2], [1, 2]))
        left_boundary = np.tensordot(left_boundary, contraction, axes=([0, 1], [0, 2]))

    right_boundary = np.ones((1, 1))
    return np.trace(left_boundary @ right_boundary)


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
    return np.trace(left_boundary @ right_boundary)


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

    dim_mid_0, _, _, dim_mid_3 = mps.cores[mps.order // 2].shape
    left_boundary = np.ones((dim_mid_0, dim_mid_0))
    dim_r_0, _, _, dim_r_3 = mps.cores[mps.order // 2 + r].shape
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

    mean_product = np.trace(left_boundary @ right_boundary)

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
