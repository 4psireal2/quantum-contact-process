import logging
import unittest
from copy import deepcopy

import numpy as np

import scikit_tt.tensor_train as tt

import src.utilities.utils as utils
import src.models.contact_process_model as model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

L = 10
BOND_DIM = 3
basis_0 = np.array([1, 0])
basis_1 = np.array([0, 1])


class UtilityFunctions(unittest.TestCase):

    def test_mixed_canonical_form(self):
        ortho_center = 5
        mps = tt.ones(row_dims=L * [4], col_dims=L * [1], ranks=BOND_DIM)
        mps = utils.orthogonalize_mps(mps, ortho_center)
        mps_dag = mps.transpose(conjugate=True)
        mps = utils.canonicalize_mps(mps)
        mps_dag = utils.canonicalize_mps(mps_dag)

        # check left orthogonalization
        a = np.tensordot(mps.cores[0], mps_dag.cores[0], axes=([0, 1, 2], [0, 1, 2]))
        assert np.allclose(a, np.eye(BOND_DIM))

        # check right orthogonalization
        b = np.tensordot(mps.cores[6], mps_dag.cores[6], axes=([1, 2, 3], [1, 2, 3]))
        assert np.allclose(b, np.eye(BOND_DIM))

        # check norm
        norm = np.tensordot(mps.cores[ortho_center], mps.cores[ortho_center], axes=([0, 1, 2, 3], [0, 1, 2, 3]))
        assert np.isclose(norm, mps.norm()**2)

        # check orthonormalization
        mps = utils.orthonormalize_mps(mps, ortho_center)
        norm = np.tensordot(mps.cores[ortho_center], mps.cores[ortho_center], axes=([0, 1, 2, 3], [0, 1, 2, 3]))
        assert np.isclose(norm, mps.norm()**2)
        assert np.isclose(norm, 1.0)

    def test_num_op(self):

        mps = utils.construct_basis_mps(L, basis=[np.kron(basis_1, basis_1)] * L)
        hermit_mps = deepcopy(mps)
        mps_dag = mps.transpose(conjugate=True)

        for k in range(L):
            hermit_mps.cores[k] = (mps.cores[k] + mps_dag.cores[k]) / 2
        particle_nums = utils.compute_site_expVal(hermit_mps, model.construct_num_op(L))

        assert np.array_equal(particle_nums, 2 * np.ones(L))

        mps = utils.construct_basis_mps(L, basis=[np.kron(basis_0, basis_0)] * L)
        hermit_mps = deepcopy(mps)
        mps_dag = mps.transpose(conjugate=True)

        for k in range(L):
            hermit_mps.cores[k] = (mps.cores[k] + mps_dag.cores[k]) / 2
        particle_nums = utils.compute_site_expVal(hermit_mps, model.construct_num_op(L))

        assert np.array_equal(particle_nums, np.zeros(L))

    def test_purity(self):
        mps_1 = utils.construct_basis_mps(L, basis=[np.kron(basis_1, basis_1)] * L)
        assert np.isclose(utils.compute_purity(mps_1), 1.0)

        mps_2 = utils.construct_basis_mps(L, basis=[np.kron(basis_1, basis_0)] * L)
        mixed_mps = 1 / np.sqrt(2) * (mps_1 + mps_2)
        assert np.isclose(utils.compute_purity(mixed_mps), 0.5)

    def test_correlation(self):
        mps_1 = utils.construct_basis_mps(L, basis=[np.kron(basis_1, basis_1)] * L)
        an_op = model.construct_num_op(1)
        corr = np.zeros(L // 2)

        for j in range(L // 2):
            corr[j] = abs(utils.compute_correlation(mps_1, an_op, r=j))

        assert np.array_equal(corr, np.zeros(L // 2))
