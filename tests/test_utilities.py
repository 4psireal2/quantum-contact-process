"""
Test utility functions
"""

import logging
import unittest

import numpy as np

import scikit_tt.tensor_train as tt

import src.utilities.utils as utils
import src.models.contact_process_model as model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

L = 5
BOND_DIM = 3
basis_0 = np.array([1, 0])
basis_1 = np.array([0, 1])


class UtilityFunctions(unittest.TestCase):

    def test_mixed_canonical_form(self):
        ortho_center = 3
        mps = tt.ones(row_dims=L * [4], col_dims=L * [1], ranks=BOND_DIM)
        mps = utils.orthogonalize_mps(mps, ortho_center)
        mps_dag = mps.transpose(conjugate=True)
        mps = utils.canonicalize_mps(mps)
        mps_dag = utils.canonicalize_mps(mps_dag)

        # check left orthogonalization
        a = np.tensordot(mps.cores[0], mps_dag.cores[0], axes=([0, 1, 2], [0, 1, 2]))
        assert np.allclose(a, np.eye(BOND_DIM))

        # check right orthogonalization
        b = np.tensordot(mps.cores[ortho_center + 1], mps_dag.cores[ortho_center + 1], axes=([1, 2, 3], [1, 2, 3]))
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
        mps_tests = [
            utils.construct_basis_mps(L, basis=[np.kron(basis_1, basis_1)] * L),
            utils.construct_basis_mps(L, basis=[np.kron(basis_1, basis_0)] * L),
            utils.construct_basis_mps(L, basis=[np.kron(basis_0, basis_1)] * L),
            utils.construct_basis_mps(L, basis=[np.kron(basis_0, basis_0)] * L)
        ]
        num_op_chain = model.construct_num_op(L)

        # expected arrays of single site expectation values
        site_vals = [2 * np.ones(L), np.ones(L), np.ones(L), np.zeros(L)]
        for i, mps in enumerate(mps_tests):
            site_expVal = utils.compute_site_expVal_vMPO(mps, num_op_chain)
            assert np.array_equal(site_expVal, site_vals[i])

    def test_purity(self):
        mps_1 = utils.construct_basis_mps(L, basis=[np.kron(basis_1, basis_1)] * L)
        assert np.isclose(utils.compute_purity(mps_1), 1.0)

        mps_2 = utils.construct_basis_mps(L, basis=[np.kron(basis_1, basis_0)] * L)
        mixed_mps = 1 / np.sqrt(2) * (mps_1 + mps_2)
        assert np.isclose(utils.compute_purity(mixed_mps), 0.5)

    # def test_correlation(self):
    #     mps_1 = utils.construct_basis_mps(L, basis=[np.kron(basis_1, basis_1)] * L)
    #     an_op = model.construct_num_op(1)
    #     corr = np.zeros(L // 2)

    #     for j in range(L // 2):
    #         corr[j] = abs(utils.compute_correlation_vMPO(mps_1, an_op, r=j))

    #     assert np.array_equal(corr, np.zeros(L // 2))
