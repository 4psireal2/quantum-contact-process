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
        basis_0 = np.array([1, 0])
        basis_1 = np.array([0, 1])

        gs_mps = utils.construct_basis_mps(L, basis=[np.kron(basis_1, basis_1)] * L)
        hermit_mps = deepcopy(gs_mps)
        gs_mps_dag = gs_mps.transpose(conjugate=True)

        for k in range(L):
            hermit_mps.cores[k] = (gs_mps.cores[k] + gs_mps_dag.cores[k]) / 2
        particle_nums = utils.compute_site_expVal(hermit_mps, model.construct_num_op(L))

        assert np.array_equal(particle_nums, 2 * np.ones(L))

        gs_mps = utils.construct_basis_mps(L, basis=[np.kron(basis_0, basis_0)] * L)
        hermit_mps = deepcopy(gs_mps)
        gs_mps_dag = gs_mps.transpose(conjugate=True)

        for k in range(L):
            hermit_mps.cores[k] = (gs_mps.cores[k] + gs_mps_dag.cores[k]) / 2
        particle_nums = utils.compute_site_expVal(hermit_mps, model.construct_num_op(L))

        assert np.array_equal(particle_nums, np.zeros(L))
