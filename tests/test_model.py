"""
Test model construction: dissipative Ising chain and quantum contact process
"""

import logging
import unittest

import numpy as np

import src.models.contact_process_model as cp_model
import src.models.diss_ising_model as diss_ising_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

L = 5
GAMMA = 1.0
OMEGA = 1.5
V = 5.0
DELTA = -4.0

d = 2
basis_0 = np.array([1, 0])


class ModelFunctions(unittest.TestCase):

    def test_hermiticity(self):
        ising_lindblad = diss_ising_model.construct_lindblad(gamma=GAMMA, V=V, omega=OMEGA, delta=DELTA, L=L)
        ising_lindblad_dag = diss_ising_model.construct_lindblad_dag(gamma=GAMMA, V=V, omega=OMEGA, delta=DELTA, L=L)
        assert np.array_equal(ising_lindblad.transpose(conjugate=True).matricize(), ising_lindblad_dag.matricize())

        ising_lindblad_hermitian = ising_lindblad_dag @ ising_lindblad
        assert np.array_equal(np.conj(ising_lindblad_hermitian.matricize().T), ising_lindblad_hermitian.matricize())

        cp_lindblad = cp_model.construct_lindblad(gamma=GAMMA, omega=OMEGA, L=L)
        cp_lindblad_dag = cp_model.construct_lindblad_dag(gamma=GAMMA, omega=OMEGA, L=L)
        assert np.array_equal(cp_lindblad.transpose(conjugate=True).matricize(), cp_lindblad_dag.matricize())

        cp_lindblad_hermitian = cp_lindblad_dag @ cp_lindblad
        assert np.array_equal(np.conj(cp_lindblad_hermitian.matricize().T), cp_lindblad_hermitian.matricize())

    def test_exact_diagonalization(self):
        cp_lindblad = cp_model.construct_lindblad(gamma=GAMMA, omega=OMEGA, L=L)
        cp_lindblad_dag = cp_model.construct_lindblad_dag(gamma=GAMMA, omega=OMEGA, L=L)
        cp_lindblad_hermitian = cp_lindblad_dag @ cp_lindblad

        evals, evecs = np.linalg.eigh(cp_lindblad_hermitian.matricize())
        assert np.isclose(np.min(evals.real), 0.0)

        # overlap with dark state
        dark_state = np.outer(basis_0, basis_0)
        for _ in range(L - 1):
            basis = np.outer(basis_0, basis_0)
            dark_state = np.kron(dark_state, basis)

        dark_state = dark_state.reshape(d**(L * 2))
        assert np.array_equal(evecs[np.argmin(evals.real)], dark_state)

        ising_lindblad = diss_ising_model.construct_lindblad(gamma=GAMMA, V=V, omega=OMEGA, delta=DELTA, L=L)
        ising_lindblad_dag = diss_ising_model.construct_lindblad_dag(gamma=GAMMA, V=V, omega=OMEGA, delta=DELTA, L=L)
        ising_lindblad_hermitian = ising_lindblad_dag @ ising_lindblad

        evals, evecs = np.linalg.eigh(ising_lindblad_hermitian.matricize())
        print(np.min(evals.real))
        assert 1 == 0
        assert np.isclose(np.min(evals.real), 0.0)
        # print(np.min(evals.real))

        # overlap with dark state
        dark_state = np.outer(basis_0, basis_0)
        for _ in range(L - 1):
            basis = np.outer(basis_0, basis_0)
            dark_state = np.kron(dark_state, basis)

        dark_state = dark_state.reshape(d**(L * 2))
        print(f"{np.linalg.norm(evecs[np.argmin(evals.real)] - dark_state)=}")
        assert np.array_equal(evecs[np.argmin(evals.real)], dark_state)
