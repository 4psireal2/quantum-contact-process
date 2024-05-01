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

L = 3
GAMMA = 1.0
OMEGA = 1.5
V = 5.0
DELTA = 0.0


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

        evals, _ = np.linalg.eig(cp_lindblad_hermitian.matricize())
        assert np.isclose(np.min(evals.real), 0.0)

        ising_lindblad = diss_ising_model.construct_lindblad(gamma=GAMMA, V=V, omega=OMEGA, delta=DELTA, L=L)
        ising_lindblad_dag = diss_ising_model.construct_lindblad_dag(gamma=GAMMA, V=V, omega=OMEGA, delta=DELTA, L=L)
        ising_lindblad_hermitian = ising_lindblad_dag @ ising_lindblad

        evals, _ = np.linalg.eig(ising_lindblad_hermitian.matricize())
        assert np.isclose(np.min(evals.real), 0.0)
