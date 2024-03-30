import numpy as np
import matplotlib.pyplot as plt

import scikit_tt.tensor_train as tt
from scikit_tt.solvers.evp import als

from src.models.ising_chain_model import create_mpo

# path for results
PATH = "/home/psireal42/study/quantum-contact-process-1D/results"

# system parameters
GAMMA = 1
OMEGA = 1.5
V = 5
DELTAS = np.linspace(-4, 6, 30)

# compute spectral gap of L†L
L = 3
gaps = []

for DELTA in DELTAS:
    lindblad = create_mpo(gamma=GAMMA, V=V, omega=OMEGA, delta=DELTA, L=L)
    lindblad_hermitian = lindblad.transpose(conjugate=True) @ lindblad

    psi = tt.uniform(L * [4], ranks=10)
    eigenvalues, eigentensors, _ = als(lindblad_hermitian, psi, number_ev=2, repeats=10, sigma=0)
    # assert abs(eigenvalues[0]) < 1e-5
    # gaps.append(eigenvalues[1] - eigenvalues[0])

plt.figure()
plt.plot(DELTAS, gaps, 'o', label="L=3")
plt.xlabel(r"$\Delta$")
plt.title(r"Gap of $L^\dagger$ L for $\gamma = 1, V=5, \Omega = 1.5$")
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig(PATH + "/spectrum_gap_ising_chain.png")
