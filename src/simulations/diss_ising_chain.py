import scikit_tt.tensor_train as tt
from scikit_tt.solvers.evp import als

from models.diss_ising_model import construct_lindblad, construct_lindblad_dag

# path for results
# PATH = "/home/psireal42/study/quantum-contact-process-1D/results"

# system parameters
GAMMA = 1.0
OMEGA = 1.5
V = 5
DELTAS = [0.0]

# TN algorithm parameters
conv_eps = 1e-6
bond_dim = 16

# compute spectral gap of L†L
L = 10
gaps = []

for DELTA in DELTAS:
    lindblad = construct_lindblad(gamma=GAMMA, V=V, omega=OMEGA, delta=DELTA, L=L)
    lindblad_hermitian = construct_lindblad_dag(gamma=GAMMA, V=V, omega=OMEGA, delta=DELTA, L=L) @ lindblad

    mps = tt.ones(row_dims=L * [4], col_dims=L * [1], ranks=bond_dim)
    mps = mps.ortho()
    mps = (1 / mps.norm()**2) * mps
    eigenvalues, eigentensors, _ = als(lindblad_hermitian, mps, number_ev=1, repeats=10, conv_eps=conv_eps, sigma=0)
    print(f"{eigenvalues=}")
    # assert abs(eigenvalues[0]) < 1e-5
    # gaps.append(eigenvalues[1] - eigenvalues[0])

# plt.figure()
# plt.plot(DELTAS, gaps, 'o', label="L=3")
# plt.xlabel(r"$\Delta$")
# plt.title(r"Gap of $L^\dagger$ L for $\gamma = 1, V=5, \Omega = 1.5$")
# plt.legend()
# plt.tight_layout()
# plt.grid()
# plt.savefig(PATH + "/spectrum_gap_ising_chain.png")

# lindblad = construct_lindblad(gamma=GAMMA, V=V, omega=OMEGA, delta=DELTA, L=L)
# lindblad_hermitian =  construct_lindblad_dag(gamma=GAMMA, V=V, omega=OMEGA, delta=DELTA, L=L) @ lindblad
