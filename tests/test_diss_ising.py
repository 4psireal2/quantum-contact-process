import numpy as np
from datetime import datetime

import scikit_tt.tensor_train as tt
from scikit_tt.solvers.evp import als

import src.models.diss_ising_model as diss_ising_model
import src.utilities.utils as utils

# path for results
PATH = "/scratch/nguyed99/qcp-1d/results/"

L = 10
GAMMA = 1.0
OMEGA = 1.5
V = 5.0
# DELTAS = np.linspace(-4, 4, 9)
DELTAS = [0.0]

bond_dim = 8
conv_eps = 1e-6

purities = np.zeros(len(DELTAS))
staggered_mag = np.zeros(len(DELTAS))
correlations = np.zeros((3, L // 2 - 1))
polarizations = np.zeros((3, L))

for i, DELTA in enumerate(DELTAS):
    ising_lindblad = diss_ising_model.construct_lindblad(gamma=GAMMA, V=V, omega=OMEGA, delta=DELTA, L=L)
    ising_lindblad_dag = diss_ising_model.construct_lindblad_dag(gamma=GAMMA, V=V, omega=OMEGA, delta=DELTA, L=L)
    ising_lindblad_hermitian = ising_lindblad_dag @ ising_lindblad

    mps = tt.ones(row_dims=L * [4], col_dims=L * [1], ranks=bond_dim)
    mps = mps.ortho()
    mps = (1 / mps.norm()**2) * mps
    eigenvalue, eigentensor, _ = als(ising_lindblad_hermitian, mps, number_ev=1, repeats=10, conv_eps=conv_eps, sigma=0)

    gs_mps = utils.canonicalize_mps(eigentensor)
    gs_mps_dag = gs_mps.transpose(conjugate=True)

    # compute non-Hermitian part of mps
    non_hermit_mps = (1 / 2) * (gs_mps - gs_mps_dag)
    print(f"The norm of the non-Hermitian part: {non_hermit_mps.norm()**2}")

    # compute Hermitian part of mps
    hermit_mps = (1 / 2) * (gs_mps + gs_mps_dag)

    purities[i] = utils.compute_purity(gs_mps)

    # compute observables
    staggered_mag[i] = diss_ising_model.compute_staggered_mag(hermit_mps)

    j = 0
    if DELTA in [-4, 4, 0]:
        correlations[j, :] = diss_ising_model.commpute_half_chain_corr(hermit_mps)

        # compute local polarization
        sigmaz = 0.5 * np.array([[1, 0], [0, -1]])
        sigmaz = np.kron(sigmaz, np.eye(2)) + np.kron(np.eye(2), sigmaz)
        an_op = tt.TT(sigmaz)
        an_op.cores[0] = an_op.cores[0].reshape(2, 2, 2, 2)

        polarizations[j, :] = utils.compute_site_expVal_vMPO(hermit_mps, an_op)

        j += 1

time3 = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())

np.savetxt(PATH + f"ising_purities_L_{L}_{time3}.txt", purities, delimiter=',')
np.savetxt(PATH + f"ising_staggered_mag_L_{L}_{time3}.txt", staggered_mag, delimiter=',')
np.savetxt(PATH + f"ising_correlations_L_{L}_{time3}.txt", correlations, delimiter=',')
np.savetxt(PATH + f"ising_polarizations_L_{L}_{time3}.txt", polarizations, delimiter=',')
