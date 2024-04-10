import logging
import matplotlib.pyplot as plt
import numpy as np
import time
from copy import deepcopy
from datetime import datetime

import scikit_tt.tensor_train as tt
from scikit_tt.solvers.evp import als

from src.models.contact_process_model import (construct_lindblad, construct_num_op)
from src.utilities.utils import (canonicalize_mps, compute_correlation, compute_site_expVal)

logger = logging.getLogger(__name__)
log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(filename="/home/psireal42/study/quantum-contact-process-1D/playground/logging/" + log_filename,
                    level=logging.INFO)

# path for results
PATH = "/home/psireal42/study/quantum-contact-process-1D/results/"

# system parameters
L = 10
GAMMA = 1
OMEGAS = np.linspace(0, 10, 10)

# TN algorithm parameters
bond_dims = np.array([8, 16, 20])
conv_eps = 1e-6

### Stationary simulation
logger.info("Stationary simulation")
spectral_gaps = []
n_s = np.zeros((bond_dims.shape[0], OMEGAS.shape[0]))
purities = []
correlations = np.zeros((len(OMEGAS), L // 2))

for i, OMEGA in enumerate(OMEGAS):
    for j, bond_dim in enumerate(bond_dims):
        logger.info(f"Run ALS for {L=}, {OMEGA=} and {bond_dim=}")
        lindblad = construct_lindblad(gamma=GAMMA, omega=OMEGA, L=L)
        lindblad_hermitian = lindblad.transpose(conjugate=True) @ lindblad

        mps = tt.ones(row_dims=L * [4], col_dims=L * [1], ranks=bond_dim)
        mps = mps.ortho()
        mps = (1 / mps.norm()**2) * mps
        time1 = time.time()
        eigenvalues, eigentensors, _ = als(lindblad_hermitian, mps, number_ev=2, repeats=10, conv_eps=conv_eps, sigma=0)
        logger.info(f"{(lindblad_hermitian@eigentensors[0] - eigenvalues[0]*eigentensors[0]).norm()**2=}")
        time2 = time.time()
        logger.info(f"Elapsed time: {time2 - time1} seconds")
        logger.info(f"Ground state energy per site E = {eigenvalues/L}")
        logger.info(f"Norm of ground state: {eigentensors[0].norm()**2}")

        logger.info("Reshape MPS of ground state")
        gs_mps = eigentensors[0]
        gs_mps = canonicalize_mps(gs_mps)

        logger.info("Compute expectation value")
        # compute Hermitian part of mps
        hermit_mps = deepcopy(gs_mps)
        gs_mps_dag = gs_mps.transpose(conjugate=True)
        for k in range(L):
            hermit_mps.cores[k] = (gs_mps.cores[k] + gs_mps_dag.cores[k]) / 2

        logger.info("Compute particle numbers")
        particle_nums = compute_site_expVal(hermit_mps, construct_num_op(L))
        n_s[j, i] = np.mean(particle_nums)
        logger.info(f"Mean particle number = {n_s[j, i]}")

        if bond_dim == bond_dims[-1]:
            logger.info("Compute spectral gap of L†L for largest bond dimension")
            spectral_gaps.append(abs(eigenvalues[1] - eigenvalues[0]))

            logger.info("Compute purity of state for largest bond dimension")
            purities.append(gs_mps.norm()**2)
            logger.info(f"purity={gs_mps.norm()**2}")

            logger.info("Compute half-chain density correlation for largest bond dimension")
            an_op = construct_num_op(1)
            for k in range(L // 2):
                correlations[i, k] = abs(compute_correlation(gs_mps, an_op, r=k))

# plot spectral gaps
plt.figure()
plt.plot(OMEGAS, spectral_gaps, 'o-')
plt.xlabel(r"$\Omega$")
plt.title(f"{L=}, $\chi=${bond_dims[-1]}")
plt.grid()
plt.tight_layout()
plt.savefig(PATH + f"spectral_gaps_{L=}.png")

# plot stationary densities
plt.figure()
for i, bond_dim in enumerate(bond_dims):
    plt.plot(OMEGAS, n_s[i, :], 'o-', label=f"$\chi=${bond_dim}")
plt.xlabel(r"$\Omega$")
plt.ylabel(r"$n_s$")
plt.legend()
plt.title(f"{L=}")
plt.grid()
plt.tight_layout()
plt.savefig(PATH + f"stationary_density_{L=}.png")

# plot purities
plt.figure()
plt.plot(OMEGAS, purities, 'o-')
plt.xlabel(r"$\Omega$")
plt.ylabel(r"tr($\rho^{2}$)")
plt.title(f"{L=}, $\chi=${bond_dims[-1]}")
plt.grid()
plt.tight_layout()
plt.savefig(PATH + f"purities_{L=}.png")

# plot correlations
plt.figure()
for i, OMEGA in enumerate(OMEGAS):
    plt.plot(list(range(L // 2)), correlations[i, :], 'o-', label=f"$\Omega=${OMEGA}")
plt.xlabel("r")
plt.ylabel(r"$|C^{L/2}_{nn}(r)|$")
plt.legend()
plt.title(f"{L=}, $\chi=${bond_dims[-1]}")
plt.grid()
plt.tight_layout()
plt.savefig(PATH + f"correlations_{L=}.png")
