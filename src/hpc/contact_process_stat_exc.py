"""
Do static simulations for first excited state
"""
import logging
import numpy as np
import sys
import time

import scikit_tt.tensor_train as tt
from scikit_tt.solvers.evp import als

from src.models.contact_process_model import (construct_lindblad)
from src.utilities.utils import (canonicalize_mps, compute_correlation, compute_dens_dens_corr, compute_purity,
                                 compute_site_expVal, compute_expVal, compute_entanglement_spectrum,
                                 construct_basis_mps, compute_overlap)

OUTPUT_PATH = "/scratch/nguyed99/qcp-1d/results/"
LOG_PATH = "/scratch/nguyed99/qcp-1d/logging/"

# system parameters
L = 50
d = 2

# TN algorithm parameters
bond_dims = np.array([35, 50])
conv_eps = 1e-6

### observable operator
number_op = np.array([[0, 0], [0, 1]])
number_op.reshape((1, 2, 2, 1))
number_mpo = [None]
number_mpo = tt.TT(number_op)


def main():
    # parse environment variables
    OMEGA, bond_dim, SLURM_ARRAY_JOB_ID = float(sys.argv[1]), int(sys.argv[2]), str(sys.argv[3])
    logger = logging.getLogger(__name__)
    log_filename = f"L_{L}_OMEGA_{OMEGA}_D_{bond_dim}_{SLURM_ARRAY_JOB_ID}.log"
    logging.basicConfig(filename=LOG_PATH + log_filename, level=logging.INFO)

    logger.info("Stationary simulation")
    logger.info(f"Run ALS for {L=}, {OMEGA=} and {bond_dim=}")
    lindblad = construct_lindblad(gamma=1.0, omega=OMEGA, L=L)
    lindblad_hermitian = lindblad.transpose(conjugate=True) @ lindblad

    mps = tt.ones(row_dims=L * [4], col_dims=L * [1], ranks=bond_dim)
    mps = mps.ortho()
    mps = (1 / mps.norm()**2) * mps
    time1 = time.time()
    eigenvalues, eigentensors, _ = als(lindblad_hermitian, mps, number_ev=2, repeats=10, conv_eps=conv_eps, sigma=0)
    time2 = time.time()
    logger.info(f"Elapsed time: {time2 - time1} seconds")

    np.savetxt(OUTPUT_PATH + f"eval_0_exc_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
               eigenvalues[0],
               delimiter=',')
    np.savetxt(OUTPUT_PATH + f"eval_1_exc_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
               eigenvalues[1],
               delimiter=',')

    evp_residual = (lindblad_hermitian @ eigentensors[1] - eigenvalues[1] * eigentensors[1]).norm()**2
    logger.info(f"{eigenvalues=}")
    logger.info(f"Eigensolver error for first excited state: {evp_residual}")

    logger.info(f"First excited state energy per site: {eigenvalues[1]/L}")
    logger.info(f"Norm of first excited state: {eigentensors[1].norm()**2}")

    mps = canonicalize_mps(eigentensors[1])
    mps_dag = mps.transpose(conjugate=True)

    # compute non-Hermitian part of mps
    non_hermit_mps = (1 / 2) * (mps - mps_dag)
    logger.info(f"The norm of the non-Hermitian part: {non_hermit_mps.norm()**2}")
    logger.info(
        f"expVal_1, eval_1_als, eval_1: {compute_expVal(mps, lindblad_hermitian)}, {eigenvalues[1]}, {mps_dag@lindblad_hermitian@mps}"
    )

    # compute Hermitian part of mps
    hermit_mps = (1 / 2) * (mps + mps_dag)

    # compute observables
    logger.info("Compute particle numbers")
    particle_nums = compute_site_expVal(hermit_mps, number_mpo)
    logger.info(f"Particle number/site: {particle_nums}")
    n_s = np.mean(particle_nums)
    logger.info(f"Mean Particle number: {n_s}")
    np.savetxt(OUTPUT_PATH + f"n_s_exc_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt", n_s, delimiter=',')

    if bond_dim == bond_dims[-1]:
        logger.info(f"{bond_dim=}")
        logger.info("Compute spectral gap of L†L for largest bond dimension")
        spectral_gaps = abs(eigenvalues[1] - eigenvalues[0])
        logger.info(f"{spectral_gaps=}")
        np.savetxt(OUTPUT_PATH + f"spectral_gaps_exc_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
                   spectral_gaps,
                   delimiter=',')

        logger.info("Compute purity of state for largest bond dimension")
        purities = compute_purity(mps)
        logger.info(f"Purity: {purities}")
        np.savetxt(OUTPUT_PATH + f"purities_exc_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
                   purities,
                   delimiter=',')

        logger.info("Compute two-point correlation for largest bond dimension")
        correlations = np.zeros(L - 1)
        for k in range(L - 1):
            correlations[k] = compute_correlation(mps, number_mpo, r0=0, r1=k + 1)
        logger.info(f"{correlations=}")
        np.savetxt(OUTPUT_PATH + f"correlations_exc_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
                   correlations,
                   delimiter=',')

        logger.info("Compute density-density correlation for largest bond dimension")
        dens_dens_corr = np.zeros(L - 1)
        for k in range(L - 1):
            dens_dens_corr[k] = compute_dens_dens_corr(mps, number_mpo, r=k + 1)
        logger.info(f"{dens_dens_corr=}")
        np.savetxt(OUTPUT_PATH + f"dens_dens_corr_exc_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
                   dens_dens_corr,
                   delimiter=',')

        logger.info("Compute half-chain entanglement spectrum for largest bond dimension")
        entanglement_spectrum = compute_entanglement_spectrum(mps)
        logger.info(f"{entanglement_spectrum=}")
        np.savetxt(OUTPUT_PATH + f"entanglement_spectrum_exc_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
                   entanglement_spectrum,
                   delimiter=',')

        basis_0 = np.array([1, 0])
        dark_state = construct_basis_mps(L, basis=[np.outer(basis_0, basis_0)] * L)
        logger.info(f"Overlap with dark state: {compute_overlap(dark_state, mps)}")  #NOTE: negative?


if __name__ == "__main__":
    main()
