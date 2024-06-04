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
                                 compute_site_expVal_mpo, compute_site_expVal_mps, compute_expVal,
                                 compute_entanglement_spectrum, construct_basis_mps, compute_overlap)

OUTPUT_PATH = "/scratch/nguyed99/qcp-1d/results/"
LOG_PATH = "/scratch/nguyed99/qcp-1d/logging/"

# TN algorithm parameters
conv_eps = 1e-6

### observable operator
number_op = np.array([[0, 0], [0, 1]])


def main():
    # parse environment variables
    L, OMEGA, bond_dim, SLURM_ARRAY_JOB_ID = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), str(sys.argv[4])
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
    eigenvalues, eigentensors, _ = als(lindblad_hermitian, mps, number_ev=1, repeats=10, conv_eps=conv_eps, sigma=0)
    time2 = time.time()
    logger.info(f"Elapsed time: {time2 - time1} seconds")
    logger.info(f"First excited state energy per site: {eigenvalues/L}")
    logger.info(f"Norm of first excited state: {eigentensors.norm()**2}")

    logger.info(f"{eigenvalues=}")

    evp_residual_ground = (lindblad_hermitian @ eigentensors - eigenvalues * eigentensors).norm()**2
    logger.info(f"Eigensolver error for ground state: {evp_residual_ground}")

    np.savetxt(OUTPUT_PATH + f"evp_residual_ground_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
               np.array([evp_residual_ground]),
               fmt='%.6f')

    mps = canonicalize_mps(eigentensors)
    mps_dag = mps.transpose(conjugate=True)

    logger.info(
        f"expVal_0, eval_0_als, eval_0: {compute_expVal(mps, lindblad_hermitian)}, {eigenvalues}, {eigentensors.transpose(conjugate=True) @ lindblad_hermitian @ eigentensors}"
    )

    # compute non-Hermitian part of mps
    non_hermit_mps = (1 / 2) * (mps - mps_dag)
    logger.info(f"The norm of the non-Hermitian part: {non_hermit_mps.norm()**2}")
    np.savetxt(OUTPUT_PATH + f"non_hermit_ground_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
               np.array([non_hermit_mps.norm()**2]),
               fmt='%.6f')

    # compute Hermitian part of mps
    hermit_mps = (1 / 2) * (mps + mps_dag)

    # compute observables
    logger.info("Compute particle numbers mps style")
    particle_nums = compute_site_expVal_mps(hermit_mps, number_op)
    logger.info(f"Particle number/site: {particle_nums}")
    n_s = np.mean(particle_nums)
    logger.info(f"Mean Particle number: {n_s}")
    np.savetxt(OUTPUT_PATH + f"n_s_mps_ground_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
               np.array([n_s]),
               fmt='%.6f')

    logger.info("Compute particle numbers mpo style")
    particle_nums = compute_site_expVal_mpo(hermit_mps, number_op)
    logger.info(f"Particle number/site: {particle_nums}")
    n_s = np.mean(particle_nums)
    logger.info(f"Mean Particle number: {n_s}")
    np.savetxt(OUTPUT_PATH + f"n_s_mpo_ground_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
               np.array([n_s]),
               fmt='%.6f')

    # logger.info("Compute spectral gap of L†L")
    # spectral_gaps = abs(eigenvalues[1] - eigenvalues[0])
    # logger.info(f"{spectral_gaps=}")
    # np.savetxt(OUTPUT_PATH + f"spectral_gaps_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
    #            np.array([spectral_gaps]),
    #            fmt='%.6f')

    logger.info("Compute purity of state")
    purities = compute_purity(mps)
    logger.info(f"Purity: {purities}")
    np.savetxt(OUTPUT_PATH + f"purities_ground_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
               np.array([purities]),
               fmt='%.6f')

    logger.info("Compute two-point correlation")
    correlations = np.zeros(L - 1)
    for k in range(L - 1):
        correlations[k] = compute_correlation(mps, number_op, r0=0, r1=k + 1)
    logger.info(f"{correlations=}")
    np.savetxt(OUTPUT_PATH + f"correlations_ground_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
               correlations,
               fmt='%.6f')

    logger.info("Compute density-density correlation")
    dens_dens_corr = np.zeros(L - 1)
    for k in range(L - 1):
        dens_dens_corr[k] = compute_dens_dens_corr(mps, number_op, r=k + 1)
    logger.info(f"{dens_dens_corr=}")
    np.savetxt(OUTPUT_PATH + f"dens_dens_corr_ground_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
               dens_dens_corr,
               fmt='%.6f')

    logger.info("Compute half-chain entanglement spectrum")
    entanglement_spectrum = compute_entanglement_spectrum(mps)
    logger.info(f"{entanglement_spectrum=}")
    np.savetxt(OUTPUT_PATH + f"entanglement_spectrum_ground_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
               entanglement_spectrum,
               fmt='%.6f')

    basis_0 = np.array([1, 0])
    dark_state = construct_basis_mps(L, basis=[np.outer(basis_0, basis_0)] * L)
    overlap = compute_overlap(dark_state, mps)
    logger.info(f"Overlap with dark state: {overlap}")  #NOTE: negative?
    np.savetxt(OUTPUT_PATH + f"darkst_overlap_ground_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
               np.array([overlap]),
               fmt='%.6f')

    # store states and eigenvalues
    # mps_0 = np.empty(L, dtype=object)
    # for i, core in enumerate(eigentensors[0].cores):
    #     mps_0[i] = core

    # mps_1 = np.empty(L, dtype=object)
    # for i, core in enumerate(eigentensors[1].cores):
    #     mps_1[i] = core

    # np.savez_compressed(OUTPUT_PATH + f"states_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.npz", mps_0, mps_1)
    # np.savetxt(OUTPUT_PATH + f"evals_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt", eigenvalues)


if __name__ == "__main__":
    main()
