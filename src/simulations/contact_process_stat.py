"""
Do static simulations for first excited state
"""
import numpy as np
import sys
import time

import scikit_tt.tensor_train as tt
from scikit_tt.solvers.evp import als

from src.models.contact_process_model import (construct_lindblad)
from src.utilities.utils import (canonicalize_mps)

OUTPUT_PATH = "/home/psireal42/study/quantum-contact-process-1D/playground/"
LOG_PATH = "/home/psireal42/study/quantum-contact-process-1D/playground/"

# TN algorithm parameters
conv_eps = 1e-6

### observable operator
number_op = np.array([[0, 0], [0, 1]])


def main():
    # parse environment variables
    L, OMEGA, bond_dim = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), str(sys.argv[4])

    print("Stationary simulation")
    print(f"Run ALS for {L=}, {OMEGA=} and {bond_dim=}")
    lindblad = construct_lindblad(gamma=1.0, omega=OMEGA, L=L)
    lindblad_hermitian = lindblad.transpose(conjugate=True) @ lindblad

    mps = tt.ones(row_dims=L * [4], col_dims=L * [1], ranks=bond_dim)
    mps = mps.ortho()
    mps = (1 / mps.norm()**2) * mps

    time1 = time.time()
    eigenvalue, eigentensor, _ = als(lindblad_hermitian, mps, number_ev=1, repeats=10, conv_eps=conv_eps, sigma=0)
    time2 = time.time()
    print(f"Elapsed time: {time2 - time1} seconds")
    print(f"Ground state energy per site: {eigenvalue/L}")
    print(f"Norm of ground state: {eigentensor.norm()**2}")

    evp_residual_ground = (lindblad_hermitian @ eigentensor - eigenvalue * eigentensor).norm()**2
    print(f"Eigensolver error for ground state: {evp_residual_ground}")

    _eigentensor = canonicalize_mps(eigentensor)
    non_hermit = (1 / 2) * (_eigentensor - _eigentensor.transpose(conjugate=True))
    print(f"Non-Hermitian contribution: {non_hermit.norm()}")  # should be close to 0

    print("Search for excited state")
    mps = tt.ones(row_dims=L * [4], col_dims=L * [1], ranks=bond_dim)
    mps = mps.ortho()
    mps = (1 / mps.norm()**2) * mps

    time1 = time.time()
    eval_1, estate_1, _ = als(lindblad_hermitian,
                              initial_guess=mps,
                              previous=[eigentensor],
                              shift=1.0,
                              number_ev=1,
                              repeats=10,
                              conv_eps=conv_eps,
                              sigma=0)
    time2 = time.time()
    print(f"Elapsed time: {time2 - time1} seconds")
    print(f"First excited state energy per site: {eval_1/L}")
    print(f"Norm of first excited state: {estate_1.norm()**2}")

    evp_residual_exc = (lindblad_hermitian @ estate_1 - eval_1 * estate_1).norm()**2
    print(f"Eigensolver error for first excited state: {evp_residual_exc}")

    _estate_1 = canonicalize_mps(estate_1)
    non_hermit = (1 / 2) * (_estate_1 - _estate_1.transpose(conjugate=True))
    print(f"Non-Hermitian contribution: {non_hermit.norm()}")  # should be close to 0

    # np.savetxt(OUTPUT_PATH + f"evp_residual_ground_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
    #            np.array([evp_residual_ground]),
    #            fmt='%.6f')

    # mps = canonicalize_mps(eigentensor)
    # mps_dag = mps.transpose(conjugate=True)

    # print(
    #     f"expVal_0, eval_0_als, eval_0: {compute_expVal(mps, lindblad_hermitian)}, {eigenvalue}, {eigentensor.transpose(conjugate=True) @ lindblad_hermitian @ eigentensor}"
    # )

    # # compute non-Hermitian part of mps
    # non_hermit_mps = (1 / 2) * (mps - mps_dag)
    # print(f"The norm of the non-Hermitian part: {non_hermit_mps.norm()**2}")
    # np.savetxt(OUTPUT_PATH + f"non_hermit_ground_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
    #            np.array([non_hermit_mps.norm()**2]),
    #            fmt='%.6f')

    # # compute Hermitian part of mps
    # hermit_mps = (1 / 2) * (mps + mps_dag)

    # # compute observables
    # print("Compute particle numbers mps style")
    # particle_nums = compute_site_expVal_mps(hermit_mps, number_op)
    # print(f"Particle number/site: {particle_nums}")
    # n_s = np.mean(particle_nums)
    # print(f"Mean Particle number: {n_s}")
    # np.savetxt(OUTPUT_PATH + f"n_s_mps_ground_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
    #            np.array([n_s]),
    #            fmt='%.6f')

    # print("Compute particle numbers mpo style")
    # particle_nums = compute_site_expVal_mpo(hermit_mps, number_op)
    # print(f"Particle number/site: {particle_nums}")
    # n_s = np.mean(particle_nums)
    # print(f"Mean Particle number: {n_s}")
    # np.savetxt(OUTPUT_PATH + f"n_s_mpo_ground_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
    #            np.array([n_s]),
    #            fmt='%.6f')

    # # print("Compute spectral gap of L†L")
    # # spectral_gaps = abs(eigenvalue[1] - eigenvalue[0])
    # # print(f"{spectral_gaps=}")
    # # np.savetxt(OUTPUT_PATH + f"spectral_gaps_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
    # #            np.array([spectral_gaps]),
    # #            fmt='%.6f')

    # print("Compute purity of state")
    # purities = compute_purity(mps)
    # print(f"Purity: {purities}")
    # np.savetxt(OUTPUT_PATH + f"purities_ground_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
    #            np.array([purities]),
    #            fmt='%.6f')

    # print("Compute two-point correlation")
    # correlations = np.zeros(L - 1)
    # for k in range(L - 1):
    #     correlations[k] = compute_correlation(mps, number_op, r0=0, r1=k + 1)
    # print(f"{correlations=}")
    # np.savetxt(OUTPUT_PATH + f"correlations_ground_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
    #            correlations,
    #            fmt='%.6f')

    # print("Compute density-density correlation")
    # dens_dens_corr = np.zeros(L - 1)
    # for k in range(L - 1):
    #     dens_dens_corr[k] = compute_dens_dens_corr(mps, number_op, r=k + 1)
    # print(f"{dens_dens_corr=}")
    # np.savetxt(OUTPUT_PATH + f"dens_dens_corr_ground_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
    #            dens_dens_corr,
    #            fmt='%.6f')

    # print("Compute half-chain entanglement spectrum")
    # entanglement_spectrum = compute_entanglement_spectrum(mps)
    # print(f"{entanglement_spectrum=}")
    # np.savetxt(OUTPUT_PATH + f"entanglement_spectrum_ground_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
    #            entanglement_spectrum,
    #            fmt='%.6f')

    # basis_0 = np.array([1, 0])
    # dark_state = construct_basis_mps(L, basis=[np.outer(basis_0, basis_0)] * L)
    # overlap = compute_overlap(dark_state, mps)
    # print(f"Overlap with dark state: {overlap}")  #NOTE: negative?
    # np.savetxt(OUTPUT_PATH + f"darkst_overlap_ground_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt",
    #            np.array([overlap]),
    #            fmt='%.6f')

    # store states and eigenvalue
    # mps_0 = np.empty(L, dtype=object)
    # for i, core in enumerate(eigentensor[0].cores):
    #     mps_0[i] = core

    # mps_1 = np.empty(L, dtype=object)
    # for i, core in enumerate(eigentensor[1].cores):
    #     mps_1[i] = core

    # np.savez_compressed(OUTPUT_PATH + f"states_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.npz", mps_0, mps_1)
    # np.savetxt(OUTPUT_PATH + f"evals_L_{L}_D_{bond_dim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt", eigenvalue)


if __name__ == "__main__":
    main()
