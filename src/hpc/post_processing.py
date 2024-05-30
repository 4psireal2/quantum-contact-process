import numpy as np
import matplotlib.pyplot as plt
import src.utilities.utils as utils
import scikit_tt.tensor_train as tt
from src.models.contact_process_model import (construct_lindblad)

PATH = "/home/psireal42/study/quantum-contact-process-1D/hpc/results/"
SLURM_ARRAY_JOB_ID = "298321"

# system parameters
L = 10
d = 2
GAMMA = 1
OMEGAS = np.arange(0, 11, 0.9)

basis_0 = np.array([1, 0])
dark_state = utils.construct_basis_mps(L, basis=[np.outer(basis_0, basis_0)] * L)

# TN algorithm parameters
bond_dims = np.array([10, 25])

# spectral_gaps_10 = []
# evp_res_10 = [1.07e-19, 0.140, 0.285, 0.446, 0.546, 0.745, 0.967, 1.299, 1.616, 1.419, 1.750, 1.982, 2.261]
# for OMEGA in OMEGAS:
#     spectral_gaps_10.append(
#         np.diff(np.loadtxt(PATH + f"evals_L_{L}_D_{bond_dims[0]}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt")))
#     n_s_10.append(np.loadtxt(PATH + f"n_s_exc_L_{L}_D_{bond_dims[0]}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt"))

# spectral_gaps_25 = []
# evp_res_25 = [4.823e-24, 0.450, 0.281, 0.389, 0.495, 0.658, 0.893, 1.220, 1.655, 2.199, 2.867, 3.631, 4.526]
# for OMEGA in OMEGAS:
#     spectral_gaps_25.append(
#         np.diff(np.loadtxt(PATH + f"evals_L_{L}_D_{bond_dims[1]}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt")))
#     n_s_25.append(np.loadtxt(PATH + f"n_s_exc_L_{L}_D_{bond_dims[1]}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt"))

# plt.figure()
# plt.plot(OMEGAS, evp_res_10, label=r'$\chi$_10')
# plt.plot(OMEGAS, evp_res_25, label=r'$\chi$_25')
# plt.xlabel(r"$\Omega$")
# plt.title('Excited state')
# plt.ylabel(r"$||L^\dagger L \rho_1 - \lambda_1 \rho_1||$")
# plt.legend()
# plt.savefig(PATH + f'residual_exc_L_{L}.png')

# plt.figure()
# plt.plot(OMEGAS, spectral_gaps_10, label=r'$\chi$_10')
# plt.plot(OMEGAS, spectral_gaps_25, label=r'$\chi$_25')
# plt.xlabel(r"$\Omega$")
# plt.ylabel(r"$|\lambda_2 - \lambda_1|$")
# plt.legend()
# plt.savefig(PATH + f'spectral_gap_L_{L}.png')

############# observables for GROUND state
# bonddim = 10
# # number_op = np.array([[0, 0], [0, 1]])

# chains = []
# evp_res_10_ground = []
# var_10_ground = []
# for OMEGA in OMEGAS:
#     with np.load(PATH + f'states_L_{L}_D_{bonddim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.npz', allow_pickle=True) as data:
#         mps_load = data['arr_0']

#     mps_reformat = [None] * L
#     for i in range(L):
#         mps_reformat[i] = mps_load[i]

#     mps_reformat = tt.TT(mps_reformat)

#     chains.append(mps_reformat)

# for i, mps_i in enumerate(chains):
#     eigenvalues = np.loadtxt(PATH + f"evals_L_{L}_D_{bonddim}_O_{OMEGAS[i]}_{SLURM_ARRAY_JOB_ID}.txt")
#     lindblad = construct_lindblad(gamma=1.0, omega=OMEGAS[i], L=L)
#     lindblad_hermitian = lindblad.transpose(conjugate=True) @ lindblad
#     diff_tensor = lindblad_hermitian @ mps_i - eigenvalues[0] * mps_i

#     evp_res_10_ground.append(diff_tensor.norm()**2)

# purities_10_ground = [utils.compute_purity(mps_i) for mps_i in chains]
# n_s_10_ground_MPS = [np.mean(utils.compute_site_expVal_mps(mps_i, number_op)) for mps_i in chains]
# n_s_10_ground_MPO = [np.mean(utils.compute_site_expVal_mpo(mps_i, number_op)) for mps_i in chains]
# overlap_10_ground = [utils.compute_overlap(dark_state, mps_i) for mps_i in chains]

# bonddim = 25

# chains = []
# evp_res_25_ground = []
# for OMEGA in OMEGAS:
#     with np.load(PATH + f'states_L_{L}_D_{bonddim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.npz', allow_pickle=True) as data:
#         mps_load = data['arr_0']

#     mps_reformat = [None] * L
#     for i in range(L):
#         mps_reformat[i] = mps_load[i]
#     # mps_reformat = utils.canonicalize_mps(tt.TT(mps_reformat))
#     mps_reformat = tt.TT(mps_reformat)

#     chains.append(mps_reformat)

# for i, mps_i in enumerate(chains):
#     eigenvalues = np.loadtxt(PATH + f"evals_L_{L}_D_{bonddim}_O_{OMEGAS[i]}_{SLURM_ARRAY_JOB_ID}.txt")
#     lindblad = construct_lindblad(gamma=1.0, omega=OMEGAS[i], L=L)
#     lindblad_hermitian = lindblad.transpose(conjugate=True) @ lindblad
#     diff_tensor = lindblad_hermitian @ mps_i - eigenvalues[0] * mps_i
#     evp_res_25_ground.append(diff_tensor.norm()**2)

# purities_25_ground = [utils.compute_purity(mps_i) for mps_i in chains]
# n_s_25_ground_MPS = [np.mean(utils.compute_site_expVal_mps(mps_i, number_op)) for mps_i in chains]
# n_s_25_ground_MPO = [np.mean(utils.compute_site_expVal_mpo(mps_i, number_op)) for mps_i in chains]

# overlap_25_ground = [utils.compute_overlap(dark_state, mps_i) for mps_i in chains]

# plt.figure()
# plt.plot(OMEGAS, evp_res_10_ground, label=r'$\chi$_10')
# plt.plot(OMEGAS, evp_res_25_ground, label=r'$\chi$_25')
# plt.xlabel(r"$\Omega$")
# plt.ylabel(r"$||L^\dagger L \rho_0 - \lambda_0 \rho_0||$")
# plt.title('Ground state')
# plt.legend()
# plt.savefig(PATH + f'residual_ground_L_{L}.png')

# plt.figure()
# plt.plot(OMEGAS, purities_10_ground, label=r'$\chi$_10')
# plt.plot(OMEGAS, purities_25_ground, label=r'$\chi$_25')
# plt.xlabel(r"$\Omega$")
# plt.ylabel(r"$Tr(\rho \cdot \rho)$")
# plt.title('Ground state')
# plt.legend()
# plt.savefig(PATH + f'purity_ground_L_{L}.png')

# plt.figure()
# plt.plot(OMEGAS, n_s_10_ground_MPS, '--', label=r'$\chi_{10}$, MPS')
# plt.plot(OMEGAS, n_s_10_ground_MPO, label=r'$\chi_{10}$, MPO')
# plt.plot(OMEGAS, n_s_25_ground_MPS, '<', label=r'$\chi_{25}$, MPS')
# plt.plot(OMEGAS, n_s_25_ground_MPO, '*', label=r'$\chi_{25}$, MPO')
# plt.xlabel(r"$\Omega$")
# plt.ylabel("Stationary density")
# plt.title('Ground state')
# plt.legend()
# plt.savefig(PATH + f'n_s_ground_L_{L}.png')

# plt.figure()
# plt.plot(OMEGAS, overlap_10_ground, label=r'$\chi_{10}$')
# plt.plot(OMEGAS, overlap_25_ground, label=r'$\chi_{25}$')

# plt.xlabel(r"$\Omega$")
# plt.ylabel("Overlap with dark state")
# plt.title('Ground state')
# plt.legend()
# plt.savefig(PATH + f'overlap_ground_L_{L}.png')

############# more observables for EXCITED state
bonddim = 10
# number_op = np.array([[0, 0], [0, 1]])

chains = []
evp_res_10_exc = []

for OMEGA in OMEGAS:
    with np.load(PATH + f'states_L_{L}_D_{bonddim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.npz', allow_pickle=True) as data:
        mps_load = data['arr_1']

    mps_reformat = [None] * L
    for i in range(L):
        mps_reformat[i] = mps_load[i]
    # mps_reformat = utils.canonicalize_mps(tt.TT(mps_reformat))
    mps_reformat = tt.TT(mps_reformat)

    chains.append(mps_reformat)

for i, mps_i in enumerate(chains):
    eigenvalues = np.loadtxt(PATH + f"evals_L_{L}_D_{bonddim}_O_{OMEGAS[i]}_{SLURM_ARRAY_JOB_ID}.txt")
    lindblad = construct_lindblad(gamma=1.0, omega=OMEGAS[i], L=L)
    lindblad_hermitian = lindblad.transpose(conjugate=True) @ lindblad
    diff_tensor = lindblad_hermitian @ mps_i - eigenvalues[1] * mps_i

    evp_res_10_exc.append(diff_tensor.norm()**2)

# n_s_10_exc_MPS = [np.mean(utils.compute_site_expVal_mps(mps_i, number_op)) for mps_i in chains]
# n_s_10_exc_MPO = [np.mean(utils.compute_site_expVal_mpo(mps_i, number_op)) for mps_i in chains]
# purities_10_exc = [utils.compute_purity(mps_i) for mps_i in chains]
# overlap_10_exc = [utils.compute_overlap(dark_state, mps_i) for mps_i in chains]

bonddim = 25

chains = []
evp_res_25_exc = []

for OMEGA in OMEGAS:
    with np.load(PATH + f'states_L_{L}_D_{bonddim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.npz', allow_pickle=True) as data:
        mps_load = data['arr_1']

    mps_reformat = [None] * L
    for i in range(L):
        mps_reformat[i] = mps_load[i]
    # mps_reformat = utils.canonicalize_mps(tt.TT(mps_reformat))
    mps_reformat = tt.TT(mps_reformat)

    chains.append(mps_reformat)

for i, mps_i in enumerate(chains):
    eigenvalues = np.loadtxt(PATH + f"evals_L_{L}_D_{bonddim}_O_{OMEGAS[i]}_{SLURM_ARRAY_JOB_ID}.txt")
    lindblad = construct_lindblad(gamma=1.0, omega=OMEGAS[i], L=L)
    lindblad_hermitian = lindblad.transpose(conjugate=True) @ lindblad
    diff_tensor = lindblad_hermitian @ mps_i - eigenvalues[1] * mps_i

    evp_res_25_exc.append(diff_tensor.norm()**2)

# n_s_25_exc_MPS = [np.mean(utils.compute_site_expVal_mps(mps_i, number_op)) for mps_i in chains]
# n_s_25_exc_MPO = [np.mean(utils.compute_site_expVal_mpo(mps_i, number_op)) for mps_i in chains]
# purities_25_exc = [utils.compute_purity(mps_i) for mps_i in chains]
# overlap_25_exc = [utils.compute_overlap(dark_state, mps_i) for mps_i in chains]

plt.figure()
plt.plot(OMEGAS, evp_res_10_exc, label=r'$\chi$_10')
plt.plot(OMEGAS, evp_res_25_exc, label=r'$\chi$_25')
plt.xlabel(r"$\Omega$")
plt.title('Excited state')
plt.ylabel(r"$||L^\dagger L \rho_1 - \lambda_1 \rho_1||$")
plt.legend()
plt.savefig(PATH + f'residual_exc_L_{L}.png')

# plt.figure()
# plt.plot(OMEGAS, purities_10_exc, label=r'$\chi$_10')
# plt.plot(OMEGAS, purities_25_exc, label=r'$\chi$_25')
# plt.xlabel(r"$\Omega$")
# plt.ylabel(r"$Tr(\rho \cdot \rho)$")
# plt.title('Excited state')
# plt.legend()
# plt.savefig(PATH + f'purity_exc_L_{L}.png')

# plt.figure()
# plt.plot(OMEGAS, n_s_10_exc_MPS, '--', label=r'$\chi_{10}$, MPS')
# plt.plot(OMEGAS, n_s_10_exc_MPO, label=r'$\chi_{10}$, MPO')
# plt.plot(OMEGAS, n_s_25_exc_MPS, '<', label=r'$\chi_{25}$, MPS')
# plt.plot(OMEGAS, n_s_25_exc_MPO, '*', label=r'$\chi_{25}$, MPO')
# plt.xlabel(r"$\Omega$")
# plt.ylabel("Stationary density")
# plt.title('Excited state')
# plt.legend()
# plt.savefig(PATH + f'n_s_exc_L_{L}.png')

# plt.figure()
# plt.plot(OMEGAS, overlap_10_exc, label=r'$\chi_{10}$')
# plt.plot(OMEGAS, overlap_25_exc, label=r'$\chi_{25}$')

# plt.xlabel(r"$\Omega$")
# plt.ylabel("Overlap with dark state")
# plt.title('Excited state')
# plt.legend()
# plt.savefig(PATH + f'overlap_exc_L_{L}.png')

####### together

# bonddim = 10
# number_op = np.array([[0, 0], [0, 1]])

# chains_10_ground = []
# for OMEGA in OMEGAS:
#     with np.load(PATH + f'states_L_{L}_D_{bonddim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.npz', allow_pickle=True) as data:
#         mps_load = data['arr_0']

#     mps_reformat = [None] * L
#     for i in range(L):
#         mps_reformat[i] = mps_load[i]
#     mps_reformat = utils.canonicalize_mps(tt.TT(mps_reformat))
#     # mps_reformat = tt.TT(mps_reformat)

#     chains_10_ground.append(mps_reformat)

# bonddim = 25
# number_op = np.array([[0, 0], [0, 1]])

# chains_25_ground = []
# for OMEGA in OMEGAS:
#     with np.load(PATH + f'states_L_{L}_D_{bonddim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.npz', allow_pickle=True) as data:
#         mps_load = data['arr_0']

#     mps_reformat = [None] * L
#     for i in range(L):
#         mps_reformat[i] = mps_load[i]
#     mps_reformat = utils.canonicalize_mps(tt.TT(mps_reformat))
#     # mps_reformat = tt.TT(mps_reformat)

#     chains_25_ground.append(mps_reformat)

# bonddim = 10

# chains_10_exc = []
# for OMEGA in OMEGAS:
#     with np.load(PATH + f'states_L_{L}_D_{bonddim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.npz', allow_pickle=True) as data:
#         mps_load = data['arr_1']

#     mps_reformat = [None] * L
#     for i in range(L):
#         mps_reformat[i] = mps_load[i]
#     mps_reformat = utils.canonicalize_mps(tt.TT(mps_reformat))

#     chains_10_exc.append(mps_reformat)

# bonddim = 25

# chains_25_exc = []
# for OMEGA in OMEGAS:
#     with np.load(PATH + f'states_L_{L}_D_{bonddim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.npz', allow_pickle=True) as data:
#         mps_load = data['arr_1']

#     mps_reformat = [None] * L
#     for i in range(L):
#         mps_reformat[i] = mps_load[i]
#     mps_reformat = utils.canonicalize_mps(tt.TT(mps_reformat))

#     chains_25_exc.append(mps_reformat)

# ground_exc_overlap_10 = [utils.compute_overlap(chains_10_ground[i], chains_10_exc[i]) for i in range(len(OMEGAS))]
# ground_exc_overlap_25 = [utils.compute_overlap(chains_25_ground[i], chains_25_exc[i]) for i in range(len(OMEGAS))]

# plt.figure()
# plt.plot(OMEGAS, ground_exc_overlap_10, label=r'$\chi_{10}$')
# plt.plot(OMEGAS, ground_exc_overlap_25, label=r'$\chi_{25}$')
# plt.xlabel(r"$\Omega$")
# plt.ylabel("Overlap between ground and excited state")
# plt.legend()
# plt.savefig(PATH + f'overlaps_L_{L}.png')
