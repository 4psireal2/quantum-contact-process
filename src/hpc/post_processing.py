import numpy as np
import matplotlib.pyplot as plt
import src.utilities.utils as utils
import scikit_tt.tensor_train as tt

PATH = "/home/psireal42/study/quantum-contact-process-1D/hpc/results/"
SLURM_ARRAY_JOB_ID = "298321"

# system parameters
L = 10
d = 2
GAMMA = 1
OMEGAS = np.arange(0, 11, 0.9)

# TN algorithm parameters
bond_dims = np.array([10, 25])

spectral_gaps_10 = []
n_s_10 = []
evp_res_10 = []
purities_10 = []
evp_res_10 = [1.07e-19, 0.140, 0.285, 0.446, 0.546, 0.745, 0.967, 1.299, 1.616, 1.419, 1.750, 1.982, 2.261]
for OMEGA in OMEGAS:
    spectral_gaps_10.append(
        np.diff(np.loadtxt(PATH + f"evals_L_{L}_D_{bond_dims[0]}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt")))
    n_s_10.append(np.loadtxt(PATH + f"n_s_exc_L_{L}_D_{bond_dims[0]}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt"))
    # purities_10.append(np.loadtxt(PATH + f"purities_exc_L_{L}_D_{bond_dims[0]}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt"))

spectral_gaps_25 = []
n_s_25 = []
evp_res_25 = []
purities_25 = []
evp_res_25 = [4.823e-24, 0.450, 0.281, 0.389, 0.495, 0.658, 0.893, 1.220, 1.655, 2.199, 2.867, 3.631, 4.526]
for OMEGA in OMEGAS:
    spectral_gaps_25.append(
        np.diff(np.loadtxt(PATH + f"evals_L_{L}_D_{bond_dims[1]}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt")))
    n_s_25.append(np.loadtxt(PATH + f"n_s_exc_L_{L}_D_{bond_dims[1]}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt"))
    # purities_25.append(np.loadtxt(PATH + f"purities_exc_L_{L}_D_{bond_dims[1]}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.txt"))

# plt.figure()
# plt.plot(OMEGAS, evp_res_10, label=r'residual_$\chi$_10')
# plt.plot(OMEGAS, evp_res_25, label=r'residual_$\chi$_25')
# plt.xlabel(r'$\Omega$')
# plt.legend()
# plt.savefig(PATH + f'residual_exc_L_{L}.png')

# plt.figure()
# plt.plot(OMEGAS, n_s_10, label='n_s_$\chi$_10')
# plt.plot(OMEGAS, n_s_25, label='n_s_$\chi$_25')
# plt.xlabel(r'$\Omega$')
# plt.title(r'$\frac{1}{L}\sum_k \rho N_k$')
# plt.legend()
# plt.savefig(PATH + f'n_s_exc_L_{L}.png')

bonddim = 10
number_op = np.array([[0, 0], [0, 1]])

chains = []
for OMEGA in OMEGAS:
    with np.load(PATH + f'states_L_{L}_D_{bonddim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.npz', allow_pickle=True) as data:
        mps_load = data['arr_1']

    mps_reformat = [None] * L
    for i in range(L):
        mps_reformat[i] = mps_load[i]
    mps_reformat = utils.canonicalize_mps(tt.TT(mps_reformat))

    chains.append(mps_reformat)

n_s_10 = [utils.compute_site_expVal_mps(mps_i, number_op) for mps_i in chains]

L = 10
bonddim = 25
number_op = np.array([[0, 0], [0, 1]])

chains = []
for OMEGA in OMEGAS:
    with np.load(PATH + f'states_L_{L}_D_{bonddim}_O_{OMEGA}_{SLURM_ARRAY_JOB_ID}.npz', allow_pickle=True) as data:
        mps_load = data['arr_1']

    mps_reformat = [None] * L
    for i in range(L):
        mps_reformat[i] = mps_load[i]
    mps_reformat = utils.canonicalize_mps(tt.TT(mps_reformat))

    chains.append(mps_reformat)

n_s_25 = [utils.compute_site_expVal_mps(mps_i, number_op) for mps_i in chains]

plt.figure()
plt.plot(OMEGAS, n_s_10, label='n_s_$\chi$_10')
plt.plot(OMEGAS, n_s_25, label='n_s_$\chi$_25')
plt.xlabel(r'$\Omega$')
plt.title('MPS style')
plt.legend()
plt.savefig(PATH + f'n_s_exc_L_{L}_mps.png')
