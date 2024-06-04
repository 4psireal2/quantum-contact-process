import numpy as np
import matplotlib.pyplot as plt

ground_ID = "348185"
exc_ID = "348186"

PATH = "/home/psireal42/study/quantum-contact-process-1D/hpc/results/"

L = 10
OMEGAS = np.arange(0, 11, 0.9)

# n_s_ground_D_10_mps = []
# n_s_ground_D_25_mps = []
# n_s_ground_D_10_mpo = []
# n_s_ground_D_25_mpo = []

# evp_res_ground_10 = []
# evp_res_ground_25 = []
# evp_res_ground_10_mix = []
# evp_res_ground_25_mix = []

# non_hermit_ground_D_10 = []
# non_hermit_ground_D_25 = []

darkst_overlap_ground_D_10 = []
darkst_overlap_ground_D_25 = []

for OMEGA in OMEGAS:
    darkst_overlap_ground_D_10.append(np.loadtxt(PATH + f"darkst_overlap_ground_L_10_D_10_O_{OMEGA}_{ground_ID}.txt"))
    darkst_overlap_ground_D_25.append(np.loadtxt(PATH + f"darkst_overlap_ground_L_10_D_25_O_{OMEGA}_{ground_ID}.txt"))
#     evp_res_ground_10.append(np.loadtxt(PATH + f"evp_residual_ground_L_10_D_10_O_{OMEGA}_{ground_ID}.txt"))
#     evp_res_ground_25.append(np.loadtxt(PATH + f"evp_residual_ground_L_10_D_25_O_{OMEGA}_{ground_ID}.txt"))
#     evp_res_ground_10_mix.append(np.loadtxt(PATH + f"evp_residual_L_10_D_10_O_{OMEGA}_{exc_ID}.txt")[0])
#     evp_res_ground_25_mix.append(np.loadtxt(PATH + f"evp_residual_L_10_D_25_O_{OMEGA}_{exc_ID}.txt")[0])

#     n_s_ground_D_10_mps.append(np.loadtxt(PATH + f"n_s_mps_ground_L_10_D_10_O_{OMEGA}_{ground_ID}.txt"))
#     n_s_ground_D_25_mps.append(np.loadtxt(PATH + f"n_s_mps_ground_L_10_D_25_O_{OMEGA}_{ground_ID}.txt"))
#     n_s_ground_D_10_mpo.append(np.loadtxt(PATH + f"n_s_mpo_ground_L_10_D_10_O_{OMEGA}_{ground_ID}.txt"))
#     n_s_ground_D_25_mpo.append(np.loadtxt(PATH + f"n_s_mpo_ground_L_10_D_25_O_{OMEGA}_{ground_ID}.txt"))

#     non_hermit_ground_D_10.append(np.loadtxt(PATH + f"non_hermit_ground_L_10_D_10_O_{OMEGA}_{ground_ID}.txt"))
#     non_hermit_ground_D_25.append(np.loadtxt(PATH + f"non_hermit_ground_L_10_D_25_O_{OMEGA}_{ground_ID}.txt"))

plt.figure()
plt.plot(OMEGAS, darkst_overlap_ground_D_10, '--', label=r'$\chi$=10')
plt.plot(OMEGAS, darkst_overlap_ground_D_25, ':', label=r'$\chi$=25')
plt.xlabel(r"$\Omega$")
plt.title('Ground state')
plt.ylabel(r"Overlap with dark state")
plt.legend()
plt.savefig(PATH + f'darkst_overlap_ground_new_L_{L}.png')

# plt.figure()
# plt.plot(OMEGAS, evp_res_ground_10, '--', label=r'$\chi$=10, only ground')
# plt.plot(OMEGAS, evp_res_ground_25, ':', label=r'$\chi$=25, only ground')
# plt.plot(OMEGAS, evp_res_ground_10_mix, '<', label=r'$\chi$=10, mix')
# plt.plot(OMEGAS, evp_res_ground_25_mix, '*', label=r'$\chi$=25, mix')
# plt.xlabel(r"$\Omega$")
# plt.title('Ground state')
# plt.ylabel(r"$||L^\dagger L \rho_0 - \lambda_0 \rho_0||$")
# plt.legend()
# plt.savefig(PATH + f'residual_ground_new_L_{L}.png')

# plt.figure()
# plt.plot(OMEGAS, n_s_ground_D_10_mps, '--', label=r'$\chi$=10, mps')
# plt.plot(OMEGAS, n_s_ground_D_25_mps, ':', label=r'$\chi$=25, mps')
# plt.plot(OMEGAS, n_s_ground_D_10_mpo, '<', label=r'$\chi$=10, mpo')
# plt.plot(OMEGAS, n_s_ground_D_25_mpo, '*', label=r'$\chi$=25, mpo')
# plt.xlabel(r"$\Omega$")
# plt.title('Ground state')
# plt.ylabel(r"Stationary density")
# plt.legend()
# plt.savefig(PATH + f'n_s_ground_new_L_{L}.png')

# plt.figure()
# plt.plot(OMEGAS, non_hermit_ground_D_10, '--', label=r'$\chi$=10')
# plt.plot(OMEGAS, non_hermit_ground_D_25, ':', label=r'$\chi$=25')
# plt.xlabel(r"$\Omega$")
# plt.title('Ground state')
# plt.ylabel(r"Non-hermitian contribution")
# plt.legend()
# plt.savefig(PATH + f'non_hermit_ground_new_L_{L}.png')

########## excited state

# evp_res_exc_10 = []
# evp_res_exc_25 = []

# non_hermit_exc_D_10 = []
# non_hermit_exc_D_25 = []

# n_s_exc_D_10_mps = []
# n_s_exc_D_25_mps = []
# n_s_exc_D_10_mpo = []
# n_s_exc_D_25_mpo = []

# darkst_overlap_D_10= []
# darkst_overlap_D_25 = []

# for OMEGA in OMEGAS:
#     evp_res_exc_10.append(np.loadtxt(PATH + f"evp_residual_L_10_D_10_O_{OMEGA}_{exc_ID}.txt")[1])
#     evp_res_exc_25.append(np.loadtxt(PATH + f"evp_residual_L_10_D_25_O_{OMEGA}_{exc_ID}.txt")[1])

#     non_hermit_exc_D_10.append(np.loadtxt(PATH + f"non_hermit_exc_L_10_D_10_O_{OMEGA}_{exc_ID}.txt"))
#     non_hermit_exc_D_25.append(np.loadtxt(PATH + f"non_hermit_exc_L_10_D_25_O_{OMEGA}_{exc_ID}.txt"))

#     n_s_exc_D_10_mps.append(np.loadtxt(PATH + f"n_s_mps_exc_L_10_D_10_O_{OMEGA}_{exc_ID}.txt"))
#     n_s_exc_D_25_mps.append(np.loadtxt(PATH + f"n_s_mps_exc_L_10_D_25_O_{OMEGA}_{exc_ID}.txt"))
#     n_s_exc_D_10_mpo.append(np.loadtxt(PATH + f"n_s_mpo_exc_L_10_D_10_O_{OMEGA}_{exc_ID}.txt"))
#     n_s_exc_D_25_mpo.append(np.loadtxt(PATH + f"n_s_mpo_exc_L_10_D_25_O_{OMEGA}_{exc_ID}.txt"))

#     darkst_overlap_D_10.append(np.loadtxt(PATH + f"darkst_overlap_exc_L_10_D_10_O_{OMEGA}_{exc_ID}.txt"))
#     darkst_overlap_D_25.append(np.loadtxt(PATH + f"darkst_overlap_exc_L_10_D_25_O_{OMEGA}_{exc_ID}.txt"))

# plt.figure()

# plt.plot(OMEGAS, evp_res_exc_10, '<', label=r'$\chi$=10')
# plt.plot(OMEGAS, evp_res_exc_25, '*', label=r'$\chi$=25')
# plt.xlabel(r"$\Omega$")
# plt.title('Excited state')
# plt.ylabel(r"$||L^\dagger L \rho_1 - \lambda_1 \rho_1||$")
# plt.legend()
# plt.savefig(PATH + f'residual_exc_new_L_{L}.png')

# plt.figure()
# plt.plot(OMEGAS, non_hermit_exc_D_10, '--', label=r'$\chi$=10')
# plt.plot(OMEGAS, non_hermit_exc_D_25, ':', label=r'$\chi$=25')
# plt.xlabel(r"$\Omega$")
# plt.title('Excited state')
# plt.ylabel(r"Non-hermitian contribution")
# plt.legend()
# plt.savefig(PATH + f'non_hermit_exc_new_L_{L}.png')

# plt.figure()
# plt.plot(OMEGAS, darkst_overlap_D_10, '--', label=r'$\chi$=10')
# plt.plot(OMEGAS, darkst_overlap_D_25, ':', label=r'$\chi$=25')
# plt.xlabel(r"$\Omega$")
# plt.title('Excited state')
# plt.ylabel(r"Overlap with dark state")
# plt.legend()
# plt.savefig(PATH + f'darkst_overlap_exc_new_L_{L}.png')

# plt.figure()
# plt.plot(OMEGAS, n_s_exc_D_10_mps, '--', label=r'$\chi$=10, mps')
# plt.plot(OMEGAS, n_s_exc_D_25_mps, ':', label=r'$\chi$=25, mps')
# plt.plot(OMEGAS, n_s_exc_D_10_mpo, '<', label=r'$\chi$=10, mpo')
# plt.plot(OMEGAS, n_s_exc_D_25_mpo, '*', label=r'$\chi$=25, mpo')
# plt.xlabel(r"$\Omega$")
# plt.title('Excited state')
# plt.ylabel(r"Stationary density")
# plt.legend()
# plt.savefig(PATH + f'n_s_exc_new_L_{L}.png')
