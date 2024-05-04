import matplotlib.pyplot as plt
import numpy as np

PATH = "/home/psireal42/study/quantum-contact-process-1D/hpc/results/"
time_string = "2024_05_01_16_12_13"

# system parameters
L = 50
d = 2
GAMMA = 1
OMEGAS = np.linspace(0, 10, 10)

# TN algorithm parameters
bond_dims = np.array([8, 16, 20])

# load arrays
spectral_gaps = np.loadtxt(PATH + f"spectral_gaps_L_{L}_{time_string}.txt", delimiter=',')
n_s = np.loadtxt(PATH + f"n_s_L_{L}_{time_string}.txt", delimiter=',')
eval_0 = np.loadtxt(PATH + f"eval_0_L_{L}_{time_string}.txt", delimiter=',')
eval_1 = np.loadtxt(PATH + f"eval_1_L_{L}_{time_string}.txt", delimiter=',')
ent_ent_spectrum = np.loadtxt(PATH + f"ent_ent_spectrum_L_{L}_{time_string}.txt", delimiter=',')
purities = np.loadtxt(PATH + f"purities_L_{L}_{time_string}.txt", delimiter=',')
correlations = np.loadtxt(PATH + f"correlations_L_{L}_{time_string}.txt", delimiter=',')
evp_residual = np.loadtxt(PATH + f"evp_residual_L_{L}_{time_string}.txt", delimiter=',')

# plot spectral gaps
plt.figure()
plt.plot(OMEGAS, spectral_gaps, 'o-')
plt.xlabel(r"$\Omega$")
plt.title(f"{L=}, $\chi=${bond_dims[-1]}")
plt.grid()
plt.tight_layout()
plt.savefig(PATH + f"spectral_gaps_L_{L}_{time_string}.png")

# plot eigenvalues
plt.figure()
for i, bond_dim in enumerate(bond_dims):
    plt.plot(OMEGAS, eval_0[i, :], 'o-', label=f"$\chi=${bond_dim}")
plt.xlabel(r"$\Omega$")
plt.ylabel(r"$\lambda_0$")
plt.legend()
plt.title(f"{L=}")
plt.grid()
plt.tight_layout()
plt.savefig(PATH + f"eval_0_L_{L}_{time_string}.png")

plt.figure()
for i, bond_dim in enumerate(bond_dims):
    plt.plot(OMEGAS, eval_1[i, :], 'o-', label=f"$\chi=${bond_dim}")
plt.xlabel(r"$\Omega$")
plt.ylabel(r"$\lambda_1$")
plt.legend()
plt.title(f"{L=}")
plt.grid()
plt.tight_layout()
plt.savefig(PATH + f"eval_1_L_{L}_{time_string}.png")

# plot eigenvalue spectrum
plt.figure()
for i, omega in enumerate(OMEGAS):
    plt.plot(list(range(bond_dims[-1] * d**2)), ent_ent_spectrum[i, :], 'o-', label=f"$\Omega=${omega}")
plt.xlabel(r"$\lambda_i$")
plt.legend()
plt.title(f"{L=}, $\chi=${bond_dims[-1]}")
plt.grid()
plt.tight_layout()
plt.savefig(PATH + f"ent_ent_spectrum_L_{L}_{time_string}.png")

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
plt.savefig(PATH + f"stationary_density_L_{L}_{time_string}.png")

# plot eigensolver residuals
plt.figure()
for i, bond_dim in enumerate(bond_dims):
    plt.plot(OMEGAS, evp_residual[i, :], 'o-', label=f"$\chi=${bond_dim}")
plt.xlabel(r"$\Omega$")
plt.ylabel(r"$|| \hat{L} \rho_0 - \lambda_0 \rho_0 ||$")
plt.legend()
plt.title(f"{L=}")
plt.grid()
plt.tight_layout()
plt.savefig(PATH + f"evp_residual_L_{L}_{time_string}.png")

# plot purities
plt.figure()
plt.plot(OMEGAS, purities, 'o-')
plt.xlabel(r"$\Omega$")
plt.ylabel(r"tr($\rho^{2}$)")
plt.title(f"{L=}, $\chi=${bond_dims[-1]}")
plt.grid()
plt.tight_layout()
plt.savefig(PATH + f"purities_L_{L}_{time_string}.png")

# plot correlations
plt.figure()
for i, OMEGA in enumerate(OMEGAS):
    plt.plot(list(range(1, L)), correlations[i, :], 'o-', label=f"$\Omega=${OMEGA}")
plt.xlabel("r")
plt.ylabel(r"$|C_{nn}(r)|$")
plt.legend()
plt.title(f"{L=}, $\chi=${bond_dims[-1]}")
plt.grid()
plt.tight_layout()
plt.savefig(PATH + f"correlations_L_{L}_{time_string}.png")
