import matplotlib.pyplot as plt
import numpy as np

PATH = "/home/psireal42/study/quantum-contact-process-1D/hpc/results/"

# system parameters
L = 10
GAMMA = 1
OMEGAS = np.linspace(0, 10, 10)

# TN algorithm parameters
bond_dims = np.array([8, 16, 20])
conv_eps = 1e-6

# load arrays
spectral_gaps = np.loadtxt(PATH + f"spectral_gaps_L_{L}.txt", delimiter=',')
n_s = np.loadtxt(PATH + f"n_s_L_{L}.txt", delimiter=',')
purities = np.loadtxt(PATH + f"purities_L_{L}.txt", delimiter=',')
correlations = np.loadtxt(PATH + f"correlations_L_{L}.txt", delimiter=',')
evp_residual = np.loadtxt(PATH + f"evp_residual_L_{L}.txt", delimiter=',')

# plot spectral gaps
plt.figure()
plt.plot(OMEGAS, spectral_gaps, 'o-')
plt.xlabel(r"$\Omega$")
plt.title(f"{L=}, $\chi=${bond_dims[-1]}")
plt.grid()
plt.tight_layout()
plt.savefig(PATH + f"spectral_gaps_L_{L}.png")

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
plt.savefig(PATH + f"stationary_density_L_{L}.png")

# plot eigensolver residuals
# plot stationary densities
plt.figure()
for i, bond_dim in enumerate(bond_dims):
    plt.plot(OMEGAS, evp_residual[i, :], 'o-', label=f"$\chi=${bond_dim}")
plt.xlabel(r"$\Omega$")
plt.ylabel(r"$|| \hat{L} \rho_0 - \lambda_0 \rho_0 ||$")
plt.legend()
plt.title(f"{L=}")
plt.grid()
plt.tight_layout()
plt.savefig(PATH + f"evp_residual_L_{L}.png")

# plot purities
plt.figure()
plt.plot(OMEGAS, purities, 'o-')
plt.xlabel(r"$\Omega$")
plt.ylabel(r"tr($\rho^{2}$)")
plt.title(f"{L=}, $\chi=${bond_dims[-1]}")
plt.grid()
plt.tight_layout()
plt.savefig(PATH + f"purities_L_{L}.png")

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
plt.savefig(PATH + f"correlations_L_{L}.png")