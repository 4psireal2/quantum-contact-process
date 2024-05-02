import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

import scikit_tt.tensor_train as tt
from scikit_tt.solvers.ode import hod

from src.models.contact_process_model import (construct_lindblad, construct_num_op)
from src.utilities.utils import (canonicalize_mps, compute_site_expVal)

# # system parameters
L = 21
OMEGA = 6.0

# # dynamics parameter
step_size = 1
number_of_steps = 7
max_rank = 50

lindblad = construct_lindblad(gamma=1.0, omega=OMEGA, L=L)
lindblad_hermitian = lindblad.transpose(conjugate=True) @ lindblad
num_op = construct_num_op(L)
mps = tt.unit(L * [4], inds=L // 2 * [0] + [3] + L // 2 * [0])
mps_t = hod(lindblad_hermitian,
            mps,
            step_size=step_size,
            number_of_steps=number_of_steps,
            normalize=2,
            max_rank=max_rank)

t = np.linspace(0, step_size * number_of_steps, number_of_steps + 1)
particle_num_t = np.zeros((len(t), L))

for i in range(len(t)):
    hermit_mps = deepcopy(mps_t[i])
    hermit_mps = canonicalize_mps(hermit_mps)
    mps_dag = hermit_mps.transpose(conjugate=True)
    for k in range(L):
        hermit_mps.cores[k] = (hermit_mps.cores[k] + mps_dag.cores[k]) / 2
    particle_num_t[i, :] = compute_site_expVal(hermit_mps, construct_num_op(L))

plt.figure()
plt.imshow(particle_num_t)
plt.xticks([])
plt.colorbar()
plt.tight_layout()
plt.show()
