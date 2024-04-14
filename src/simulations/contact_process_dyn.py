import numpy as np

import scikit_tt.tensor_train as tt
from scikit_tt.solvers.ode import hod

from src.models.contact_process_model import (construct_lindblad, construct_num_op)

# # system parameters
L = 51
OMEGA = 6.0
GAMMA = 1.0

# # dynamics parameter
step_size = 1
number_of_steps = 10
max_rank = 100

lindblad = construct_lindblad(gamma=GAMMA, omega=OMEGA, L=L)
lindblad_hermitian = lindblad.transpose(conjugate=True) @ lindblad
num_op = construct_num_op(L)
mps = tt.unit(L * [4], inds=25 * [0] + [3] + 25 * [0])
mps_t = hod(lindblad_hermitian,
            mps,
            step_size=step_size,
            number_of_steps=number_of_steps,
            normalize=2,
            max_rank=max_rank)

t = np.linspace(0, step_size * number_of_steps, number_of_steps)
particle_num_t = np.zeros((len(t), L))
