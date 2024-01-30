import numpy as np

from models.qcp_model_tenpy import wfmc

from tenpy.algorithms.tebd import TEBDEngine

# set the system parameters
L = 11
model_params = {'L': L, 'gamma': 1, 'omega': 6, 'bc_MPS': 'finite', 'explicit_plus_hc': False}

# set the solver parameters
solver_params = {
    'ntraj': 1,
    'preserve_norm': False,
    'n_time_steps': 6,  # total number of time steps
    'N_steps': 1,  # for engine
    'solver': TEBDEngine,
    'dt': 0.005,
    'trunc_params': {
        'chi_max': 350,
        'svd_min': 1.e-12
    }
}

basis_0 = np.array([1, 0])
basis_1 = np.array([0, 1])

first_half_index = L // 2
if L % 2 == 1:
    second_half_index = L // 2
else:
    second_half_index = L // 2 - 1
init_state = [basis_0] * (first_half_index) + [basis_1] + [basis_0] * (second_half_index)

trajs = wfmc(psi0=init_state, model_params=model_params, solver_params=solver_params)
