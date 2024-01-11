"""
This module simulates the 1D Quantum Contact Process
(see https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.100604)
using QuTiP (see https://qutip.org/docs/latest/index.html)
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, qload

from utils.analysis import n_t
from utils.visualization import density_plot_t

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

current_script_directory = os.path.dirname(os.path.abspath(__file__))

## Test for WFMC method

# set the system parameters
L = 11
GAMMA = 1
OMEGA = 6 * GAMMA

# solver parameter
ntraj = 250

# create initial state
first_half_index = L // 2
if L % 2 == 1:
    second_half_index = L // 2
else:
    second_half_index = L // 2 - 1
init_state = [basis(dimensions=2, n=0)] * (first_half_index) + [basis(
    dimensions=2, n=1)] + [basis(dimensions=2, n=0)] * (second_half_index)  # n=1: up, n=0: down

# dynamics variables
time = np.linspace(0, 10, 5)

# ## dynamics with MC
# qcp_1d = QCPModel(init_state=init_state, omega=OMEGA, gamma=GAMMA, time=time, solver="mcsolve")
# states_t = qcp_1d.time_evolution().states
# qsave(data=states_t, name=current_script_directory + "/MC_states_t")
states_t = qload(name=current_script_directory + "/MC_states_t")
n_t = n_t(time, states_t, solver="mcsolve", L=L)
density_plot_t(time=time, L=L, states_t=states_t, ntraj=ntraj)

plt.figure()
plt.plot(time, n_t)
plt.xlabel("t")
plt.ylabel("n(t)")
plt.title("Monte Carlo Solver")
plt.show()

## dynamics with ME
# qcp_1d = QCPModel(init_state=init_state, omega=OMEGA, gamma=GAMMA, time=time, solver="mesolve")
# states_t = qcp_1d.time_evolution().states
# qsave(data=states_t, name=current_script_directory + "/ME_states_t")
# n_t = n_t(time, states_t, solver="mesolve", L=L)

# plt.figure()
# plt.plot(time, n_t)
# plt.xlabel("t")
# plt.ylabel("n(t)")
# plt.title("ME Solver")
# plt.show()
