"""
This module simulates the 1D Quantum Contact Process
(see https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.100604)
using QuTiP (see https://qutip.org/docs/latest/index.html)
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from qutip import basis

from models.qcp_model import QCPModel
from analysis.utils import n_t

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

## Test for WFMC method

# set the system parameters
L = 11
GAMMA = 1
OMEGA = 5 * GAMMA

# solver parameter
ntraj = 250

# create initial state
first_half_index = L // 2
if L % 2 == 1:
    second_half_index = L // 2
else:
    second_half_index = L // 2 - 1

# dynamics variables
time = np.linspace(0, 10, 5)

init_state = [basis(dimensions=2, n=0)] * (first_half_index) + [basis(
    dimensions=2, n=1)] + [basis(dimensions=2, n=0)] * (second_half_index)  # n=1: up, n=0: down
qcp_1d = QCPModel(init_state=init_state, omega=OMEGA, gamma=GAMMA, time=time, solver="mcsolve")
states_t = qcp_1d.time_evolution().states
n_t = n_t(time, states_t, solver="mcsolve", L=L)

plt.figure()
plt.plot(time, n_t)
plt.xlabel("t")
plt.ylabel("n(t)")
plt.title("Monte Carlo Solver")
plt.show()
