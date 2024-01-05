"""
This module visualizes various (time-dependent) properties of  the 1D Quantum Contact Process
(see https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.100604)
using QuTiP (see https://qutip.org/docs/latest/index.html)
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from qutip import (tensor, qeye, num)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def density_plot_t(time: np.ndarray, L: int, states_t: np.ndarray, ntraj: int):
    """
    Density plot of the site-resolved average density for quantum trajectories
    starting from a single seed
    """

    density_t = np.zeros((len(time), L))
    num_list = []
    for i in range(L):
        op_list = [qeye(2)] * L

        op_list[i] = num(2)
        num_list.append(tensor(op_list))

    for t in range(len(time)):
        particle_numbers_trajs = np.zeros(L)
        for traj in range(ntraj):
            particle_numbers = states_t[traj][t].dag() * num_list * states_t[traj][t]
            particle_numbers_trajs += np.array(
                [np.real(diagonal_element.full()[0, 0]) for diagonal_element in particle_numbers])

        density_t[t] = (1 / ntraj / L) * particle_numbers_trajs

    plt.imshow(density_t, aspect='auto', cmap='viridis', extent=[1, L, time[-1], time[0]])
    plt.colorbar(label='Average Particle Number')
    plt.xlabel('Site Index')
    plt.ylabel('Time')
    plt.title('Density Plot of Particle Numbers')
    plt.show()
