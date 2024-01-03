"""
Functions for calculating various properties of 1D-QCP model
"""

import logging

import numpy as np

from qutip import tensor, qeye
from qutip.operators import num

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def n_t(time: np.array,
        states_t,
        solver: str,
        L: int,
        ntraj: int = 250) -> np.ndarray:  # TODO: Is there a more efficient way? with e_ops?

    n_t = np.zeros(len(time), dtype=float)

    if solver == "mesolve":
        for t in range(len(time)):
            no_of_particles = 0
            for i in range(L):
                no_of_particles += np.trace(states_t[t].ptrace(i) * num(2)).real

            n_t[t] += 1 / L * no_of_particles

    if solver == "mcsolve":
        num_list = []
        for i in range(L):
            op_list = [qeye(2)] * L

            op_list[i] = num(2)
            num_list.append(tensor(op_list))

        for t in range(len(time)):
            no_of_particles = np.zeros(ntraj, dtype=float)
            for traj in range(ntraj):
                no_of_particles[traj] += (1 / L) * float(sum(np.sum(
                    num_list * states_t[traj][t])).real)  # 1st sum to get (L,0),
                # 2nd sum to get no. of alive cells in the chain

            n_t[t] += np.mean(no_of_particles)

    return n_t
