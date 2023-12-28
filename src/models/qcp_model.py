"""1D-QCP model"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from qutip import (mcsolve, mesolve, tensor, qeye)
from qutip.operators import num, sigmax, destroy
from qutip.qobj import Qobj
from qutip.solver import Options, Result
from qutip.superoperator import lindblad_dissipator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class TimePropagator:
    hermitian_op: Qobj
    non_hermitian_op: Qobj


@dataclass
class QCPModel:
    """
    Ref: https://link.aps.org/doi/10.1103/PhysRevLett.123.100604
    """
    init_state: field(default_factory=List[Qobj])
    omega: field(default_factory=float)
    gamma: field(default_factory=float)
    time: field(default_factory=np.array)
    solver: field(default_factory=str)  # either "mesolve" or "mcsolve"
    rhs_reuse: Optional[bool] = False  # set True for calculation of various initial state
    ntraj: Optional[int] = 250  # for mcsolve
    shared_op: [] = None

    def __post_init__(self):
        if not (self.solver == "mesolve" or self.solver == "mcsolve"):
            raise ValueError("Only 2 solvers are implemented: mesolve and mcsolve")

    def time_propagators(self):
        L = len(self.init_state)
        sx_list, num_list, destroy_list = [], [], []
        for i in range(L):
            op_list = [qeye(2)] * L

            op_list[i] = sigmax()
            sx_list.append(tensor(op_list))
            op_list[i] = num(2)
            num_list.append(tensor(op_list))

            op_list[i] = destroy(2)
            destroy_list.append(tensor(op_list))

        self.shared_op = num_list

        H = 0
        for i in range(L - 1):
            H += self.omega * (sx_list[i] * num_list[i + 1] + num_list[i] * sx_list[i + 1])

        dissipation_op = []
        if self.solver == "mesolve":
            for i in range(L):
                dissipation_op.append(self.gamma * lindblad_dissipator(a=destroy_list[i]))

        if self.solver == "mcsolve":
            for i in range(L):
                dissipation_op.append(self.gamma * destroy_list[i])

        return TimePropagator(hermitian_op=H, non_hermitian_op=dissipation_op)

    def time_evolution(self) -> Result:
        psi0 = tensor(self.init_state)
        state0 = psi0

        # setting options for solver
        options = Options(rhs_reuse=self.rhs_reuse, store_states=True)

        if self.solver == "mesolve":
            state0 = psi0 * psi0.dag()

        propagators = self.time_propagators()
        H, dissipation_op = propagators.hermitian_op, propagators.non_hermitian_op
        if self.solver == "mesolve":
            result = mesolve(H, state0, tlist=self.time, c_ops=dissipation_op, options=options, progress_bar=True)

        if self.solver == "mcsolve":
            result = mcsolve(H, state0, tlist=self.time, c_ops=dissipation_op, options=options, progress_bar=True)

        return result

    def n_t(self) -> np.ndarray:  # TODO: Is there a more efficient way? with e_ops?
        n_t = np.zeros(len(self.time), dtype=float)
        L = len(self.init_state)
        states_t = self.time_evolution().states

        if self.solver == "mesolve":
            for t in range(len(self.time)):
                no_of_particles = 0
                for i in range(L):
                    no_of_particles += np.trace(states_t[t].ptrace(i) * num(2)).real

                n_t[t] += 1 / L * no_of_particles

        if self.solver == "mcsolve":
            for t in range(len(self.time)):
                no_of_particles = np.zeros(self.ntraj, dtype=float)
                for traj in range(self.ntraj):
                    no_of_particles[traj] += (1 / L) * float(sum(np.sum(
                        self.shared_op * states_t[traj][t])).real)  # 1st sum to get (L,0),
                    # 2nd sum to get no. of alive cells in the chain

                n_t[t] += np.mean(no_of_particles)

        return n_t
