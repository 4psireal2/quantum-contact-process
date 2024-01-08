"""
Ref: https://www.phys.ens.fr/~dalibard/publi3/osa_93.pdf
"""
import logging
import os

import numpy as np

from tenpy.models.lattice import Chain
from tenpy.models.model import (CouplingModel, MPOModel, NearestNeighborModel)
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite

current_script_directory = os.path.dirname(os.path.abspath(__file__))
log_file_name = current_script_directory + '/time_evolution_tenpy.log'
logger = logging.getLogger(__name__)
logging.basicConfig(filename=log_file_name, level=logging.INFO)

handler = logging.FileHandler(log_file_name, mode='w')  # 'w' mode creates a new file
handler.setLevel(logging.INFO)
logging.getLogger().addHandler(handler)


class QCPModel(CouplingModel, MPOModel, NearestNeighborModel):
    """
    Implementation of part 1 in Section 2B
    """

    def __init__(self, params):
        L = params.get('L')
        omega = params.get('omega')
        gamma = params.get('gamma')
        bc_MPS = params.get('bc_MPS', 'finite')

        site = SpinHalfSite(conserve='None')  # no conserved quantities

        # Add operators to sites
        sigmax = np.array([[0, 1], [1, 0]])
        number_op = np.array([[0, 0], [0, 1]])
        annihilation_op = np.array([[0, 1], [0, 0]])
        site.add_op('sigmax', sigmax)
        site.add_op('number_op', number_op)
        site.add_op('annihilation_op', annihilation_op)

        oned_lattice = Chain(L=L, site=site, bc='open', bc_MPS=bc_MPS)

        CouplingModel.__init__(self, oned_lattice)
        for i in range(L - 1):
            self.add_coupling_term(strength=omega, i=i, j=i + 1, op_i='sigmax', op_j='number_op', plus_hc=False)
            self.add_coupling_term(strength=omega, i=i, j=i + 1, op_i='number_op', op_j='sigmax', plus_hc=False)

        for i in range(L):
            self.add_onsite_term(strength=-gamma * 1j / 2, i=i, op='number_op')

        # generate MPO for DMRG
        MPOModel.__init__(self, oned_lattice, self.calc_H_MPO())

        # generate H_bond for TEBD
        NearestNeighborModel.__init__(self, oned_lattice, self.calc_H_bond())


def wfmc(psi0: np.ndarray, model_params: dict, solver_params: dict) -> list:
    """
    Implementation of part 2 in Section 2B
    """

    # dynamical variables
    dt = solver_params.get('dt', 1)
    n_time_steps = solver_params.get('N_steps', 1)

    qcp_model = QCPModel(model_params)

    psi = MPS.from_product_state(sites=qcp_model.lat.mps_sites(), p_state=psi0)
    logging.info(f"Norm of initial state: {psi.norm}")

    trajs = [[] for _ in range(solver_params['ntraj'])]

    for i in range(solver_params['ntraj']):
        psi_t = psi.copy()

        for _ in range(n_time_steps):
            dp = dt * psi_t.expectation_value('number_op')  # an array
            dp_l = dp / np.sum(dp)

            epsilon = np.random.uniform()

            # usual time evolution - Eq.11
            if np.sum(dp) < epsilon:
                solver = solver_params['solver'](psi_t, qcp_model, solver_params)
                solver.run()
                logging.info(f"Norm of state after 1 time step evolution without jump: {psi_t.norm}")
                logging.info(f"Norm diff: {np.abs(psi_t.norm - (1 - np.sum(dp)**(1/2)))}")

                psi_t.norm = psi_t.norm / (1 - np.sum(dp))**(1 / 2)
                logging.info(f"Norm of state after normalization: {psi_t.norm}")

            else:  # Eq.12 - Only 1 jump
                for site_l in range(model_params['L']):
                    if np.random.uniform() < dp_l[site_l]:
                        logging.info(f"Norm of state before 1 quantum jump: {psi_t.norm}")
                        psi_t.apply_local_op(i=site_l, op='annihilation_op', renormalize=False)
                        break

                logging.info(f"Norm of state after 1 quantum jump: {psi_t.norm}")
                psi_t.norm = psi_t.norm * (dp_l[site_l] / dt)**(1 / 2)
                logging.info(f"Norm of state after normalization: {psi_t.norm}")

        trajs[i].append(psi_t)

    return trajs
