"""
Ref: https://www.phys.ens.fr/~dalibard/publi3/osa_93.pdf
"""

import numpy as np

from tenpy.algorithms import algorithm
from tenpy.models.lattice import Chain
from tenpy.models.model import (CouplingModel, MPOModel, NearestNeighborModel)
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite


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
        for i in range(L):
            self.add_onsite_term(strength=-gamma * 1j / 2, i=i, op='number_op')

        # generate MPO for DMRG
        MPOModel.__init__(self, oned_lattice, self.calc_H_MPO())

        # generate H_bond for TEBD
        NearestNeighborModel.__init__(self, oned_lattice, self.calc_H_bond())


def wfmc(psi0: np.ndarray, final_time: float, model_params: dict, TEBD_params: dict, ntraj: int,
         engine: algorithm) -> str:
    """
    Implementation of part 2 in Section 2B
    """

    # dynamical variables
    dt = TEBD_params.get('dt', 1)
    n_time_steps = int(final_time // dt)

    qcp_model = QCPModel(model_params)

    psi = MPS.from_product_state(sites=qcp_model.lat.mps_sites(), p_state=psi0)
    psi_t = psi.copy()
    trajs = [[] for _ in range(ntraj)]
    for i in range(ntraj):
        for _ in range(n_time_steps):
            dp = dt * psi_t.psi_t.expectation_value('number_op')
            dp_l = dp / np.sum(dp)

            epsilon = np.random.uniform()

            # usual time evolution - Eq.11
            if np.sum(dp) < epsilon:
                engine = engine(psi_t, qcp_model, TEBD_params)
                engine.run()
                psi_t.norm = psi_t.norm / (1 - np.sum(dp))**(1 / 2)  #TODO: How do TENPY ppl do renormalization
            else:  # Eq.12
                for site_l in range(model_params['L']):
                    if np.random.uniform() < dp_l[site_l]:
                        psi_t.apply_local_op(i=site_l, op='annihilation_op', renormalize=False)
                        psi_t.norm = psi_t.norm * (dp_l[site_l] / dt)**(1 / 2)

        trajs[i].append(psi_t)


chain = QCPModel({'omega': 6, 'gamma': 1, 'L': 10, 'bc_MPS': 'finite', 'explicit_plus_hc': False})

basis_0 = np.array([1, 0])
basis_1 = np.array([0, 1])
