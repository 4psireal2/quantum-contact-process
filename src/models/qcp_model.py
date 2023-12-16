"""1D-QCP model"""

import logging
from dataclasses import dataclass

import numpy as np

from tenpy.models.lattice import Chain
from tenpy.models.model import MPOModel, NearestNeighborModel, CouplingModel
from tenpy.networks.site import SpinHalfSite



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class QCPModel(CouplingModel, NearestNeighborModel, MPOModel):
    omega: float # branching rate
    gamma: float # decaying rate
    L: int # number of sites
    bc_MPS: str # boundary condition of MPS

    # define local operators
    sigma_x = np.array([[0., 1.], [1., 0.]])
    sigma_plus = np.array([[0., 1.], [0., 0.]])
    sigma_minus = np.array([[0., 0.], [1., 0.]])
    number_op = np.array([[1., 0.], [0., 0.]]) # = np.dot(sigma_plus, sigma_minus)

    # initialize the physical site and add local operators to it
    site = SpinHalfSite(conserve=None)
    site.add_op('sigma_x', sigma_x)          # σ_x (or sigma_1)
    site.add_op('sigma_plus', sigma_plus)    # σ_+
    site.add_op('sigma_minus', sigma_minus)  # σ_-
    site.add_op('number', number_op)         # n

    # define chain
    chain = Chain(L, site, bc='open', bc_MPS=bc_MPS)

    # initialize CouplingModel
    model = CouplingModel.__init__(chain)

    # Hamiltonian
    for i in range(L-1):
        model.add_coupling_term(omega, i, i+1, 'sigma_x', 'number', plus_hc=False)
        model.add_coupling_term(omega, i, i+1, 'number', 'omega_x', plus_hc=False)
    
    # Dissipative
    for i in range(L): # TODO: There are 2 terms missing... Where do they go?
        model.add_onsite_term(-1j/2. *gamma, i, 'number') 
    
    ################################################ TODO: We stopped here for today!

    # initialize H_MPO
    MPOModel.__init__(model, chain, model.calc_H_MPO())

    # initialize H_bond (the order of 7/8 doesn't matter)
    NearestNeighborModel.__init__(model, chain, model.calc_H_bond())