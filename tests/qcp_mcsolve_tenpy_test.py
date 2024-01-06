import numpy as np

from qcp_mcsolve_tenpy import QCPModel
from tenpy.networks.mps import MPS

L = 10
basis_0 = np.array([1, 0])
basis_1 = np.array([0, 1])

chain = QCPModel({'omega': 6, 'gamma': 1, 'L': 10, 'bc_MPS': 'finite', 'explicit_plus_hc': False})
sites = chain.lat.mps_sites()

first_half_index = L // 2
if L % 2 == 1:
    second_half_index = L // 2
else:
    second_half_index = L // 2 - 1
init_state = [basis_0] * (first_half_index) + [basis_1] + [basis_0] * (second_half_index)

psi = MPS.from_product_state(sites, p_state=init_state)

### calculate total number of particles in a chain
number_particles = psi.expectation_value('number_op')
