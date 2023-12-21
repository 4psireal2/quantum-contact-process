"""
This module contains an implementation of a code simulating the Quantum contact
process url=http://dx.doi.org/10.1103/PhysRevLett.123.100604 with matrix
product state using the TeNPy module by Johannes Hauschild et al.
url=https://tenpy.readthedocs.io/
"""
import logging
import numpy as np

import sys
sys.path.append("C:/Users/robin/Desktop/quantum_contact_process")

from own_TeNPy import QRBasedTEBDEngine, SVDBasedTEBDEngine, normalization
from tenpy.models.model import MPOModel, NearestNeighborModel, CouplingModel
from tenpy.models.lattice import Chain

from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tenpy.networks.site import SpinHalfSite

from numpy.random import default_rng


logger = logging.getLogger(__name__)

rng = default_rng()


class initial_state:
    """
    This class is for creating the initial state. Each function
    creates a list of strings namely ['up'] and ['down'] which can
    later be used to create an Matrix product state by the TeNPy function
    MPS.from_product_state(). The w_state function is different. This
    state is build by applying an MPO to an product state. More details
    of the method are explaind here:
    https://tenpy.johannes-hauschild.de/viewtopic.php?t=553
    """
    def down_state(L):
        # print("Create state |11...1>")
        state = []
        for i in range(L):
            state.append('down')
        return state

    def up_state(L):
        # print("Create state |00...0>")
        state = []
        for i in range(L):
            state.append('up')
        return state

    def single_state(L):
        # print("Create state |00..1..0>")
        state = []
        for i in range(L):
            if i == L//2:
                state.append('down')
            else:
                state.append('up')
        return state

    def w_state(L):
        # print("Create state 1/sqrt(N) (|10 ..0> + |010 ..0> +... +|0 .. 01>))
        site = SpinHalfSite(conserve='None', sort_charge=True)
        op_number = np.array([[0., 0.], [0., 1.]])
        op_omega_plus = np.array([[0., 1.], [0., 0.]])
        site.add_op('number', op_number)
        site.add_op('omega_plus', op_omega_plus)
        lat = Chain(L, site, bc_MPS='finite')
        Sp, Id = site.Sp, site.Id
        W_m = [[Id, Sp], [None, Id]]
        grids = np.array([W_m]*L)
        initial = MPS.from_product_state(
            lat.mps_sites(), initial_state.down_state(L))
        H = MPO.from_grids(lat.mps_sites(), grids, IdL=0, IdR=-1)
        options = {'trunc_params': {'chi_max': 1500, 'svd_min': 1.e-14},
                   'compression_method': 'SVD'}
        H.apply(initial, options)
        for i in range(L):
            initial.apply_local_op(i, 'Sigmax')
        initial.canonical_form()
        return initial


class QCPModel(CouplingModel, NearestNeighborModel, MPOModel):
    """
    This class is for initialising the Model of the effective
    Hamiltonian H' = H - ih/2 * γ * σ_+ σ_- to later simulate
    it via the TEBD algorithm and get the time evolution of a state.
    There are some basic steps (also explained here:
    https://tenpy.readthedocs.io/en/latest/intro/model.html#the-couplingmodel-general-structure)
    how to build models in TeNPy.
    """

    def __init__(self, model_params):
        # read out parameters
        Ω = model_params.get('omega')
        γ = model_params.get('gamma')
        L = model_params.get('L')
        bc_MPS = model_params.get('bc_MPS', 'finite')

        # define local operators
        op_omega_x = np.array([[0., 1.], [1., 0.]])
        op_omega_plus = np.array([[0., 1.], [0., 0.]])
        op_omega_minus = np.array([[0., 0.], [1., 0.]])
        op_number = np.array([[0., 0.], [0., 1.]])
        op_o = op_o = np.array([[1., 0.], [0., 0.]])

        # initialize the physical site and add onsite operators to it
        site = SpinHalfSite(conserve='None')
        site.add_op('oo', op_o)                     # |0><0|
        site.add_op('omega_x', op_omega_x)          # σ_x
        site.add_op('omega_plus', op_omega_plus)    # σ_+
        site.add_op('omega_minus', op_omega_minus)  # σ_-
        site.add_op('number', op_number)            # |1><1| = n = σ_+ σ_-

        # define lattice
        lat = Chain(L, site,
                    bc='open',
                    bc_MPS=bc_MPS
                    )
        # initialize CouplingModel
        CouplingModel.__init__(self, lat)
        for i in range(L-1):
            self.add_coupling_term(
                Ω, i, i+1, 'omega_x', 'number', plus_hc=False)
            self.add_coupling_term(
                Ω, i, i+1, 'number', 'omega_x', plus_hc=False)
        for i in range(L):
            self.add_onsite_term(-1j/2. * γ, i, 'number')
        # initialize H_MPO

        MPOModel.__init__(self, lat, self.calc_H_MPO())
        # initialize H_bond (the order of 7/8 doesn't matter)
        NearestNeighborModel.__init__(self, lat, self.calc_H_bond())


class wave_function_monte_carlo():
    """
    This class contains the main function of the wave-function Monte-Carlo method.
    This method is based on the paper
        Klaus Mølmer, Yvan Castin, and Jean Dalibard,
        "Monte Carlo wave-function method in quantum optics,"
        J. Opt. Soc. Am. B 10, 524-538 (1993)
    """
    def execute(psi_t_initial, t_final, parameter, Ntraj=1, engine=QRBasedTEBDEngine):
        """
        execute the wave-function Monte-Carlo Method of an initial state till t_final

        Parameters
        ----------
        psi_t_initial : list of {int | str | 1D Array}
            Choose between the 4 predefined states of the class initial_state
            or create an own state e.g. [['down'],['up'],['down']] = |010>
        t_final : int
            The time limit till which time the state should evolve
        parameter : dict
            A dictionary with the following entries
                model_params : dict
                    'L':  {int} size of the system
                    'omega' : {int} strength of coagulation
                    'gamma' : {int} strength dissipation
                tebd_params : dict
                    'dt': {float}
                    'trunc_params': dict
                        chi_max {int}
        Ntraj : int
            Number of how many trajectories you want to calculate
        engine: func
            function which defines the engine for simulating the
            time evolution e.g. QRBasedTEBDEngine or SVDBasedTEBDEngine

        Returns
        -------
        result_list : list
        """

        dt = 2 * parameter.get('tebd_params').get('dt', 0)

        # initialize Model
        M = QCPModel(parameter.get('model_params'))
        res = []

        # initialize initial state
        # to use the W-state as initial state, just replace
        # "MPS.from_product_state(M.lat.mps_sites(), psi_t_initial)"
        # with "initial_state.w_state(L)"
        psi = MPS.from_product_state(M.lat.mps_sites(), psi_t_initial)
        for k in range(Ntraj):
            psi_t = psi.copy()
            # save density of |ψ(0)> by calculating <N(0)> = ∑_i <ψ|n ψ>_i
            state_list = [np.sum(psi_t.expectation_value('number'))]
            for j in range(0, t_final):
                # calculate jump probability for each site i
                # dp = dt * <ψ(t)|σ_+ σ_- ψ(t)>
                dp = np.real(dt*psi_t.expectation_value('number'))

                # contruct random list for each site i
                r_arr = [rng.random() for _ in range(psi_t.L)]

                # calculate time evolution |ψ(t+dt)> with TEBD algorithm
                eng = engine(psi_t, M, parameter.get('tebd_params'))
                eng.run()

                # check for each site if jump prob. is bigger than random
                # number, if so apply σ_+ on site i -> |ψ>_i = σ_+|ψ>_i
                for i in range(0, psi_t.L):
                    if dp[i] > r_arr[i]:
                        psi_t.apply_local_op(
                           i, 'omega_plus', unitary=False, renormalize=True)

                # renormalize the resulting state
                for i in range(psi_t.L-1):
                    normalization.normalize_site(psi_t, i, {'chi_max': 1200, 'svd_min': 1.e-12})
                psi_t.norm = psi_t.overlap(psi_t)/(psi_t.norm**2)
                # calculate <N(t)> = ∑_i <ψ|σ_+ σ_- ψ> = ∑_i <ψ|n ψ>_i
                state_list.append(np.sum(psi_t.expectation_value('number')))

            # save density of trajectorie <N(t)>_k
            res.append(state_list)

        # calculate 1/Ntraj * ∑_k <N(t)>_k
        return np.sum(res, axis=0)/len(res)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    parameter = {'model_params': {
            'bc_MPS': 'finite',
            'L': 34,
            'omega': 6,
            'gamma': 1,
            'boundary_conditions': 'open'
        }, 'tebd_params': {
                'N_steps': 2,
                'dt': 0.05,
                'order': 2,
                'trunc_params': {'chi_max': 350, 'svd_min': 1.e-12}
                }
        }

    data_res = wave_function_monte_carlo.execute(initial_state.single_state(parameter.get('model_params').get('L')),
                                                 t_final=20, parameter=parameter, Ntraj=1, engine=QRBasedTEBDEngine)
    print(plt.plot(data_res))