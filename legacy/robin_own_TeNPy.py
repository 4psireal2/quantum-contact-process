"""
This module is mainly original TeNPy code expect for the normaliziation
class and some changes in the update_bond function of the SVD based
Engine and the QR based Engine. TeNPy always normalize states and keeps
track of the norm by directly changeing psi.norm = ... Since we use an
hermitian Hamiltonian the state shouldn't be normalized, so i deleted the
part "/ renormalize" in line 131 and added a the term "S *= renormalize"
in line 138. Respectively i did the same for the SVD decomposition.
"""

import numpy as np
import logging
logger = logging.getLogger(__name__)

from tenpy.linalg import np_conserved as npc
from tenpy.algorithms.truncation import svd_theta, TruncationError, truncate
from tenpy.algorithms.tebd import TEBDEngine
from tenpy.algorithms.algorithm import TimeEvolutionAlgorithm

import warnings
import time

class normaliziation():
    """
    This function is used for normalizing the state on a site i. Since TeNPy intrinsically normalize 
    the matrix S when doing a SVD decomposition, i just use that function and later renormalize also the
    B_L matrix. If you use a high value (> 1200) for chi_max in the trunc_params there should be no change to the state 
    except the normaliziation.
    """
    def normalize_site(psi, i, trunc_params):
        i0, i1 = i , i +1
        logger.debug("Update sites (%d, %d)", i0, i1)
        C = psi.get_theta(i0, n=2, formL=0.)  # the two B without the S on the left
        C.itranspose(['vL', 'p0', 'p1', 'vR'])
        theta = C.scale_axis(psi.get_SL(i0), 'vL')
        theta = theta.combine_legs([('vL', 'p0'), ('p1', 'vR')], qconj=[+1, -1])
        U, S, V, trunc_err, renormalize = svd_theta(theta,
                                                    trunc_params,
                                                    [psi.get_B(i0, None).qtotal, None],
                                                    inner_labels=['vR', 'vL'])
        B_R = V.split_legs(1).ireplace_label('p1', 'p')
        B_L = npc.tensordot(C.combine_legs(('p1', 'vR'), pipes=theta.legs[1]),
                            V.conj(),
                            axes=['(p1.vR)', '(p1*.vR*)'])
        B_L.ireplace_labels(['vL*', 'p0'], ['vR', 'p'])
        B_L /= renormalize  # re-normalize to <psi|psi> = 1
        psi.norm *= renormalize
        psi.set_SR(i0, S)
        psi.set_B(i0, B_L, form='B')
        psi.set_B(i1, B_R, form='B')
        return trunc_err



class QRBasedTEBDEngine(TEBDEngine):
    r"""Version of TEBD that relies on QR decompositions rather than SVD.

    As introduced in :arxiv:`2212.09782`.

    .. todo ::
        To use `use_eig_based_svd == True`, which makes sense on GPU only, we need to implement
        the `_eig_based_svd` for "non-square" matrices.
        This means that :math:`M^{\dagger} M` and :math:`M M^{\dagger}` dont have the same size,
        and we need to disregard those eigenvectors of the larger one, that have eigenvalue zero,
        since we dont have corresponding eigenvalues of the smaller one.

    Options
    -------
    .. cfg:config :: QRBasedTEBDEngine
        :include: TEBDEngine

        cbe_expand : float
            Expansion rate. The QR-based decomposition is carried out at an expanded bond dimension
            ``eta = (1 + cbe_expand) * chi``, where ``chi`` is the bond dimension before the time step.
            Default is `0.1`.
        cbe_expand_0 : float
            Expansion rate at low ``chi``.
            If given, the expansion rate decreases linearly from ``cbe_expand_0`` at ``chi == 1``
            to ``cbe_expand`` at ``chi == trunc_params['chi_max']``, then remains constant.
            If not given, the expansion rate is ``cbe_expand`` at all ``chi``.
        cbe_min_block_increase : int
            Minimum bond dimension increase for each block. Default is `1`.
        use_eig_based_svd : bool
            Whether the SVD of the bond matrix :math:`\Xi` should be carried out numerically via
            the eigensystem. This is faster on GPUs, but less accurate.
            It makes no sense to do this on CPU. It is currently not supported for update_imag.
            Default is `False`.
        compute_err : bool
            Whether the truncation error should be computed exactly.
            Compared to SVD-based TEBD, computing the truncation error is significantly more expensive.
            If `True` (default), the full error is computed.
            Otherwise, the truncation error is set to NaN.
    """

    def _expansion_rate(self, i):
        """get expansion rate for updating bond i"""
        expand = self.options.get('cbe_expand', 0.1)
        expand_0 = self.options.get('cbe_expand_0', None)

        if expand_0 is None or expand_0 == expand:
            return expand

        chi_max = self.trunc_params.get('chi_max', None)
        if chi_max is None:
            raise ValueError('Need to specify trunc_params["chi_max"] in order to use cbe_expand_0.')

        chi = min(self.psi.get_SL(i).shape)
        return max(expand_0 - chi / chi_max * (expand_0 - expand), expand)

    def update_bond(self, i, U_bond):
        i0, i1 = i - 1, i
        expand = self._expansion_rate(i)
        logger.debug(f'Update sites ({i0}, {i1}). CBE expand={expand}')
        # Construct the theta matrix
        C = self.psi.get_theta(i0, n=2, formL=0.)  # the two B without the S on the left
        C = npc.tensordot(U_bond, C, axes=(['p0*', 'p1*'], ['p0', 'p1']))  # apply U
        C.itranspose(['vL', 'p0', 'p1', 'vR'])
        theta = C.scale_axis(self.psi.get_SL(i0), 'vL')
        theta = theta.combine_legs([('vL', 'p0'), ('p1', 'vR')], qconj=[+1, -1])

        min_block_increase = self.options.get('cbe_min_block_increase', 1)
        Y0 = _qr_tebd_cbe_Y0(B_L=self.psi.get_B(i0, 'B'), B_R=self.psi.get_B(i1, 'B'), theta=theta,
                             expand=expand, min_block_increase=min_block_increase)
        A_L, S, B_R, trunc_err, renormalize = _qr_based_decomposition(
            theta=theta, Y0=Y0, use_eig_based_svd=self.options.get('use_eig_based_svd', False),
            need_A_L=False, compute_err=self.options.get('compute_err', True),
            trunc_params=self.trunc_params
        )
        B_L = npc.tensordot(C.combine_legs(('p1', 'vR'), pipes=theta.legs[1]),
                            B_R.conj(),
                            axes=[['(p1.vR)'], ['(p*.vR*)']]) #/ renormalize
        
        #B_L = npc.tensordot(C.combine_legs(('p1', 'vR'), pipes=theta.legs[1]),
        #                    B_R.conj(),
        #                    axes=[['(p1.vR)'], ['(p*.vR*)']]) / renormalize
        #
        #deleted the /renormalize to make it work for non hermitian Hamitltonian
        S *= renormalize
        #
        B_L.ireplace_labels(['p0', 'vL*'], ['p', 'vR'])
        B_R = B_R.split_legs(1)
        
        #self.psi.norm *= renormalize
        #
        #deleted self.psi.norm because we need to calculate the norm on our own
        self.psi.set_B(i0, B_L, form='B')
        self.psi.set_SL(i1, S)
        self.psi.set_B(i1, B_R, form='B')
        self._trunc_err_bonds[i] = self._trunc_err_bonds[i] + trunc_err
        return trunc_err

    def update_bond_imag(self, i, U_bond):
        i0, i1 = i - 1, i
        expand = self._expansion_rate(i)
        logger.debug(f'Update sites ({i0}, {i1}). CBE expand={expand}')
        # Construct the theta matrix
        theta = self.psi.get_theta(i0, n=2)
        theta = npc.tensordot(U_bond, theta, axes=(['p0*', 'p1*'], ['p0', 'p1']))
        theta.itranspose(['vL', 'p0', 'p1', 'vR'])
        theta = theta.combine_legs([('vL', 'p0'), ('p1', 'vR')], qconj=[+1, -1])

        use_eig_based_svd = self.options.get('use_eig_based_svd', False)

        if use_eig_based_svd:
            # see todo comment in _eig_based_svd
            raise NotImplementedError('update_bond_imag does not (yet) support eig based SVD')

        min_block_increase = self.options.get('cbe_min_block_increase', 1)
        Y0 = _qr_tebd_cbe_Y0(B_L=self.psi.get_B(i0, 'B'), B_R=self.psi.get_B(i1, 'B'), theta=theta,
                             expand=expand, min_block_increase=min_block_increase)
        A_L, S, B_R, trunc_err, renormalize = _qr_based_decomposition(
            theta=theta, Y0=Y0, use_eig_based_svd=use_eig_based_svd,
            need_A_L=True, compute_err=self.options.get('compute_err', True),
            trunc_params=self.trunc_params
        )
        A_L = A_L.split_legs(0)
        B_R = B_R.split_legs(1)

        self.psi.norm *= renormalize
        self.psi.set_B(i0, A_L, form='A')
        self.psi.set_SL(i1, S)
        self.psi.set_B(i1, B_R, form='B')
        self._trunc_err_bonds[i] = self._trunc_err_bonds[i] + trunc_err

        return trunc_err
def _qr_tebd_cbe_Y0(B_L: npc.Array, B_R: npc.Array, theta: npc.Array, expand: float, min_block_increase: int):
    """Generate the initial guess Y0 for the left isometry in QR based TEBD

    Parameters
    ----------
    B_L : Array with legs [vL, p, vR]
    B_R : Array with legs [vL, p, vR]
    theta : Array with legs [(vL.p0), (p1.vR)]
    expand : float or None

    Returns
    -------
    Y0 : Array with legs [vL, (p1.vR)]
    """
    if expand is None or expand == 0:
        return B_R.combine_legs(['p', 'vR']).ireplace_labels('(p.vR)', '(p1.vR)')

    assert min_block_increase >= 0

    Y0 = theta.copy(deep=False)
    Y0.legs[0] = Y0.legs[0].to_LegCharge()
    Y0.ireplace_label('(vL.p0)', 'vL')
    if any(B_L.qtotal != 0):
        Y0.gauge_total_charge('vL', new_qtotal=B_R.qtotal)
    vL_old = B_R.get_leg('vL')
    if not vL_old.is_blocked():
        vL_old = vL_old.sort()[1]
    vL_new = Y0.get_leg('vL')  # is blocked, since created from pipe

    # vL_old is guaranteed to be a slice of vL_new by charge rule in B_L
    piv = np.zeros(vL_new.ind_len, dtype=bool)  # indices to keep in vL_new
    increase_per_block = max(min_block_increase, int(vL_old.ind_len * expand // vL_new.block_number))
    sizes_old = vL_old.get_block_sizes()
    sizes_new = vL_new.get_block_sizes()
    # iterate over charge blocks in vL_new and vL_old at the same time
    j_old = 0
    q_old = vL_old.charges[j_old, :]
    qdata_order = np.argsort(Y0._qdata[:, 0])
    qdata_idx = 0
    for j_new, q_new in enumerate(vL_new.charges):
        if all(q_new == q_old):  # have charge block in both vL_new and vL_old
            s_new = sizes_old[j_old] + increase_per_block
            # move to next charge block in next loop iteration
            j_old += 1
            if j_old < len(vL_old.charges):
                q_old = vL_old.charges[j_old, :]
        else:  # charge block only in vL_new
            s_new = increase_per_block
        s_new = min(s_new, sizes_new[j_new])  # don't go beyond block

        if Y0._qdata[qdata_order[qdata_idx], 0] != j_new:
            # block does not exist
            # while we could set corresponding piv entries to True, it would not help, since
            # the corresponding "entries" of Y0 are zero anyway
            continue

        # block has axis [vL, (p1.vR)]. want to keep the s_new slices of the vL axis
        #  that have the largest norm
        norms = np.linalg.norm(Y0._data[qdata_order[qdata_idx]], axis=1)
        kept_slices = np.argsort(-norms)[:s_new]  # negative sign so we sort large to small
        start = vL_new.slices[j_new]
        piv[start + kept_slices] = True

        qdata_idx += 1
        if qdata_idx >= Y0._qdata.shape[0]:
            break

    Y0.iproject(piv, 'vL')
    return Y0
def _qr_based_decomposition(theta: npc.Array, Y0: npc.Array, use_eig_based_svd: bool, trunc_params,
                            need_A_L: bool, compute_err: bool):
    """Perform the decomposition step of QR based TEBD

    Parameters
    ----------
    theta : Array with legs [(vL.p0), (p1.vR)]
    Y0 : Array with legs [vL, (p1.vR)]
    ...

    Returns
    -------
    A_L : array with legs [(vL.p), vR] or None
    S : 1D numpy array
    B_R : array with legs [vL, (p.vR)]
    trunc_err : TruncationError
    renormalize : float
    """

    if compute_err:
        need_A_L = True

    # QR based updates
    theta_i0 = npc.tensordot(theta, Y0.conj(), ['(p1.vR)', '(p1*.vR*)']).ireplace_label('vL*', 'vR')
    A_L, _ = npc.qr(theta_i0, inner_labels=['vR', 'vL'])
    # A_L: [(vL.p0), vR]
    theta_i1 = npc.tensordot(A_L.conj(), theta, ['(vL*.p0*)', '(vL.p0)']).ireplace_label('vR*', 'vL')
    theta_i1.itranspose(['(p1.vR)', 'vL'])
    B_R, Xi = npc.qr(theta_i1, inner_labels=['vL', 'vR'], inner_qconj=-1)
    B_R.itranspose(['vL', '(p1.vR)'])
    Xi.itranspose(['vL', 'vR'])

    # SVD of bond matrix Xi
    if use_eig_based_svd:
        U, S, Vd, trunc_err, renormalize = _eig_based_svd(
            Xi, inner_labels=['vR', 'vL'], need_U=need_A_L, trunc_params=trunc_params
        )
    else:
        U, S, Vd, _, renormalize = svd_theta(Xi, trunc_params)
    B_R = npc.tensordot(Vd, B_R, ['vR', 'vL'])
    if need_A_L:
        A_L = npc.tensordot(A_L, U, ['vR', 'vL'])
    else:
        A_L = None

    if compute_err:
        theta_approx = npc.tensordot(A_L.scale_axis(S, axis='vR'), B_R, ['vR', 'vL'])
        eps = npc.norm(theta - theta_approx) ** 2
        trunc_err = TruncationError(eps, 1. - 2. * eps)
    else:
        trunc_err = TruncationError(np.nan, np.nan)

    B_R = B_R.ireplace_label('(p1.vR)', '(p.vR)')
    if need_A_L:
        A_L = A_L.ireplace_label('(vL.p0)', '(vL.p)')

    return A_L, S, B_R, trunc_err, renormalize
def _eig_based_svd(A, need_U: bool = True, need_Vd: bool = True, inner_labels=[None, None],
                   trunc_params=None):
    """Computes the singular value decomposition of a matrix A via eigh

    Singular values and vectors are obtained by diagonalizing the "square" A.hc @ A and/or A @ A.hc,
    i.e. with two eigh calls instead of an svd call.

    Truncation if performed if and only if trunc_params are given.
    This performs better on GPU, but is not really useful on CPU.
    If isometries U or Vd are not needed, their computation can be omitted for performance.

    Does not (yet) support computing both U and Vd
    """

    assert A.rank == 2

    if need_U and need_Vd:
        # TODO (JU) just doing separate eighs for U, S and for S, Vd is not sufficient
        #  the phases of U / Vd are arbitrary.
        #  Need to put in more work in that case...
        raise NotImplementedError

    if need_U:
        Vd = None
        A_Ahc = npc.tensordot(A, A.conj(), [1, 1])
        L, U = npc.eigh(A_Ahc, sort='>')
        S = np.sqrt(np.abs(L))  # abs to avoid `nan` due to accidentially negative values close to zero
        U = U.ireplace_label('eig', inner_labels[0])
    elif need_Vd:
        U = None
        Ahc_A = npc.tensordot(A.conj(), A, [0, 0])
        L, V = npc.eigh(Ahc_A, sort='>')
        S = np.sqrt(np.abs(L))  # abs to avoid `nan` due to accidentially negative values close to zero
        Vd = V.iconj().itranspose().ireplace_label('eig*', inner_labels[1])
    else:
        U = None
        Vd = None
        # use the smaller of the two square matrices -- they have the same eigenvalues
        if A.shape[1] >= A.shape[0]:
            A2 = npc.tensordot(A, A.conj(), [1, 0])
        else:
            A2 = npc.tensordot(A.conj(), A, [1, 0])
        L = npc.eigvalsh(A2)
        S = np.sqrt(np.abs(L))  # abs to avoid `nan` due to accidentially negative values close to zero

    if trunc_params is not None:
        piv, renormalize, trunc_err = truncate(S, trunc_params)
        S = S[piv]
        S /= renormalize
        if need_U:
            U.iproject(piv, 1)
        if need_Vd:
            Vd.iproject(piv, 0)
    else:
        renormalize = np.linalg.norm(S)
        S /= renormalize
        trunc_err = TruncationError()

    return U, S, Vd, trunc_err, renormalize


class SVDBasedTEBDEngine(TimeEvolutionAlgorithm):
    def __init__(self, psi, model, options, **kwargs):
        TimeEvolutionAlgorithm.__init__(self, psi, model, options, **kwargs)
        self.trunc_err = self.options.get('start_trunc_err', TruncationError())
        self._U = None
        self._U_param = {}
        self._trunc_err_bonds = [TruncationError() for i in range(psi.L + 1)]
        self._update_index = None

    @property
    def TEBD_params(self):
        warnings.warn("renamed self.TEBD_params -> self.options", FutureWarning, stacklevel=2)
        return self.options

    @property
    def trunc_err_bonds(self):
        """truncation error introduced on each non-trivial bond."""
        return self._trunc_err_bonds[self.psi.nontrivial_bonds]

    def run(self):
        """Run TEBD real time evolution by `N_steps`*`dt`."""
        # initialize parameters
        delta_t = self.options.get('dt', 0.1)
        N_steps = self.options.get('N_steps', 10)
        TrotterOrder = self.options.get('order', 2)
        E_offset = self.options.get('E_offset', None)

        self.calc_U(TrotterOrder, delta_t, type_evo='real', E_offset=E_offset)

        Sold = np.mean(self.psi.entanglement_entropy())
        start_time = time.time()

        self.update(N_steps)
        
        S = self.psi.entanglement_entropy()
        logger.info(
            "--> time=%(t)3.3f, max(chi)=%(chi)d, max(S)=%(S).5f, "
            "avg DeltaS=%(dS).4e, since last update: %(wall_time).1fs", {
                't': self.evolved_time.real,
                'chi': max(self.psi.chi),
                'S': max(S),
                'dS': np.mean(S) - Sold,
                'wall_time': time.time() - start_time,
            })

    def run_GS(self):
        # initialize parameters
        delta_tau_list = self.options.get(
            'delta_tau_list',
            [0.1, 0.01, 0.001, 1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8, 1.e-9, 1.e-10, 1.e-11, 0.])
        max_error_E = self.options.get('max_error_E', 1.e-13)
        N_steps = self.options.get('N_steps', 10)
        TrotterOrder = self.options.get('order', 2)

        Eold = np.mean(self.model.bond_energies(self.psi))
        Sold = np.mean(self.psi.entanglement_entropy())
        start_time = time.time()

        for delta_tau in delta_tau_list:
            logger.info("delta_tau=%e", delta_tau)
            self.calc_U(TrotterOrder, delta_tau, type_evo='imag')
            DeltaE = 2 * max_error_E
            step = 0
            while (DeltaE > max_error_E):
                if self.psi.finite and TrotterOrder == 2:
                    self.update_imag(N_steps)
                else:
                    self.update(N_steps)
                step += N_steps
                E = np.mean(self.model.bond_energies(self.psi))
                DeltaE = abs(Eold - E)
                Eold = E
                S = self.psi.entanglement_entropy()
                max_S = max(S)
                S = np.mean(S)
                DeltaS = S - Sold
                Sold = S
                logger.info(
                    "--> step=%(step)6d, beta=%(beta)3.3f, max(chi)=%(max_chi)d,"
                    "DeltaE=%(dE).2e, E_bond=%(E).10f, Delta_S=%(dS).4e, "
                    "max(S)=%(max_S).10f, time simulated: %(wall_time).1fs", {
                        'step': step,
                        'beta': -self.evolved_time.imag,
                        'max_chi': max(self.psi.chi),
                        'dE': DeltaE,
                        'E': E.real,
                        'dS': DeltaS,
                        'max_S': max_S,
                        'wall_time': time.time() - start_time,
                    })
        # done

    @staticmethod
    def suzuki_trotter_time_steps(order):
        if order == 1:
            return [1.]
        elif order == 2:
            return [0.5, 1.]
        elif order == 4:
            t1 = 1. / (4. - 4.**(1 / 3.))
            t3 = 1. - 4. * t1
            return [t1 / 2., t1, (t1 + t3) / 2., t3]
        elif order == '4_opt':
            # Eq (30a) of arXiv:1901.04974
            a1 = 0.095848502741203681182
            b1 = 0.42652466131587616168
            a2 = -0.078111158921637922695
            b2 = -0.12039526945509726545
            return [a1, b1, a2, b2, 0.5 - a1 - a2, 1. - 2 * (b1 + b2)]  # a1 b1 a2 b2 a3 b3
        # else
        raise ValueError("Unknown order %r for Suzuki Trotter decomposition" % order)

    @staticmethod
    def suzuki_trotter_decomposition(order, N_steps):
        even, odd = 0, 1
        if N_steps == 0:
            return []
        if order == 1:
            a = (0, odd)
            b = (0, even)
            return [a, b] * N_steps
        elif order == 2:
            a = (0, odd)  # dt/2
            a2 = (1, odd)  # dt
            b = (1, even)  # dt
            # U = [a b a]*N
            #   = a b [a2 b]*(N-1) a
            return [a, b] + [a2, b] * (N_steps - 1) + [a]
        elif order == 4:
            a = (0, odd)  # t1/2
            a2 = (1, odd)  # t1
            b = (1, even)  # t1
            c = (2, odd)  # (t1 + t3) / 2 == (1 - 3 * t1)/2
            d = (3, even)  # t3 = 1 - 4 * t1
            # From Schollwoeck 2011 (:arxiv:`1008.3477`):
            # U = U(t1) U(t2) U(t3) U(t2) U(t1)
            # with U(dt) = U(dt/2, odd) U(dt, even) U(dt/2, odd) and t1 == t2
            # Using above definitions, we arrive at:
            # U = [a b a2 b c d c b a2 b a] * N
            #   = [a b a2 b c d c b a2 b] + [a2 b a2 b c d c b a2 b a] * (N-1) + [a]
            steps = [a, b, a2, b, c, d, c, b, a2, b]
            steps = steps + [a2, b, a2, b, c, d, c, b, a2, b] * (N_steps - 1)
            steps = steps + [a]
            return steps
        elif order == '4_opt':
            # symmetric: a1 b1 a2 b2 a3 b3 a2 b2 a2 b1 a1
            steps = [(0, odd), (1, even), (2, odd), (3, even), (4, odd),  (5, even),
                     (4, odd), (3, even), (2, odd), (1, even), (0, odd)]  # yapf: disable
            return steps * N_steps
        # else
        raise ValueError("Unknown order {0!r} for Suzuki Trotter decomposition".format(order))

    def calc_U(self, order, delta_t, type_evo='real', E_offset=None):
        U_param = dict(order=order, delta_t=delta_t, type_evo=type_evo, E_offset=E_offset)
        if type_evo == 'real':
            U_param['tau'] = delta_t
        elif type_evo == 'imag':
            U_param['tau'] = -1.j * delta_t
        else:
            raise ValueError("Invalid value for `type_evo`: " + repr(type_evo))
        if self._U_param == U_param:  # same keys and values as cached
            logger.debug("Skip recalculation of U with same parameters as before")
            return  # nothing to do: U is cached
        self._U_param = U_param
        logger.info("Calculate U for %s", U_param)

        L = self.psi.L
        self._U = []
        for dt in self.suzuki_trotter_time_steps(order):
            U_bond = [
                self._calc_U_bond(i_bond, dt * delta_t, type_evo, E_offset) for i_bond in range(L)
            ]
            self._U.append(U_bond)
        # done

    def update(self, N_steps):
        #print("update norm =",self.psi.norm)
        #print("psi =",self.psi)
        trunc_err = TruncationError()
        order = self._U_param['order']
        for U_idx_dt, odd in self.suzuki_trotter_decomposition(order, N_steps):
            trunc_err += self.update_step(U_idx_dt, odd)
        self.evolved_time = self.evolved_time + N_steps * self._U_param['tau']
        self.trunc_err = self.trunc_err + trunc_err  # not += : make a copy!
        # (this is done to avoid problems of users storing self.trunc_err after each `update`)
        return trunc_err

    def update_step(self, U_idx_dt, odd):
        Us = self._U[U_idx_dt]
        trunc_err = TruncationError()
        for i_bond in np.arange(int(odd) % 2, self.psi.L, 2):
            if Us[i_bond] is None:
                continue  # handles finite vs. infinite boundary conditions
            self._update_index = (U_idx_dt, i_bond)
            trunc_err += self.update_bond(i_bond, Us[i_bond])
        self._update_index = None
        return trunc_err

    def update_bond(self, i, U_bond):
        #print("beginning update bond =", self.psi.norm)
        #print("position = ",i)
        i0, i1 = i - 1, i
        logger.debug("Update sites (%d, %d)", i0, i1)
        # Construct the theta matrix
        C = self.psi.get_theta(i0, n=2, formL=0.)  # the two B without the S on the left
        C = npc.tensordot(U_bond, C, axes=(['p0*', 'p1*'], ['p0', 'p1']))  # apply U
        C.itranspose(['vL', 'p0', 'p1', 'vR'])
        theta = C.scale_axis(self.psi.get_SL(i0), 'vL')
        # now theta is the same as if we had done
        #   theta = self.psi.get_theta(i0, n=2)
        #   theta = npc.tensordot(U_bond, theta, axes=(['p0*', 'p1*'], ['p0', 'p1']))  # apply U
        # but also have C which is the same except the missing "S" on the left
        # so we don't have to apply inverses of S (see below)

        theta = theta.combine_legs([('vL', 'p0'), ('p1', 'vR')], qconj=[+1, -1])
        # Perform the SVD and truncate the wavefunction
        U, S, V, trunc_err, renormalize = svd_theta(theta,
                                                    self.trunc_params,
                                                    [self.psi.get_B(i0, None).qtotal, None],
                                                    inner_labels=['vR', 'vL'])

        # Split tensor and update matrices
        B_R = V.split_legs(1).ireplace_label('p1', 'p')


        B_L = npc.tensordot(C.combine_legs(('p1', 'vR'), pipes=theta.legs[1]),
                            V.conj(),
                            axes=['(p1.vR)', '(p1*.vR*)'])
        B_L.ireplace_labels(['vL*', 'p0'], ['vR', 'p'])
        #print("renormalize =", renormalize,"")
        #print("B_L = ",B_L)
        #print("norm = ",B_L.norm())
        #print("singulaerwert Norm = ",LA.norm(S))
        #print("B_R = ",B_R)
        #re-normalize to <psi|psi> = 1
        #print("before renormalize =",self.psi.norm)
        
        
        
        S = S * renormalize
        #Der Vektor S der Singulärwerte wurde bereits mit dem Faktor "renormalize" normalisiert, muss dies 
        #Rückgängig gemacht werden?
        
        
        #print("renormalizefactor =", renormalize)
        #B_L /= renormalize 
        #Wenn ich die renormalisierung von B_L unterlasse, ändert sich dann die Norm wie folgt? 
        #self.psi.norm *= renormalize
        #Vergleich zu 
        #update_bond_imag(self, i, U_bond):
        #dort wird die norm mit dem Faktor multipliziert
        
        #print("norm = ",B_L.norm())
        self.psi.set_SR(i0, S)
        self.psi.set_B(i0, B_L, form='B')
        self.psi.set_B(i1, B_R, form='B')
        
        #print("norm =", self.psi.norm, type(self.psi))
        self._trunc_err_bonds[i] = self._trunc_err_bonds[i] + trunc_err
        #print("end renormalize =", self.psi.norm)
        return trunc_err

    def update_imag(self, N_steps):
        trunc_err = TruncationError()
        order = self._U_param['order']
        # allow only second order evolution
        if order != 2 or not self.psi.finite:
            # Would lead to loss of canonical form. What about DMRG?
            raise NotImplementedError("Use DMRG instead...")
        U_idx_dt = 0  # always with dt=0.5
        assert (self.suzuki_trotter_time_steps(order)[U_idx_dt] == 0.5)
        assert (self.psi.finite)  # finite or segment bc
        Us = self._U[U_idx_dt]
        for _ in range(N_steps):
            # sweep right
            for i_bond in range(self.psi.L):
                if Us[i_bond] is None:
                    continue  # handles finite vs. infinite boundary conditions
                self._update_index = (U_idx_dt, i_bond)
                trunc_err += self.update_bond_imag(i_bond, Us[i_bond])
            # sweep left
            for i_bond in range(self.psi.L - 1, -1, -1):
                if Us[i_bond] is None:
                    continue  # handles finite vs. infinite boundary conditions
                self._update_index = (U_idx_dt, i_bond)
                trunc_err += self.update_bond_imag(i_bond, Us[i_bond])
        self._update_index = None
        self.evolved_time = self.evolved_time + N_steps * self._U_param['tau']
        self.trunc_err = self.trunc_err + trunc_err  # not += : make a copy!
        # (this is done to avoid problems of users storing self.trunc_err after each `update`)
        return trunc_err

    def update_bond_imag(self, i, U_bond):
        i0, i1 = i - 1, i
        logger.debug("Update sites (%d, %d)", i0, i1)
        # Construct the theta matrix
        theta = self.psi.get_theta(i0, n=2)  # 'vL', 'vR', 'p0', 'p1'
        theta = npc.tensordot(U_bond, theta, axes=(['p0*', 'p1*'], ['p0', 'p1']))
        theta = theta.combine_legs([('vL', 'p0'), ('vR', 'p1')], qconj=[+1, -1])
        # Perform the SVD and truncate the wavefunction
        U, S, V, trunc_err, renormalize = svd_theta(theta,
                                                    self.trunc_params,
                                                    inner_labels=['vR', 'vL'])
       
        self.psi.norm *= renormalize
        # Split legs and update matrices
        B_R = V.split_legs(1).ireplace_label('p1', 'p')
        A_L = U.split_legs(0).ireplace_label('p0', 'p')
        self.psi.set_SR(i0, S)
        self.psi.set_B(i0, A_L, form='A')
        self.psi.set_B(i1, B_R, form='B')
        self._trunc_err_bonds[i] = self._trunc_err_bonds[i] + trunc_err
        return trunc_err

    def _calc_U_bond(self, i_bond, dt, type_evo, E_offset):
        """Calculate exponential of a bond Hamitonian.
        * ``U_bond = exp(-i dt (H_bond-E_offset_bond))`` for ``type_evo='real'``, or
        * ``U_bond = exp(- dt H_bond)`` for ``type_evo='imag'``.
        """
        h = self.model.H_bond[i_bond]
        if h is None:
            return None  # don't calculate exp(i H t), if `H` is None
        H2 = h.combine_legs([('p0', 'p1'), ('p0*', 'p1*')], qconj=[+1, -1])
        if type_evo == 'imag':
            H2 = (-dt) * H2
        elif type_evo == 'real':
            if E_offset is not None:
                H2 = H2 - npc.diag(E_offset[i_bond], H2.legs[0])
            H2 = (-1.j * dt) * H2
        else:
            raise ValueError("Expect either 'real' or 'imag'inary time, got " + repr(type_evo))
        U = npc.expm(H2)
#print("calc_U_bond =", U)
        assert (tuple(U.get_leg_labels()) == ('(p0.p1)', '(p0*.p1*)'))
        return U.split_legs()


class Engine(SVDBasedTEBDEngine):
    """Deprecated old name of :class:`TEBDEngine`.
    .. deprecated : v0.8.0
        Renamed the `Engine` to `TEBDEngine` to have unique algorithm class names.
    """
    def __init__(self, psi, model, options):
        msg = "Renamed `Engine` class to `TEBDEngine`."
        warnings.warn(msg, category=FutureWarning, stacklevel=2)
        SVDBasedTEBDEngine.__init__(self, psi, model, options)

