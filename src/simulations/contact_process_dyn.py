import matplotlib.pyplot as plt
import numpy as np

import scikit_tt.tensor_train as tt
from scikit_tt.solvers.ode import hod

from src.models.contact_process_model import (construct_lindblad, construct_num_op)
from src.utilities.utils import (canonicalize_mps, compute_site_expVal)
### TODO: euler verfahren mit Matrizierung ?, hod funktioniert nur bei (near)-Hermitian system ?
# # system parameters
L = 5
OMEGA = 6.0

# # dynamics parameter
step_size = 0.1
number_of_steps = 7
max_rank = 15

lindblad = construct_lindblad(gamma=1.0, omega=OMEGA, L=L)
# lindblad_hermitian = lindblad.transpose(conjugate=True) @ lindblad
num_op = construct_num_op(L)
mps = tt.unit(L * [4], inds=L // 2 * [0] + [3] + L // 2 * [0])
# for i in range(max_rank - 1) :
#     mps = mps + tt.unit(L * [4], inds=L // 2 * [0] + [3] + L // 2 * [0])

# mps = mps.ortho()
# mps = (1 / mps.norm()) * mps
# mps_t = tdvp(-1j * lindblad,
#             mps,
#             step_size=step_size,
#             number_of_steps=number_of_steps,
#             normalize=2)

mps_t = hod(lindblad, mps, step_size=step_size, number_of_steps=number_of_steps, normalize=2, max_rank=max_rank)

# mps_t = [mps]
# for i in range(number_of_steps):
#     vec_t = scipy.linalg.expm( lindblad.matricize() * i * step_size) @ mps.matricize() # mps_t
#     vec_t = vec_t.reshape([2] * 2*L, order='F')
#     mps_t.append(TT(vec_t))

t = np.linspace(0, step_size * number_of_steps, number_of_steps + 1)
particle_num_t = np.zeros((len(t), L))

number_op = np.array([[0, 0], [0, 1]])
number_op.reshape((1, 2, 2, 1))
number_mpo = [None]
number_mpo = tt.TT(number_op)

for i in range(len(t)):
    gs_mps = canonicalize_mps(mps_t[i])
    gs_mps_dag = gs_mps.transpose(conjugate=True)
    hermit_mps = (1 / 2) * (gs_mps + gs_mps_dag)
    particle_num_t[i, :] = compute_site_expVal(hermit_mps, number_mpo)

plt.figure()
plt.imshow(particle_num_t)
plt.xticks([])
plt.colorbar()
plt.tight_layout()
plt.show()
