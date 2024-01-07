# 1D-quantum-contact-process
TODO:
- [x] Simulate QCP-1D in QuTiP using WFMC, ODE solver
- [x] Consult QuTiP community
- [x] Replicate Fig. 2c in https://link.aps.org/doi/10.1103/PhysRevLett.123.100604
- [] Test TenPy code baby, update normalization scheme hehe. Include logging to see change in norm, write norm test
- [] 2 time evolution algorithms: SingleSiteTDVP, TEBD (QRBasedTEBDEngine). Test out both
- [] Check out Adam's and BDF method -> What exactly is going on in `mesolve`?
- [] steady-state solution
- [] update `n(t)` for v5.0.0a2
- [] Check whether the decay rates are smaller than the minium energy splitting in the system Hamiltonian -> approximations for the validity of Lindblad Master equation (https://qutip.org/docs/latest/guide/dynamics/dynamics-master.html)