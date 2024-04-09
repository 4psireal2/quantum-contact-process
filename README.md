# quantum-contact-process-1D
## TODO (theory):
- [] (*TenPy*) 2 time evolution algorithms: SingleSiteTDVP and TEBD (SVDBasedTEBDEngine + QRBasedTEBDEngine). Test both
- [] (*TenPy*) WFMC implementation for [higher order](https://www.sciencedirect.com/science/article/pii/S0010465512000835?via%3Dihub)
- [] (*TenPy*) Reproduce density plot in Hendrik's paper
- [] (*scikit_tt*) TEBD, TDVP
- [] (*scikit_tt*) ALS gave error for eigenspectrum computation for L=15
- [] (*QuTip*) check what can be computed for dynamical simulation
- [] (*theory*) Check whether the decay rates are smaller than the minium energy splitting in the system Hamiltonian -> approximations for the [validity](https://qutip.org/docs/latest/guide/dynamics/dynamics-master.html) of Lindblad Master equation
- [] (*theory*) Check Jen's [approach](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.237201), LDPO?
- [] (*theory) Dynamics of entanglement spectrum
- [] (*technicality*) Putting code also at group's repo `itp0.physik.tu-berlin.de/home/agweimer`? (Check [wiki](https://www3.itp.tu-berlin.de/dokuwiki/agweimer:start))
- [] (*artist*) draw tensor network diagrams hehe

## TODO (simulations):
- Finite system: Half-chain correlation, purity of state, check for ||Ax - \lambda x|| < eps
- Thermodynamic limits for critical exponents-> VUMPS?
- OBSERVATION: convergence for E doesn't require large bond dimensions, but for other observables alr!
- scikit-tt norm needs to be squared
- Write tests for my functions

## Technical details 
Required tools: `poetry`

### Do this once (create and activate the environment)

```
make init
```

### Do this happily ever after (Update the environment)

```
# Add new dependencies (from PyPI) and re-generate poetry.lock
poetry add --lock PACKAGE_NAME
# Update the Poetry virtual environment based on the lock file
poetry install
# To update lock file based on pyproject.toml
poetry lock --no-update
```

### HPC-access
```
ssh  username@sheldon.physik.fu-berlin.de
```