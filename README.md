# 1D-quantum-contact-process
TODO:
- [x] Simulate QCP-1D in QuTiP using WFMC, ODE solver
- [x] Consult QuTiP community
- [x] Replicate Fig. 2c in https://link.aps.org/doi/10.1103/PhysRevLett.123.100604
- [x] Include logging to see change in norm
- [] 2 time evolution algorithms: SingleSiteTDVP, TEBD (QRBasedTEBDEngine). Test both
- [] Check out Adam's and BDF method -> What exactly is going on in `mesolve`?
- [] steady-state solution
- [] update `n(t)` for v5.0.0a2
- [] Check whether the decay rates are smaller than the minium energy splitting in the system Hamiltonian -> approximations for the validity of Lindblad Master equation (https://qutip.org/docs/latest/guide/dynamics/dynamics-master.html)
- [] overlap statt norm


## Technical details (Only for linux-64, osx-64, win-64 )
Required tools: `conda`, `conda-lock`, `poetry`

### Do this once (create and activate the environment)

```
conda-lock install --name YOURENV conda-lock.yml
conda activate YOURENV
make init
```

### Do this happily ever after (Update the environment)

```
# Re-generate Conda lock file(s) based on environment.yml
conda-lock -f environment.yml -p osx-64 -p linux-64 -p win-64
# Update Conda packages based on re-generated lock file
conda update --file conda-linux-64.lock
# Update Poetry packages and re-generate poetry.lock
poetry update
```

#### *Bugs that are not mine ... So try at your own peril anything in []*
- `conda-lock` somehow doesn't work for all platforms [`conda-lock -f environment.yml` or `conda-lock --lockfile conda-lock.yml`] (Something about mamba and conda that don't quite go well together ... see [issue](https://github.com/conda/conda-libmamba-solver/issues/418))
