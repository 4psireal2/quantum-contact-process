# 1D-quantum-contact-process
TODO:
- [] (*TenPy*) 2 time evolution algorithms: SingleSiteTDVP and TEBD (SVDBasedTEBDEngine + QRBasedTEBDEngine). Test both
- [] (*TenPy*) `overlap` statt `norm`
- [] (*scikit_tt*) check SLIM representation
- [] (*scikit_tt*) compute steady-state solution
- [] (*QuTip*) check what can be computed for dynamical simulation
- [] (*theory*) Check whether the decay rates are smaller than the minium energy splitting in the system Hamiltonian -> approximations for the validity of Lindblad Master equation (https://qutip.org/docs/latest/guide/dynamics/dynamics-master.html)


## Technical details 
Required tools: `conda`, `conda-lock`, `poetry`

### Do this once (create and activate the environment)

```
conda-lock install --name YOURENV linux-64.conda.lock
conda activate YOURENV
make init
```

### Do this happily ever after (Update the environment)

```
# Re-generate Conda lock file(s) based on environment.yml
conda-lock -f environment.yml -p linux-64 -k explicit --filename-template "linux-64.conda.lock"
# Update Conda packages based on re-generated lock file
conda update --file linux-64.conda.lock
# Update Poetry packages and re-generate poetry.lock
poetry update
```

#### *Bugs that are not mine ... So try at your own peril anything in []*
- `conda-lock` somehow doesn't work for all platforms [`conda-lock -f environment.yml` or `conda-lock --lockfile conda-lock.yml`] (Something about mamba and conda that don't quite go well together ... see [issue](https://github.com/conda/conda-libmamba-solver/issues/418))
