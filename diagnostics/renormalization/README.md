# Deterministic smooth-halo renormalization

The official renormalization calculation is deterministic. It computes the
expected mass inside the individual NFW scale radius of the MHD subhalo
population,

\[
M_{\rm sub,total}=\int_{0.1}^{120}\!\frac{dN}{dV_{\max}}
\left[\int M_{\rm sub}(<r_s;V_{\max},c_V)
p(\log_{10}c_V\mid V_{\max})\,d\log_{10}c_V\right]dV_{\max}.
\]

The inner expectation uses Gauss--Hermite quadrature for the Gaussian scatter
in \(\log_{10}c_V\). The outer SHVF integral uses a uniform
\(\log_{10}V_{\max}\) grid. Scientific parameters come only from
[`configs/input_hydro.yml`](../../configs/input_hydro.yml): the scenario SHVF,
the \([0.1,120]\) km/s range, the Moliné+21 concentration relation and
scatter, the NFW conversion, and the host/cosmological quantities.

Run from the repository root:

```bash
python3 scripts/renormalize_mw_profile.py both
```

The command only reports values. To save a JSON summary without changing a
template or scientific output, use an explicit path:

```bash
python3 scripts/renormalize_mw_profile.py both --output /tmp/renormalization.json
```

The calculation keeps the smooth-host NFW scale radius and outer matching
radius fixed, and rescales only \(\rho_0\). The same factor rescales
`gMW_RHOSOL`.

## Reference results

With 80 Gauss--Hermite nodes and 16,385 \(V_{\max}\) grid points:

| Scenario | \(M_{\rm sub,total}\) [\(M_\odot\)] | \(m_{\rm sub}\) | \(\rho_{0,\rm new}/\rho_{0,\rm old}\) | `gMW_RHOSOL_new` [GeV/cm³] |
|---|---:|---:|---:|---:|
| fragile | \(9.67884547\times10^9\) | 0.679114% | 0.99320886 | 0.3949787768 |
| resilient | \(1.56972493\times10^{10}\) | 1.101394% | 0.98898606 | 0.3932994559 |

These are the approved effective `gMW_RHOSOL` values in the corresponding
CLUMPY templates under `configs/clumpy_templates/`.

No Monte Carlo mass estimator is part of this repository's official
renormalization flow. The historical implementation and its diagnostic
products remain outside this repository.
