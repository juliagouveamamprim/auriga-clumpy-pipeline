# Auriga + CLUMPY Pipeline

Pipeline for generating Auriga-inspired Galactic subhalo repopulations and complete full-sky dark-matter annihilation J-factor maps.

The workflow separates extended and pointlike subhalos, renders the extended component with CLUMPY, applies a halo-by-halo normalization correction, and combines all components into a final HEALPix map.

## Overview

For each independent repopulation and hydro scenario, the pipeline:

1. Generates a repopulated subhalo catalog.
2. Stores it in a reduced HDF5 table.
3. Separates extended and pointlike subhalos.
4. Writes the extended component as a CLUMPY halo list.
5. Builds a pointlike-only HEALPix FITS map.
6. Runs CLUMPY with the original extended-halo normalizations.
7. Corrects each halo density using the J-factor rendered by CLUMPY.
8. Reruns CLUMPY using the corrected list.
9. Combines the corrected CLUMPY map with the pointlike map.
10. Optionally produces Mollweide plots of all components.

Supported hydro scenarios:

- `resilient`
- `fragile`

## Pipeline workflow

```text
Auriga repopulation
        |
        v
10-column HDF5 catalog
        |
        v
extended / pointlike separation
        |
        +------------------------------+
        |                              |
        v                              v
extended CLUMPY list           pointlike HEALPix FITS
        |                              |
        v                              |
raw CLUMPY run                         |
        |                              |
        v                              |
halo-by-halo rho_s correction          |
        |                              |
        v                              |
corrected CLUMPY run                   |
        |                              |
        +--------------+---------------+
                       |
                       v
                 total FITS
                       |
                       v
             optional HEALPix plots
```

## HDF5 catalogs

Each repopulation is stored under:

```text
outputs/repop_XXXX/
```

with one file per scenario:

```text
fullrepop_hydro_resilient.h5
fullrepop_hydro_fragile.h5
```

The main datasets are:

```text
iteration_0/data
iteration_0/halo_name
```

### Reduced catalog schema

The current HDF5 table contains ten columns:

| Column | Unit | Description |
|---|---|---|
| `Js` | GeV² cm⁻⁵ | Analytic J-factor integrated within the halo scale radius |
| `D_Earth` | kpc | Distance from the observer |
| `Vmax` | km s⁻¹ | Maximum circular velocity |
| `theta_s` | deg | Angular scale radius |
| `Cv` | dimensionless | Concentration-related quantity |
| `r_s` | kpc | NFW scale radius |
| `rho_s` | M☉ kpc⁻³ | NFW scale density |
| `Xearth` | kpc | Earth-centered Galactic Cartesian coordinate |
| `Yearth` | kpc | Earth-centered Galactic Cartesian coordinate |
| `Zearth` | kpc | Earth-centered Galactic Cartesian coordinate |

Galactocentric coordinates may still be used internally, but are not stored in the final table.

Before storage, invalid objects are removed, including Earth-engulfing and Roche-disrupted subhalos. The surviving catalog is sorted by decreasing `Js`.

## Extended and pointlike separation

The separation is performed by:

```text
scripts/prepare_subhalo_components.py
```

For the default resolution:

```text
NSIDE = 1024
```

the equivalent HEALPix pixel size is approximately `0.057258 deg`. The threshold is rounded conservatively to:

```text
theta_cut = 0.06 deg
```

The classification is:

```text
extended:  theta_s >= 0.06 deg
pointlike: theta_s <  0.06 deg
```

Both components are generated during the same chunked pass through the HDF5 catalog. This prevents overlaps, missing halos, or inconsistent cuts.

### Extended component

The extended halos are written to:

```text
outputs/clumpy/<scenario>/lists/raw/
    repop_XXXX_raw_nopointlike.txt
```

The CLUMPY list format is:

```text
Name Type l b d z Rdelta rhos rs prof #1 #2 #3
```

The NFW profile is represented as:

```text
prof = kZHAO
alpha, beta, gamma = 1, 3, 1
Rdelta = r_s
```

The HDF5 catalog stores the NFW scale density `rho_s`, whereas CLUMPY expects the density evaluated at the scale radius:

```text
rho(rs) = rho_s / 4
```

### Pointlike component

Each pointlike halo is assigned to a Galactic HEALPix pixel using its Earth-centered coordinates.

When several halos fall in the same pixel, their integrated J-factors are summed:

```text
Jpointlike(pixel) = sum_i Js_i
```

The output is:

```text
outputs/clumpy/<scenario>/pointlike/
    repop_XXXX_pointlike_nside1024.fits
```

The map uses:

```text
PIXTYPE  = HEALPIX
ORDERING = NESTED
NSIDE    = 1024
COORDSYS = G
INDXSCHM = EXPLICIT
```

It contains two HDUs:

```text
JFACTOR
    PIXEL
    Jpointlike

JFACTOR_PER_SR
    PIXEL
    Jpointlike_per_sr
```

with:

```text
Jpointlike_per_sr = Jpointlike / pixel_area_sr
```

Empty pointlike pixels are stored as zero.

## CLUMPY processing

The extended component is processed in two stages.

### Raw run

The raw run uses the original halo normalizations and writes its products under:

```text
outputs/clumpy/<scenario>/outputs/raw_clumpy/repop_XXXX/
```

The correction step requires a patched CLUMPY installation that produces:

```text
*.halo_rendered.log
```

### Halo-by-halo density correction

The script:

```text
scripts/correct_rhos_from_clumpy_raw.py
```

compares the J-factor rendered by CLUMPY with the analytic target for each halo.

For an NFW halo truncated at `r_s`:

```text
J_expected = (56 pi / 3) rho(rs)^2 r_s^3 / D^2
```

The rendered ratio is:

```text
R_raw = J_rendered / J_expected
```

Since the J-factor scales as the square of the density normalization:

```text
rho_new = rho_old / sqrt(R_raw)
```

The corrected list is written to:

```text
outputs/clumpy/<scenario>/lists/corrected/
    repop_XXXX_rhocorr.txt
```

### Corrected run

The corrected CLUMPY map is written under:

```text
outputs/clumpy/<scenario>/outputs/corrected_clumpy/repop_XXXX/
```

It contains the smooth Milky Way halo and corrected extended subhalos. The pointlike component is added in the next stage.

## Complete map

The script:

```text
scripts/combine_clumpy_pointlike.py
```

combines the corrected CLUMPY map with the pointlike map.

Because the CLUMPY `Jtot` column already contains the smooth and extended components:

```text
total = CLUMPY Jtot + pointlike
```

The smooth component is not added twice.

The final product is:

```text
outputs/clumpy/<scenario>/outputs/total/repop_XXXX/
    auriga_total_nside1024.fits
```

The integrated-J HDU contains:

```text
PIXEL
Jtot
Jsmooth
Jextended
Jpointlike
```

The per-solid-angle HDU contains:

```text
PIXEL
Jtot_per_sr
Jsmooth_per_sr
Jextended_per_sr
Jpointlike_per_sr
```

## Repository structure

```text
.
├── configs/
│   ├── input_hydro.yml
│   └── clumpy_templates/
├── LICENSE
├── LICENSES/
│   └── REPOLLO-BSD-3-Clause.txt
├── README.md
├── scripts/
│   ├── combine_clumpy_pointlike.py
│   ├── correct_rhos_from_clumpy_raw.py
│   ├── generate_clumpy_params.py
│   ├── plot_healpix_components.py
│   ├── prepare_subhalo_components.py
│   ├── run_clumpy_one_case.sh
│   └── run_repopulation.py
├── src/
│   └── repop_algorithm.py
└── tests/
    ├── test_build_table.py
    ├── test_combine_clumpy_pointlike.py
    └── test_prepare_subhalo_components.py
```

Generated catalogs, parameter files, logs, FITS maps, and plots are stored under `outputs/` and excluded from version control.

## Requirements

The Python environment requires scientific packages including:

```text
numpy
h5py
pandas
matplotlib
healpy
astropy
PyYAML
```

The complete workflow also requires:

- Bash;
- `/usr/bin/time`;
- a functional CLUMPY installation;
- the CLUMPY patch that produces `*.halo_rendered.log`.

CLUMPY is an external dependency and is not distributed with this repository.

## Usage

Run commands from the repository root.

### Generate one repopulation

```bash
python3 scripts/run_repopulation.py 0
```

This creates:

```text
outputs/repop_0000/
```

containing the resilient and fragile HDF5 catalogs.

### Configure CLUMPY

Set the CLUMPY executable or wrapper:

```bash
export CLUMPY_EXECUTABLE=/path/to/clumpy_wrapper
```

If this variable is not set, the wrapper searches for `clumpy` in the current `PATH`.

The executable must produce the patched halo-rendered log.

### Run one complete case

```bash
bash scripts/run_clumpy_one_case.sh 0 resilient
```

or:

```bash
bash scripts/run_clumpy_one_case.sh 0 fragile
```

The wrapper performs seven sequential steps:

1. Prepare the extended list and pointlike FITS.
2. Generate the raw CLUMPY parameter file.
3. Run raw CLUMPY.
4. Correct the halo density normalizations.
5. Generate the corrected CLUMPY parameter file.
6. Run corrected CLUMPY.
7. Combine the corrected and pointlike maps.

Independent repopulations may run in parallel, but the internal steps for one case must remain sequential.

### Plot the components

Plotting is separate from the production wrapper:

```bash
python3 scripts/plot_healpix_components.py 0 resilient
```

It produces Mollweide maps for the smooth, extended, pointlike, and total components, together with a four-panel comparison under:

```text
outputs/clumpy/<scenario>/plots/repop_XXXX/
```

The plotted quantity is `log10(J / sr)`.

## Tests

The repository includes lightweight synthetic tests:

```bash
python3 tests/test_build_table.py
python3 tests/test_prepare_subhalo_components.py
python3 tests/test_combine_clumpy_pointlike.py
```

They verify:

- the reduced ten-column HDF5 schema;
- catalog filtering, ordering, coordinates, and metadata;
- complementary extended/pointlike classification;
- conservation of the pointlike J-factor;
- NESTED pixel placement;
- consistency between integrated and per-sr maps;
- map combination without double counting;
- preservation of all map components.

## Upstream code and attribution

The implementation in:

```text
src/repop_algorithm.py
```

is derived from REPOLLO, originally developed by Sara Porras Bedmar.

This repository contains modifications for the Auriga + CLUMPY workflow, including HDF5 output, catalog filtering, Earth-centered coordinates, the reduced schema, extended/pointlike separation, CLUMPY integration, halo-by-halo density correction, and complete HEALPix map generation.

The original REPOLLO code is distributed under the BSD 3-Clause License. Its original copyright notice, license conditions, and disclaimer are preserved in:

```text
LICENSES/REPOLLO-BSD-3-Clause.txt
```

Upstream project:

```text
https://github.com/saraporrasbedmar/repollo
```

The modifications and additional pipeline code are distributed under the license in:

```text
LICENSE
```

## Status

Version 2.0 is under active development and validation.

The established production pipeline and its existing outputs remain separate while this portable implementation is tested.
