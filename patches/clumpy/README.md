# CLUMPY patches for Auriga production

These patches target the CLUMPY checkout at upstream commit:

```text
101aff8ecd2da20a71f7eb83fb6be2f8d4e99679
```

Apply them in this order:

1. `clumpy_halo_rendered_log.patch`
2. `clumpy_external_list_physical_boundary.patch`

## Patch purposes

`clumpy_halo_rendered_log.patch` is the patch used by the RAW-to-CORRECTED
Auriga workflow. It records the J-factor actually deposited by every external
list halo in `*.halo_rendered.log`. It also contains the small renderer-edge
and interpolation-index changes present in, and validated with, the Lattes
production setup; it must not be described or treated as a logging-only patch.

The verified SHA-256 of this patch is:

```text
93b01e90745e1f10423bb48bfa5ab4e8dcbc7da6d76d4ce2da070fbf7270a53a
```

`clumpy_external_list_physical_boundary.patch` changes only the
external-list `add_halo_in_map()` call in `gal_j2D()`. It passes a zero
background to the drawing-radius calculation so that external-list halos are
rendered to their supplied physical boundary instead of stopping when they
fall below the local smooth-Milky-Way background threshold.

The patch does not disable or remove the smooth Milky Way. The smooth component
is still calculated and written normally. It does not change `gSIM_EPS`,
`gSIM_EPS_DRAWN`, or any other `add_halo_in_map()` call.

## Check and apply

Start from a clean CLUMPY checkout at the commit above. Set
`AURIGA_PIPELINE` to this repository and run the checks immediately before
each corresponding application:

```bash
git status --short
git rev-parse HEAD

git apply --check \
  "$AURIGA_PIPELINE/patches/clumpy/clumpy_halo_rendered_log.patch"
git apply \
  "$AURIGA_PIPELINE/patches/clumpy/clumpy_halo_rendered_log.patch"

git apply --check \
  "$AURIGA_PIPELINE/patches/clumpy/clumpy_external_list_physical_boundary.patch"
git apply \
  "$AURIGA_PIPELINE/patches/clumpy/clumpy_external_list_physical_boundary.patch"
```

After applying both patches, inspect the resulting diff before building:

```bash
git diff --stat
git diff -- src/healpix_fits.cc src/janalysis.cc
```

## Build and runtime verification

For the existing source-tree CMake setup, configure with the production
installation prefix, rebuild, and install as appropriate for that checkout:

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/path/to/clumpy-install
cmake --build build
cmake --install build
```

If the build directory is already configured with the intended installation
prefix, only the last two commands are needed. Do not assume that rebuilding
`build/cli/clumpy` also updates an older installed `libclumpy.so`.

Verify the executable that production will run and the library it will load:

```bash
command -v clumpy
ldd "$(command -v clumpy)" | grep libclumpy
```

The pipeline accepts an explicit `CLUMPY_EXECUTABLE`. It validates and exports
`CLUMPY_DATA` before launching CLUMPY; for the normal source layouts it infers
`<root>/data` from either `<root>/bin/clumpy` or
`<root>/build/cli/clumpy`.

## Scientific validation

The physical-boundary patch was validated on Lattes with a post-cut sample of
20,000 resilient halos. With the existing RAW-to-CORRECTED workflow:

```text
median R_raw  = 0.950259669099
median R_corr = 0.999999999956
```

Here `R = J_rendered / J_expected`, and the unchanged correction is
`rhos_new = rhos_old / sqrt(R_raw)`. All 20,000 corrected halos were within
0.1% of unity. The smooth Milky Way remained present and unchanged; only the
external-halo rendering regions changed.
