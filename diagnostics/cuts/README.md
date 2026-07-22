# Subhalo cut diagnostics

Catalogue-wise diagnostics used to select pointlike and extended subhalo cuts for the NSIDE=2048 all-sky templates.

## Central-pixel scan

The script `scripts/scan_central_pixel_cuts.py` builds a combined HEALPix proxy map using:

- `Js` in the source pixel for pointlike halos;
- `Jtheta` in the central pixel for extended halos.

For each pair `(pointlike_f, extended_f)`, the cuts are defined by:

    J_cut = f * J_pixel_ref

Here, `J_pixel_ref` is the maximum individual pointlike or extended central-pixel proxy.

The main diagnostic is:

    max(discarded combined map) / max(final combined map)

This follows the pixel-wise metric previously used with CLUMPY-rendered discarded maps.

The output also records the numbers of pointlike and extended halos kept after each cut.

## Conservative extended-halo envelope

The conservative `theta-s` envelope can be enabled with:

    --extended-envelope-modes theta-s

For each discarded extended halo, the same `Jtheta` contribution is
repeated in every HEALPix pixel intersected by a disc of radius
`theta_s`. This is an intentionally extreme stress test; `theta_s` is a
characteristic angular scale, not a physical outer boundary.

The central-pixel result and the `theta-s` envelope are stored as
separate pixel-wise metrics.

The envelope is not a physical rendered map. It deliberately repeats
the same `Jtheta` contribution across multiple pixels and must not be
used for integrated or cumulative J-factor quantities. All integrated
quantities in the CSV continue to use the central-pixel proxy only.

## Scope

This diagnostic does not render the radial surface-brightness profiles
of extended halos. The `theta-s` envelope provides a conservative pixel-wise bound for
choosing cut candidates before a smaller number of full CLUMPY tests.

Large HDF5 catalogues, FITS maps, temporary outputs, and full scan
results should not be committed.

## Multiple-repopulation scan

The script `scripts/scan_multiple_repops.py` runs the individual cut
diagnostic over a configurable range of repopulations and aggregates the
results.

The repopulation range is selected with:

    --repop-start 0
    --n-repops 10

This example evaluates `repop_0000` through `repop_0009`.

The scenarios can be selected independently or together:

    --scenarios fragile,resilient

Each repopulation and scenario still produces an individual CSV and log.
Existing compatible individual CSV files are reused by default. Use
`--overwrite` to recompute them.

After all scans, the driver produces:

- a combined CSV containing every individual result;
- a summary CSV grouped by scenario, NSIDE, `Jtheta` aperture, envelope
  mode, and pair of cut fractions.

For each numerical diagnostic, the summary records:

- mean;
- sample standard deviation;
- median;
- minimum;
- maximum.

For the pixel-impact ratios, the summary also records the repopulation
that produced the maximum value. The mean describes the typical
behaviour, while the maximum across repopulations should be checked when
deciding whether a cut satisfies the adopted tolerance.

A ten-repopulation production scanning
`pointlike_f, extended_f = 1e-2, 1e-3, 1e-4, 1e-5` can be launched with:

    python -u diagnostics/cuts/scripts/scan_multiple_repops.py \
        --repop-start 0 \
        --n-repops 10 \
        --scenarios fragile,resilient \
        --input-root /dados5/julia/Auriga_outputs_hdf5_v2 \
        --nside 2048 \
        --pointlike-f-values 1e-2,1e-3,1e-4,1e-5 \
        --extended-f-values 1e-2,1e-3,1e-4,1e-5 \
        --theta-aperture-deg 0.015 \
        --extended-envelope-modes theta-s

Use `--aggregate-only` to regenerate the combined and summary CSV files
without rerunning the individual scans.
