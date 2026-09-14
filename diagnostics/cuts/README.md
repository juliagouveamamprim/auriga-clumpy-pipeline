# Subhalo cut diagnostics

Catalogue-wise diagnostics used to select pointlike and extended subhalo cuts for the NSIDE=2048 all-sky templates.

## Conservative pixel-level scan

The script `scripts/scan_central_pixel_cuts.py` builds combined HEALPix
proxy maps using:

- `Js` in the source pixel for pointlike halos;
- the projected, CLUMPY-like central-pixel proxy for extended halos.

For extended halos, the proxy uses the circular integration aperture

    alpha_int = hp.max_pixrad(NSIDE)

and the exact projected NFW-squared fraction for a profile truncated at
`r_s`, followed by the CLUMPY aperture-to-pixel area rescaling. See
`../central_pixel_validation/README.md` for the complete derivation and
validation.

For each pair `(pointlike_f, extended_f)`, the cuts are defined by:

    J_cut = f * J_pixel_ref

Here, `J_pixel_ref` is the maximum individual pointlike or extended
central-pixel proxy.

The discarded pointlike contribution is assigned to the source pixel.
For each discarded extended halo, its central-pixel proxy is repeated
in every HEALPix pixel intersected by a disc of radius `theta_s`.
Contributions from overlapping halos are summed pixel by pixel.

The only pixel-level validation metric is

    max(conservative discarded proxy map) / J_pixel_ref

It is stored as
`ratio_max_discarded_theta_s_envelope_to_j_pixel_ref`. The denominator
is computed from the complete catalogue before applying any cuts, as
the maximum between the brightest pointlike `Js` and the brightest
extended central-pixel proxy.

When reading legacy CSVs without this ratio column, the aggregation and
plotting scripts derive it in memory from `j_pixel_ref` and
`max_discarded_combined_theta_s_envelope_pixel`. The input CSVs are not
modified, and non-finite or non-positive reference values are rejected.

The `theta-s` envelope is always applied. It is an intentionally extreme
stress test: `theta_s` is a characteristic angular scale, not a physical
outer boundary. The resulting map is not a physical rendered map and
must not be used for integrated or cumulative J-factor quantities.

Separately from the proxy maps, the output records the discarded
catalogue-integrated fraction

    sum(Js_discarded) / sum(Js_full)

using the original `Js` values for both pointlike and extended halos.
Neither the extended central-pixel proxy nor its `theta-s` envelope
enters this integrated quantity. The output also records the numbers of
pointlike and extended halos kept after each cut.

## Scope

This diagnostic does not render the radial surface-brightness profiles
of extended halos. The `theta-s` envelope provides a conservative pixel-wise bound for
choosing cut candidates before a smaller number of full CLUMPY tests.

Large HDF5 catalogues, FITS maps, temporary outputs, and full scan
results should not be committed.

## Relative-J cumulative distribution

The script `scripts/scan_relative_js_cumulative.py` characterizes how
faint the subhalo population is relative to the brightest object in
each catalogue. For every catalogue, it computes the cumulative
distribution of

    Js / Js,max

where `Js,max` is evaluated independently for each repopulation. The
catalogues are read in chunks and are never concatenated in memory.
The saved NPZ contains one CDF per catalogue together with the
repopulation IDs, catalogue sizes, maximum `log10(Js)`, and invalid or
out-of-range counts.

This is a descriptive population diagnostic. It motivates the need for
catalogue reduction but does not select or validate a particular value
of `f`.

Example for scanning 500 repopulations in both disruption scenarios:

    python -u diagnostics/cuts/scripts/scan_relative_js_cumulative.py \
        --input-root /path/to/Auriga_outputs_hdf5_v2 \
        --repop-start 0 \
        --n-repops 500 \
        --scenarios fragile,resilient \
        --output-npz outputs/diagnostics/cuts/relative_js_cumulative/relative_js_cumulative_0500repops.npz

The script `scripts/plot_relative_js_cumulative.py` reads the saved
NPZ without reopening the HDF5 catalogues. It plots the median CDF and
the 16th–84th percentile interval across repopulations for each
scenario. The dotted vertical lines mark the relative `Js` at which
the median CDF reaches 99%. The 50%, 90%,
and 99% thresholds are also printed to the terminal.

Example:

    python diagnostics/cuts/scripts/plot_relative_js_cumulative.py \
        --input-npz outputs/diagnostics/cuts/relative_js_cumulative/relative_js_cumulative_0500repops.npz \
        --output-dir diagnostics/cuts/plots \
        --formats png,pdf

The generated NPZ files and figures should not be committed.

## Discarded-Js distribution

The script `scripts/scan_discarded_js_distribution.py` reapplies the
pointlike and extended catalogue cuts and saves one compact histogram of
`Js / Js,max` per catalogue and population (`all`, `pointlike`, and
`extended`). `Js,max` is looked up by the exact `(scenario, repop_id)` key
in the CSV produced by `scan_catalogue_jsmax.py`; catalogues are never
combined before their relative `Js` values are calculated. The selected
row's `n_saved` must also match the HDF5 dataset row count. The default
range and resolution match the cumulative diagnostic (`-16 <= log10(j_rel)
<= 0`, 320 bins), and underflow, overflow, invalid values, totals, maxima,
and source provenance are stored explicitly. The NPZ is checkpointed
atomically after every catalogue and is resumed automatically.

    python -u diagnostics/cuts/scripts/scan_discarded_js_distribution.py \
        --input-root /path/to/Auriga_outputs_hdf5_v2 \
        --jsmax-csv outputs/diagnostics/repop_jsmax_diagnostic_all_repopulations.csv \
        --output-npz outputs/diagnostics/cuts/discarded_js_distribution.npz

The companion `scripts/plot_discarded_js_distribution.py` first sums pairs of
adjacent 0.05-dex bins by default, producing 0.10-dex steps; this can be
configured with `--rebin-factor`. It then normalizes every catalogue before
calculating the percentage of discarded subhalos in each rebinned interval
(`100 * n_bin / n_discarded`). The arithmetic mean and pointwise 16th--84th
percentiles are calculated across repopulations after this normalization.
The shaded step areas show those percentile intervals, including
zero-percentage catalogues in every bin. Individual catalogue histograms sum
to 100%; the pointwise percentile curves are not expected to. Select
`--population all`, `pointlike`, or `extended`; PNG and PDF are produced by
default.
The displayed x range is limited to
`1e-12 <= Js / Js,max^cat <= 1e-2`; this changes only the view and does not
truncate the stored histograms or their normalization.

Suggested figure caption: For each catalogue, the histogram is normalized by
the total number of discarded subhalos. Solid lines show the arithmetic mean
across the 500 repopulations of each scenario. In each bin separately, the
shaded regions delimit the 16th--84th percentile interval of the
discarded-subhalo fractions across the repopulations. The percentile bounds
are therefore pointwise summaries and do not represent individual catalogue
histograms.

    python diagnostics/cuts/scripts/plot_discarded_js_distribution.py \
        --input-npz outputs/diagnostics/cuts/discarded_js_distribution.npz \
        --output-dir diagnostics/cuts/plots \
        --population all

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
- a summary CSV grouped by scenario, NSIDE, CLUMPY integration aperture, envelope
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

Example for scanning ten repopulations with
`pointlike_f, extended_f = 1e-2, 1e-3, 1e-4, 1e-5`:

    python -u diagnostics/cuts/scripts/scan_multiple_repops.py \
        --repop-start 0 \
        --n-repops 10 \
        --scenarios fragile,resilient \
        --input-root /path/to/Auriga_outputs_hdf5_v2 \
        --nside 2048 \
        --pointlike-f-values 1e-2,1e-3,1e-4,1e-5 \
        --extended-f-values 1e-2,1e-3,1e-4,1e-5

Use `--aggregate-only` to regenerate the combined and summary CSV files
without rerunning the individual scans.

## Multiple-repopulation plots

The script `scripts/plot_multi_repop_cut_diagnostics.py` reads the
combined CSV produced by the multiple-repopulation scan and generates
one figure for each disruption scenario.

Only symmetric cuts along the diagonal

    pointlike_f = extended_f = f

are shown.

The upper panel presents the conservative combined-map impact from the
`theta-s` envelope:

    max(conservative discarded proxy map) / J_pixel_ref

Individual repopulations are shown together with the mean and the
16th–84th percentile interval. The dashed 1% reference line is retained.

The lower panel shows the mean retained catalogue fraction separately
for pointlike and extended subhalos. The annotations report the mean
numbers of kept (`K`) and discarded (`D`) objects.

Example:

    python diagnostics/cuts/scripts/plot_multi_repop_cut_diagnostics.py \
        --combined-csv multi_repop_nside2048_combined.csv \
        --output-dir plots \
        --formats png,pdf

The generated figures and the large combined scan tables should not be
committed to the repository.

## Adopted NSIDE=2048 cuts

The production configuration uses symmetric pointlike and extended cuts:

    pointlike_f = extended_f = 1e-3

Both fractions are applied to the same `J_pixel_ref` defined above.
Pointlike halos are compared using `Js`, while extended halos are compared
using the projected central-pixel proxy described above.
